"""Regression tests for the preview multi-resolution cache on the real MGDS pipeline."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from modules.dataLoader.cache.AspectBucketing import AspectBucketing
from modules.dataLoader.cache.MultiResolutionDiskCache import MultiResolutionDiskCache
from modules.dataLoader.cache.MultiResolutionVariation import encode_multi_resolution_variation
from modules.dataLoader.cache.SingleAspectCalculation import SingleAspectCalculation

from mgds.MGDS import MGDS
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModules.DiskCache import DiskCache
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch


class ImageSource(PipelineModule, RandomAccessPipelineModule):
    """Small source rows; all cache, bucketing and epoch logic remains real."""

    def __init__(self, concepts, resolution=(1, 384, 384)):
        super().__init__()
        self.concepts = concepts
        self.resolution = resolution

    def length(self):
        return len(self.concepts)

    def get_inputs(self):
        return []

    def get_outputs(self):
        return ["original_resolution", "image_path", "dpo_pair_key", "concept",
                "tokens", "tokens_mask", "text_encoder_hidden_state", "prompt", "image"]

    def get_item(self, variation, index, requested_name=None):
        return {
            "original_resolution": self.resolution,
            "image_path": f"image-{index}.png",
            "dpo_pair_key": f"pair-{index}",
            "concept": self.concepts[index],
            "tokens": torch.tensor([index + 10, index + 20]),
            "tokens_mask": torch.ones(2, dtype=torch.bool),
            "text_encoder_hidden_state": torch.full((2, 2, 4), float(index)),
            "prompt": f"caption-{index}",
            "image": torch.zeros(3, *self.resolution),
        }


class LatentProbe(PipelineModule, RandomAccessPipelineModule):
    """Record every encoding request and produce tiny deterministic CPU latents."""

    def __init__(self, bad_pair=False):
        super().__init__()
        self.calls = []
        self.bad_pair = bad_pair

    def length(self):
        return self._get_previous_length("crop_resolution")

    def get_inputs(self):
        return ["crop_resolution"]

    def get_outputs(self):
        return ["latent_image", "latent_image_rejected", "latent_mask", "augmentation", "crop_offset"]

    def get_item(self, variation, index, requested_name=None):
        crop = self._get_previous_item(variation, "crop_resolution", index)
        shape = (2, *crop[:-2], crop[-2] // 8, crop[-1] // 8)
        augmentation = self._get_rand(variation, index).randint(0, 1000000)
        self.calls.append((index, variation, tuple(crop)))
        chosen = torch.full(shape, float(index + 1))
        rejected = chosen[..., :-1] if self.bad_pair else chosen + 10
        return {
            "latent_image": chosen,
            "latent_image_rejected": rejected,
            "latent_mask": torch.ones((1, *shape[1:])),
            "augmentation": augmentation,
            "crop_offset": (0, 0),
        }


def concept(seed=17, override=None, variations=1):
    return {
        "path": f"concept-{seed}",
        "seed": seed,
        "enabled": True,
        "image_variations": variations,
        "text_variations": 1,
        "balancing": 1.0,
        "balancing_strategy": "REPEATS",
        "image": {"enable_resolution_override": override is not None,
                  "resolution_override": override or "128", "enable_crop_jitter": False},
        "text": {},
        "include_subdirectories": False,
        "dpo_chosen_pattern": "",
        "dpo_rejected_pattern": "",
    }


class MultiResolutionCacheTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_pipeline(self, resolutions="128,256,384", *, concepts=None, bucketed=True,
                      initial_epoch=0, initial_index=0, directory="cache", bad_pair=False):
        concepts = concepts or [concept(), concept()]
        state = PipelineState(2)
        self.addCleanup(state.executor.shutdown, wait=True)
        source = ImageSource(concepts)
        kwargs = {
            "resolution_in_name": "original_resolution",
            "target_resolution_in_name": "settings.target_resolution",
            "enable_target_resolutions_override_in_name": "concept.image.enable_resolution_override",
            "target_resolutions_override_in_name": "concept.image.resolution_override",
            "scale_resolution_out_name": "scale_resolution",
            "crop_resolution_out_name": "crop_resolution",
            "possible_resolutions_out_name": "possible_resolutions",
            "resolution_variants_out_name": "resolution_variants",
        }
        if bucketed:
            bucket = AspectBucketing(quantization=64, target_frames_in_name="settings.target_frames",
                                     frame_dim_enabled=False, **kwargs)
        else:
            bucket = SingleAspectCalculation(**kwargs)
        probe = LatentProbe(bad_pair)
        prepared = Mock()
        split = ["latent_image", "latent_image_rejected", "latent_mask", "augmentation"]
        aggregate = ["crop_resolution", "image_path"]
        groups = ["concept.path", "concept.seed", "concept.image"]
        cache = MultiResolutionDiskCache(
            str(self.root / directory / "image"), split, aggregate,
            variations_in_name="concept.image_variations",
            balancing_in_name="concept.balancing",
            balancing_strategy_in_name="concept.balancing_strategy",
            variations_group_in_name=groups, group_enabled_in_name="concept.enabled",
            before_cache_fun=prepared,
        )
        text_cache = DiskCache(
            str(self.root / directory / "text"), ["tokens"], [],
            variations_in_name="concept.text_variations",
            balancing_in_name="concept.balancing",
            balancing_strategy_in_name="concept.balancing_strategy",
            variations_group_in_name=["concept.path", "concept.seed", "concept.text"],
            group_enabled_in_name="concept.enabled",
        )
        output = OutputPipelineModule(split + aggregate + ["tokens"])
        ds = MGDS(torch.device("cpu"), concepts,
                  {"target_resolution": resolutions, "target_frames": "1"},
                  [source, bucket, probe, cache, text_cache, output], 1, state,
                  initial_epoch=initial_epoch, initial_epoch_sample=initial_index)
        return ds, cache, probe, prepared

    def epoch(self, ds):
        ds.start_next_epoch()
        return list(ds)

    def test_every_resolution_is_cached_and_selected_without_reencoding(self):
        ds, cache, probe, prepared = self.make_pipeline()
        seen = {f"image-{index}.png": set() for index in range(2)}
        for _ in range(6):
            rows = self.epoch(ds)
            self.assertEqual(len(rows), 2)
            for row in rows:
                crop = row["crop_resolution"]
                seen[row["image_path"]].add(crop[-1])
                self.assertEqual(row["latent_image"].shape, (2, 1, crop[-2] // 8, crop[-1] // 8))
                torch.testing.assert_close(row["latent_image_rejected"], row["latent_image"] + 10)
                self.assertEqual(row["latent_mask"].shape[1:], row["latent_image"].shape[1:])
                index = int(row["image_path"].split("-")[1].split(".")[0])
                torch.testing.assert_close(row["tokens"], torch.tensor([index + 10, index + 20]))
        self.assertEqual(list(seen.values()), [{128, 256, 384}, {128, 256, 384}])
        self.assertEqual(len(probe.calls), 6)
        prepared.assert_called_once()
        payloads = list(Path(cache.cache_dir).glob("*/variation-0/[0-9]*.pt"))
        self.assertEqual(len(payloads), 2)
        for path in payloads:
            payload = torch.load(path, weights_only=False)
            self.assertEqual(payload["resolution_variants"], [128, 256, 384])
            self.assertEqual(len(payload["variants"]), 3)
            self.assertEqual(len({v["augmentation"] for v in payload["variants"]}), 1)

    def test_resume_reuses_disk_cache_and_preserves_epoch_selection(self):
        ds, _, _, _ = self.make_pipeline()
        for _ in range(6):
            expected = self.epoch(ds)
        resumed, _, probe, prepared = self.make_pipeline(initial_epoch=5, initial_index=1)
        actual = self.epoch(resumed)
        self.assertEqual(len(actual), 1)
        self.assertEqual(actual[0]["image_path"], expected[1]["image_path"])
        self.assertEqual(actual[0]["crop_resolution"], expected[1]["crop_resolution"])
        torch.testing.assert_close(actual[0]["latent_image"], expected[1]["latent_image"])
        self.assertEqual(probe.calls, [])
        prepared.assert_not_called()

    def test_concept_overrides_have_independent_resolution_sets(self):
        ds, _, probe, _ = self.make_pipeline(concepts=[concept(11, "128,256"), concept(22, "384")])
        choices = {"image-0.png": set(), "image-1.png": set()}
        for _ in range(2):
            for row in self.epoch(ds):
                choices[row["image_path"]].add(row["crop_resolution"][-1])
        self.assertEqual(choices, {"image-0.png": {128, 256}, "image-1.png": {384}})
        self.assertEqual(len(probe.calls), 3)

    def test_duplicate_resolutions_do_not_skip_a_distinct_resolution(self):
        ds, cache, probe, _ = self.make_pipeline("128,128,256", concepts=[concept()])
        seen = {self.epoch(ds)[0]["crop_resolution"][-1] for _ in range(2)}
        self.assertEqual(seen, {128, 256})
        self.assertEqual(len(probe.calls), 2)
        item = next(Path(cache.cache_dir).glob("*/variation-0/0.pt"))
        payload = torch.load(item, weights_only=False)
        self.assertEqual(payload["resolution_variants"], [128, 256])

    def test_non_bucketed_krea_frame_dimension_is_preserved(self):
        ds, _, probe, _ = self.make_pipeline("128,256", bucketed=False, concepts=[concept()])
        for _ in range(2):
            row = self.epoch(ds)[0]
            self.assertEqual(row["crop_resolution"][0], 1)
            self.assertEqual(row["latent_image"].ndim, 4)
            self.assertEqual(row["latent_image"].shape[1], 1)
        self.assertEqual(len(probe.calls), 2)

    def test_single_resolution_still_caches_once(self):
        ds, _, probe, _ = self.make_pipeline("256", concepts=[concept()])
        for _ in range(3):
            self.assertEqual(self.epoch(ds)[0]["crop_resolution"], (1, 256, 256))
        self.assertEqual(len(probe.calls), 1)

    def test_dpo_shape_mismatch_cannot_publish_complete_cache(self):
        ds, cache, _, _ = self.make_pipeline(concepts=[concept()], bad_pair=True)
        with self.assertRaisesRegex(RuntimeError, "chosen/rejected cache shape mismatch"):
            ds.start_next_epoch()
        self.assertEqual(list(Path(cache.cache_dir).glob("*/variation-*/aggregate.pt")), [])

    def test_resolution_cycles_cover_all_sizes_and_do_not_repeat_at_boundaries(self):
        for count in (2, 3, 4, 5):
            for key in ("image-0", "image-1", "pair-78"):
                order = [MultiResolutionDiskCache._selected_resolution_index(key, epoch, count)
                         for epoch in range(count * 8)]
                for start in range(0, len(order), count):
                    self.assertEqual(set(order[start:start + count]), set(range(count)))
                self.assertTrue(all(a != b for a, b in zip(order, order[1:], strict=False)))

    def test_resolution_selection_preserves_augmentation_seed(self):
        for base in (0, 1, 100):
            for resolution in range(5):
                encoded = encode_multi_resolution_variation(base, resolution)
                self.assertEqual(hash((42, 7, encoded, 3)), hash((42, 7, base, 3)))


if __name__ == "__main__":
    unittest.main()
