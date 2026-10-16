"""Exercise Krea/Kroma's real shared cache assembly with small CPU inputs."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from modules.dataLoader.cache.MultiResolutionDiskCache import MultiResolutionDiskCache
from modules.dataLoader.Krea2BaseDataLoader import Krea2BaseDataLoader
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod

from mgds.MGDS import MGDS
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch

from PIL import Image
from test_multi_resolution_cache import ImageSource, LatentProbe, concept


class KreaMultiResolutionIntegrationTest(unittest.TestCase):
    def test_krea_cache_assembly_with_and_without_aspect_bucketing(self):
        for bucketed in (True, False):
            with self.subTest(bucketed=bucketed), tempfile.TemporaryDirectory() as directory:
                config = TrainConfig.default_values()
                config.model_type = ModelType.KREA_2
                config.training_method = TrainingMethod.LORA
                config.text_encoder.train = False
                config.resolution = "128,256"
                config.cache_dir = directory
                config.aspect_ratio_bucketing = bucketed
                config.image_caching = True
                config.text_caching = True
                config.rlhf_enabled = False
                loader = object.__new__(Krea2BaseDataLoader)
                loader.train_device = torch.device("cpu")
                loader.temp_device = torch.device("cpu")
                model = SimpleNamespace(to=Mock(), vae_to=Mock(), eval=Mock())
                setup = SimpleNamespace(prepare_text_caching=Mock())
                modules = loader._cache_modules(config, model, setup)
                self.assertIsInstance(modules[0], MultiResolutionDiskCache)
                source = ImageSource([concept()])
                probe = LatentProbe()
                bucketing = loader._aspect_bucketing_in(config, 64)
                output = OutputPipelineModule(["latent_image", "crop_resolution", "tokens",
                                               "text_encoder_hidden_state"])
                state = PipelineState(1)
                try:
                    ds = MGDS(torch.device("cpu"), [concept()],
                              {"target_resolution": config.resolution, "target_frames": "1"},
                              [source, bucketing, probe, modules, output], 1, state)
                    seen = set()
                    for _ in range(2):
                        ds.start_next_epoch()
                        rows = list(ds)
                        self.assertEqual(len(rows), 1)
                        seen.add(rows[0]["crop_resolution"][-1])
                        self.assertEqual(rows[0]["latent_image"].shape[1], 1)
                        torch.testing.assert_close(rows[0]["tokens"], torch.tensor([10, 20]))
                    self.assertEqual(seen, {128, 256})
                    self.assertEqual(len(probe.calls), 2)
                    self.assertTrue(list(Path(directory).glob("image/*/variation-0/0.pt")))
                    setup.prepare_text_caching.assert_called_once()
                finally:
                    state.executor.shutdown(wait=True)


class FileFeatures(PipelineModule, RandomAccessPipelineModule):
    """Tiny encodings behind the real directory enumeration and cache pipeline."""

    def get_inputs(self):
        return ["image_path"]

    def get_outputs(self):
        return ["original_resolution", "tokens", "tokens_mask", "text_encoder_hidden_state", "prompt", "image"]

    def length(self):
        return self._get_previous_length("image_path")

    def get_item(self, variation, index, requested_name=None):
        path = Path(self._get_previous_item(variation, "image_path", index)).with_suffix(".txt")
        value = int(path.read_text()) if path.is_file() else 0
        return {
            "original_resolution": (1, 16, 16),
            "image": torch.zeros(3, 1, 16, 16),
            "tokens": torch.tensor([value, value + 1000]),
            "tokens_mask": torch.ones(2, dtype=torch.bool),
            "text_encoder_hidden_state": torch.full((2, 2, 4), float(value)),
            "prompt": str(value),
        }


class DatasetCacheRestartTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.images = self.root / "images"
        self.images.mkdir()

    def write_image(self, name, caption):
        path = self.images / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16)).save(path)
        path.with_suffix(".txt").write_text(str(caption))
        return path

    def make_pipeline(self, *, initial_epoch=0, initial_index=0, resolution="64", paired=False,
                      balancing=1.0, strategy="REPEATS", seed=17):
        config = TrainConfig.default_values()
        config.model_type = ModelType.KREA_2
        config.training_method = TrainingMethod.LORA
        config.text_encoder.train = False
        config.resolution = resolution
        config.cache_dir = str(self.root / "cache")
        config.aspect_ratio_bucketing = False
        config.image_caching = config.text_caching = True
        config.rlhf_enabled = True
        value = ConceptConfig.default_values()
        value.path = str(self.images)
        value.seed = seed
        value.image_variations = value.text_variations = 1
        value.balancing = balancing
        value.balancing_strategy = strategy
        value.include_subdirectories = paired
        if paired:
            value.dpo_chosen_pattern = "chosen/{}.png"
            value.dpo_rejected_pattern = "rejected/{}.png"
        loader = object.__new__(Krea2BaseDataLoader)
        loader.train_device = loader.temp_device = torch.device("cpu")
        model = SimpleNamespace(to=Mock(), vae_to=Mock(), eval=Mock())
        setup = SimpleNamespace(prepare_text_caching=Mock())
        probe = LatentProbe()
        modules = loader._cache_modules(config, model, setup)
        state = PipelineState(2)
        self.addCleanup(state.executor.shutdown, wait=True)
        ds = MGDS(
            torch.device("cpu"), [value.to_dict()],
            {"target_resolution": resolution, "target_frames": "1"},
            [loader._enumerate_input_modules(config), FileFeatures(),
             loader._aspect_bucketing_in(config, 64), probe, modules,
             OutputPipelineModule(["image_path", "tokens", "latent_image", "crop_resolution"])],
            1, state, initial_epoch=initial_epoch, initial_epoch_sample=initial_index,
        )
        ds.start_next_epoch()
        return ds, list(ds), probe, setup

    def assert_captions_match(self, rows):
        for row in rows:
            value = int(Path(row["image_path"]).with_suffix(".txt").read_text())
            torch.testing.assert_close(row["tokens"], torch.tensor([value, value + 1000]))

    def test_restart_grows_615_to_923_and_resume_keeps_current_length(self):
        for i in range(615):
            self.write_image(f"{i:04}.png", i)
        first, rows, _, _ = self.make_pipeline()
        self.assertEqual(first.approximate_length(), 615)
        self.assertEqual(len(rows), 615)
        for i in range(615, 923):
            self.write_image(f"{i:04}.png", i)
        restarted, rows, _, _ = self.make_pipeline()
        self.assertEqual(restarted.approximate_length(), 923)
        self.assertEqual(len(rows), 923)
        self.assert_captions_match(rows)
        resumed, rows, probe, setup = self.make_pipeline(initial_epoch=2, initial_index=7)
        self.assertEqual(resumed.approximate_length(), 923)
        self.assertEqual(len(rows), 916)
        self.assertEqual(probe.calls, [])
        setup.prepare_text_caching.assert_not_called()
        self.assert_captions_match(rows)

    def test_same_count_replacement_and_reordering_rebuild_both_caches(self):
        old = self.write_image("002.png", 2)
        self.write_image("003.png", 3)
        self.make_pipeline()
        old.unlink()
        old.with_suffix(".txt").unlink()
        self.write_image("001.png", 1)
        _, rows, probe, setup = self.make_pipeline()
        self.assertEqual([Path(row["image_path"]).name for row in rows], ["001.png", "003.png"])
        self.assertEqual(len(probe.calls), 2)
        setup.prepare_text_caching.assert_called_once()
        self.assert_captions_match(rows)

    def test_caption_change_rebuilds_text_and_reuses_images(self):
        path = self.write_image("001.png", 1)
        self.make_pipeline()
        path.with_suffix(".txt").write_text("99999")
        _, rows, probe, setup = self.make_pipeline()
        self.assertEqual(probe.calls, [])
        setup.prepare_text_caching.assert_called_once()
        self.assert_captions_match(rows)

    def test_rejected_image_change_invalidates_paired_image_cache(self):
        self.write_image("chosen/001.png", 1)
        rejected = self.write_image("rejected/001.png", 2)
        self.make_pipeline(paired=True)
        Image.new("RGB", (32, 32), "red").save(rejected)
        _, rows, probe, setup = self.make_pipeline(paired=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(probe.calls), 1)
        setup.prepare_text_caching.assert_not_called()

    def test_changed_resolution_rebuilds_images_only(self):
        self.write_image("001.png", 1)
        self.make_pipeline(resolution="64")
        _, rows, probe, setup = self.make_pipeline(resolution="128")
        self.assertEqual(tuple(rows[0]["crop_resolution"]), (1, 128, 128))
        self.assertEqual(len(probe.calls), 1)
        setup.prepare_text_caching.assert_not_called()

    def test_balancing_is_recomputed_without_reencoding_unchanged_sources(self):
        self.write_image("001.png", 1)
        self.write_image("002.png", 2)
        self.make_pipeline()
        for strategy, balancing, count in [("REPEATS", 2.0, 4), ("SAMPLES", 3.0, 3)]:
            with self.subTest(strategy=strategy):
                ds, rows, probe, setup = self.make_pipeline(strategy=strategy, balancing=balancing)
                self.assertEqual(ds.approximate_length(), count)
                self.assertEqual(len(rows), count)
                self.assertEqual(probe.calls, [])
                setup.prepare_text_caching.assert_not_called()
                self.assert_captions_match(rows)

    def test_legacy_persisted_layout_cannot_override_current_sources(self):
        self.write_image("001.png", 1)
        self.make_pipeline()
        cache = self.root / "cache" / "image-rlhf-mixed-v2"
        torch.save({"layout_version": 2, "group_indices": {"obsolete": list(range(615))},
                    "group_output_samples": {"obsolete": 615}}, cache / "cache_layout.pt")
        self.write_image("002.png", 2)
        ds, rows, _, _ = self.make_pipeline()
        self.assertEqual(ds.approximate_length(), 2)
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
