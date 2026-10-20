"""Focused MOSS layout/conditioning tests; no model weights or GPU required."""
import copy
import hashlib
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from modules.util import create  # noqa: F401

# isort: split
from modules.dataLoader.moss.MOSSImageCache import MOSSImageCache
from modules.dataLoader.moss.MOSSLayout import MOSSMaskRotateCrop, MOSSRandomFlip, MOSSRandomRotate
from modules.dataLoader.MOSSImageBaseDataLoader import MOSSImageBaseDataLoader
from modules.model.MOSSImageModel import MOSSImageModel
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.moss_bucketing import _Source, bucket_image
from modules.util.moss_layout import (
    empty_layout,
    layout_prompt,
    parse_layout,
    read_layout_source,
    sample_layout_prompt,
    transform_layout,
)
from modules.util.TrainProgress import TrainProgress

from mgds.PipelineModule import PipelineState

import torch
from torchvision.transforms import functional as TF

from PIL import Image


def annotation(width=192, height=128, caption="a red object", box=None):
    value = empty_layout(width, height, "a red object on a white background")
    value["items"] = [{"label": "object", "caption": caption, "bbox": box or [-32, -64, 32, 0]}]
    return value


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def __call__(self, text, padding=False, truncation=False, max_length=None, return_tensors=None):
        rows = [text] if isinstance(text, str) else text
        ids = [[1] + list(value.encode("utf-8")) for value in rows]
        if return_tensors is None:
            return SimpleNamespace(input_ids=ids)
        tokens = torch.zeros((len(ids), max_length), dtype=torch.long)
        mask = torch.zeros_like(tokens)
        for i, row in enumerate(ids):
            row = row[:max_length]
            tokens[i, :len(row)] = torch.tensor(row)
            mask[i, :len(row)] = 1
        return SimpleNamespace(input_ids=tokens, attention_mask=mask)

    def batch_decode(self, tokens, **kwargs):
        return ["unused"] * len(tokens)


class TinyModel:
    tokenize = MOSSImageModel.tokenize

    def __init__(self):
        self.tokenizer = Tokenizer()
        self.text_encoder = SimpleNamespace(config=SimpleNamespace(_commit_hash="test-encoder"))
        self.vae_train_dtype = self.train_dtype = DataType.FLOAT_32
        self.vae_autocast_context = self.autocast_context = nullcontext()
        self.vae = SimpleNamespace(encode=self.encode_image, parameters=lambda: iter([torch.zeros(1)]))
        self.image_calls = 0
        self.text_calls = 0

    def encode_image(self, image):
        self.image_calls += 1
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: TF.resize(image, [image.shape[-2] // 8, image.shape[-1] // 8])))

    def encode_tokens(self, tokens, mask):
        self.text_calls += 1
        return tokens.float().unsqueeze(-1).repeat(1, 1, 4)

    def to(self, device):
        pass

    def vae_to(self, device):
        pass

    def eval(self):
        pass


class GeometryTests(unittest.TestCase):
    def test_resize_crop_removes_invisible_and_clips_partial_items(self):
        value = annotation(200, 100, box=[-40, -90, 40, -60])
        value["items"].append({"label": "text", "caption": "title", "text": "Hi", "bbox": [-20, -60, 20, 60]})
        original = copy.deepcopy(value)
        result = transform_layout(parse_layout(value), 100, 100, offset=(0, 50))
        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["items"][0]["bbox"], [-20, -50, 20, 50])
        self.assertEqual(result["items"][0]["text"], "Hi")
        self.assertEqual(value, original)
        prompt = json.loads(layout_prompt(result))
        self.assertNotIn("canvas_width", prompt)
        self.assertNotIn("_polygon", prompt["items"][0])

    def test_prompt_uses_official_integer_compact_format(self):
        value = annotation(box=[-32.5, -64.25, 31.2, 0.0])
        value["items"].append({"label": "object", "caption": "sliver", "bbox": [10, -20, 10.5, 20]})
        self.assertEqual(
            layout_prompt(parse_layout(value)),
            '{"summary":{"meta":[],"canvas":"single","caption":"a red object on a white background"},'
            '"items":[{"label":"object","caption":"a red object","bbox":[-33,-65,32,0]}],'
            '"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}',
        )

    def test_flip_and_positive_rotation_use_image_coordinates(self):
        layout = parse_layout(annotation(100, 100, box=[-40, -30, -10, -10]))
        flipped = transform_layout(layout, 100, 100, flip=True)
        self.assertEqual(flipped["items"][0]["bbox"], [-40, 10, -10, 30])
        rotated = transform_layout(layout, 100, 100, angle=90)
        for actual, expected in zip(rotated["items"][0]["bbox"], [10, -40, 30, -10], strict=True):
            self.assertAlmostEqual(actual, expected)

    def test_invalid_annotations_fail_with_specific_errors(self):
        for box in ([0, 0, 0, 5], [0, 0, float("nan"), 1], [True, 0, 1, 1]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                parse_layout(annotation(box=box))
        with self.assertRaisesRegex(ValueError, "does not match"):
            parse_layout(annotation(), 512, 512, match_image=True)
        value = annotation()
        value["bbox_format"] = "xyxy"
        with self.assertRaisesRegex(ValueError, "yxyx"):
            parse_layout(value)

    def test_official_json_without_canvas_uses_image_size(self):
        value = annotation()
        del value["canvas_width"], value["canvas_height"]
        result = parse_layout(value, 192, 128, match_image=True)
        self.assertEqual(result["canvas_width"], 192)

    def test_sidecar_priority_multiline_and_mixed_caption_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            txt = path.with_suffix(".txt")
            txt.write_text("ordinary caption\nsecond caption", encoding="utf-8")
            self.assertEqual(read_layout_source(path), ("", ""))
            txt.write_text(json.dumps(annotation(), indent=2), encoding="utf-8")
            self.assertTrue(read_layout_source(path)[0])
            sidecar = path.with_suffix(".json")
            sidecar.write_text(json.dumps(annotation(caption="json sidecar")), encoding="utf-8")
            self.assertIn("json sidecar", read_layout_source(path)[0])
            explicit = path.with_suffix(".layout.json")
            explicit.write_text(json.dumps(annotation(caption="preferred")), encoding="utf-8")
            self.assertIn("preferred", read_layout_source(path)[0])
            self.assertEqual(read_layout_source(path, "off"), ("", ""))
            explicit.write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "layout.json"):
                read_layout_source(path)
            with self.assertRaisesRegex(ValueError, "missing MOSS"):
                read_layout_source(Path(directory) / "absent.png", "required")

    def test_sampler_scales_layout_and_preserves_plain_prompts(self):
        sample = SampleConfig.default_values(ModelType.MOSS_IMAGE)
        sample.width, sample.height = 385, 257
        sample.prompt = "soft daylight"
        self.assertEqual(sample_layout_prompt(sample, 384, 256), "soft daylight")
        sample.moss_layout_json = json.dumps(annotation())
        sample.moss_layout_enabled = True
        result = json.loads(sample_layout_prompt(sample, 384, 256))
        self.assertEqual(result["items"][0]["bbox"], [-64, -128, 64, 0])
        self.assertIn("soft daylight", result["summary"]["caption"])
        restored = SampleConfig.default_values().from_dict(sample.to_dict())
        self.assertEqual(restored.moss_layout_json, sample.moss_layout_json)

    def test_json_prompt_mode_and_empty_enabled_layout(self):
        sample = SampleConfig.default_values(ModelType.MOSS_IMAGE)
        sample.width, sample.height = 192, 128
        sample.prompt = json.dumps(annotation())
        self.assertEqual(json.loads(sample_layout_prompt(sample, 192, 128))["items"], annotation()["items"])
        sample.moss_layout_enabled = True
        with self.assertRaisesRegex(ValueError, "empty"):
            sample_layout_prompt(sample, 192, 128)

    def test_enabling_editor_for_a_json_prompt_does_not_embed_json_in_caption(self):
        sample = SampleConfig.default_values(ModelType.MOSS_IMAGE)
        sample.width, sample.height = 192, 128
        sample.prompt = sample.moss_layout_json = json.dumps(annotation())
        sample.moss_layout_enabled = True
        result = json.loads(sample_layout_prompt(sample, 192, 128))
        self.assertEqual(result["summary"]["caption"], annotation()["summary"]["caption"])

    def test_layout_token_overflow_fails_but_plain_prompt_keeps_existing_truncation(self):
        model = TinyModel()
        prompt = layout_prompt(parse_layout(annotation()))
        with self.assertRaisesRegex(ValueError, "Sequence Length"):
            model.tokenize(prompt, 16)
        self.assertEqual(model.tokenize("a" * 100, 16).input_ids.shape, (1, 16))
        self.assertEqual(model.tokenize(prompt, 1024).input_ids.shape, (1, 1024))

    def test_tool_bucket_geometry_matches_native_scale_and_crop(self):
        image = Image.new("RGB", (201, 127), "red")
        result = bucket_image(image, resolution="128,256", variant=0)
        scale, crop = result["scale_resolution"], result["crop_resolution"]
        self.assertEqual(result["image"].size, (crop[-1], crop[-2]))
        self.assertTrue(all(value % 16 == 0 for value in crop))
        expected = TF.crop(TF.resize(TF.pil_to_tensor(image).float() / 255, list(scale), antialias=True),
                           *result["crop_offset"], *crop)
        torch.testing.assert_close(TF.pil_to_tensor(result["image"]).float() / 255, expected, atol=1 / 255, rtol=0)
        second = bucket_image(image, resolution="128,256", variant=1)
        self.assertGreater(second["image"].width, result["image"].width)
        fixed = bucket_image(image, resolution="160x128")
        self.assertEqual(fixed["image"].size, (160, 128))

    def test_geometric_stages_match_transformed_visible_pixels(self):
        image = torch.zeros(3, 100, 100)
        image[:, 10:40, 20:40] = 1
        value = parse_layout(annotation(100, 100, box=[-40, -30, -10, -10]))
        source = _Source({"image": image, "mask": image[:1], "moss_layout": value,
                          "enabled": False, "fixed": True, "angle": 90})
        for cls in (MOSSRandomFlip, MOSSRandomRotate):
            kwargs = {"names": ["image"], "enabled_in_name": "enabled", "fixed_enabled_in_name": "fixed"}
            if cls is MOSSRandomRotate:
                kwargs["max_angle_in_name"] = "angle"
            module = cls(**kwargs)
            result = self.run_module(source, module)
            ys, xs = torch.where(result["image"][0] > 0.5)
            actual = [ys.min().item() - 50, xs.min().item() - 50,
                      ys.max().item() + 1 - 50, xs.max().item() + 1 - 50]
            for a, b in zip(actual, result["moss_layout"]["items"][0]["bbox"], strict=True):
                self.assertAlmostEqual(a, b, places=3)

    def run_module(self, source, module):
        state = PipelineState(1)
        self.addCleanup(state.executor.shutdown, wait=True)
        pipeline = SimpleNamespace(modules=[source, module], device=torch.device("cpu"))
        source.init(pipeline, 42, 0, state)
        module.init(pipeline, 42, 1, state)
        return module.get_item(0, 0)

    def test_mask_rotate_crop_records_actual_mask_geometry(self):
        image = torch.zeros(3, 100, 120)
        image[:, 20:70, 30:80] = 1
        source = _Source({"image": image, "mask": image[:1], "enabled": True,
                          "moss_layout": parse_layout(annotation(120, 100, box=[-30, -30, 20, 20]))})
        module = MOSSMaskRotateCrop(
            mask_name="mask", additional_names=["image"], enabled_in_name="enabled",
            min_size=16, min_padding_percent=10, max_padding_percent=10, max_rotate_angle=0,
        )
        result = self.run_module(source, module)
        h, w = result["image"].shape[-2:]
        ys, xs = torch.where(result["image"][0] > 0.5)
        expected = [ys.min().item() - h / 2, xs.min().item() - w / 2,
                    ys.max().item() + 1 - h / 2, xs.max().item() + 1 - w / 2]
        self.assertEqual(result["moss_layout"]["items"][0]["bbox"], expected)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        for name in ("layout", "caption"):
            Image.new("RGB", (192, 128), "white").save(self.data / (name + ".png"))
        (self.data / "layout.layout.json").write_text(json.dumps(annotation()), encoding="utf-8")
        (self.data / "caption.txt").write_text("ordinary, blue, sky", encoding="utf-8")

    def make(self, *, image_cache=True, text_cache=True, epoch=0, index=0,
             resolution="128,256", concept=None, cache="cache", rlhf=False, batch_size=1,
             extra_concepts=(), prepend=False):
        config = TrainConfig.default_values()
        config.model_type = ModelType.MOSS_IMAGE
        config.train_device = config.temp_device = "cpu"
        config.train_dtype = config.fallback_train_dtype = DataType.FLOAT_32
        config.base_model_name = "tiny-moss"
        config.text_encoder_sequence_length = 1024
        config.text_encoder.train = False
        config.image_caching, config.text_caching = image_cache, text_cache
        config.cache_dir = str(self.root / cache)
        config.resolution = resolution
        config.batch_size, config.caching_threads = batch_size, 2
        config.aspect_ratio_bucketing = True
        config.rlhf_enabled = rlhf
        concept = concept or ConceptConfig.default_values()
        concept.path = str(self.data)
        concept.seed = 123
        concept.image.enable_crop_jitter = False
        config.concepts = [*extra_concepts, concept] if prepend else [concept, *extra_concepts]
        if rlhf:
            normal = ConceptConfig.default_values()
            normal.path, normal.seed = str(self.data), 123
            normal.image.enable_crop_jitter = False
            config.concepts.insert(0, normal)
        model = TinyModel()
        setup = SimpleNamespace(prepare_text_caching=lambda model, cfg: None)
        progress = TrainProgress()
        progress.epoch, progress.epoch_sample = epoch, index
        loader = MOSSImageBaseDataLoader(torch.device("cpu"), torch.device("cpu"), config, model, setup, progress)
        ds = loader.get_data_set()
        state = ds.loading_pipeline.modules[0]._state
        self.addCleanup(state.executor.shutdown, wait=True)
        return loader, model, config

    @staticmethod
    def epoch(loader):
        loader.get_data_set().start_next_epoch()
        return list(loader.get_data_set())

    def assert_conditioning(self, row):
        encoded = TinyModel().tokenize(row["prompt"], 1024)
        torch.testing.assert_close(row["tokens"], encoded.input_ids[0])
        torch.testing.assert_close(row["tokens_mask"], encoded.attention_mask[0])
        torch.testing.assert_close(row["text_encoder_hidden_state"][:, 0], row["tokens"].float())

    def test_real_pipeline_mixes_layouts_and_captions_across_resolutions(self):
        loader, model, _ = self.make()
        seen = set()
        for _ in range(4):
            rows = self.epoch(loader)
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assert_conditioning(row)
                if Path(row["image_path"]).stem == "layout":
                    h, w = row["crop_resolution"][-2:]
                    seen.add((h, w))
                    prompt = json.loads(row["prompt"])
                    self.assertEqual(prompt["bbox_format"], "yxyx")
                    self.assertEqual(row["latent_image"].shape[-2:], (h // 8, w // 8))
                    for y0, x0, y1, x1 in [item["bbox"] for item in prompt["items"]]:
                        self.assertTrue(-h / 2 <= y0 < y1 <= h / 2)
                        self.assertTrue(-w / 2 <= x0 < x1 <= w / 2)
                else:
                    self.assertEqual(row["prompt"], "ordinary, blue, sky")
        self.assertEqual(len(seen), 2)
        self.assertEqual(model.image_calls, 4)
        self.assertEqual(model.text_calls, 3)  # two layouts + one shared caption

    def test_all_four_cache_combinations_keep_geometry_and_conditioning_aligned(self):
        for image_cache in (False, True):
            for text_cache in (False, True):
                with self.subTest(image_cache=image_cache, text_cache=text_cache):
                    loader, _, _ = self.make(image_cache=image_cache, text_cache=text_cache,
                                              cache=f"cache-{image_cache}-{text_cache}")
                    for _ in range(2):
                        for row in self.epoch(loader):
                            self.assert_conditioning(row)

    def test_layout_and_plain_caption_collate_in_the_same_batch(self):
        loader, _, _ = self.make(resolution="128", batch_size=2)
        loader.get_data_set().start_next_epoch()
        batches = list(loader.get_data_loader())
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]["tokens"].shape, (2, 1024))
        self.assertEqual(sum(prompt.startswith("{") for prompt in batches[0]["prompt"]), 1)

    def test_balanced_image_variations_resume_mid_epoch_without_layout_drift(self):
        concept = ConceptConfig.default_values()
        concept.balancing = 2.0
        concept.image_variations = 2
        concept.text_variations = 2
        concept.image.enable_random_flip = True
        loader, _, _ = self.make(concept=concept)
        for _ in range(3):
            expected = self.epoch(loader)
        self.assertEqual(len(expected), 4)
        resumed, model, _ = self.make(concept=concept, epoch=2, index=1)
        actual = self.epoch(resumed)
        self.assertEqual([r["prompt"] for r in actual], [r["prompt"] for r in expected[1:]])
        self.assertEqual(model.image_calls, 0)
        self.assertEqual(model.text_calls, 0)
        for row in actual:
            self.assert_conditioning(row)

    def test_resume_reuses_cache_and_preserves_selected_layout(self):
        loader, _, _ = self.make()
        self.epoch(loader)
        expected = self.epoch(loader)
        resumed, model, _ = self.make(epoch=1)
        actual = self.epoch(resumed)
        self.assertEqual([r["prompt"] for r in actual], [r["prompt"] for r in expected])
        self.assertEqual(model.image_calls, 0)
        self.assertEqual(model.text_calls, 0)

    def test_layout_edit_invalidates_geometry_cache_and_conditioning(self):
        loader, _, _ = self.make(resolution="128")
        self.epoch(loader)
        path = self.data / "layout.layout.json"
        path.write_text(json.dumps(annotation(caption="edited object")), encoding="utf-8")
        resumed, model, _ = self.make(resolution="128")
        rows = self.epoch(resumed)
        self.assertTrue(any("edited object" in r["prompt"] for r in rows))
        self.assertGreater(model.image_calls, 0)
        self.assertEqual(model.text_calls, 1)  # ordinary caption still reusable

    def test_caption_edit_only_reencodes_text(self):
        loader, _, _ = self.make(resolution="128")
        self.epoch(loader)
        (self.data / "caption.txt").write_text("changed ordinary caption", encoding="utf-8")
        resumed, model, _ = self.make(resolution="128")
        self.assertIn("changed ordinary caption", [row["prompt"] for row in self.epoch(resumed)])
        self.assertEqual(model.image_calls, 0)
        self.assertEqual(model.text_calls, 1)

    def test_tag_dropout_never_damages_layout_json(self):
        concept = ConceptConfig.default_values()
        concept.text.tag_dropout_enable = True
        concept.text.tag_dropout_probability = 1.0
        concept.text.keep_tags_count = 0
        concept.text.enable_tag_shuffling = True
        concept.image.enable_fixed_flip = True
        loader, _, _ = self.make(concept=concept, resolution="128")
        for row in self.epoch(loader):
            if Path(row["image_path"]).stem == "layout":
                self.assertEqual(json.loads(row["prompt"])["items"][0]["caption"], "a red object")

    def test_dataset_growth_and_loss_weight_change_are_not_stale(self):
        loader, _, _ = self.make(resolution="128")
        self.epoch(loader)
        Image.new("RGB", (192, 128), "blue").save(self.data / "new.png")
        (self.data / "new.txt").write_text("new caption", encoding="utf-8")
        concept = ConceptConfig.default_values()
        concept.loss_weight = 0.25
        resumed, _, _ = self.make(resolution="128", concept=concept)
        rows = self.epoch(resumed)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["loss_weight"] == 0.25 for row in rows))

    def extra_concept(self):
        path = self.root / "additional"
        path.mkdir(exist_ok=True)
        Image.new("RGB", (192, 128), "blue").save(path / "extra.png")
        (path / "extra.txt").write_text("new concept caption", encoding="utf-8")
        concept = ConceptConfig.default_values()
        concept.path, concept.seed, concept.loss_weight = str(path), 456, 0.75
        concept.image.enable_crop_jitter = False
        return concept

    def test_prepend_reorder_and_remove_concept_reuse_existing_caches(self):
        concept = ConceptConfig.default_values()
        concept.loss_weight = 0.25
        concept.text.enable_tag_shuffling = True
        (self.data / "caption.txt").write_text("one, two, three\nfour, five, six", encoding="utf-8")
        original, _, _ = self.make(concept=concept, resolution="128")
        expected = {r["image_path"]: r["prompt"] for r in self.epoch(original)}
        extra = self.extra_concept()
        added, model, _ = self.make(concept=concept, resolution="128",
                                     extra_concepts=[extra], prepend=True)
        rows = self.epoch(added)
        self.assertEqual(model.image_calls, 1)
        self.assertEqual(model.text_calls, 1)
        for row in rows:
            if row["image_path"] in expected:
                self.assertEqual(row["prompt"], expected[row["image_path"]])
                self.assertEqual(row["loss_weight"], 0.25)
            else:
                self.assertEqual(row["loss_weight"], 0.75)
        for extras in ([extra], []):
            resumed, model, _ = self.make(concept=concept, resolution="128", extra_concepts=extras)
            self.epoch(resumed)
            self.assertEqual(model.image_calls, 0)
            self.assertEqual(model.text_calls, 0)

    def legacy_cache(self, loader):
        """Turn current manifests into the exact previous global-row format."""
        cache = next(m for m in loader.get_data_set().loading_pipeline.modules if isinstance(m, MOSSImageCache))
        files = {}
        for group, indices in cache.group_indices.items():
            digest = hashlib.sha256(b"moss-layout-v1")
            for index in indices:
                row = loader.layout_inventory.rows[index]
                digest.update(json.dumps([index, row["path"], row["source"], row["raw"]]).encode())
            fingerprint = digest.hexdigest()
            values = [fingerprint if name == "concept._moss_layout_fingerprint"
                      else cache._get_previous_item(0, name, indices[0])
                      for name in cache.variations_group_in_names]
            old_group = cache._hash(values)
            directory = Path(cache._cache_dir(group, 0))
            manifest = cache._manifest(str(directory))
            for item, index in zip(manifest["items"], indices, strict=True):
                for variant in item["variants"]:
                    del variant["moss_source_key"]
                    variant["moss_source_index"] = index
            manifest["cache_layout"]["concept"]["_moss_layout_fingerprint"] = fingerprint
            cache._save(manifest, str(directory / "aggregate.pt"), False)
            destination = directory.parent.parent / old_group
            if destination != directory.parent:
                directory.parent.rename(destination)
            for path in (destination / "variation-0").glob("[0-9]*.pt"):
                files[path.name + old_group] = (path.stat().st_mtime_ns, path.read_bytes())
        return files

    def test_legacy_cache_upgrade_reuses_latents_and_conditioning_after_prepend(self):
        extra = self.extra_concept()
        original, _, _ = self.make(resolution="128", extra_concepts=[extra], prepend=True)
        expected = {r["image_path"]: r["prompt"] for r in self.epoch(original)}
        payloads = self.legacy_cache(original)
        resumed, model, _ = self.make(resolution="128", extra_concepts=[extra])
        actual = {r["image_path"]: r["prompt"] for r in self.epoch(resumed)}
        self.assertEqual(actual, expected)
        self.assertEqual(model.image_calls, 0)
        self.assertEqual(model.text_calls, 0)
        current = [(p.stat().st_mtime_ns, p.read_bytes()) for p in
                   (self.root / "cache").rglob("*.pt") if p.stem.isdecimal()]
        self.assertCountEqual(current, payloads.values())

    def test_legacy_cache_with_edited_layout_is_not_reused(self):
        original, _, _ = self.make(resolution="128")
        self.epoch(original)
        self.legacy_cache(original)
        (self.data / "layout.layout.json").write_text(
            json.dumps(annotation(caption="changed annotation")), encoding="utf-8")
        resumed, model, _ = self.make(resolution="128")
        self.assertTrue(any("changed annotation" in row["prompt"] for row in self.epoch(resumed)))
        self.assertEqual(model.image_calls, 2)
        self.assertEqual(model.text_calls, 1)


    def test_mixed_dpo_keeps_layout_with_chosen_and_rejected_resolution(self):
        for folder in ("chosen", "rejected"):
            (self.data / folder).mkdir()
            Image.new("RGB", (192, 128), "red").save(self.data / folder / "pair.png")
        (self.data / "chosen" / "pair.layout.json").write_text(json.dumps(annotation()), encoding="utf-8")
        concept = ConceptConfig.default_values()
        concept.include_subdirectories = True
        concept.dpo_chosen_pattern = "chosen/{}.png"
        concept.dpo_rejected_pattern = "rejected/{}.png"
        loader, _, _ = self.make(concept=concept, rlhf=True)
        rows = self.epoch(loader)
        self.assertEqual(len(rows), 3)
        paired = [row for row in rows if row["dpo_is_paired"]]
        self.assertEqual(len(paired), 1)
        self.assertEqual(paired[0]["latent_image"].shape, paired[0]["latent_image_rejected"].shape)
        self.assertEqual(json.loads(paired[0]["prompt"])["items"][0]["caption"], "a red object")
        self.assert_conditioning(paired[0])


if __name__ == "__main__":
    unittest.main()
