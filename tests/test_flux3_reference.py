"""Numerical reference, real tiny Qwen-VL, exports and actual sampler."""
import copy
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from modules.model.Flux3Model import Flux3Model
from modules.modelLoader.Flux3ModelLoader import Flux3ModelLoader
from modules.modelSampler.Flux3Sampler import Flux3Sampler, sampling_sigmas
from modules.modelSaver.flux3.Flux3ModelSaver import Flux3ModelSaver
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.ConceptConfig import ConceptConfig
from modules.dataLoader.Flux3BaseDataLoader import Flux3BaseDataLoader
from modules.util.TrainProgress import TrainProgress
from PIL import Image
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.convert_util import convert
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.quantization_util import replace_linear_with_quantized_layers

import torch

from diffusers import AutoencoderKLFlux2
from transformers import (
    Qwen2Tokenizer,
    Qwen2VLImageProcessor,
    Qwen3VLConfig,
    Qwen3VLModel,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
)

import test_flux3 as fixtures


def real_components(model, root):
    model.text_encoder = Qwen3VLModel(Qwen3VLConfig(
        text_config={
            "vocab_size": 32, "hidden_size": 8, "intermediate_size": 16,
            "num_hidden_layers": 32, "num_attention_heads": 1, "num_key_value_heads": 1,
            "head_dim": 8, "rope_parameters": {"rope_type": "default", "mrope_section": [1, 1, 2]},
        },
        vision_config={
            "depth": 1, "hidden_size": 8, "intermediate_size": 16, "num_heads": 1,
            "patch_size": 2, "spatial_merge_size": 2, "out_hidden_size": 8,
            "deepstack_visual_indexes": [],
        },
    )).eval()
    model.vae = AutoencoderKLFlux2(
        block_out_channels=(8, 8, 8, 8), decoder_block_out_channels=(8, 8, 8, 8), latent_channels=2, norm_num_groups=8,
        layers_per_block=1, patch_size=(2, 2),
    ).eval()
    vocab = Path(root)/"vocab.json"
    merges = Path(root)/"merges.txt"
    vocab.write_text('{"a":0,"b":1,"c":2,"<|endoftext|>":3,"<|im_start|>":4,"<|im_end|>":5}')
    merges.write_text("#version: 0.2\n")
    tokenizer = Qwen2Tokenizer(vocab={"a":0,"b":1,"c":2,"<|endoftext|>":3,"<|im_start|>":4,"<|im_end|>":5}, merges=[], pad_token="<|endoftext|>")
    template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    model.processor = Qwen3VLProcessor(
        tokenizer=tokenizer, image_processor=Qwen2VLImageProcessor(),
        video_processor=Qwen3VLVideoProcessor(), chat_template=template,
    )
    model.tokenizer = tokenizer
    model.tokenizer.chat_template = template


class Flux3ReferenceTest(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.Flux3Test()
        self.f.setUp()
        self.model, self.setup, self.config = self.f.model, self.f.setup, self.f.config

    def test_bfl_reference_velocity_and_gradients_base_and_flow_map(self):
        # CI fetches this one pinned source file from BFL, never model weights.
        path = os.environ.get("FLUX3_REFERENCE_DIT")
        self.assertTrue(path and Path(path).exists(), "Set FLUX3_REFERENCE_DIT to the pinned BFL dit.py")
        spec = importlib.util.spec_from_file_location("flux3_bfl_reference", path)
        ref = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = ref
        spec.loader.exec_module(ref)
        ref.HIDDEN, ref.HEADS, ref.DEPTH, ref.DEPTH_SINGLE = 16, 2, 1, 2
        ref.AXES_DIM = (2, 2, 2, 2)
        ref.IN_CHANNELS = {"image": 8, "image_cond": 8, "text": 64}
        for flow_map in (False, True):
            for use_reference in (False, True):
                with self.subTest(flow_map=flow_map, reference=use_reference):
                    self.model.transformer = fixtures.tiny_transformer(flow_map)
                    official = ref.DiT(flow_map)
                    official.load_state_dict(convert(
                        self.model.transformer.state_dict(), self.model.checkpoint_diffusers_to_original(),
                    ), strict=True)
                    x, text = torch.randn(1, 8, 2, 2), torch.randn(1, 80, 64)
                    condition = torch.randn_like(x) if use_reference else None
                    mask, t = torch.ones(1, 80, dtype=torch.bool), torch.tensor([.7])
                    target = torch.tensor([.3]) if flow_map else None
                    actual = self.model.pack_latents(self.model.predict_velocity(
                        x, t, text, mask, target_timestep=target, reference=condition,
                    ))[0]
                    data = {
                        "image": self.model.pack_latents(x)[0], "text": text[0],
                        "image_cond": self.model.pack_latents(condition)[0] if use_reference else torch.empty(0, 8),
                    }
                    ids = {
                        "image": self.model.image_ids(2, 2, "cpu"), "text": self.model.text_ids(80, "cpu"),
                        "image_cond": self.model.image_ids(2, 2, "cpu", 0) if use_reference else torch.empty(0, 4),
                    }
                    packed = ref.Packed(
                        data, ids, {m: torch.full((len(v),), .7 if m == "image" else 0.) for m, v in data.items()},
                        {m: [(0, len(v))] for m, v in data.items()},
                    )
                    expected = official(packed, timesteps_ref=target)
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                    actual.square().mean().backward()
                    expected.square().mean().backward()
                    gradients = convert(
                        {k: p.grad for k, p in self.model.transformer.named_parameters() if p.grad is not None},
                        self.model.checkpoint_diffusers_to_original(),
                    )
                    for name, gradient in gradients.items():
                        torch.testing.assert_close(gradient, official.get_parameter(name).grad, atol=2e-6, rtol=2e-4)

    def test_real_qwen_padding_matches_reference_and_quantized_cache(self):
        with tempfile.TemporaryDirectory() as root:
            real_components(self.model, root)
            batch = self.f.batch()
            ids, mask = batch["tokens"], batch["tokens_mask"]
            baseline = self.model.encode_tokens(ids, mask)
            for i, length in enumerate((80, 96)):
                valid = mask[i:i+1, :length]
                pos = (valid.long().cumsum(-1)-1).masked_fill(~valid, 1)
                out = self.model.text_encoder(
                    input_ids=ids[i:i+1, :length], attention_mask=valid,
                    position_ids=pos[None].expand(4, -1, -1), output_hidden_states=True, use_cache=False,
                )
                reference = torch.stack([out.hidden_states[k] for k in (4,8,12,16,20,24,28,32)], 1)
                reference = reference.permute(0, 2, 1, 3).reshape(1, length, -1)
                torch.testing.assert_close(baseline[i, :length], reference[0])
            for dtype in (DataType.INT_8, DataType.NFLOAT_4, DataType.FLOAT_8):
                with self.subTest(dtype=dtype):
                    encoder = copy.deepcopy(self.model.text_encoder)
                    replace_linear_with_quantized_layers(encoder, dtype, copy_parameters=True)
                    with patch.object(self.model, "text_encoder", encoder):
                        self.setup.setup_optimizations(self.model, self.config)
                        hidden = self.model.encode_tokens(ids, mask)
                    self.assertTrue(torch.isfinite(hidden).all())
                    self.assertTrue((hidden[0, 80:] == 0).all())
                    similarity = torch.nn.functional.cosine_similarity(hidden[:, :80], baseline[:, :80], dim=-1)
                    self.assertGreater(similarity.min().item(), .85)

    def test_full_export_reload_native_override_and_quantized_encoder(self):
        with tempfile.TemporaryDirectory() as root:
            real_components(self.model, root)
            replace_linear_with_quantized_layers(self.model.text_encoder, DataType.INT_8, copy_parameters=True)
            self.setup.setup_optimizations(self.model, self.config)
            batch = self.f.batch()
            hidden = self.model.encode_tokens(batch["tokens"], batch["tokens_mask"])
            dest = str(Path(root)/"export")
            Flux3ModelSaver().save(self.model, ModelFormat.DIFFUSERS, dest, torch.float32)
            loaded = Flux3Model(ModelType.FLUX_3_KLEIN)
            dtypes = ModelWeightDtypes.from_single_dtype(DataType.FLOAT_32)
            Flux3ModelLoader().load(
                loaded, ModelType.FLUX_3_KLEIN, ModelNames(base_model=dest), dtypes,
                QuantizationConfig.default_values(),
            )
            roundtrip = loaded.encode_tokens(batch["tokens"], batch["tokens_mask"])
            torch.testing.assert_close(hidden, roundtrip, atol=.03, rtol=.04)
            for key, value in self.model.transformer.state_dict().items():
                torch.testing.assert_close(value, loaded.transformer.state_dict()[key])
            native = str(Path(root)/"model.safetensors")
            Flux3ModelSaver().save(self.model, ModelFormat.ORIGINAL_TRANSFORMER, native, torch.float32)
            overridden = Flux3Model(ModelType.FLUX_3_KLEIN)
            Flux3ModelLoader().load(
                overridden, ModelType.FLUX_3_KLEIN,
                ModelNames(base_model=dest, transformer_model=native), dtypes, QuantizationConfig.default_values(),
            )
            for key, value in loaded.transformer.state_dict().items():
                torch.testing.assert_close(value, overridden.transformer.state_dict()[key])

    def test_sampler_cfg_euler_and_flowmap_target_timesteps(self):
        self.model.train_dtype = DataType.FLOAT_32
        for flow_map, cfg in ((False, 0.), (False, 1.), (False, 4.), (True, 1.)):
            with self.subTest(flow_map=flow_map, cfg=cfg):
                self.model.transformer = fixtures.tiny_transformer(flow_map)
                positive, negative = torch.randn(1, 80, 64), torch.randn(1, 80, 64)
                mask = torch.ones(1, 80, dtype=torch.bool)
                config = SampleConfig.default_values(ModelType.FLUX_3_KLEIN)
                config.width = config.height = 32
                config.seed, config.random_seed, config.diffusion_steps, config.cfg_scale = 42, False, 3, cfg
                schedule = sampling_sigmas(3)
                latents = self.model.unpack_latents(
                    torch.randn((1, 4, 8), generator=torch.Generator().manual_seed(42)).bfloat16().float(), 2, 2,
                )
                with torch.no_grad():
                    for i in range(3):
                        target = schedule[i+1:i+2] if flow_map else None
                        c = self.model.predict_velocity(latents, schedule[i:i+1], positive, mask, target_timestep=target)
                        if cfg != 1:
                            n = self.model.predict_velocity(latents, schedule[i:i+1], negative, mask)
                            c = n + cfg*(c-n)
                        latents += (schedule[i+1]-schedule[i])*c
                expected = self.model.unpatchify_latents(self.model.unscale_latents(latents))
                self.model.encode_text = Mock(side_effect=[(positive, mask), (negative, mask)])
                sampler = Flux3Sampler(torch.device("cpu"), torch.device("cpu"), self.model, ModelType.FLUX_3_KLEIN)
                sampler.save_sampler_output = Mock()
                sampler.sample(config, "unused")
                torch.testing.assert_close(self.model.vae.decoded, expected)

    def test_real_dataloader_cache_restart_and_multiple_resolutions(self):
        self.f.config.text_encoder_sequence_length = 8
        self.f.config.image_caching = self.f.config.text_caching = True
        self.f.config.aspect_ratio_bucketing = False
        self.f.config.resolution = "32,64"
        self.f.config.caching_threads = 1
        with tempfile.TemporaryDirectory() as root:
            real_components(self.model, root)
            images = Path(root) / "images"
            images.mkdir()
            self.f.config.cache_dir = str(Path(root) / "cache")
            concept = ConceptConfig.default_values()
            concept.path = str(images)
            concept.image_variations = concept.text_variations = 1
            self.f.config.concepts = [concept]
            for count in (1, 2):
                Image.new("RGB", (64, 64), "red").save(images / f"{count}.png")
                (images / f"{count}.txt").write_text("abc")
                loader = object.__new__(Flux3BaseDataLoader)
                loader.train_device = loader.temp_device = torch.device("cpu")
                ds = loader._create_dataset(self.f.config, self.model, self.setup, TrainProgress())
                try:
                    seen = set()
                    for _ in range(2):
                        ds.start_next_epoch()
                        rows = list(ds)
                        self.assertEqual(len(rows), count)
                        for row in rows:
                            self.assertEqual(row["latent_image"].shape[0], 2)
                            self.assertEqual(row["latent_image"].dtype, torch.float32)
                            self.assertTrue(torch.isfinite(row["text_encoder_hidden_state"]).all())
                            seen.add(row["latent_image"].shape[-1] * 8)
                    self.assertEqual(seen, {32, 64})
                finally:
                    ds.loading_pipeline.modules[0]._state.executor.shutdown(wait=True)
