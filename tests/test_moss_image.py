"""CPU tests use randomly initialized tiny models; no model weights are downloaded."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from random import Random
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules.dataLoader.MOSSImageBaseDataLoader import MOSSImageBaseDataLoader
from modules.model.MOSSImageModel import MOSSImageModel
from modules.modelLoader.MOSSImageModelLoader import MOSSImageLoRALoader, MOSSImageModelLoader
from modules.modelSampler.MOSSImageSampler import MOSSImageSampler
from modules.modelSaver.moss.MOSSImageLoRASaver import MOSSImageLoRASaver
from modules.modelSaver.moss.MOSSImageModelSaver import MOSSImageModelSaver
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.module.moss.pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler, MOSSImageTransformer
from modules.trainer.GenericTrainer import GenericTrainer
from modules.util import create
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import QuantizationConfig, TrainConfig
from modules.util.enum.ConceptType import ConceptType
from modules.util.enum.DataType import DataType
from modules.util.enum.DPORefMode import DPORefMode
from modules.util.enum.ModelType import ModelType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.TrainProgress import TrainProgress

import torch
from torch import nn
from torch.nn import functional as F

from diffusers import AutoencoderKLFlux2
from transformers import Qwen2Tokenizer, Qwen3_5Config, Qwen3_5Model

from PIL import Image
from safetensors.torch import load_file, save_file


class PaddingSensitiveEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.lengths = []

    @property
    def device(self):
        return self.embedding.weight.device

    def forward(self, input_ids, attention_mask, **kwargs):
        self.lengths.append(input_ids.shape[1])
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids).cumsum(1))


class TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(128)
        self.anchor = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(batch_norm_eps=1e-4)
        self.decoded = None

    @property
    def dtype(self):
        return self.anchor.dtype

    def decode(self, latent, return_dict=False):
        assert latent.dtype == torch.float32
        self.decoded = latent.clone()
        return (F.interpolate(latent[:, :3], scale_factor=8, mode="nearest").tanh(),)


def tiny_model():
    model = MOSSImageModel(ModelType.MOSS_IMAGE)
    model.transformer = MOSSImageTransformer(
        dim=32, ffn_dim=64, num_heads=4, num_layers=3, text_dim=8, freq_dim=8,
    )
    model.noise_scheduler = MOSSImageScheduler()
    model.vae = TinyVAE().eval()
    model.text_encoder = PaddingSensitiveEncoder()
    return model


def train_config():
    config = TrainConfig.default_values()
    config.model_type = ModelType.MOSS_IMAGE
    config.training_method = TrainingMethod.LORA
    config.train_device = config.temp_device = "cpu"
    config.train_dtype = config.lora_weight_dtype = DataType.FLOAT_32
    config.text_encoder.train = False
    config.text_encoder.dropout_probability = 0
    config.text_encoder.offload_fraction = config.transformer.offload_fraction = 0
    config.text_encoder.activation_offloading = config.transformer.activation_offloading = False
    config.compile = False
    config.layer_filter = "layers."
    config.lora_rank = config.lora_alpha = 2
    return config


class MOSSImageTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        self.model = tiny_model()
        self.config = train_config()
        self.setup = create.get_model_setup_class(ModelType.MOSS_IMAGE, TrainingMethod.LORA)(
            torch.device("cpu"), torch.device("cpu"), False,
        )

    def attach_lora(self):
        self.model.transformer.requires_grad_(False)
        self.model.transformer_lora = LoRAModuleWrapper(
            self.model.transformer, "transformer", self.config, ["layers."],
        )
        self.model.transformer_lora.hook_to_module()
        return list(self.model.transformer_lora.parameters())

    def batch(self):
        return {
            "latent_image": torch.randn(2, 32, 4, 4),
            "latent_image_rejected": torch.randn(2, 32, 4, 4),
            "tokens": torch.tensor([[1, 2, 0, 0], [1, 2, 3, 0]]),
            "tokens_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]),
            "text_encoder_hidden_state": torch.randn(2, 4, 8),
            "loss_weight": torch.ones(2),
            "dpo_is_paired": torch.tensor([False, True]),
            "concept_type": [ConceptType.STANDARD] * 2,
            "dpo_pair_key": ["", "moss/pair"],
        }

    def test_text_unpads_before_encoder_and_cache_matches_live(self):
        ids = torch.tensor([[0, 1, 2, 0], [1, 2, 3, 0]])
        mask = torch.tensor([[0, 1, 1, 0], [1, 1, 1, 0]])
        live, actual_mask = self.model.encode_text("cpu", tokens=ids, tokens_mask=mask)
        self.assertEqual(self.model.text_encoder.lengths, [2, 3])
        self.assertTrue((live[~mask.bool()] == 0).all())
        cached, cached_mask = self.model.encode_text(
            "cpu", tokens_mask=mask, text_encoder_output=live,
        )
        torch.testing.assert_close(live, cached)
        torch.testing.assert_close(actual_mask, cached_mask)
        direct = self.model.text_encoder(ids[:1, 1:3], torch.ones(1, 2)).last_hidden_state
        torch.testing.assert_close(live[:1, 1:3], direct)

    def test_dropout_uses_valid_empty_prompt_and_preserves_cache(self):
        hidden, mask = torch.randn(2, 4, 8), torch.ones(2, 4, dtype=torch.bool)
        original = hidden.clone()
        self.model.empty_text[4] = (torch.full((1, 4, 8), 3.0), torch.tensor([[1, 1, 0, 0]]))
        dropped, actual_mask = self.model.encode_text(
            "cpu", rand=Random(1), text_encoder_output=hidden, tokens_mask=mask,
            text_encoder_dropout_probability=1.0,
        )
        self.assertTrue((dropped == 3).all())
        self.assertEqual(actual_mask.sum().item(), 4)
        torch.testing.assert_close(hidden, original)
        self.assertTrue(mask.all())

    def test_dit_padding_cannot_change_output(self):
        x, text = torch.randn(2, 128, 2, 2), torch.randn(2, 4, 8)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        actual = self.model.transformer(x, torch.tensor([20, 700]), text, mask).sample
        text[~mask.bool()] = 1e6
        other = self.model.transformer(x, torch.tensor([20, 700]), text, mask).sample
        torch.testing.assert_close(actual, other, rtol=0, atol=0)

    def test_training_timestep_matches_noise_level(self):
        batch = self.batch()
        with patch.object(self.setup, "_get_timestep_discrete", return_value=torch.tensor([0, 999])), \
                patch.object(self.setup, "_create_noise", return_value=torch.ones(2, 128, 2, 2)), \
                patch.object(self.model.transformer, "forward", wraps=self.model.transformer.forward) as forward:
            result = self.setup.predict(self.model, batch, self.config, TrainProgress())
        args = forward.call_args.kwargs
        torch.testing.assert_close(args["timestep"], torch.tensor([1.0, 1000.0]))
        clean = self.model.scale_latents(self.model.patchify_latents(batch["latent_image"]))
        torch.testing.assert_close(args["hidden_states"][0], clean[0] * .999 + .001)
        torch.testing.assert_close(args["hidden_states"][1], torch.ones_like(clean[1]))
        torch.testing.assert_close(result["timestep"], torch.tensor([0, 999]))

    def test_checkpointed_lora_backward_and_optimizer_update(self):
        parameters = self.attach_lora()
        self.setup.setup_optimizations(self.model, self.config)
        batch = self.batch()
        optimizer = torch.optim.AdamW(parameters, lr=1e-3)
        before = [p.detach().clone() for p in parameters]
        output = self.setup.predict(self.model, batch, self.config, TrainProgress())
        loss = self.setup.calculate_loss(self.model, batch, output, self.config)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters))
        optimizer.step()
        self.assertTrue(any(not torch.equal(p, old) for p, old in zip(parameters, before, strict=True)))
        self.assertTrue(all(p.grad is None for p in self.model.transformer.parameters()))

    def test_mixed_rlhf_three_weights_have_independent_gradients(self):
        parameters = self.attach_lora()
        self.setup.setup_optimizations(self.model, self.config)
        self.config.rlhf_enabled = True
        self.config.rlhf_dpo_ref_mode = DPORefMode.NEW_ADAPTER
        self.setup.initialize_dpo_reference(self.model, self.config)
        batch = self.batch()
        trainer = object.__new__(GenericTrainer)
        trainer.model, trainer.model_setup, trainer.config = self.model, self.setup, self.config
        trainer.train_device = torch.device("cpu")
        trainer._dpo_metric_sums, trainer._dpo_timestep_sums = {}, {}
        trainer._dpo_metric_weight = 0.0
        self.setup._write_dpo_pair_csv_log = Mock()
        self.setup._write_dpo_bad_pair_csv_log = Mock()

        def gradient(weights):
            self.config.rlhf_normal_weight, self.config.rlhf_dpo_weight, self.config.rlhf_chosen_weight = weights
            with self.setup.preference_dropout_disabled(self.model):
                loss = trainer._GenericTrainer__calculate_mixed_rlhf_training_loss(batch, TrainProgress())
                grads = torch.autograd.grad(loss, parameters)
            return loss.detach(), torch.cat([g.flatten() for g in grads])

        parts = [gradient(w) for w in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
        value, grad = gradient((2, 3, 4))
        self.assertTrue(torch.isfinite(grad).all())
        torch.testing.assert_close(value, sum(w * p[0] for w, p in zip((2, 3, 4), parts, strict=True)))
        torch.testing.assert_close(grad, sum(w * p[1] for w, p in zip((2, 3, 4), parts, strict=True)), atol=1e-6, rtol=1e-4)

    def test_native_lora_formats_roundtrip(self):
        self.attach_lora()
        original = self.model.transformer_lora.state_dict()
        for fmt in ModelType.MOSS_IMAGE.supported_lora_formats():
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory() as root:
                path = str(Path(root) / "adapter.safetensors")
                MOSSImageLoRASaver().save(self.model, fmt, path, torch.float32)
                restored = MOSSImageLoRALoader()._to_canonical(self.model, load_file(path))
                self.assertEqual(set(original), set(restored))
                for key in original:
                    torch.testing.assert_close(original[key].to(torch.float32), restored[key], rtol=0, atol=0)

    def test_transformer_override_strict_and_vae_fp32(self):
        with tempfile.TemporaryDirectory() as root:
            self.model.transformer.save_config(Path(root) / "transformer")
            path = str(Path(root) / "teacher.safetensors")
            save_file(self.model.transformer.state_dict(), path)
            prefix = "modules.modelLoader.MOSSImageModelLoader."
            with patch(prefix + "Qwen2Tokenizer.from_pretrained"), \
                    patch(prefix + "Qwen3_5Model.from_pretrained"), \
                    patch(prefix + "MOSSImageScheduler.from_pretrained"), \
                    patch(prefix + "AutoencoderKLFlux2.from_pretrained") as vae:
                loaded = MOSSImageModel(ModelType.MOSS_IMAGE)
                MOSSImageModelLoader().load(
                    loaded, ModelType.MOSS_IMAGE,
                    ModelNames(base_model=root, transformer_model=path),
                    ModelWeightDtypes.from_single_dtype(DataType.FLOAT_32), QuantizationConfig.default_values(),
                )
                self.assertEqual(vae.call_args.kwargs["torch_dtype"], torch.float32)
            for key, value in self.model.transformer.state_dict().items():
                torch.testing.assert_close(value, loaded.transformer.state_dict()[key])

    def test_sampler_matches_official_pipeline(self):
        positive, negative = torch.randn(1, 3, 8), torch.randn(1, 2, 8)
        pm, nm = torch.ones(1, 3, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool)
        for cfg in (0.0, 1.0, 3.0):
            with self.subTest(cfg=cfg):
                official = MOSSImagePipeline(
                    self.model.transformer, copy.deepcopy(self.model.noise_scheduler), vae=self.model.vae,
                )
                expected = official(
                    prompt_embeds=positive, prompt_attention_mask=pm,
                    negative_prompt_embeds=negative, negative_prompt_attention_mask=nm,
                    height=32, width=32, num_inference_steps=3, guidance_scale=cfg,
                    generator=torch.Generator().manual_seed(42), uncond_skip_layers=[1],
                ).images[0]
                expected_latent = self.model.vae.decoded.clone()
                self.model.encode_text = Mock(side_effect=[(positive, pm), (negative, nm)])
                config = SampleConfig.default_values(ModelType.MOSS_IMAGE)
                config.width = config.height = 32
                config.diffusion_steps, config.cfg_scale, config.moss_uncond_skip_layers = 3, cfg, "1"
                sampler = MOSSImageSampler(torch.device("cpu"), torch.device("cpu"), self.model, ModelType.MOSS_IMAGE)
                sampler.save_sampler_output = Mock()
                sampler.sample(config, "unused")
                torch.testing.assert_close(self.model.vae.decoded, expected_latent, rtol=0, atol=0)
                self.assertEqual(sampler.save_sampler_output.call_args.args[0].data.tobytes(), expected.tobytes())

    def test_latent_patch_and_bn_roundtrip(self):
        raw = torch.randn(2, 32, 4, 6)
        self.model.vae.bn.running_mean.normal_()
        self.model.vae.bn.running_var.uniform_(.2, 2)
        decoded = self.model.unpatchify_latents(
            self.model.unscale_latents(self.model.scale_latents(self.model.patchify_latents(raw))),
        )
        torch.testing.assert_close(raw, decoded)

    def test_scheduler_grid_and_serialization(self):
        scheduler = MOSSImageScheduler()
        scheduler.set_timesteps(50)
        t = scheduler.timesteps
        self.assertEqual(t[0], 1000)
        self.assertTrue((t[1:] < t[:-1]).all())
        torch.testing.assert_close(t, t.round(), atol=1e-4, rtol=0)
        with tempfile.TemporaryDirectory() as root:
            scheduler.save_pretrained(root)
            restored = MOSSImageScheduler.from_pretrained(root)
            restored.set_timesteps(50)
            torch.testing.assert_close(scheduler.sigmas, restored.sigmas, rtol=0, atol=0)

    def test_registration_presets_and_defaults(self):
        for method in ModelType.MOSS_IMAGE.supported_training_methods():
            self.assertIsNotNone(create.get_model_setup_class(ModelType.MOSS_IMAGE, method))
        config = SampleConfig.default_values(ModelType.MOSS_IMAGE)
        self.assertEqual((config.diffusion_steps, config.cfg_scale, config.moss_uncond_skip_layers), (50, 3, "22"))
        self.assertTrue(ModelType.MOSS_IMAGE.is_flow_matching())
        for path in Path("training_presets/MOSS Image").glob("*.json"):
            preset = TrainConfig.default_values().from_dict(json.loads(path.read_text()), migrate=False)
            self.assertEqual(preset.model_type, ModelType.MOSS_IMAGE)
            self.assertEqual(preset.vae.weight_dtype, DataType.FLOAT_32)

    def test_dataloader_uses_fp32_vae_and_multires_cache(self):
        loader = object.__new__(MOSSImageBaseDataLoader)
        loader.train_device = torch.device("cpu")
        self.model.tokenizer = Mock()
        modules = loader._preparation_modules(self.config, self.model)
        encoder = next(m for m in modules if type(m).__name__ == "EncodeVAE")
        self.assertEqual(encoder.dtype, torch.float32)
        self.assertIs(encoder.autocast_contexts[0], self.model.vae_autocast_context)
        with patch.object(loader, "_cache_modules_from_names", return_value=[]) as cache:
            loader._cache_modules(self.config, self.model, self.setup)
            self.assertIn("tokens_mask", cache.call_args.kwargs["text_split_names"])
            self.assertIn("latent_image", cache.call_args.kwargs["image_split_names"])


    def real_components(self):
        self.model.text_encoder = Qwen3_5Model(Qwen3_5Config(
            text_config={
                "vocab_size": 32, "hidden_size": 8, "intermediate_size": 16,
                "num_hidden_layers": 2, "num_attention_heads": 1,
                "num_key_value_heads": 1, "head_dim": 8,
                "layer_types": ["linear_attention", "full_attention"],
                "linear_key_head_dim": 8, "linear_value_head_dim": 8,
                "linear_num_key_heads": 1, "linear_num_value_heads": 1,
                "pad_token_id": 0, "eos_token_id": 0,
                "rope_parameters": {
                    "rope_type": "default", "rope_theta": 10000,
                    "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2],
                    "mrope_interleaved": True,
                },
            },
            vision_config={
                "depth": 1, "hidden_size": 8, "intermediate_size": 16,
                "num_heads": 1, "out_hidden_size": 8, "patch_size": 2,
                "spatial_merge_size": 1, "num_position_embeddings": 16,
            },
            image_token_id=29, video_token_id=30, vision_start_token_id=28, vision_end_token_id=27,
        )).eval().requires_grad_(False)
        self.model.tokenizer = Qwen2Tokenizer(
            vocab={"<|endoftext|>": 0, "a": 1, "b": 2, "c": 3}, merges=[],
            chat_template="{% for m in messages %}a{{ m['content'] }}{% endfor %}",
        )
        self.model.vae = AutoencoderKLFlux2(
            block_out_channels=(8, 8, 8, 8), decoder_block_out_channels=(8, 8, 8, 8),
            norm_num_groups=8, layers_per_block=1, sample_size=32,
        ).eval().requires_grad_(False)

    def test_real_qwen35_encoder_cache_parity(self):
        self.real_components()
        tokens = torch.tensor([[0, 1, 2, 0], [1, 2, 3, 0]])
        mask = torch.tensor([[0, 1, 1, 0], [1, 1, 1, 0]])
        hidden, _ = self.model.encode_text("cpu", tokens=tokens, tokens_mask=mask)
        direct = self.model.text_encoder(
            input_ids=tokens[:1, 1:3], attention_mask=torch.ones(1, 2, dtype=torch.long),
            use_cache=False,
        ).last_hidden_state
        torch.testing.assert_close(hidden[:1, 1:3], direct)
        self.assertTrue(torch.isfinite(hidden).all())

    def test_full_export_and_public_loader_roundtrip_preserve_fp32_vae(self):
        self.real_components()
        expected_vae = {k: v.clone() for k, v in self.model.vae.state_dict().items()}
        with tempfile.TemporaryDirectory() as root:
            MOSSImageModelSaver().save(self.model, ModelFormat.DIFFUSERS, root, torch.bfloat16)
            loaded = MOSSImageModel(ModelType.MOSS_IMAGE)
            dtypes = ModelWeightDtypes.from_single_dtype(DataType.BFLOAT_16)
            MOSSImageModelLoader().load(
                loaded, ModelType.MOSS_IMAGE, ModelNames(base_model=root), dtypes,
                QuantizationConfig.default_values(),
            )
            for key, value in expected_vae.items():
                torch.testing.assert_close(value, loaded.vae.state_dict()[key], rtol=0, atol=0)
            for key, value in self.model.transformer.state_dict().items():
                expected = value if key == "rope.bands" else value.bfloat16()
                torch.testing.assert_close(expected, loaded.transformer.state_dict()[key], rtol=0, atol=0)
            self.assertTrue((Path(root) / "pipeline_moss_image.py").is_file())
            self.assertTrue((Path(root) / "MOSS-LICENSE").is_file())
            index = json.loads((Path(root) / "model_index.json").read_text())
            self.assertEqual(index["transformer"], ["pipeline_moss_image", "MOSSImageTransformer"])
            hidden, mask = loaded.encode_text("cpu", text="abc", text_encoder_sequence_length=8)
            self.assertTrue(torch.isfinite(hidden).all())
            self.assertGreater(mask.sum(), 0)

    def test_real_dataloader_cache_restart_and_multiple_resolutions(self):
        self.real_components()
        self.config.text_encoder_sequence_length = 8
        self.config.image_caching = self.config.text_caching = True
        self.config.aspect_ratio_bucketing = False
        self.config.resolution = "32,64"
        self.config.caching_threads = 1
        with tempfile.TemporaryDirectory() as root:
            images = Path(root) / "images"
            images.mkdir()
            self.config.cache_dir = str(Path(root) / "cache")
            concept = ConceptConfig.default_values()
            concept.path = str(images)
            concept.image_variations = concept.text_variations = 1
            self.config.concepts = [concept]
            for count in (1, 2):
                Image.new("RGB", (64, 64), "red").save(images / f"{count}.png")
                (images / f"{count}.txt").write_text("abc")
                loader = object.__new__(MOSSImageBaseDataLoader)
                loader.train_device = loader.temp_device = torch.device("cpu")
                ds = loader._create_dataset(self.config, self.model, self.setup, TrainProgress())
                try:
                    seen = set()
                    for _ in range(2):
                        ds.start_next_epoch()
                        rows = list(ds)
                        self.assertEqual(len(rows), count)
                        for row in rows:
                            self.assertEqual(row["latent_image"].shape[0], 32)
                            self.assertEqual(row["latent_image"].dtype, torch.float32)
                            self.assertTrue(torch.isfinite(row["text_encoder_hidden_state"]).all())
                            seen.add(row["latent_image"].shape[-1] * 8)
                    self.assertEqual(seen, {32, 64})
                finally:
                    ds.loading_pipeline.modules[0]._state.executor.shutdown(wait=True)

    def test_cached_restart_prepares_empty_dropout_conditioning(self):
        self.config.text_encoder.dropout_probability = .1
        self.config.text_caching = True
        self.model.prepare_empty_text = Mock()
        self.setup.setup_train_device(self.model, self.config)
        self.model.prepare_empty_text.assert_called_once_with(self.config.text_encoder_sequence_length)

    def test_unsupported_offload_rejected_before_training(self):
        self.config.transformer.offload_fraction = .5
        with self.assertRaisesRegex(ValueError, "per-sample"):
            self.setup.setup_optimizations(self.model, self.config)


if __name__ == "__main__":
    unittest.main()
