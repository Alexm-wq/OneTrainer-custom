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
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.module.moss.pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler, MOSSImageTransformer
from modules.trainer.GenericTrainer import GenericTrainer
from modules.util import create
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import QuantizationConfig, TrainConfig
from modules.util.enum.ConceptType import ConceptType
from modules.util.enum.DataType import DataType
from modules.util.enum.DPORefMode import DPORefMode
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.TrainProgress import TrainProgress

import torch
from torch import nn
from torch.nn import functional as F

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
                    torch.testing.assert_close(original[key], restored[key])

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

    def test_unsupported_offload_rejected_before_training(self):
        self.config.transformer.offload_fraction = .5
        with self.assertRaisesRegex(ValueError, "per-sample"):
            self.setup.setup_optimizations(self.model, self.config)


if __name__ == "__main__":
    unittest.main()
