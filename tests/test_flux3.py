"""Focused FLUX.3 integration tests with tiny random weights; no model downloads."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from random import Random
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules.dataLoader.Flux3BaseDataLoader import Flux3BaseDataLoader
from modules.model.Flux3Model import Flux3Model
from modules.modelLoader.Flux3ModelLoader import Flux3LoRALoader
from modules.modelSampler.Flux3Sampler import sampling_sigmas
from modules.modelSaver.flux3.Flux3LoRASaver import Flux3LoRASaver
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.trainer.GenericTrainer import GenericTrainer
from modules.util import create
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.convert_util import convert
from modules.util.enum.ConceptType import ConceptType
from modules.util.enum.DataType import DataType
from modules.util.enum.DPORefMode import DPORefMode
from modules.util.enum.ModelType import ModelType, PeftType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.optimizer.muon_util import build_muon_adam_key_fn
from modules.util.quantization_util import replace_linear_with_quantized_layers
from modules.util.TrainProgress import TrainProgress

import torch
from torch import nn

from diffusers import FlowMatchEulerDiscreteScheduler, Flux3Transformer2DModel
from diffusers.loaders.single_file_utils import convert_flux3_transformer_checkpoint_to_diffusers

from safetensors.torch import load_file


def tiny_transformer(flow_map=False):
    return Flux3Transformer2DModel(
        in_channels=8, cond_in_channels=8, num_layers=1, num_single_layers=2,
        num_attention_heads=2, attention_head_dim=8, joint_attention_dim=64,
        timestep_channels=256, axes_dims_rope=(2, 2, 2, 2), flow_map=flow_map,
    )


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.calls = []

    @property
    def device(self):
        return self.embedding.weight.device

    def forward(self, input_ids, attention_mask, position_ids, **kwargs):
        self.calls.append((input_ids.clone(), attention_mask.clone(), position_ids.clone()))
        hidden = self.embedding(input_ids) + position_ids[0, :, :, None]
        return SimpleNamespace(hidden_states=tuple(hidden + i for i in range(33)))


class TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(8, affine=False)
        self.anchor = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(batch_norm_eps=1e-4)
        self.decoded = None

    def decode(self, latent, return_dict=False):
        self.decoded = latent.clone()
        rgb = torch.cat((latent, latent[:, :1]), dim=1)
        return (torch.nn.functional.interpolate(rgb, scale_factor=8).tanh(),)


def tiny_model():
    model = Flux3Model(ModelType.FLUX_3_KLEIN)
    model.transformer = tiny_transformer()
    model.text_encoder = FakeEncoder()
    model.vae = TinyVAE().eval()
    model.noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=1)
    return model


def train_config():
    config = TrainConfig.default_values()
    config.model_type = ModelType.FLUX_3_KLEIN
    config.training_method = TrainingMethod.LORA
    config.train_device = config.temp_device = "cpu"
    config.train_dtype = config.lora_weight_dtype = config.fallback_train_dtype = DataType.FLOAT_32
    config.text_encoder.train = False
    config.text_encoder.dropout_probability = 0
    config.text_encoder.offload_fraction = config.transformer.offload_fraction = 0
    config.text_encoder.activation_offloading = config.transformer.activation_offloading = False
    config.transformer.gradient_checkpointing = True
    config.compile = False
    config.layer_filter = "mode_blocks.image.,txt_mode_blocks.,single_transformer_blocks."
    config.lora_rank = config.lora_alpha = 2
    config.text_encoder_sequence_length = 96
    return config


class TrackingConductor:
    def __init__(self):
        self.pending = set()
        self.keep = False
        self.starts = []

    def offload_activated(self):
        return True

    def add_layer(self, *args):
        pass

    def start_forward(self, keep):
        if self.pending:
            raise AssertionError("A new forward overwrote an unfinished activation arena")
        self.keep = keep
        self.starts.append(keep)

    def before_layer(self, layer, call_id, args):
        if torch.is_grad_enabled():
            if call_id not in self.pending:
                raise AssertionError("Backward lost its saved activations")
            self.pending.remove(call_id)
        return args

    def after_layer(self, layer, call_id, args):
        if self.keep and not torch.is_grad_enabled():
            self.pending.add(call_id)


def track_offload(model):
    from modules.module.flux3_window_attention import enable_flux3_window_attention
    from modules.util.checkpointing_util import create_checkpoint
    enable_flux3_window_attention(model.transformer)
    conductor = TrackingConductor()
    lists = [model.transformer.txt_mode_blocks, model.transformer.mode_blocks["image"],
             model.transformer.single_transformer_blocks]
    index = 0
    for layers in lists:
        for i, layer in enumerate(layers):
            layers[i] = create_checkpoint(layer, torch.device("cpu"), [], conductor, True, index)
            index += 1
    model.transformer_offload_conductor = conductor
    return conductor


class Flux3Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        self.model, self.config = tiny_model(), train_config()
        self.setup = create.create_model_setup(
            ModelType.FLUX_3_KLEIN, torch.device("cpu"), torch.device("cpu"), TrainingMethod.LORA,
        )

    def batch(self):
        return {
            "latent_image": torch.randn(2, 2, 4, 4),
            "latent_image_rejected": torch.randn(2, 2, 4, 4),
            "tokens": torch.ones(2, 96, dtype=torch.long),
            "tokens_mask": torch.arange(96)[None] < torch.tensor([[5], [81]]),
            "text_encoder_hidden_state": torch.randn(2, 96, 64),
            "loss_weight": torch.ones(2),
            "concept_type": [ConceptType.STANDARD] * 2,
            "dpo_is_paired": torch.tensor([False, True]),
            "dpo_pair_key": ["", "flux3/pair"],
        }

    def attach_lora(self):
        self.model.transformer.requires_grad_(False)
        self.model.transformer_lora = LoRAModuleWrapper(
            self.model.transformer, "transformer", self.config, self.config.layer_filter.split(","),
        )
        self.model.transformer_lora.hook_to_module()
        return list(self.model.transformer_lora.parameters())

    def test_text_layers_padding_positions_cache_and_dropout(self):
        batch = self.batch()
        live, mask = self.model.encode_text("cpu", tokens=batch["tokens"], tokens_mask=batch["tokens_mask"])
        self.assertEqual([x[0].shape[-1] for x in self.model.text_encoder.calls], [80, 96])
        self.assertEqual(live.shape, (2, 96, 64))
        self.assertTrue((live[0, 80:] == 0).all())
        self.assertFalse((live[0, 5:80] == 0).all())
        for ids, valid, positions in self.model.text_encoder.calls:
            self.assertEqual(positions.shape, (4, 1, ids.shape[1]))
            self.assertTrue((positions[:, ~valid] == 1).all())
        torch.testing.assert_close(live[0, 0, 8:16] - live[0, 0, :8], torch.full((8,), 4.0))
        cached, cached_mask = self.model.encode_text("cpu", tokens_mask=mask, text_encoder_output=live)
        torch.testing.assert_close(live, cached)
        torch.testing.assert_close(mask, cached_mask)
        before = live.clone()
        self.model.empty_text[96] = (live[:1].clone(), mask[:1].clone())
        dropped, dm = self.model.encode_text(
            "cpu", tokens_mask=mask, text_encoder_output=live,
            text_encoder_dropout_probability=1, rand=Random(42),
        )
        torch.testing.assert_close(dropped[1], live[0])
        torch.testing.assert_close(dm[1], mask[0])
        torch.testing.assert_close(live, before)

    def test_variable_window_forward_and_gradients_match_individual_calls(self):
        batch = self.batch()
        x, t = torch.randn(2, 8, 2, 2), torch.tensor([.2, .7])
        text, mask = batch["text_encoder_hidden_state"], batch["tokens_mask"]
        actual = self.model.predict_velocity(x, t, text, mask)
        expected = torch.cat([
            self.model.unpack_latents(self.model.transformer(
                hidden_states=self.model.pack_latents(x[i:i+1]),
                encoder_hidden_states=text[i:i+1, :length], timestep=t[i:i+1],
                img_ids=self.model.image_ids(2, 2, "cpu"), txt_ids=self.model.text_ids(length, "cpu"),
            ).sample, 2, 2) for i, length in enumerate((80, 96))
        ])
        torch.testing.assert_close(actual, expected)
        params = tuple(self.model.transformer.parameters())
        ga = torch.autograd.grad(actual.square().mean(), params, allow_unused=True)
        ge = torch.autograd.grad(expected.square().mean(), params, allow_unused=True)
        for a, e in zip(ga, ge, strict=True):
            if a is not None:
                torch.testing.assert_close(a, e)
        self.assertTrue(any(a is not None for a in ga))

    def test_training_sigma_targets_masked_loss_and_paired_rng(self):
        self.setup.setup_optimizations(self.model, self.config)
        params = self.attach_lora()
        batch = self.batch()
        self.config.masked_training = True
        batch["latent_mask"] = torch.ones(2, 1, 4, 4)
        with patch.object(self.model, "predict_velocity", wraps=self.model.predict_velocity) as forward:
            result = self.setup.predict(self.model, batch, self.config, TrainProgress())
        torch.testing.assert_close(forward.call_args.args[1], (result["timestep"].float()+1)/1000)
        loss = self.setup.calculate_loss(self.model, batch, result, self.config)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params))
        self.setup._dpo_paired_half = 1
        batch["latent_image"][1] = batch["latent_image"][0]
        paired = self.setup.predict(self.model, batch, self.config, TrainProgress())
        torch.testing.assert_close(paired["target"][0], paired["target"][1])
        self.assertEqual(paired["timestep"][0], paired["timestep"][1])
        score = self.setup.rlhf_logp_per_sample(self.model, batch, paired, self.config)
        self.assertEqual(score.shape, (2,))

    def test_quantized_peft_backward_and_attention_filter(self):
        for dtype, peft in (
            (DataType.INT_8, PeftType.LORA), (DataType.NFLOAT_4, PeftType.LORA),
            (DataType.FLOAT_8, PeftType.LORA), (DataType.INT_W8A8, PeftType.LORA),
            (DataType.FLOAT_W8A8, PeftType.LORA), (DataType.INT_8, PeftType.LOHA),
            (DataType.INT_8, PeftType.LOKR), (DataType.INT_8, PeftType.OFT_2),
        ):
            with self.subTest(dtype=dtype, peft=peft):
                self.setUp()
                self.config.peft_type, self.config.oft_block_size = peft, 8
                replace_linear_with_quantized_layers(self.model.transformer, dtype, copy_parameters=True)
                self.setup.setup_optimizations(self.model, self.config)
                params = self.attach_lora()
                optimizer = torch.optim.AdamW(params, lr=.001)
                before = [p.detach().clone() for p in params]
                out = self.setup.predict(self.model, self.batch(), self.config, TrainProgress())
                self.assertTrue(torch.isfinite(out["predicted"]).all(), "nonfinite quantized prediction")
                out["predicted"].square().mean().backward()
                bad = [n for n, p in enumerate(params)
                       if p.grad is None or not torch.isfinite(p.grad).all()]
                self.assertEqual(bad, [])
                optimizer.step()
                self.assertTrue(any(not torch.equal(p, b) for p, b in zip(params, before, strict=True)))

    def test_compile_executes_forward_and_backward(self):
        self.config.compile = True
        original = nn.Module.compile
        def compile_eager(module, **kwargs):
            return original(module, backend="eager", **kwargs)
        with patch.object(nn.Module, "compile", compile_eager):
            self.setup.setup_optimizations(self.model, self.config)
        params = self.attach_lora()
        result = self.setup.predict(self.model, self.batch(), self.config, TrainProgress())
        result["predicted"].square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in params))

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

    def test_native_checkpoint_and_lora_formats_roundtrip(self):
        original = self.model.transformer.state_dict()
        native = convert(original, self.model.checkpoint_diffusers_to_original())
        restored = convert_flux3_transformer_checkpoint_to_diffusers(dict(native))
        self.assertEqual(set(original), set(restored))
        for key in original:
            torch.testing.assert_close(original[key], restored[key], atol=0, rtol=0)
        self.attach_lora()
        state = self.model.transformer_lora.state_dict()
        for fmt in ModelType.FLUX_3_KLEIN.supported_lora_formats():
            with self.subTest(format=fmt), tempfile.TemporaryDirectory() as root:
                dest = str(Path(root)/"lora.safetensors")
                Flux3LoRASaver().save(self.model, fmt, dest, torch.float32)
                restored = Flux3LoRALoader()._to_canonical(self.model, load_file(dest))
                self.assertEqual(set(state), set(restored))
                for key in state:
                    torch.testing.assert_close(state[key].float(), restored[key], atol=0, rtol=0)

    def test_registration_presets_sampling_grid_and_muon_groups(self):
        from modules.ui.TopBarController import TopBarController
        self.assertIn(("FLUX.3 Image Klein", ModelType.FLUX_3_KLEIN), TopBarController(self.config).get_model_types())
        for method in ModelType.FLUX_3_KLEIN.supported_training_methods():
            self.assertIsNotNone(create.create_model_loader(ModelType.FLUX_3_KLEIN, method))
            self.assertIsNotNone(create.create_model_saver(ModelType.FLUX_3_KLEIN, method))
            self.assertIsNotNone(create.get_model_setup_class(ModelType.FLUX_3_KLEIN, method))
        for path in Path("training_presets/FLUX.3 Klein").glob("*.json"):
            config = TrainConfig.default_values().from_dict(json.loads(path.read_text()), migrate=False)
            self.assertEqual(config.model_type, ModelType.FLUX_3_KLEIN)
            self.assertFalse(config.text_encoder.train)
            self.assertEqual(config.timestep_shift, 6.93)
        sample = SampleConfig.default_values(ModelType.FLUX_3_KLEIN)
        self.assertEqual((sample.diffusion_steps, sample.cfg_scale, sample.flux3_timestep_shift), (50, 4, 16))
        torch.testing.assert_close(sampling_sigmas(2), torch.tensor([1, 16/17, 0]))
        groups = build_muon_adam_key_fn(self.model, self.config)
        self.assertEqual(groups[id(self.model.transformer.mode_blocks["image"][0].attn.to_q.weight)], "muon")
        self.assertEqual(groups[id(self.model.transformer.context_embedder.weight)], "adam")

    def test_dataloader_mean_vae_and_multires_cache(self):
        loader = object.__new__(Flux3BaseDataLoader)
        loader.train_device = torch.device("cpu")
        self.model.tokenizer = Mock()
        preparation = loader._preparation_modules(self.config, self.model)
        encoder = next(m for m in preparation if type(m).__name__ == "EncodeVAE")
        self.assertEqual(encoder.dtype, torch.float32)
        self.assertEqual(next(m for m in preparation if type(m).__name__ == "SampleVAEDistribution").mode, "mean")
        with patch.object(loader, "_cache_modules_from_names", return_value=[]) as cache:
            loader._cache_modules(self.config, self.model, self.setup)
        kwargs = cache.call_args.kwargs
        self.assertIn("text_encoder_hidden_state", kwargs["text_split_names"])
        self.assertIn("flux3-", kwargs["config"].cache_dir)

    def test_mixed_rlhf_rejects_premature_fused_updates(self):
        self.config.rlhf_enabled = True
        self.config.optimizer.fused_back_pass = True
        with self.assertRaisesRegex(ValueError, "must finish accumulating"):
            self.setup.setup_optimizations(self.model, self.config)

    def test_distilled_training_guard(self):
        self.model.transformer = tiny_transformer(flow_map=True)
        with self.assertRaisesRegex(ValueError, "flow-map"):
            self.setup.setup_optimizations(self.model, self.config)


    def test_multiple_windows_keep_offload_activation_lifetimes_separate(self):
        baseline = copy.deepcopy(self.model)
        conductor = track_offload(self.model)
        batch = self.batch()
        x, t = torch.randn(2, 8, 2, 2), torch.tensor([.2, .7])
        hidden, mask = batch["text_encoder_hidden_state"], batch["tokens_mask"]
        # Multiple windows and independent chosen/normal forwards are outstanding
        # together, as in mixed RLHF, before the combined backward.
        actual = sum(self.model.predict_velocity(x+i, t, hidden, mask).square().mean() for i in (0, 1))
        expected = sum(baseline.predict_velocity(x+i, t, hidden, mask).square().mean() for i in (0, 1))
        actual.backward()
        expected.backward()
        self.assertFalse(conductor.pending)
        self.assertEqual(conductor.starts.count(True), 2)
        for a, e in zip(self.model.transformer.parameters(), baseline.transformer.parameters(), strict=True):
            if e.grad is not None:
                torch.testing.assert_close(a.grad, e.grad, atol=1e-6, rtol=1e-4)



    def test_full_finetune_updates_dit_and_keeps_encoder_vae_frozen(self):
        self.config.training_method = TrainingMethod.FINE_TUNE
        self.config.transformer.weight_dtype = DataType.FLOAT_32
        self.config.transformer.train = True
        setup = create.create_model_setup(
            ModelType.FLUX_3_KLEIN, torch.device("cpu"), torch.device("cpu"), TrainingMethod.FINE_TUNE,
        )
        setup.setup_optimizations(self.model, self.config)
        setup.setup_model(self.model, self.config)
        self.assertFalse(any(p.requires_grad for p in self.model.text_encoder.parameters()))
        self.assertFalse(any(p.requires_grad for p in self.model.vae.parameters()))
        parameters = [p for p in self.model.transformer.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=.001)
        before = [p.detach().clone() for p in parameters]
        batch = self.batch()
        result = setup.predict(self.model, batch, self.config, TrainProgress())
        setup.calculate_loss(self.model, batch, result, self.config).backward()
        self.assertTrue(any(p.grad is not None for p in parameters))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters))
        optimizer.step()
        self.assertTrue(any(not torch.equal(p, old) for p, old in zip(parameters, before, strict=True)))

    def test_offloaded_fused_step_accumulates_all_prompt_windows_once(self):
        from collections import Counter

        baseline = copy.deepcopy(self.model)
        conductor = track_offload(self.model)
        counts = Counter()
        handles = []
        for name, parameter in self.model.transformer.named_parameters():
            def step(parameter, name=name):
                counts[name] += 1
                with torch.no_grad():
                    parameter.add_(parameter.grad, alpha=-.01)
                parameter.grad = None
            handles.append(parameter.register_post_accumulate_grad_hook(step))
        batch = self.batch()
        x, t = torch.randn(2, 8, 2, 2), torch.tensor([.2, .7])
        hidden, mask = batch["text_encoder_hidden_state"], batch["tokens_mask"]
        self.model.predict_velocity(x, t, hidden, mask).square().mean().backward()
        baseline.predict_velocity(x, t, hidden, mask).square().mean().backward()
        with torch.no_grad():
            for name, parameter in baseline.transformer.named_parameters():
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-.01)
                    self.assertEqual(counts[name], 1)
                torch.testing.assert_close(self.model.transformer.get_parameter(name), parameter, atol=1e-6, rtol=1e-4)
        self.assertFalse(conductor.pending)
        self.assertEqual(conductor.starts.count(True), 1)
        for handle in handles:
            handle.remove()

    def test_flash_varlen_prefix_contract_matches_sdpa_with_reference(self):
        from diffusers.models import attention_dispatch

        batch = self.batch()
        x, t, ref = torch.randn(2, 8, 2, 2), torch.tensor([.2, .7]), torch.randn(2, 8, 2, 2)
        hidden, mask = batch["text_encoder_hidden_state"], batch["tokens_mask"]
        expected = self.model.predict_velocity(x, t, hidden, mask, reference=ref)

        def cpu_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, **kwargs):
            outputs = []
            for i in range(len(cu_seqlens_q)-1):
                qs = q[int(cu_seqlens_q[i]):int(cu_seqlens_q[i+1])].transpose(0, 1)
                ks = k[int(cu_seqlens_k[i]):int(cu_seqlens_k[i+1])].transpose(0, 1)
                vs = v[int(cu_seqlens_k[i]):int(cu_seqlens_k[i+1])].transpose(0, 1)
                outputs.append(torch.nn.functional.scaled_dot_product_attention(qs, ks, vs).transpose(0, 1))
            return torch.cat(outputs)

        def dispatch(q, k, v, attn_mask=None, **kwargs):
            return attention_dispatch._flash_varlen_attention(q, k, v, attn_mask=attn_mask)

        # Exercise upstream packing and the actual model mask, substituting only
        # the unavailable CUDA kernel with its exact CPU attention operation.
        with patch.object(attention_dispatch, "flash_attn_varlen_func", cpu_varlen), patch(
            "diffusers.models.transformers.transformer_flux3.dispatch_attention_fn", dispatch,
        ):
            actual = self.model.predict_velocity(x, t, hidden, mask, reference=ref)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-4)
        params = tuple(self.model.transformer.parameters())
        ga = torch.autograd.grad(actual.square().mean(), params, allow_unused=True)
        ge = torch.autograd.grad(expected.square().mean(), params, allow_unused=True)
        for a, e in zip(ga, ge, strict=True):
            if e is not None:
                torch.testing.assert_close(a, e, atol=1e-6, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
