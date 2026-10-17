"""Focused CPU coverage for MOSS quantization/offloading. No pretrained weights."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_moss_image as fixtures

from modules.model.MOSSImageModel import MOSSImageModel
from modules.modelLoader.MOSSImageModelLoader import MOSSImageModelLoader
from modules.modelSaver.moss.MOSSImageModelSaver import MOSSImageModelSaver
from modules.modelSetup.BaseMOSSImageSetup import BaseMOSSImageSetup
from modules.module.moss import pipeline_moss_image as moss
from modules.module.quantized.LinearInt8 import LinearInt8
from modules.module.quantized.mixin.QuantizedLinearMixin import QuantizedLinearMixin
from modules.util import create
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.ModuleFilter import ModuleFilter
from modules.util.quantization_util import (
    get_offload_tensors, get_unquantized_weight, offload_quantized,
    quantize_layers, replace_linear_with_quantized_layers,
)
from modules.util.TrainProgress import TrainProgress

import torch
from torch import nn


class MOSSFeaturesTest(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.MOSSImageTest()
        self.f.setUp()
        self.model, self.config, self.setup = self.f.model, self.f.config, self.f.setup

    def convert(self, module, dtype, quantization=None):
        replace_linear_with_quantized_layers(
            module, dtype, keep_in_fp32_modules=[], quantization=quantization,
            copy_parameters=True,
        )

    def test_batched_variable_lengths_match_single_rows_and_gradients(self):
        reference = copy.deepcopy(self.model.transformer)
        x, text = torch.randn(2, 128, 2, 2), torch.randn(2, 5, 8)
        mask = torch.tensor([[0, 1, 0, 1, 0], [1, 1, 1, 0, 0]])
        times = torch.tensor([200, 700])
        # Each reference call has one genuinely unpadded row, as upstream does.
        for skip in ([], [1], [0, 2]):
            self.model.transformer.zero_grad()
            reference.zero_grad()
            actual = self.model.transformer(x, times, text, mask, skip_layers=skip).sample
            expected = torch.cat([
                reference(x[i:i + 1], times[i:i + 1], text[i:i + 1, mask[i].bool()],
                          skip_layers=skip).sample for i in range(2)
            ])
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            actual.square().mean().backward()
            expected.square().mean().backward()
            for a, b in zip(self.model.transformer.parameters(), reference.parameters(), strict=True):
                if a.grad is not None:
                    torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-4)

    def test_quantized_checkpointed_lora_backward_and_sampling_forward(self):
        for dtype in (DataType.INT_8, DataType.NFLOAT_4, DataType.FLOAT_8,
                      DataType.INT_W8A8, DataType.FLOAT_W8A8,
                      DataType.BFLOAT_16_A8_INT, DataType.BFLOAT_16_A8_FLOAT):
            with self.subTest(dtype=dtype):
                self.setUp()
                self.config.transformer.weight_dtype = dtype
                self.convert(self.model.transformer, dtype)
                self.setup.setup_optimizations(self.model, self.config)
                parameters = self.f.attach_lora()
                self.model.transformer.to("cpu")
                optimizer = torch.optim.AdamW(parameters, lr=1e-3)
                before = [p.detach().clone() for p in parameters]
                batch = self.f.batch()
                prediction = self.setup.predict(self.model, batch, self.config, TrainProgress())
                prediction["predicted"].square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters))
                optimizer.step()
                self.assertTrue(any(not torch.equal(p, old) for p, old in zip(parameters, before, strict=True)))
                with torch.no_grad():
                    sampled = self.model.transformer(
                        torch.randn(2, 128, 2, 2), 500, batch["text_encoder_hidden_state"],
                        batch["tokens_mask"], skip_layers=[1],
                    ).sample
                self.assertTrue(torch.isfinite(sampled).all())
                self.assertTrue(all(p.grad is None for p in self.model.transformer.parameters()))

    def test_int8_recursive_moves_and_allocator_keep_scales_and_adapter_weight(self):
        model = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 16)).requires_grad_(False)
        self.convert(model, DataType.INT_8)
        quantize_layers(model, torch.device("cpu"), DataType.FLOAT_32, self.config)
        x = torch.randn(3, 32)
        expected = model(x).detach()
        for layer in model:
            self.assertIsInstance(layer, LinearInt8)
            self.assertEqual(layer.weight.dtype, torch.int8)
            weight = get_unquantized_weight(layer, torch.float32, torch.device("cpu"))
            self.assertLess(weight.abs().max(), 1)
            self.assertEqual(len(get_offload_tensors(layer)), 3)
        model.to(dtype=torch.bfloat16).to(dtype=torch.float32).to("cpu")
        for layer in model:
            offload_quantized(layer, torch.device("cpu"), allocator=lambda t: torch.empty_like(t))
            self.assertEqual(layer.state.CB.data_ptr(), layer.weight.data_ptr())
            self.assertIsNotNone(layer.state.SCB)
            self.assertEqual(layer.state.SCB.dtype, torch.float32)
        torch.testing.assert_close(expected, model(x), rtol=.02, atol=.005)
        model(x.requires_grad_()).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_attention_quantization_filter_selects_moduledict_leaves(self):
        preset = BaseMOSSImageSetup.LAYER_PRESETS["attention"]
        quant = QuantizationConfig.default_values()
        quant.layer_filter = ",".join(preset["patterns"])
        quant.layer_filter_regex = preset["regex"]
        self.convert(self.model.transformer, DataType.INT_8, quant)
        selected = [
            name for name, module in self.model.transformer.named_modules()
            if isinstance(module, QuantizedLinearMixin)
        ]
        self.assertEqual(len(selected), 3 * 4)
        self.assertTrue(all(name.rsplit(".", 1)[-1] in ("q", "k", "v", "o") for name in selected))
        filter_ = ModuleFilter(quant.layer_filter, use_regex=True)
        self.assertTrue(filter_.matches("layers.1.proj.q.weight"))
        self.assertFalse(filter_.matches("layers.1.proj.ffn_down.weight"))

    def test_svd_int8_retains_floating_residual_and_backward(self):
        quant = QuantizationConfig.default_values()
        quant.svd_dtype = DataType.FLOAT_32
        quant.svd_rank = 2
        quant.cache_dir = None
        self.convert(self.model.transformer, DataType.INT_8, quant)
        self.setup.setup_optimizations(self.model, self.config)
        parameters = self.f.attach_lora()
        self.model.transformer.to("cpu")
        result = self.setup.predict(self.model, self.f.batch(), self.config, TrainProgress())
        result["predicted"].square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters))
        layer = self.model.transformer.layers[0].proj["q"]
        self.assertEqual(layer.svd_up.dtype, torch.float32)
        self.assertEqual(layer.weight.dtype, torch.int8)

    def test_real_qwen_quantized_encode_and_portable_export(self):
        self.f.real_components()
        self.convert(self.model.text_encoder, DataType.INT_8)
        self.setup.setup_optimizations(self.model, self.config)
        self.model.text_encoder.to("cpu")
        tokens = torch.tensor([[0, 1, 2, 0], [1, 2, 3, 0]])
        mask = torch.tensor([[0, 1, 1, 0], [1, 1, 1, 0]])
        hidden, _ = self.model.encode_text("cpu", tokens=tokens, tokens_mask=mask)
        self.assertTrue(torch.isfinite(hidden).all())
        self.assertTrue((hidden[~mask.bool()] == 0).all())
        with tempfile.TemporaryDirectory() as root:
            MOSSImageModelSaver().save(self.model, fixtures.ModelFormat.DIFFUSERS, root, torch.float32)
            loaded = MOSSImageModel(ModelType.MOSS_IMAGE)
            dtypes = ModelWeightDtypes.from_single_dtype(DataType.FLOAT_32)
            MOSSImageModelLoader().load(
                loaded, ModelType.MOSS_IMAGE, ModelNames(base_model=root), dtypes,
                QuantizationConfig.default_values(),
            )
            exported, _ = loaded.encode_text("cpu", tokens=tokens, tokens_mask=mask)
            torch.testing.assert_close(hidden, exported, rtol=.04, atol=.04)
            self.assertFalse(any(isinstance(m, LinearInt8) for m in loaded.text_encoder.modules()))
            self.assertTrue((Path(root) / "model_index.json").is_file())

    def test_quantized_native_hf_override_uses_strict_prefixed_state(self):
        self.f.real_components()
        prefix = "modules.modelLoader.MOSSImageModelLoader."
        state = {"model.diffusion_model." + k: v for k, v in self.model.transformer.state_dict().items()}
        with tempfile.TemporaryDirectory() as root:
            self.model.transformer.save_config(Path(root) / "transformer")
            with patch(prefix + "load_single_file_checkpoint", return_value=state) as download, \
                    patch(prefix + "Qwen2Tokenizer.from_pretrained", return_value=self.model.tokenizer), \
                    patch(prefix + "Qwen3_5Model.from_pretrained", return_value=self.model.text_encoder), \
                    patch(prefix + "MOSSImageScheduler.from_pretrained", return_value=self.model.noise_scheduler), \
                    patch(prefix + "AutoencoderKLFlux2.from_pretrained", return_value=self.model.vae):
                loaded = MOSSImageModel(ModelType.MOSS_IMAGE)
                url = "https://huggingface.co/example/moss/blob/main/teacher.safetensors"
                dtypes = ModelWeightDtypes.from_single_dtype(DataType.FLOAT_32)
                dtypes.transformer = DataType.INT_8
                MOSSImageModelLoader().load(
                    loaded, ModelType.MOSS_IMAGE, ModelNames(base_model=root, transformer=url), dtypes,
                    QuantizationConfig.default_values(),
                )
                self.assertEqual(download.call_args.args[0], url)
                self.assertIsInstance(loaded.transformer.layers[0].proj["q"], LinearInt8)
                self.setup.setup_optimizations(loaded, self.config)
                output = loaded.transformer(
                    torch.randn(1, 128, 2, 2), 300, torch.randn(1, 2, 8),
                ).sample
                self.assertTrue(torch.isfinite(output).all())

    def test_attention_backend_dispatch_and_block_compile(self):
        observed = []
        original = moss.dispatch_attention_fn
        def dispatch(q, k, v, backend):
            observed.append((backend, q.shape[1]))
            return original(q, k, v, backend="native")
        self.model.transformer.set_attention_backend("flash")
        with patch.object(moss, "dispatch_attention_fn", side_effect=dispatch):
            self.model.transformer(
                torch.randn(2, 128, 2, 2), 400, torch.randn(2, 3, 8),
                torch.tensor([[1, 0, 0], [1, 1, 1]]),
            )
        self.assertEqual(set(x[0] for x in observed), {"flash"})
        self.assertEqual(set(x[1] for x in observed), {1, 3, 5, 7})
        self.model.transformer.set_attention_backend("native")
        compile_original = nn.Module.compile
        self.config.compile = True
        def compile_eager(module, **kwargs):
            return compile_original(module, backend="eager", **kwargs)
        with patch.object(nn.Module, "compile", compile_eager):
            self.setup.setup_optimizations(self.model, self.config)
        parameters = self.f.attach_lora()
        prediction = self.setup.predict(self.model, self.f.batch(), self.config, TrainProgress())
        prediction["predicted"].square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters))

    def test_full_finetune_rejects_quantized_trainable_base(self):
        self.config.training_method = TrainingMethod.FINE_TUNE
        self.config.transformer.weight_dtype = DataType.INT_8
        setup = create.get_model_setup_class(ModelType.MOSS_IMAGE, TrainingMethod.FINE_TUNE)(
            torch.device("cpu"), torch.device("cpu"), False,
        )
        with self.assertRaisesRegex(ValueError, "floating weights"):
            setup.setup_optimizations(self.model, self.config)


if __name__ == "__main__":
    unittest.main()
