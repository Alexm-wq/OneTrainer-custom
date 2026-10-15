"""CPU regression tests: python -m unittest discover -s tests -p test_krea2_checkpoint.py"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.model.Krea2Model import Krea2Model
from modules.modelLoader.krea2.Krea2LoRALoader import Krea2LoRALoader
from modules.modelLoader.krea2.Krea2ModelLoader import Krea2ModelLoader
from modules.modelSaver.krea2.Krea2LoRASaver import Krea2LoRASaver
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.module.quantized.LinearFp8 import LinearFp8
from modules.util.config.TrainConfig import QuantizationConfig, TrainConfig
from modules.util.convert_util import convert
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

import torch

from diffusers import AutoencoderKLQwenImage, Krea2Pipeline, Krea2Transformer2DModel

from safetensors.torch import load_file, save_file


class Krea2CheckpointTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = Krea2Model(ModelType.KREA_2)
        self.source = Krea2Transformer2DModel(
            in_channels=8, num_layers=2, attention_head_dim=8,
            num_attention_heads=2, num_key_value_heads=1, intermediate_size=32,
            timestep_embed_dim=8, text_hidden_dim=8, num_text_layers=3,
            text_num_attention_heads=2, text_num_key_value_heads=2,
            text_intermediate_size=16, num_layerwise_text_blocks=1,
            num_refiner_text_blocks=1, axes_dims_rope=(2, 2, 4),
        )
        self.source.save_config(self.root / "transformer")
        self.native = convert(self.source.state_dict(), self.model.checkpoint_diffusers_to_original())
        self.checkpoint = self.root / "kroma.safetensors"
        self.save(self.native)
        self.loader = Krea2ModelLoader()
        self.dtypes = ModelWeightDtypes.from_single_dtype(DataType.FLOAT_32)
        self.quantization = QuantizationConfig.default_values()

    def save(self, state):
        save_file({k: v.contiguous() for k, v in state.items()}, self.checkpoint)

    def load(self):
        return self.loader._load_transformer_checkpoint(
            self.model, str(self.root), str(self.checkpoint), self.dtypes, self.quantization,
        )

    def assert_weights_equal(self, actual):
        expected = self.source.state_dict()
        self.assertEqual(set(expected), set(actual.state_dict()))
        for key, value in actual.state_dict().items():
            self.assertFalse(value.is_meta, key)
            torch.testing.assert_close(value, expected[key].to(value.dtype), rtol=0, atol=0)

    def test_native_and_diffusers_checkpoints(self):
        for state in (self.native, self.source.state_dict()):
            for prefix in ("", "diffusion_model.", "model.diffusion_model.", "transformer."):
                with self.subTest(native="first.weight" in state, prefix=prefix):
                    self.save({prefix + k: v for k, v in state.items()})
                    self.assert_weights_equal(self.load())

    def test_mixed_precision_and_bfloat16_loading(self):
        # Kroma Teacher/Turbo contain BF16 linear weights and FP32 norms/modulation.
        self.save({k: v.to(torch.bfloat16) if v.ndim == 2 and k.endswith("weight") else v
                   for k, v in self.native.items()})
        self.dtypes.transformer = DataType.BFLOAT_16
        actual = self.load()
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in actual.parameters()))
        self.assert_weights_equal(actual)

    def test_missing_unexpected_and_wrong_shape_fail(self):
        bad_states = []
        missing = dict(self.native)
        missing.pop("blocks.0.mod.lin")
        bad_states.append(missing)
        bad_states.append({**self.native, "first.weight_scale": torch.ones(1)})
        bad_states.append({**self.native, "first.weight": torch.zeros(3, 3)})
        for state in bad_states:
            with self.subTest(keys=len(state)):
                self.save(state)
                with self.assertRaises(RuntimeError):
                    self.load()

    def test_override_loads_through_public_loader_without_base_transformer_weights(self):
        loader_module = "modules.modelLoader.krea2.Krea2ModelLoader"
        names = ModelNames(base_model=str(self.root), transformer_model=str(self.checkpoint))
        with patch(loader_module + ".Qwen2Tokenizer.from_pretrained"), \
                patch(loader_module + ".FlowMatchEulerDiscreteScheduler.from_pretrained"), \
                patch.object(self.loader, "_load_transformers_sub_module"), \
                patch.object(self.loader, "_load_diffusers_sub_module") as load_component:
            self.loader.load(self.model, ModelType.KREA_2, names, self.dtypes, self.quantization)
        # There is only config.json in base/transformer, never a base weight file.
        load_component.assert_called_once()
        self.assertIs(load_component.call_args.args[0], AutoencoderKLQwenImage)
        self.assert_weights_equal(self.model.transformer)

    def test_existing_diffusers_load_path(self):
        loader_module = "modules.modelLoader.krea2.Krea2ModelLoader"
        with patch(loader_module + ".Qwen2Tokenizer.from_pretrained"), \
                patch(loader_module + ".FlowMatchEulerDiscreteScheduler.from_pretrained"), \
                patch.object(self.loader, "_load_transformers_sub_module"), \
                patch.object(self.loader, "_load_diffusers_sub_module") as load_component, \
                patch.object(self.loader, "_load_transformer_checkpoint") as load_override:
            self.loader.load(self.model, ModelType.KREA_2, ModelNames(base_model=str(self.root)),
                             self.dtypes, self.quantization)
        load_override.assert_not_called()
        self.assertIs(load_component.call_args.args[0], Krea2Transformer2DModel)

    def test_huggingface_file_url_uses_diffusers_download(self):
        url = "https://huggingface.co/lodestones/Kroma/blob/main/kroma-v0.3-base.safetensors"
        # Exercise the pinned Diffusers URL resolver, with only the network download mocked.
        with patch("diffusers.loaders.single_file_utils._get_model_file",
                   return_value=str(self.checkpoint)) as download:
            actual = self.loader._load_transformer_checkpoint(
                self.model, str(self.root), url, self.dtypes, self.quantization,
            )
        self.assert_weights_equal(actual)
        self.assertEqual(download.call_args.args[0], "lodestones/Kroma")
        self.assertEqual(download.call_args.kwargs["weights_name"], "kroma-v0.3-base.safetensors")

    def test_gguf_has_actionable_error(self):
        self.dtypes.transformer = DataType.GGUF
        with self.assertRaisesRegex(ValueError, "not GGUF"):
            self.load()

    def test_onetrainer_fp8_preparation(self):
        self.dtypes.transformer = DataType.FLOAT_8
        actual = self.load()
        self.assertIsInstance(actual.transformer_blocks[0].attn.to_q, LinearFp8)
        self.assertFalse(any(p.is_meta for p in actual.parameters()))
        self.assertFalse(any(b.is_meta for b in actual.buffers()))
        torch.testing.assert_close(actual.transformer_blocks[0].attn.to_q.weight,
                                   self.source.transformer_blocks[0].attn.to_q.weight.to(torch.bfloat16),
                                   rtol=0, atol=0)

    def test_lora_training_export_and_reload(self):
        self.model.transformer = self.load()
        self.model.transformer.requires_grad_(False)
        config = TrainConfig.default_values()
        config.lora_rank = 2
        config.lora_alpha = 2.0
        self.model.transformer_lora = LoRAModuleWrapper(self.model.transformer, "transformer", config)
        self.model.transformer_lora.hook_to_module()
        parameters = list(self.model.transformer_lora.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=0.01)
        before = [p.detach().clone() for p in parameters]
        inputs = {
            "hidden_states": torch.randn(1, 4, 8),
            "encoder_hidden_states": torch.randn(1, 3, 3, 8),
            "timestep": torch.tensor([0.5]),
            "position_ids": Krea2Pipeline.prepare_position_ids(3, 2, 2, torch.device("cpu")),
            "return_dict": False,
        }
        target = torch.randn(1, 4, 8)
        for _ in range(2):
            optimizer.zero_grad()
            loss = (self.model.transformer(**inputs)[0] - target).square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters))
            optimizer.step()
        self.assertTrue(any(not torch.equal(p, old) for p, old in zip(parameters, before, strict=True)))
        self.assert_weights_equal(self.model.transformer)
        with torch.no_grad():
            expected = self.model.transformer(**inputs)[0]

        for fmt in (ModelFormat.ORIGINAL_LORA, ModelFormat.COMFY_LORA, ModelFormat.DIFFUSERS_LORA):
            with self.subTest(format=fmt):
                destination = str(self.root / (fmt.value + ".safetensors"))
                Krea2LoRASaver().save(self.model, fmt, destination, torch.float32)
                exported = load_file(destination)
                if fmt == ModelFormat.COMFY_LORA:
                    self.assertTrue(all(k.startswith("diffusion_model.") for k in exported))
                restored = Krea2Model(ModelType.KREA_2)
                restored.transformer = self.load()
                restored.transformer.requires_grad_(False)
                Krea2LoRALoader().load(restored, ModelNames(lora=destination))
                restored.transformer_lora = LoRAModuleWrapper(restored.transformer, "transformer", config)
                restored.transformer_lora.load_state_dict(restored.lora_state_dict)
                restored.transformer_lora.hook_to_module()
                original_adapter = self.model.transformer_lora.state_dict()
                restored_adapter = restored.transformer_lora.state_dict()
                self.assertEqual(set(original_adapter), set(restored_adapter))
                for key, value in original_adapter.items():
                    torch.testing.assert_close(restored_adapter[key], value, rtol=0, atol=0)
                with torch.no_grad():
                    actual = restored.transformer(**inputs)[0]
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
