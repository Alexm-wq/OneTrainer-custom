"""The ComfyUI T-LoRA loader's core (comfyui/onetrainer_tlora) against OneTrainer's T-LoRA. CPU only.

ComfyUI itself is not imported: these tests cover the masking, the per-layer LoRA path and the
noise-level hand-over that the node wires into ComfyUI's model patcher.
"""
import copy
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

from modules.modelSaver.moss.MOSSImageLoRASaver import MOSSImageLoRASaver
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.ModelFormat import ModelFormat

import torch
from torch import nn

from safetensors import safe_open
from test_moss_image import tiny_model, train_config

_spec = importlib.util.spec_from_file_location(
    "tlora_core", Path(__file__).resolve().parents[1] / "comfyui" / "onetrainer_tlora" / "tlora_core.py",
)
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)


class ComfyTLoRACoreTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(1)

    def _trained(self, rank=8, min_rank=2):
        model = tiny_model()
        base = copy.deepcopy(model.transformer)
        config = train_config()
        config.lora_rank, config.lora_alpha, config.tlora, config.tlora_min_rank = rank, 4.0, True, min_rank
        config.layer_filter = "layers.,text_embedder.,shared_block_modulation"
        model.train_config = config
        model.transformer_lora = LoRAModuleWrapper(model.transformer, "transformer", config, config.layer_filter.split(","))
        model.transformer_lora.hook_to_module()
        model.enable_tlora()
        for module in model.transformer_lora.lora_modules.values():
            nn.init.normal_(module.lora_up.weight, std=0.2)
        return model, base, config

    def test_rank_schedule_matches_training(self):
        model, _, config = self._trained()
        times = torch.arange(1, 1001, 37, dtype=torch.float32)
        model.set_tlora_timestep(times)
        expected = model.transformer_lora._rank_mask
        actual = core.rank_mask(times / 1000, config.lora_rank, config.tlora_min_rank, len(times), "cpu", torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # One sigma for a larger batch, as for references run alongside the image.
        self.assertEqual(core.rank_mask(torch.tensor([1.0]), 8, 2, 3, "cpu", torch.float32).sum(1).tolist(), [2.0] * 3)
        with self.assertRaisesRegex(ValueError, "noise levels"):
            core.rank_mask(torch.tensor([1.0, 0.5]), 8, 2, 3, "cpu", torch.float32)

    def test_masked_layers_reproduce_onetrainer_samples(self):
        model, base, config = self._trained()
        plain = copy.deepcopy(base)
        schedule = core.Schedule(config.tlora_min_rank)
        hooks = []
        for name, lora in model.transformer_lora.lora_modules.items():
            hook = core.MaskedLoRA(base.get_submodule(name), schedule)
            hook.add(lora.lora_up.weight.detach(), lora.lora_down.weight.detach(), (lora.alpha / lora.rank).item())
            hook.inject()
            hooks.append(hook)
        hidden, text = torch.randn(2, 128, 2, 2), torch.randn(2, 3, 8)
        mask = torch.ones(2, 3, dtype=torch.bool)
        with torch.no_grad():
            for times in (torch.tensor([1000.0, 1000.0]), torch.tensor([250.0, 900.0]), torch.tensor([1.0, 500.0])):
                with self.subTest(times=times.tolist()):
                    expected = model.transformer(hidden, times, text, mask).sample
                    schedule.sigmas = times / 1000
                    torch.testing.assert_close(base(hidden, times, text, mask).sample, expected)
            # Outside a diffusion-model call every rank is used, as in a merged LoRA.
            schedule.sigmas = None
            model.train_config.tlora = False
            model.transformer_lora.set_rank_mask(None)
            torch.testing.assert_close(base(hidden, times, text, mask).sample, model.transformer(hidden, times, text, mask).sample)
            for hook in hooks:
                hook.eject()
            torch.testing.assert_close(base(hidden, times, text, mask).sample, plain(hidden, times, text, mask).sample)

    def test_fused_qkv_slices_match_separate_projections(self):
        q, k, v = (nn.Linear(16, 16, bias=False) for _ in range(3))
        fused = nn.Linear(16, 48, bias=False)
        fused.weight.data = torch.cat([q.weight, k.weight, v.weight]).detach()
        schedule = core.Schedule(min_rank=1)
        separate, sliced = [], core.MaskedLoRA(fused, schedule)
        for index, layer in enumerate((q, k, v)):
            up, down = torch.randn(16, 4), torch.randn(4, 16)
            hook = core.MaskedLoRA(layer, schedule)
            hook.add(up, down, 0.5)
            hook.inject()
            separate.append(hook)
            sliced.add(up, down, 0.5, index * 16, 16)
        sliced.inject()
        x = torch.randn(2, 5, 16)
        schedule.sigmas = torch.tensor([0.9, 0.2])
        torch.testing.assert_close(fused(x), torch.cat([q(x), k(x), v(x)], -1))

    def test_wrapper_hands_over_the_sampler_sigmas(self):
        schedule = core.Schedule(min_rank=1)
        seen = []

        def executor(x, timestep, context, transformer_options, **kwargs):
            seen.append(schedule.sigmas)
            if kwargs.get("fail"):
                raise RuntimeError("fail")
            return x

        schedule.wrapper(executor, "x", torch.tensor([0.5]), "c", {"sigmas": torch.tensor([0.25])})
        schedule.wrapper(executor, "x", torch.tensor([0.5]), "c", {})
        with self.assertRaises(RuntimeError):
            schedule.wrapper(executor, "x", torch.tensor([0.75]), "c", {}, fail=True)
        self.assertEqual([s.item() for s in seen], [0.25, 0.5, 0.75])
        self.assertIsNone(schedule.sigmas)

    def test_inject_and_eject_restore_the_layer(self):
        layer = nn.Linear(8, 8)
        x = torch.randn(3, 8)
        before = layer(x)
        hook = core.MaskedLoRA(layer, core.Schedule(min_rank=1))
        hook.add(torch.randn(8, 2), torch.randn(2, 8), 1.0)
        hook.inject()
        self.assertTrue(layer.comfy_force_forward)
        self.assertFalse(torch.allclose(layer(x), before))
        hook.eject()
        torch.testing.assert_close(layer(x), before)
        self.assertFalse(layer.comfy_force_forward)

    def test_saved_lora_tells_the_node_its_min_rank(self):
        model, _, _ = self._trained(min_rank=3)
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "tlora.safetensors")
            MOSSImageLoRASaver().save(model, ModelFormat.COMFY_LORA, path, torch.float32)
            with safe_open(path, "pt") as file:
                metadata = file.metadata()
        self.assertEqual(core.file_min_rank(metadata), 3)
        self.assertEqual(core.file_min_rank({"ot_config": json.dumps({"tlora": True, "tlora_min_rank": 4})}), 4)
        self.assertIsNone(core.file_min_rank({"ot_config": json.dumps({"tlora": False})}))
        self.assertIsNone(core.file_min_rank(None))
        # A gradient-only T-LoRA is a plain LoRA: no mask at inference.
        model.train_config.tlora_gradient_only = True
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "tlora.safetensors")
            MOSSImageLoRASaver().save(model, ModelFormat.COMFY_LORA, path, torch.float32)
            with safe_open(path, "pt") as file:
                metadata = file.metadata()
        self.assertEqual(metadata["ot_tlora_gradient_only"], "True")
        self.assertIsNone(core.file_min_rank(metadata))
        self.assertIsNone(core.file_min_rank({"ot_config": json.dumps({"tlora": True, "tlora_gradient_only": True})}))


if __name__ == "__main__":
    unittest.main()
