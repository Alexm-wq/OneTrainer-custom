"""SinkSGD_adv / SINKSGD_ADAPTIVE Split Fused Layers: per-part Sinkhorn normalization of fused MOSS Image layers. CPU only."""
import unittest

from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import create
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.Optimizer import Optimizer
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.optimizer.SinkSGDAdaptive import SinkSGDAdaptive
from modules.util.optimizer.split_sinkhorn import (
    SINKHORN_CHUNKS,
    configure_split_sinkhorn,
    install_adv_split_sinkhorn,
)
from modules.util.optimizer_util import change_optimizer

import torch
from torch import nn

from adv_optm import SinkSGD_adv
from adv_optm.util.sinkhorn import apply_sr_sinkhorn
from test_moss_image import tiny_model, train_config


def _step(parameter, grad, compiled):
    parameter.grad = grad.clone()
    optimizer = SinkSGD_adv([parameter], lr=0.1, momentum=0, nesterov=False, compiled_optimizer=compiled)
    optimizer.step()


class SplitSinkhornTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        torch.set_num_threads(1)

    def _lora_model(self, layer_filter="layers.,text_embedder."):
        model = tiny_model()
        config = train_config()
        model.transformer.requires_grad_(False)
        model.transformer_lora = LoRAModuleWrapper(model.transformer, "transformer", config, layer_filter.split(","))
        model.transformer_lora.requires_grad_(True)
        return model

    def test_tags_lora_up_of_fused_moss_layers(self):
        model = self._lora_model()
        configure_split_sinkhorn(model, True)
        tagged = {
            name: getattr(module.lora_up.weight, SINKHORN_CHUNKS, None)
            for name, module in model.transformer_lora.lora_modules.items()
        }
        self.assertEqual({name: chunks for name, chunks in tagged.items() if chunks}, {
            **{f"layers.{i}.proj.ffn_up_gate": 2 for i in range(3)},
            **{f"text_embedder.blocks.{i}.qkv": 3 for i in range(2)},
        })
        # Down projections are not fused; unrelated layers are untouched.
        self.assertTrue(all(getattr(m.lora_down.weight, SINKHORN_CHUNKS, None) is None
                            for m in model.transformer_lora.lora_modules.values()))
        configure_split_sinkhorn(model, False)
        self.assertFalse(any(getattr(m.lora_up.weight, SINKHORN_CHUNKS, None)
                             for m in model.transformer_lora.lora_modules.values()))

    def test_tags_full_fine_tune_weights(self):
        model = tiny_model()
        configure_split_sinkhorn(model, True)
        transformer = model.transformer
        self.assertEqual(getattr(transformer.layers[0].proj["ffn_up_gate"].weight, SINKHORN_CHUNKS), 2)
        self.assertEqual(getattr(transformer.shared_block_modulation.weight, SINKHORN_CHUNKS), 6)
        self.assertEqual(getattr(transformer.video_head_modulation.weight, SINKHORN_CHUNKS), 2)
        self.assertIsNone(getattr(transformer.layers[0].proj["q"].weight, SINKHORN_CHUNKS, None))

    def test_optimizer_normalizes_each_part(self):
        install_adv_split_sinkhorn()
        grad = torch.randn(12, 4) * torch.tensor([1.0, 5.0, 0.2]).repeat_interleave(4)[:, None]
        expected_split = -0.1 * torch.cat([apply_sr_sinkhorn(part.clone()) for part in grad.chunk(3)])
        expected_whole = -0.1 * apply_sr_sinkhorn(grad.clone())
        for compiled in (False, True):
            with self.subTest(compiled=compiled):
                split = nn.Parameter(torch.zeros(12, 4))
                setattr(split, SINKHORN_CHUNKS, 3)
                _step(split, grad, compiled)
                torch.testing.assert_close(split.detach(), expected_split)
                whole = nn.Parameter(torch.zeros(12, 4))
                _step(whole, grad, compiled)
                torch.testing.assert_close(whole.detach(), expected_whole)
        self.assertFalse(torch.allclose(expected_split, expected_whole))

    def test_adaptive_optimizer_normalizes_each_part(self):
        grad = torch.randn(12, 4) * torch.tensor([1.0, 5.0, 0.2]).repeat_interleave(4)[:, None]
        for parts in (3, None):
            with self.subTest(parts=parts):
                parameter = nn.Parameter(torch.zeros(12, 4))
                setattr(parameter, SINKHORN_CHUNKS, parts)
                optimizer = SinkSGDAdaptive(
                    [parameter], d0=1e-3, growth_rate=1.0, agreement_threshold=0.0, state_precision="fp32",
                )
                parameter.grad = grad.clone()
                optimizer.step()

                def normalize(update, parts=parts):
                    if parts:
                        return torch.cat([apply_sr_sinkhorn(part.clone()) for part in update.chunk(parts)])
                    return apply_sr_sinkhorn(update.clone())

                # Gradient and momentum are both normalized, each part on its own.
                momentum = 0.1 * normalize(grad)
                torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], momentum)
                torch.testing.assert_close(parameter.detach(), -1e-3 * normalize(momentum))

    def test_created_from_config_and_rejected_without_fused_layers(self):
        for optimizer_type, optimizer_class in ((Optimizer.SINKSGD_ADV, SinkSGD_adv),
                                                (Optimizer.SINKSGD_ADAPTIVE, SinkSGDAdaptive)):
            config = TrainConfig.default_values()
            config.model_type = ModelType.MOSS_IMAGE
            config.training_method = TrainingMethod.LORA
            config.optimizer.optimizer = optimizer_type
            config.optimizer = change_optimizer(config)
            self.assertFalse(config.optimizer.split_fused_sinkhorn)
            config.optimizer.split_fused_sinkhorn = True
            for layer_filter, fused in (("layers.", True), ("layers.0.proj.q", False)):
                with self.subTest(optimizer=optimizer_type, layer_filter=layer_filter):
                    model = self._lora_model(layer_filter)
                    collection = NamedParameterGroupCollection()
                    collection.add_group(NamedParameterGroup("transformer", model.transformer_lora.parameters(), 1e-4))
                    if fused:
                        optimizer = create.create_optimizer(collection, None, config, model)
                        self.assertIsInstance(optimizer, optimizer_class)
                        up = model.transformer_lora.lora_modules["layers.0.proj.ffn_up_gate"].lora_up.weight
                        self.assertEqual(getattr(up, SINKHORN_CHUNKS), 2)
                    else:
                        with self.assertRaisesRegex(ValueError, "none of the trained parameters"):
                            create.create_optimizer(collection, None, config, model)


if __name__ == "__main__":
    unittest.main()
