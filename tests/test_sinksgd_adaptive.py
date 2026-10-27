"""SINKSGD_ADAPTIVE (experimental): optimizer behavior and OneTrainer integration. CPU only."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from modules.trainer.GenericTrainer import GenericTrainer
from modules.util import create
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.Optimizer import Optimizer
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.optimizer.SinkSGDAdaptive import SinkSGDAdaptive, _sinkhorn
from modules.util.optimizer_util import change_optimizer
from modules.util.TrainProgress import TrainProgress

import torch
from torch import nn


def _quadratic_steps(optimizer, parameter, target, steps):
    for _ in range(steps):
        optimizer.zero_grad()
        (parameter - target).square().sum().backward()
        optimizer.step()


class SinkSGDAdaptiveTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_sinkhorn_normalizes_to_unit_rms(self):
        for shape in ((32, 8), (8, 32), (64,)):
            update = _sinkhorn(torch.randn(shape) * 7, 5)
            self.assertAlmostEqual(update.square().mean().sqrt().item(), 1.0, delta=0.05)

    def test_distance_estimate_grows_and_converges_without_tuning(self):
        parameter = nn.Parameter(torch.zeros(16, 16))
        target = torch.randn(16, 16)
        optimizer = SinkSGDAdaptive([parameter], growth_rate=1.05, d_coef=0.1, agreement_threshold=0.0)
        start = (parameter - target).square().mean().item()
        _quadratic_steps(optimizer, parameter, target, 400)
        self.assertGreater(optimizer.param_groups[0]["d_max"], 1e-3)
        self.assertLess((parameter - target).square().mean().item(), 0.05 * start)

    def test_d_shrinks_but_not_below_its_floor(self):
        parameter = nn.Parameter(torch.zeros(8, 8))
        optimizer = SinkSGDAdaptive([parameter], growth_rate=1.05, d_shrink=0.5, d_min_ratio=0.25)
        target = torch.randn(8, 8)
        _quadratic_steps(optimizer, parameter, target, 200)
        group = optimizer.param_groups[0]
        # Pure noise afterwards: no progress, so d falls towards its floor.
        for _ in range(300):
            optimizer.zero_grad()
            parameter.grad = torch.randn(8, 8)
            optimizer.step()
        self.assertLess(group["d"], group["d_max"])
        self.assertGreaterEqual(group["d"], 0.25 * group["d_max"] * (1 - 1e-6))

    def test_agreement_mask_damps_noise_but_not_consistent_gradients(self):
        def moved(gradients):
            parameter = nn.Parameter(torch.zeros(16, 16))
            optimizer = SinkSGDAdaptive([parameter], growth_rate=1.0, d0=1e-3, agreement_threshold=0.3)
            for gradient in gradients:
                parameter.grad = gradient
                optimizer.step()
            return parameter.detach().abs().mean().item()

        consistent = [torch.ones(16, 16) + 0.1 * torch.randn(16, 16) for _ in range(50)]
        noise = [torch.randn(16, 16) for _ in range(50)]
        self.assertGreater(moved(consistent), 5 * moved(noise))

    def test_drift_budget_stops_training(self):
        parameter = nn.Parameter(torch.zeros(8, 8))
        optimizer = SinkSGDAdaptive([parameter], growth_rate=1.0, d0=0.05, drift_budget=0.2, agreement_threshold=0.0)
        _quadratic_steps(optimizer, parameter, torch.full((8, 8), 10.0), 200)
        group = optimizer.param_groups[0]
        self.assertLessEqual(group["drift"], 0.2 * 1.01)
        self.assertLess(group["effective_lr"], 0.01)

    def test_validation_brake_follows_alignment(self):
        parameter = nn.Parameter(torch.zeros(8, 8))
        optimizer = SinkSGDAdaptive([parameter], validation_alignment_ref=0.5, agreement_threshold=0.0)
        parameter.grad = torch.ones(8, 8)
        optimizer.step()
        optimizer.observe_validation_gradient({parameter: torch.ones(8, 8)})
        self.assertEqual(optimizer.param_groups[0]["brake"], 1.0)
        optimizer.observe_validation_gradient({parameter: -torch.ones(8, 8)})
        optimizer.observe_validation_gradient({parameter: -torch.ones(8, 8)})
        group = optimizer.param_groups[0]
        self.assertEqual(group["brake"], 0.0)
        before = parameter.detach().clone()
        d = group["d"]
        parameter.grad = torch.ones(8, 8)
        optimizer.step()
        torch.testing.assert_close(parameter.detach(), before)
        self.assertLessEqual(group["d"], d)

    def test_bf16_parameters_use_stochastic_rounding(self):
        parameter = nn.Parameter(torch.zeros(8, 8, dtype=torch.bfloat16))
        optimizer = SinkSGDAdaptive([parameter], d0=1e-3, growth_rate=1.0, agreement_threshold=0.0)
        parameter.grad = torch.ones(8, 8, dtype=torch.bfloat16)
        optimizer.step()
        self.assertEqual(parameter.dtype, torch.bfloat16)
        # The first step moves every entry by about d0 = 1e-3 against the gradient.
        torch.testing.assert_close(parameter.float(), torch.full((8, 8), -1e-3), rtol=0.05, atol=1e-5)

    def test_state_dict_roundtrip_keeps_d_and_states(self):
        parameter = nn.Parameter(torch.zeros(8, 8))
        optimizer = SinkSGDAdaptive([parameter], growth_rate=1.05)
        _quadratic_steps(optimizer, parameter, torch.randn(8, 8), 30)
        restored_parameter = nn.Parameter(parameter.detach().clone())
        restored = SinkSGDAdaptive([restored_parameter])
        restored.load_state_dict(optimizer.state_dict())
        for key in ("d", "d_max", "d_numerator", "brake"):
            self.assertEqual(restored.param_groups[0][key], optimizer.param_groups[0][key])
        torch.testing.assert_close(restored.state[restored_parameter]["momentum"], optimizer.state[parameter]["momentum"])

    def test_created_from_config_with_defaults_and_reported_lr(self):
        config = TrainConfig.default_values()
        config.optimizer.optimizer = Optimizer.SINKSGD_ADAPTIVE
        config.optimizer = change_optimizer(config)
        config.learning_rate = 1.0
        parameter = nn.Parameter(torch.zeros(4, 4))
        collection = NamedParameterGroupCollection()
        collection.add_group(NamedParameterGroup("transformer", [parameter], learning_rate=1.0))
        optimizer = create.create_optimizer(collection, None, config)
        self.assertIsInstance(optimizer, SinkSGDAdaptive)
        group = optimizer.param_groups[0]
        self.assertEqual((group["growth_rate"], group["d_coef"], group["agreement_threshold"]), (1.05, 0.1, 0.2))
        group["d"] = 0.01
        reported = Optimizer.SINKSGD_ADAPTIVE.maybe_adjust_lrs({"transformer": 1.0}, optimizer)
        self.assertAlmostEqual(reported["transformer"], 0.01)
        self.assertFalse(Optimizer.SINKSGD_ADAPTIVE.supports_fused_back_pass())

    def test_trainer_hands_validation_gradient_to_optimizer(self):
        model = nn.Linear(4, 2)
        optimizer = SinkSGDAdaptive(model.parameters())
        optimizer.observe_validation_gradient = Mock()
        trainer = object.__new__(GenericTrainer)
        trainer.model = SimpleNamespace(optimizer=optimizer)
        trainer.config = Mock()
        batch = {"x": torch.randn(3, 4)}

        def predict(model_, batch_, config_, progress_, deterministic=False):
            self.assertTrue(deterministic and torch.is_grad_enabled())
            return {"y": model(batch_["x"])}

        trainer.model_setup = SimpleNamespace(
            predict=predict, calculate_loss=lambda model_, batch_, output, config_: output["y"].square().mean(),
        )
        trainer._GenericTrainer__observe_validation_gradient(batch, TrainProgress())
        gradients = optimizer.observe_validation_gradient.call_args.args[0]
        self.assertEqual(set(gradients), set(model.parameters()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

        model.weight.grad = torch.ones_like(model.weight)
        with self.assertRaisesRegex(RuntimeError, "pending"):
            trainer._GenericTrainer__observe_validation_gradient(batch, TrainProgress())


if __name__ == "__main__":
    unittest.main()
