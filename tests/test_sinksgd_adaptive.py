"""SINKSGD_ADAPTIVE (experimental): optimizer behavior and OneTrainer integration. CPU only."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from modules.trainer.GenericTrainer import GenericTrainer, _ValidationBrakeGradient
from modules.util import create
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ConceptType import ConceptType
from modules.util.enum.Optimizer import Optimizer
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.optimizer.SinkSGDAdaptive import SinkSGDAdaptive, _sinkhorn
from modules.util.optimizer_util import change_optimizer

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
        def moved(gradients, precision):
            parameter = nn.Parameter(torch.zeros(16, 16))
            optimizer = SinkSGDAdaptive(
                [parameter], growth_rate=1.0, d0=1e-3, agreement_threshold=0.3, state_precision=precision,
            )
            for gradient in gradients:
                parameter.grad = gradient
                optimizer.step()
            return parameter.detach().abs().mean().item()

        consistent = [torch.ones(16, 16) + 0.1 * torch.randn(16, 16) for _ in range(50)]
        noise = [torch.randn(16, 16) for _ in range(50)]
        for precision in ("fp32", "bf16_sr"):
            with self.subTest(precision=precision):
                self.assertGreater(moved(consistent, precision), 5 * moved(noise, precision))

    def test_fp32_states_are_not_modified_by_the_mask(self):
        parameter = nn.Parameter(torch.zeros(8, 8))
        optimizer = SinkSGDAdaptive([parameter], state_precision="fp32", agreement_threshold=0.2)
        parameter.grad = -torch.ones(8, 8)
        optimizer.step()
        # (1 - beta) * normalized gradient: the agreement computation must not take its absolute value in place.
        torch.testing.assert_close(optimizer.state[parameter]["momentum"], torch.full((8, 8), -0.1))

    def test_states_are_sliced_and_bf16_by_default(self):
        parameter = nn.Parameter(torch.zeros(110, 10))
        optimizer = SinkSGDAdaptive([parameter])
        parameter.grad = torch.randn(110, 10)
        optimizer.step()
        state = optimizer.state[parameter]
        self.assertEqual((state["x0"].numel(), state["s"].numel()), (100, 100))
        self.assertEqual((state["momentum"].dtype, state["magnitude"].dtype), (torch.bfloat16, torch.bfloat16))
        with self.assertRaisesRegex(ValueError, "state precisions"):
            SinkSGDAdaptive([parameter], state_precision="int8_sr")

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

    def test_small_positive_alignment_damps_but_does_not_freeze_d(self):
        results = {}
        for alignment in (0.05, -0.05):
            parameter = nn.Parameter(torch.zeros(16, 16))
            optimizer = SinkSGDAdaptive([parameter], d_coef=1.0, agreement_threshold=0.0, validation_alignment_ref=0.5)
            group = optimizer.param_groups[0]
            group["alignment"], group["brake"] = alignment, max(0.0, alignment / 0.5)
            _quadratic_steps(optimizer, parameter, torch.randn(16, 16), 100)
            results[alignment] = group["d"]
        self.assertGreater(results[0.05], 10 * 1e-5)
        self.assertEqual(results[-0.05], 1e-5)

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
        self.assertEqual((group["slice_p"], group["state_precision"]), (11, "bf16_sr"))
        group["d"] = 0.01
        reported = Optimizer.SINKSGD_ADAPTIVE.maybe_adjust_lrs({"transformer": 1.0}, optimizer)
        self.assertAlmostEqual(reported["transformer"], 0.01)
        self.assertFalse(Optimizer.SINKSGD_ADAPTIVE.supports_fused_back_pass())

    def test_validation_brake_sums_images_spread_over_concepts(self):
        model = nn.Linear(4, 2)
        optimizer = SinkSGDAdaptive(model.parameters())
        optimizer.observe_validation_gradient = Mock()
        brake = _ValidationBrakeGradient(optimizer, total=4, concepts=2)
        taken = []
        expected = [torch.zeros_like(p) for p in model.parameters()]
        # A loader that yields concept a's images first must not use up the whole budget on it.
        for key in ["a"] * 5 + ["b"] * 5:
            if not brake.wants(key):
                continue
            x = torch.randn(3, 4)
            brake.add(key, model(x).square().mean())
            for total, grad in zip(expected, torch.autograd.grad(model(x).square().mean(), list(model.parameters())), strict=True):
                total.add_(grad)
            taken.append(key)
        self.assertEqual(taken, ["a", "a", "b", "b"])
        brake.finish()
        gradients = optimizer.observe_validation_gradient.call_args.args[0]
        for parameter, total in zip(model.parameters(), expected, strict=True):
            torch.testing.assert_close(gradients[parameter], total)
        brake.clear()
        self.assertTrue(all(p.grad is None for p in model.parameters()))

        model.weight.grad = torch.ones_like(model.weight)
        with self.assertRaisesRegex(RuntimeError, "pending"):
            _ValidationBrakeGradient(optimizer, total=4, concepts=2)

    def test_validation_concepts_are_counted(self):
        trainer = object.__new__(GenericTrainer)
        concepts = []
        for concept_type, enabled in ((ConceptType.VALIDATION, True), (ConceptType.VALIDATION, False),
                                      (ConceptType.STANDARD, True), (ConceptType.VALIDATION, True)):
            concept = ConceptConfig.default_values()
            concept.type, concept.enabled = concept_type, enabled
            concepts.append(concept)
        trainer.config = SimpleNamespace(concepts=concepts)
        self.assertEqual(trainer._GenericTrainer__validation_concept_count(), 2)


if __name__ == "__main__":
    unittest.main()
