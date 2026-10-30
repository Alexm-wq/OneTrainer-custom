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
from modules.util.optimizer.adv_optm_util import load_adv_optimizer_state_dict
from modules.util.optimizer.SinkSGDAdaptive import SinkSGDAdaptive
from modules.util.optimizer_util import change_optimizer

import torch
from torch import nn

from adv_optm import SinkSGD_adv
from adv_optm.util import param_update


def _quadratic_steps(optimizer, parameter, target, steps):
    for _ in range(steps):
        optimizer.zero_grad()
        (parameter - target).square().sum().backward()
        optimizer.step()


def _fixed_step(**kwargs):
    # d stays at d0: no growth, no shrinking. The step size is then d0 * lr, as a plain lr would be.
    return {"d0": 1e-2, "growth_rate": 1.0, "d_shrink": 0.0, "agreement_threshold": 0.0, **kwargs}


class SinkSGDAdaptiveTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_matches_sinksgd_adv_with_a_fixed_step(self):
        # Every SinkSGD_adv option behaves as in SinkSGD_adv when d is fixed and the extras are off.
        configs = {
            "momentum": {"momentum": 0.9},
            "nesterov": {"momentum": 0.9, "nesterov": True},
            "normed momentum": {"momentum": 0.9, "normed_momentum": True},
            "normed nesterov": {"momentum": 0.9, "normed_momentum": True, "nesterov": True, "nesterov_coef": 0.5},
            "snr": {"momentum": 0.9, "normed_momentum": True, "snr_cond": True},
            "spectral": {"momentum": 0.9, "spectral_normalization": True},
            "cautious wd": {"momentum": 0.9, "weight_decay": 0.1, "cautious_wd": True},
            "geometric wd": {"momentum": 0.9, "normed_momentum": True, "snr_cond": True, "weight_decay": 0.1, "geometric_wd": True},
            "centered wd": {"momentum": 0.9, "centered_wd": 0.1, "centered_wd_mode": "full"},
            "orthogonal sinkhorn": {"momentum": 0.9, "orthogonal_sinkhorn": True},
            "orthograd": {"momentum": 0.9, "orthogonal_gradient": "flattened"},
            "fp32 states": {"momentum": 0.9, "state_precision": "fp32"},
            "int8 states": {"momentum": 0.9, "state_precision": "int8_sr"},
            "factored states": {"momentum": 0.9, "state_precision": "factored"},
        }
        grads = [(torch.randn(12, 6), torch.randn(6)) for _ in range(3)]
        for name, options in configs.items():
            with self.subTest(name):
                results = []
                for optimizer_class, extra in ((SinkSGD_adv, {"lr": 1e-2}),
                                               (SinkSGDAdaptive, _fixed_step(normed_momentum=False, momentum_renorm=False))):
                    kwargs = {**extra, **options}
                    if optimizer_class is SinkSGDAdaptive:
                        kwargs.setdefault("state_precision", "auto")
                    torch.manual_seed(1)
                    weight, bias = nn.Parameter(torch.randn(12, 6)), nn.Parameter(torch.randn(6))
                    param_update.set_seed(torch.device("cpu"))
                    optimizer = optimizer_class([weight, bias], **kwargs)
                    for weight_grad, bias_grad in grads:
                        weight.grad, bias.grad = weight_grad.clone(), bias_grad.clone()
                        optimizer.step()
                    results.append((weight.detach().clone(), bias.detach().clone()))
                torch.testing.assert_close(results[1], results[0])

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

    def test_high_initial_d_decays_towards_its_floor(self):
        parameter = nn.Parameter(torch.zeros(8, 8))
        optimizer = SinkSGDAdaptive([parameter], d0=1e-2, d_floor=1e-4, d_shrink=0.05, agreement_threshold=0.0)
        group = optimizer.param_groups[0]
        steps = []
        for _ in range(300):
            parameter.grad = torch.randn(8, 8)
            optimizer.step()
            steps.append(group["d"])
        # Pure noise: no progress, so the large initial step decays, but not below the floor.
        self.assertLess(steps[50], 0.5 * 1e-2)
        self.assertLess(steps[-1], 1e-3)
        self.assertGreaterEqual(min(steps), 1e-4)
        with self.assertRaisesRegex(ValueError, "D Floor"):
            SinkSGDAdaptive([parameter], d0=1e-5, d_floor=1e-4)

    def test_a_spike_of_the_estimate_cannot_raise_d_through_the_floor(self):
        parameter = nn.Parameter(torch.zeros(4, 4))
        optimizer = SinkSGDAdaptive([parameter], d0=1e-3, d_floor=1e-3, growth_rate=1.0, d_shrink=0.0)
        group = optimizer.param_groups[0]
        # An estimate of 100 that d may not follow, then one below d (the shrink branch).
        for dot in (100.0, 0.0):
            optimizer._pending = {0: [torch.tensor([dot, 1.0, 0.0]), 16, 1.0]}
            optimizer._finish_distance()
            self.assertEqual(group["d"], 1e-3)
        self.assertEqual(group["d_max"], 1e-3)

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
        for precision in ("fp32", "bf16_sr", "int8_sr", "factored"):
            with self.subTest(precision=precision):
                self.assertGreater(moved(consistent, precision), 5 * moved(noise, precision))

    def test_fp32_states_are_not_modified_by_the_mask(self):
        parameter = nn.Parameter(torch.zeros(8, 8))
        optimizer = SinkSGDAdaptive([parameter], state_precision="fp32", agreement_threshold=0.2)
        parameter.grad = -torch.ones(8, 8)
        optimizer.step()
        # (1 - beta) * normalized gradient: the agreement computation must not take its absolute value in place.
        torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], torch.full((8, 8), -0.1))

    def test_states_are_sliced_and_bf16_by_default(self):
        parameter = nn.Parameter(torch.zeros(110, 10))
        optimizer = SinkSGDAdaptive([parameter])
        parameter.grad = torch.randn(110, 10)
        optimizer.step()
        state = optimizer.state[parameter]
        self.assertEqual((state["d_x0"].numel(), state["d_s"].numel()), (100, 100))
        self.assertEqual((state["momentum_buffer"].dtype, state["magnitude"].dtype), (torch.bfloat16, torch.bfloat16))
        with self.assertRaisesRegex(ValueError, "state_precision"):
            SinkSGDAdaptive([parameter], state_precision="int4_sr")

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

    def test_tlora_aware_steps_only_the_active_ranks(self):
        # T-LoRA leaves ranks 2 and 3 without gradient at this noise level.
        for tag, shape, active in (("_is_lora_B", (12, 4), (slice(None), slice(0, 2))),
                                   ("_is_lora_A", (4, 12), (slice(0, 2), slice(None)))):
            with self.subTest(tag):
                full = nn.Parameter(torch.randn(shape))
                setattr(full, tag, True)
                start = full.detach().clone()
                sub = nn.Parameter(start[active].clone())
                setattr(sub, tag, True)
                options = _fixed_step(agreement_threshold=0.2, state_precision="fp32")
                optimizers = [SinkSGDAdaptive([full], **options), SinkSGDAdaptive([sub], **options)]
                for _ in range(3):
                    grad = torch.zeros(shape)
                    grad[active] = torch.randn(start[active].shape)
                    full.grad, sub.grad = grad, grad[active].clone()
                    for optimizer in optimizers:
                        optimizer.step()
                torch.testing.assert_close(full.detach()[active], sub.detach())
                inactive = torch.ones(shape, dtype=torch.bool)
                inactive[active] = False
                torch.testing.assert_close(full.detach()[inactive], start[inactive])
                momentum = optimizers[0].state[full]["momentum_buffer"]
                self.assertEqual(momentum[inactive].abs().sum().item(), 0.0)
                torch.testing.assert_close(momentum[active], optimizers[1].state[sub]["momentum_buffer"])

        # Switched off, the zero ranks dilute the normalization like any other entries.
        parameter = nn.Parameter(torch.zeros(12, 4))
        parameter._is_lora_B = True
        optimizer = SinkSGDAdaptive([parameter], **_fixed_step(tlora_aware=False, state_precision="fp32"))
        parameter.grad = torch.cat([torch.randn(12, 2), torch.zeros(12, 2)], 1)
        optimizer.step()
        self.assertNotAlmostEqual(parameter[:, :2].square().mean().sqrt().item(), 1e-2, delta=1e-4)

    def test_fused_back_pass_uses_the_previous_d(self):
        def run(fused):
            torch.manual_seed(3)
            parameters = [nn.Parameter(torch.randn(8, 8)), nn.Parameter(torch.randn(8, 4))]
            optimizer = SinkSGDAdaptive(parameters, d0=1e-3, growth_rate=2.0, d_coef=10.0)
            ds = []
            for _ in range(4):
                for p in parameters:
                    p.grad = p.detach() * 2
                    if fused:
                        optimizer.step_parameter(p, optimizer.param_groups[0], 0)
                        p.grad = None
                if not fused:
                    optimizer.step()
                else:
                    optimizer.step()  # no gradients left: finishes d
                ds.append(optimizer.param_groups[0]["d"])
            return ds, [p.detach().clone() for p in parameters]

        eager_d, eager = run(False)
        fused_d, fused = run(True)
        self.assertGreater(eager_d[-1], 1e-3)
        # The fused path estimates the same distances, one step later in the parameters.
        self.assertGreater(fused_d[-1], 1e-3)
        self.assertFalse(torch.allclose(eager[0], fused[0]))
        self.assertTrue(SinkSGDAdaptive([nn.Parameter(torch.zeros(2))]).supports_fused_back_pass)

    def test_compiled_normalization_matches_eager(self):
        results = []
        for compiled in (False, True):
            torch.manual_seed(4)
            parameter = nn.Parameter(torch.randn(16, 8))
            optimizer = SinkSGDAdaptive([parameter], compiled_optimizer=compiled, state_precision="fp32")
            for size in (16, 16):
                parameter.grad = torch.randn(size, 8)
                optimizer.step()
            results.append(parameter.detach().clone())
        torch.testing.assert_close(results[1], results[0], rtol=1e-5, atol=1e-7)

    def test_state_dict_roundtrip_keeps_d_and_state_precisions(self):
        parameter = nn.Parameter(torch.zeros(8, 8, dtype=torch.bfloat16))
        optimizer = SinkSGDAdaptive([parameter], growth_rate=1.05, state_precision="int8_sr")
        for _ in range(30):
            parameter.grad = (parameter.detach().float() - 1).bfloat16()
            optimizer.step()
        restored_parameter = nn.Parameter(parameter.detach().clone())
        restored = SinkSGDAdaptive([restored_parameter], state_precision="int8_sr")
        load_adv_optimizer_state_dict(restored, optimizer.state_dict())
        for key in ("d", "d_max", "d_numerator", "brake"):
            self.assertEqual(restored.param_groups[0][key], optimizer.param_groups[0][key])
        state, saved = restored.state[restored_parameter], optimizer.state[parameter]
        for key in ("d_x0", "d_s", "momentum_buffer", "magnitude"):
            self.assertEqual(state[key].dtype, saved[key].dtype, key)
            torch.testing.assert_close(state[key], saved[key])
        self.assertEqual(state["d_x0"].dtype, torch.float32)

    def test_created_from_config_with_defaults_and_reported_lr(self):
        config = TrainConfig.default_values()
        config.optimizer.optimizer = Optimizer.SINKSGD_ADAPTIVE
        config.optimizer = change_optimizer(config)
        config.learning_rate = 1.0
        parameter = nn.Parameter(torch.zeros(4, 4))
        collection = NamedParameterGroupCollection()
        collection.add_group(NamedParameterGroup("transformer", [parameter], learning_rate=1.0))
        optimizer = create.create_optimizer(collection, None, config, SimpleNamespace())
        self.assertIsInstance(optimizer, SinkSGDAdaptive)
        group = optimizer.param_groups[0]
        self.assertEqual((group["growth_rate"], group["d_coef"], group["agreement_threshold"]), (1.05, 1.0, 0.2))
        self.assertEqual((group["d0"], group["d_floor"]), (1e-5, 1e-5))
        self.assertEqual((group["slice_p"], group["state_precision"]), (11, "bf16_sr"))
        self.assertEqual((group["normed_momentum"], group["momentum_renorm"], group["tlora_aware"]), (True, True, True))
        self.assertEqual((group["nesterov"], group["spectral_normalization"], group["orthogonal_gradient"]), (False, False, "disabled"))
        group["d"] = 0.01
        reported = Optimizer.SINKSGD_ADAPTIVE.maybe_adjust_lrs({"transformer": 1.0}, optimizer)
        self.assertAlmostEqual(reported["transformer"], 0.01)
        self.assertTrue(Optimizer.SINKSGD_ADAPTIVE.supports_fused_back_pass())
        self.assertTrue(Optimizer.SINKSGD_ADAPTIVE.is_adv)

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
