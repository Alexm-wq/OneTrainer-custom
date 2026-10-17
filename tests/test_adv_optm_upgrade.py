"""Focused adv-optm upgrade checks; tiny CPU tensors, no model weights."""
import copy
import os
import sys
import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

from modules.util import create
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.Optimizer import Optimizer
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.optimizer.tag_util import tag_peft_parameters
from modules.util.optimizer_util import OPTIMIZER_DEFAULT_PARAMETERS

import torch
from torch import nn

ADV_OPTIMIZERS = tuple(optimizer for optimizer in Optimizer if optimizer.is_adv)


def make_optimizer(kind, *, state_dict=None, dtype=torch.float32, **overrides):
    torch.manual_seed(191)
    config = TrainConfig.default_values()
    config.train_device = "cpu"
    config.temp_device = "cpu"
    config.model_type = ModelType.KREA_2
    config.learning_rate = 0.002
    config.optimizer.from_dict(OPTIMIZER_DEFAULT_PARAMETERS[kind])
    config.optimizer.optimizer = kind
    # Deterministic CPU regression checks; SR/CUDA compilation is tested separately.
    config.optimizer.stochastic_rounding = False
    config.optimizer.compile = False
    for key, value in overrides.items():
        setattr(config.optimizer, key, value)
    transformer = nn.Module()
    transformer.transformer_blocks = nn.ModuleList([nn.Linear(8, 8, bias=False)])
    transformer.proj = nn.Linear(8, 8, bias=False)
    transformer.to(dtype=dtype)
    model = SimpleNamespace(model_type=config.model_type, transformer=transformer)
    groups = NamedParameterGroupCollection()
    groups.add_group(NamedParameterGroup("transformer", transformer.parameters(), config.learning_rate))
    optimizer = create.create_optimizer(groups, state_dict, config, model=model)
    return config, model, groups, optimizer


def step(optimizer, step_index, *, fused=False):
    for group in optimizer.param_groups:
        for index, parameter in enumerate(group["params"]):
            # Different signs/magnitudes on successive steps exercise momentum.
            values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
            parameter.grad = (values.add(step_index + 1).sin().mul(0.15)).to(parameter.dtype)
            if fused:
                optimizer.step_parameter(parameter, group, index)
    if not fused:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def weights(model):
    return [parameter.detach().clone() for parameter in model.transformer.parameters()]


def write_legacy_fixture(path):
    assert version("adv-optm") == "2.5.12"
    fixtures = []
    cases = [(kind, "auto", torch.float32) for kind in ADV_OPTIMIZERS]
    cases += [(Optimizer.SINKSGD_ADV, precision, dtype)
              for precision in ("factored", "fp32", "bf16_sr", "int8_sr")
              for dtype in (torch.float32, torch.bfloat16)]
    for kind, precision, dtype in cases:
        config, model, groups, optimizer = make_optimizer(kind, state_precision=precision, dtype=dtype)
        step(optimizer, 0)
        step(optimizer, 1)
        saved = copy.deepcopy(optimizer.state_dict())
        saved["param_group_mapping"] = [
            f"{group['name']}_{group['optim_type']}" for group in optimizer.param_groups
        ] if config.optimizer.MuonWithAuxAdam else groups.unique_name_mapping
        saved["param_group_optimizer_mapping"] = [kind.value] * len(optimizer.param_groups)
        fixture = {
            "kind": kind.value, "precision": precision, "dtype": str(dtype).split(".")[-1],
            "weights": weights(model), "state": saved,
        }
        step(optimizer, 2)
        fixture["expected"] = weights(model)
        fixtures.append(fixture)
    torch.save(fixtures, path)
    print(f"Saved {len(fixtures)} real 2.5.12 optimizer checkpoints on the CI runner.")


class AdvOptmUpgradeTests(unittest.TestCase):
    def test_latest_stable_is_installed(self):
        self.assertEqual(version("adv-optm"), "2.5.16")

    def test_existing_factory_defaults_step_for_all_adv_optimizers(self):
        for kind in ADV_OPTIMIZERS:
            with self.subTest(optimizer=kind):
                _, model, _, optimizer = make_optimizer(kind)
                before = weights(model)
                for index in range(3):
                    step(optimizer, index)
                for old, parameter in zip(before, model.transformer.parameters(), strict=True):
                    self.assertTrue(torch.isfinite(parameter).all())
                    self.assertFalse(torch.equal(old, parameter))

    def test_sinksgd_custom_options_and_config_roundtrip(self):
        options = {
            "momentum": 0.83, "nesterov": True, "nesterov_coef": 0.0,
            "sinkhorn_iterations": 3, "normed_momentum": True, "snr_cond": True,
            "orthogonal_sinkhorn": True, "spectral_normalization": True,
            "weight_decay": 0.02, "centered_wd": 0.03,
            "centered_wd_mode": "int8", "state_precision": "fp32",
        }
        config, model, _, optimizer = make_optimizer(Optimizer.SINKSGD_ADV, **options)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            import json
            path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
            restored = TrainConfig.default_values()
            restored.from_dict(json.loads(path.read_text(encoding="utf-8")))
        for key, expected in options.items():
            self.assertEqual(getattr(restored.optimizer, key), expected)
            self.assertEqual(optimizer.param_groups[0][key], expected)
        for index in range(3):
            step(optimizer, index)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.transformer.parameters()))

    def test_fused_sinksgd_matches_normal_steps(self):
        _, normal_model, _, normal = make_optimizer(Optimizer.SINKSGD_ADV, nesterov_coef=0.37)
        _, fused_model, _, fused = make_optimizer(Optimizer.SINKSGD_ADV, nesterov_coef=0.37)
        for index in range(3):
            step(normal, index)
            step(fused, index, fused=True)
        for actual, expected in zip(weights(fused_model), weights(normal_model), strict=True):
            torch.testing.assert_close(actual, expected)

    def test_factored_muon_and_adamuon_upstream_fix(self):
        for kind in (Optimizer.MUON_ADV, Optimizer.ADAMUON_ADV):
            with self.subTest(optimizer=kind):
                _, model, _, optimizer = make_optimizer(kind, state_precision="factored")
                for index in range(3):
                    step(optimizer, index)
                muon_parameter = optimizer.param_groups[0]["params"][0]
                self.assertTrue(optimizer.state[muon_parameter]["factored"])
                self.assertTrue(all(torch.isfinite(p).all() for p in model.transformer.parameters()))

    def test_adopt_first_step_without_kourkoutas(self):
        _, model, _, optimizer = make_optimizer(Optimizer.ADOPT_ADV, use_atan2=False, kourkoutas_beta=False)
        before = weights(model)
        step(optimizer, 0)
        for actual, expected in zip(weights(model), before, strict=True):
            torch.testing.assert_close(actual, expected)
        step(optimizer, 1)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.transformer.parameters()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(weights(model), before, strict=True)))

    def test_scale_invariant_epsilon_through_existing_factory(self):
        for kind in (Optimizer.ADAMW_ADV, Optimizer.ADOPT_ADV, Optimizer.PRODIGY_ADV):
            with self.subTest(optimizer=kind):
                _, model, _, optimizer = make_optimizer(kind, eps=None, use_atan2=False)
                self.assertIsNone(optimizer.param_groups[0]["eps"])
                for index in range(3):
                    step(optimizer, index)
                self.assertTrue(all(torch.isfinite(p).all() for p in model.transformer.parameters()))

    def test_atan2_spectral_scaling_updates_are_finite(self):
        for kind in (Optimizer.ADAMW_ADV, Optimizer.ADOPT_ADV, Optimizer.PRODIGY_ADV):
            with self.subTest(optimizer=kind):
                _, model, _, optimizer = make_optimizer(kind, use_atan2=True, spectral_normalization=True)
                for index in range(3):
                    step(optimizer, index)
                self.assertTrue(all(torch.isfinite(p).all() for p in model.transformer.parameters()))

    def test_peft_parameter_tags_are_preserved(self):
        module = nn.Module()
        module.lora_down = nn.Linear(8, 2, bias=False)
        module.lora_up = nn.Linear(2, 8, bias=False)
        module.dora_scale = nn.Parameter(torch.ones(8, 1))
        module.oft_R = nn.Linear(6, 2, bias=False)
        config = TrainConfig.default_values()
        config.oft_scaled = True
        tag_peft_parameters(SimpleNamespace(transformer=module), config)
        self.assertTrue(module.lora_down.weight._is_lora_A)
        self.assertTrue(module.lora_up.weight._is_lora_B)
        self.assertTrue(module.dora_scale._is_dora_scale)
        self.assertTrue(module.oft_R.weight._is_oft)
        self.assertGreater(module.oft_R.weight._oft_scale_factor, 0)

    def test_muon_auxiliary_custom_groups_and_learning_rates(self):
        for kind in (Optimizer.MUON_ADV, Optimizer.ADAMUON_ADV):
            with self.subTest(optimizer=kind):
                _, _, _, optimizer = make_optimizer(
                    kind, muon_adam_lr=0.0007, nesterov_coef=0.23,
                    muon_adam_config={"beta1": 0.81, "beta2": 0.96, "eps": 2e-7,
                                      "weight_decay": 0.04, "nesterov": True, "nesterov_coef": 0.31},
                )
                self.assertEqual([g["optim_type"] for g in optimizer.param_groups], ["muon", "adam"])
                self.assertEqual(optimizer.param_groups[0]["lr"], 0.002)
                self.assertEqual(optimizer.param_groups[1]["lr"], 0.0007)
                self.assertEqual(optimizer.param_groups[0]["nesterov_coef"], 0.23)
                for index in range(3):
                    step(optimizer, index)

    @unittest.skipUnless(os.environ.get("ADV_OPTM_LEGACY_FIXTURE"), "Cross-version fixture is generated only in focused CI")
    def test_resume_real_2512_states_and_preserve_next_step(self):
        fixtures = torch.load(os.environ["ADV_OPTM_LEGACY_FIXTURE"], weights_only=True)
        self.assertEqual(len(fixtures), 16)
        for fixture in fixtures:
            with self.subTest(optimizer=fixture["kind"], precision=fixture["precision"], dtype=fixture["dtype"]):
                _, model, _, optimizer = make_optimizer(
                    Optimizer(fixture["kind"]), state_dict=copy.deepcopy(fixture["state"]),
                    state_precision=fixture["precision"], dtype=getattr(torch, fixture["dtype"]),
                )
                with torch.no_grad():
                    for parameter, saved in zip(model.transformer.parameters(), fixture["weights"], strict=True):
                        parameter.copy_(saved)
                step(optimizer, 2)
                for actual, expected in zip(weights(model), fixture["expected"], strict=True):
                    # BF16 Sinkhorn now preserves native precision upstream.
                    tolerance = 0.01 if actual.dtype == torch.bfloat16 else 1e-5
                    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--write-legacy":
        write_legacy_fixture(sys.argv[2])
    else:
        unittest.main()
