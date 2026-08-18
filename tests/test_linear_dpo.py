import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util.config.TrainConfig import TrainConfig
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.enum.DPOObjective import DPOObjective
from modules.util.enum.DPORefMode import DPORefMode
from modules.util.enum.TrainingMethod import TrainingMethod


class _Adapter(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(values, dtype=torch.float32))


class _Model:
    def __init__(self, values):
        self.adapter = _Adapter(values)
        # Stand-in for unrelated Self-Flow state. DPO reference swaps must not
        # enumerate or mutate it.
        self.self_flow_projector = nn.Linear(1, 1, bias=False)

    def adapters(self):
        return [self.adapter]


class _Setup(BaseModelSetup):
    def create_parameters(self, model, config):
        raise NotImplementedError

    def setup_optimizations(self, model, config):
        raise NotImplementedError

    def setup_model(self, model, config):
        raise NotImplementedError

    def setup_train_device(self, model, config):
        raise NotImplementedError

    def predict(self, model, batch, config, train_progress, *, deterministic=False):
        raise NotImplementedError

    def calculate_loss(self, model, batch, data, config):
        raise NotImplementedError

    def after_optimizer_step(self, model, config, train_progress):
        pass


def _config(mode, decay=0.5):
    return SimpleNamespace(
        rlhf_enabled=True,
        training_method=TrainingMethod.LORA,
        rlhf_dpo_linear_ema_decay=decay,
        effective_dpo_ref_mode=lambda: mode,
    )


class LinearDPOTest(unittest.TestCase):
    def setUp(self):
        self.setup = _Setup(torch.device("cpu"), torch.device("cpu"), False)

    def test_linear_pair_loss_matches_official_sign_and_clipping(self):
        policy_chosen_score = torch.tensor([-1.0], requires_grad=True)
        policy_rejected_score = torch.tensor([-3.0], requires_grad=True)
        reference_chosen_score = torch.tensor([-2.0])
        reference_rejected_score = torch.tensor([-2.0])

        loss, utility, error_gap, margin = self.setup._linear_dpo_pair_loss(
            policy_chosen_score,
            policy_rejected_score,
            reference_chosen_score,
            reference_rejected_score,
            beta=1.0,
            eta=0.01,
        )

        torch.testing.assert_close(margin, torch.tensor([2.0]))
        torch.testing.assert_close(utility, torch.tensor([0.1]))
        torch.testing.assert_close(error_gap, torch.tensor([-2.0]))
        torch.testing.assert_close(loss, torch.tensor([-0.2]))
        loss.sum().backward()
        # Minimization lowers chosen error and raises rejected error.
        torch.testing.assert_close(policy_chosen_score.grad, torch.tensor([-0.1]))
        torch.testing.assert_close(policy_rejected_score.grad, torch.tensor([0.1]))

    def test_version_eleven_train_config_migrates_with_linear_defaults(self):
        legacy = TrainConfig.default_values().to_dict()
        legacy["__version"] = 11
        legacy.pop("rlhf_dpo_linear_eta")
        legacy.pop("rlhf_dpo_linear_ema_decay")

        loaded = TrainConfig.default_values().from_dict(legacy)
        self.assertEqual(loaded.config_version, 12)
        self.assertEqual(loaded.rlhf_dpo_linear_eta, 0.01)
        self.assertEqual(loaded.rlhf_dpo_linear_ema_decay, 0.995)

    def test_linear_config_forces_ema_and_rejects_incompatible_settings(self):
        config = TrainConfig.default_values()
        config.rlhf_enabled = True
        config.rlhf_dpo_objective = DPOObjective.LINEAR
        config.rlhf_dpo_ref_mode = DPORefMode.NEW_ADAPTER

        self.assertEqual(
            config.effective_dpo_ref_mode(),
            DPORefMode.EMA_ADAPTER,
        )
        config.validate_dpo_settings()

        # Curriculum is objective-agnostic in the new implementation and is
        # deliberately valid with Linear-DPO.
        config.rlhf_dpo_hard_pair_curriculum = True
        config.validate_dpo_settings()

        config.rlhf_dpo_adaptive_beta = True
        with self.assertRaisesRegex(ValueError, "Adaptive Beta"):
            config.validate_dpo_settings()

    def test_linear_ema_round_trip_and_reference_restore(self):
        config = _config(DPORefMode.EMA_ADAPTER)
        model = _Model([1.0, 2.0])
        self.setup.initialize_dpo_reference(model, config)

        projector_before = model.self_flow_projector.weight.detach().clone()
        model.adapter.weight.data.copy_(torch.tensor([3.0, 6.0]))
        self.setup.update_dpo_ema_reference(model, config)
        torch.testing.assert_close(
            self.setup._dpo_ema_ref_params_cpu[0][0],
            torch.tensor([2.0, 4.0]),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "onetrainer_dpo_reference.pt")
            self.setup.save_dpo_reference(path)

            resumed = _Setup(torch.device("cpu"), torch.device("cpu"), False)
            resumed_model = _Model([3.0, 6.0])
            resumed.initialize_dpo_reference(resumed_model, config, path)
            self.assertEqual(resumed._dpo_ema_ref_steps, 1)

            with resumed.reference_model(resumed_model, config):
                torch.testing.assert_close(
                    resumed_model.adapter.weight,
                    torch.tensor([2.0, 4.0]),
                )
            torch.testing.assert_close(
                resumed_model.adapter.weight,
                torch.tensor([3.0, 6.0]),
            )

        torch.testing.assert_close(
            model.self_flow_projector.weight,
            projector_before,
        )

    def test_version_one_fixed_reference_still_loads(self):
        config = _config(DPORefMode.EXISTING_ADAPTER_CPU)
        model = _Model([1.0, 2.0])
        legacy_payload = {
            "version": 1,
            "adapter_parameters": [[torch.tensor([0.25, 0.5])]],
        }

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "onetrainer_dpo_reference.pt")
            torch.save(legacy_payload, path)
            self.setup.initialize_dpo_reference(model, config, path)
            with self.setup.reference_model(model, config):
                torch.testing.assert_close(
                    model.adapter.weight,
                    torch.tensor([0.25, 0.5]),
                )
            torch.testing.assert_close(
                model.adapter.weight,
                torch.tensor([1.0, 2.0]),
            )

    def test_version_one_snapshot_cannot_fake_linear_ema_history(self):
        config = _config(DPORefMode.EMA_ADAPTER)
        model = _Model([1.0, 2.0])
        legacy_payload = {
            "version": 1,
            "adapter_parameters": [[torch.tensor([0.25, 0.5])]],
        }

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "onetrainer_dpo_reference.pt")
            torch.save(legacy_payload, path)
            with self.assertRaisesRegex(RuntimeError, "Invalid DPO reference"):
                self.setup.initialize_dpo_reference(model, config, path)

    def test_localized_dpo_weight_is_spatial_and_composes(self):
        predicted = torch.zeros((2, 1, 2, 2), dtype=torch.float32)
        target = torch.ones_like(predicted)
        existing_weight = torch.full_like(predicted, 2.0)
        data = {
            "predicted": predicted,
            "target": target,
            "element_loss_weight": existing_weight,
        }
        batch = {
            "dpo_masked": torch.tensor([True, False]),
            # Inactive concepts must stay neutral even if legacy metadata is
            # invalid; only the active localized sample is validated/applied.
            "dpo_mask_weight": torch.tensor([5.0, float("inf")]),
            "dpo_mask": torch.tensor([
                [[[1.0, 0.0], [0.5, 0.0]]],
                [[[1.0, 1.0], [1.0, 1.0]]],
            ]),
        }

        weighted = self.setup._with_dpo_localized_weight(batch, data)
        expected = torch.tensor([
            [[[10.0, 2.0], [6.0, 2.0]]],
            [[[2.0, 2.0], [2.0, 2.0]]],
        ])
        torch.testing.assert_close(weighted["element_loss_weight"], expected)
        self.assertIs(data["element_loss_weight"], existing_weight)

        error = self.setup.rlhf_linear_error_per_sample(
            None,
            batch,
            weighted,
            SimpleNamespace(),
        )
        torch.testing.assert_close(error, torch.tensor([5.0, 2.0]))
        metrics = self.setup._dpo_localized_metrics(batch, 2)
        self.assertTrue(torch.isfinite(torch.tensor(list(metrics.values()))).all())

    def test_localized_dpo_mask_resizes_without_global_loss_scaling(self):
        predicted = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
        data = {
            "predicted": predicted,
            "target": torch.ones_like(predicted),
        }
        batch = {
            "dpo_masked": [True],
            "dpo_mask_weight": [3.0],
            "dpo_mask": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
        }

        weighted = self.setup._with_dpo_localized_weight(batch, data)
        weight = weighted["element_loss_weight"]
        self.assertEqual(tuple(weight.shape), (1, 1, 4, 4))
        self.assertGreater(float(weight.max().item()), 1.0)
        self.assertEqual(float(weight.min().item()), 1.0)
        # The original prediction dictionary is not mutated, which keeps the
        # chosen supervised mix and Self-Flow auxiliaries global.
        self.assertNotIn("element_loss_weight", data)

    def test_curriculum_state_v3_loads_and_v4_round_trips(self):
        config = SimpleNamespace(
            rlhf_dpo_hard_pair_curriculum=True,
            rlhf_dpo_hard_pair_curriculum_ema=0.9,
            rlhf_dpo_hard_pair_curriculum_min_weight=0.1,
            rlhf_dpo_hard_pair_curriculum_full_margin=0.05,
        )
        pair_key = (
            "dpo-pair-path-v1\n"
            "chosen=/dataset/chosen/a.png\n"
            "rejected=/dataset/rejected/a.png"
        )

        with tempfile.TemporaryDirectory() as directory:
            legacy_path = os.path.join(directory, "legacy.json")
            with open(legacy_path, "w", encoding="utf-8") as handle:
                import json
                json.dump({
                    "version": 3,
                    "settings": {
                        "ema_decay": 0.9,
                        "minimum_weight": 0.1,
                        "full_margin": 0.05,
                        "margin_target": 0.05,
                        "margin_weight": 0.5,
                        "wrong_order_weight": 0.5,
                    },
                    "pairs": {
                        pair_key: {
                            "margin_ema": 0.025,
                            "observations": 4,
                        }
                    },
                }, handle)

            self.setup.load_dpo_curriculum_state(legacy_path, config)
            self.assertEqual(
                self.setup._dpo_curriculum_state[pair_key]["observations"],
                4,
            )
            self.assertEqual(
                self.setup._dpo_curriculum_state[pair_key]["objective"],
                str(DPOObjective.ANCHORED_REJECT),
            )

            current_path = os.path.join(directory, "current.json")
            self.setup.save_dpo_curriculum_state(current_path, config)
            resumed = _Setup(torch.device("cpu"), torch.device("cpu"), False)
            resumed.load_dpo_curriculum_state(current_path, config)
            self.assertEqual(
                resumed._dpo_curriculum_state,
                self.setup._dpo_curriculum_state,
            )

    def test_curriculum_enablement_is_objective_agnostic(self):
        for objective in DPOObjective:
            with self.subTest(objective=objective):
                config = SimpleNamespace(
                    rlhf_dpo_hard_pair_curriculum=True,
                    rlhf_dpo_objective=objective,
                )
                self.assertTrue(
                    self.setup._dpo_hard_pair_curriculum_enabled(config)
                )

    def test_curriculum_uses_objective_appropriate_competence(self):
        config = SimpleNamespace(
            rlhf_dpo_ipo_tau=20.0,
            rlhf_dpo_hard_pair_curriculum_full_margin=0.05,
        )
        margin = torch.tensor([0.0, 0.01])
        chosen = torch.tensor([-0.2, -0.1])
        rejected = torch.tensor([-0.5, -0.4])

        linear_signal, linear_threshold = (
            self.setup._dpo_curriculum_competence(
                DPOObjective.LINEAR,
                config,
                margin,
                chosen,
                rejected,
            )
        )
        torch.testing.assert_close(
            linear_signal,
            torch.tensor([0.3, 0.3]),
        )
        self.assertIsNone(linear_threshold)

        ipo_signal, ipo_threshold = self.setup._dpo_curriculum_competence(
            DPOObjective.IPO,
            config,
            margin,
            chosen,
            rejected,
        )
        self.assertIs(ipo_signal, margin)
        # IPO's optimum is 1 / (2 * tau) = 0.025, below the configured 0.05.
        self.assertEqual(ipo_threshold, 0.025)

    def test_legacy_concept_defaults_to_unmasked_dpo(self):
        legacy = ConceptConfig.default_values().to_dict()
        legacy.pop("dpo_masked")
        legacy.pop("dpo_mask_weight")

        loaded = ConceptConfig.default_values().from_dict(legacy)
        self.assertFalse(loaded.dpo_masked)
        self.assertEqual(loaded.dpo_mask_weight, 10.0)


if __name__ == "__main__":
    unittest.main()
