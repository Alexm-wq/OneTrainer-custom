import unittest
from types import SimpleNamespace

import torch

from modules.modelSetup.mixin.MageBalancedRejectV2Mixin import (
    MageBalancedRejectV2Mixin,
)
from modules.util.enum.DPOObjective import DPOObjective


class _BaseSetup:
    def __init__(self):
        self._dpo_stream_active = SimpleNamespace(get=lambda: False)
        self._reference_prediction = False
        self.recorded_metrics = {}
        self._last_dpo_metrics = {"reward_margin": 0.0}

    def rlhf_chosen_supervised_weight(self, config, objective):
        # Simulate BaseMageFlowSetup's legacy Self-Flow attenuation. BR-v2 must
        # deliberately bypass this for Balanced Reject while preserving it for
        # every other objective.
        return 0.25

    def rlhf_chosen_supervised_requires_separate_forward(self, config):
        return False

    def rlhf_logp_per_sample(self, model, batch, data, config):
        return data["score"]

    def rlhf_policy_auxiliary_loss(self, model, batch, data, config):
        value = data.get("base_aux")
        return value

    def calculate_dpo_loss(self, model, batch, config, train_progress, **kwargs):
        self._last_dpo_metrics = {"reward_margin": 0.0}
        return torch.tensor(0.0, requires_grad=True)

    def get_last_dpo_metrics(self):
        return self._last_dpo_metrics

    def _record_self_flow_metric(self, name, value):
        self.recorded_metrics[name] = float(value)

    def _dpo_reference_prediction(self):
        return self._reference_prediction

    @staticmethod
    def _dpo_pair_identity(batch, index):
        return batch["pair_key"][index]


class _Setup(MageBalancedRejectV2Mixin, _BaseSetup):
    pass


class MageBalancedRejectV2Tests(unittest.TestCase):
    @staticmethod
    def _config(objective=DPOObjective.BALANCED_REJECT):
        return SimpleNamespace(
            rlhf_dpo_objective=objective,
            concepts=None,
            self_flow_enabled=True,
            self_flow_rep_weight=1.0,
            self_flow_structural_enabled=False,
            self_flow_structural_weight=0.25,
        )

    def test_bootstrap_factor_tracks_margin_deficit(self):
        factor = MageBalancedRejectV2Mixin._brv2_bootstrap_factor
        self.assertEqual(factor(0.04, 0.03), 0.0)
        self.assertAlmostEqual(factor(0.02, 0.03), 1.0 / 3.0)
        self.assertEqual(factor(0.0, 0.03), 1.0)
        self.assertEqual(factor(-0.10, 0.03), 1.0)

    def test_budget_ema_smooths_reward_reversal(self):
        update = MageBalancedRejectV2Mixin._brv2_update_ema
        first = update(None, 0.02, 0.9)
        second = update(first, -0.02, 0.9)
        self.assertAlmostEqual(first, 0.02)
        self.assertAlmostEqual(second, 0.016)

    def test_value_replacement_preserves_gradient_identity(self):
        value = torch.tensor(2.0, requires_grad=True)
        adjusted = MageBalancedRejectV2Mixin._brv2_replace_value_preserve_gradient(
            value,
            torch.tensor(5.0),
        )
        self.assertEqual(adjusted.item(), 5.0)
        adjusted.backward()
        self.assertEqual(value.grad.item(), 1.0)

    def test_chosen_bootstrap_ignores_legacy_self_flow_attenuation(self):
        setup = _Setup()
        config = self._config()

        # Base setup reports 0.25, but Balanced Reject semantics require one
        # full positive chosen objective plus the dynamic rescue.
        self.assertAlmostEqual(
            setup.rlhf_chosen_supervised_weight(
                config,
                DPOObjective.BALANCED_REJECT,
            ),
            1.5,
        )

        setup._brv2_last_margin = 0.02
        self.assertAlmostEqual(
            setup.rlhf_chosen_supervised_weight(
                config,
                DPOObjective.BALANCED_REJECT,
            ),
            1.0 + 0.5 / 3.0,
        )

        setup._brv2_last_margin = 0.03
        self.assertAlmostEqual(
            setup.rlhf_chosen_supervised_weight(
                config,
                DPOObjective.BALANCED_REJECT,
            ),
            1.0,
        )

        # Non-Balanced objectives retain the model family's existing policy.
        self.assertAlmostEqual(
            setup.rlhf_chosen_supervised_weight(
                config,
                DPOObjective.SIGMOID,
            ),
            0.25,
        )

    def test_balanced_reject_chosen_backward_is_always_separate(self):
        setup = _Setup()
        balanced = self._config(DPOObjective.BALANCED_REJECT)
        sigmoid = self._config(DPOObjective.SIGMOID)

        self.assertTrue(
            setup.rlhf_chosen_supervised_requires_separate_forward(balanced)
        )
        self.assertFalse(
            setup.rlhf_chosen_supervised_requires_separate_forward(sigmoid)
        )

    def test_fast_policy_auxiliary_trains_rejected_half_only(self):
        setup = _Setup()
        setup._brv2_active = True
        config = self._config()
        rep = torch.tensor([1.0, 2.0, 10.0, 20.0], requires_grad=True)
        loss = setup.rlhf_policy_auxiliary_loss(
            None,
            {},
            {
                "self_flow_dpo_policy": True,
                "self_flow_training_pass": True,
                "self_flow_representation_loss_per_sample": rep,
            },
            config,
        )
        self.assertIsNotNone(loss)
        # Preserve the original 2B mean normalization with a zeroed chosen half:
        # 0.5 * mean([10, 20]) = 7.5.
        self.assertAlmostEqual(loss.item(), 7.5)
        loss.backward()
        torch.testing.assert_close(
            rep.grad,
            torch.tensor([0.0, 0.0, 0.25, 0.25]),
        )
        self.assertIn("loss/self_flow_rep_rejected_dpo", setup.recorded_metrics)

    def test_streamed_policy_auxiliary_skips_chosen_and_keeps_rejected(self):
        setup = _Setup()
        setup._brv2_active = True
        config = self._config()

        chosen_rep = torch.tensor([1.0, 2.0], requires_grad=True)
        chosen_loss = setup.rlhf_policy_auxiliary_loss(
            None,
            {setup._BRV2_BRANCH_KEY: "chosen"},
            {
                "self_flow_dpo_policy": True,
                "self_flow_training_pass": True,
                "self_flow_representation_loss_per_sample": chosen_rep,
            },
            config,
        )
        self.assertIsNone(chosen_loss)

        rejected_rep = torch.tensor([10.0, 20.0], requires_grad=True)
        rejected_loss = setup.rlhf_policy_auxiliary_loss(
            None,
            {setup._BRV2_BRANCH_KEY: "rejected"},
            {
                "self_flow_dpo_policy": True,
                "self_flow_training_pass": True,
                "self_flow_representation_loss_per_sample": rejected_rep,
            },
            config,
        )
        self.assertIsNotNone(rejected_loss)
        self.assertAlmostEqual(rejected_loss.item(), 7.5)
        rejected_loss.backward()
        torch.testing.assert_close(
            rejected_rep.grad,
            torch.tensor([0.25, 0.25]),
        )

    def test_streamed_dispatch_keeps_brv2_active_for_backward_replay(self):
        setup = _Setup()
        balanced = self._config(DPOObjective.BALANCED_REJECT)
        sigmoid = self._config(DPOObjective.SIGMOID)

        setup.calculate_dpo_loss(
            None,
            {},
            balanced,
            None,
            streamed=True,
        )
        self.assertTrue(setup._brv2_active)

        # The next non-BR dispatch must clear the state left alive for the
        # previous custom-autograd replay window.
        setup.calculate_dpo_loss(
            None,
            {},
            sigmoid,
            None,
            streamed=True,
        )
        self.assertFalse(setup._brv2_active)

        setup.calculate_dpo_loss(
            None,
            {},
            balanced,
            None,
            streamed=False,
        )
        self.assertFalse(setup._brv2_active)

    def test_budget_ema_changes_value_but_not_policy_gradient(self):
        setup = _Setup()
        setup._brv2_active = True
        config = self._config()
        batch = {"pair_key": ["a", "b", "a", "b"]}

        setup._reference_prediction = True
        reference = torch.tensor([1.0, 2.0, 10.0, 20.0])
        reference_out = setup.rlhf_logp_per_sample(
            None,
            batch,
            {"score": reference},
            config,
        )
        torch.testing.assert_close(reference_out, reference)

        setup._reference_prediction = False
        first_policy = torch.tensor(
            [1.2, 1.8, 10.0, 20.0],
            requires_grad=True,
        )
        first_out = setup.rlhf_logp_per_sample(
            None,
            batch,
            {"score": first_policy},
            config,
        )
        torch.testing.assert_close(first_out, first_policy)
        first_out.sum().backward()
        torch.testing.assert_close(
            first_policy.grad,
            torch.ones_like(first_policy),
        )

        second_policy = torch.tensor(
            [0.8, 2.2, 10.0, 20.0],
            requires_grad=True,
        )
        second_out = setup.rlhf_logp_per_sample(
            None,
            batch,
            {"score": second_policy},
            config,
        )
        # EMA=0.9: +0.20 -> -0.20 becomes +0.16; -0.20 -> +0.20 becomes -0.16.
        torch.testing.assert_close(
            second_out[:2],
            torch.tensor([1.16, 1.84]),
        )
        second_out.sum().backward()
        torch.testing.assert_close(
            second_policy.grad,
            torch.ones_like(second_policy),
        )


if __name__ == "__main__":
    unittest.main()