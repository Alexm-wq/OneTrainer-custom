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

    def rlhf_chosen_supervised_weight(self, config, objective):
        return (
            1.0
            if DPOObjective(objective) == DPOObjective.BALANCED_REJECT
            else 0.25
        )

    def rlhf_chosen_supervised_requires_separate_forward(self, config):
        return False

    def rlhf_logp_per_sample(self, model, batch, data, config):
        return data["score"]

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

    def test_chosen_bootstrap_starts_strong_and_turns_off_at_margin(self):
        setup = _Setup()
        config = self._config()

        # First pair intentionally gets the full rescue so zero-margin startup
        # cannot deadlock Balanced Reject.
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
