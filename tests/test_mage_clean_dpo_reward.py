import torch

from modules.modelSetup.mixin.MageCleanDPORewardMixin import MageCleanDPORewardMixin


class _RewardHarness(MageCleanDPORewardMixin):
    @staticmethod
    def _dpo_per_sample_tensor(value, batch_size, device, dtype, default):
        if value is None:
            return torch.full((batch_size,), default, device=device, dtype=dtype)
        if isinstance(value, torch.Tensor):
            result = value.to(device=device, dtype=dtype).reshape(-1)
        else:
            result = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
        if result.numel() == 1 and batch_size > 1:
            result = result.expand(batch_size)
        if result.numel() != batch_size:
            raise RuntimeError("test harness per-sample tensor size mismatch")
        return result

    @staticmethod
    def _resize_dpo_mask_like(mask, predicted):
        # Unit tests use an already broadcast-compatible mask so no image-space
        # interpolation machinery is needed here.
        return mask.to(device=predicted.device, dtype=torch.float32)


def _score(harness, predicted, target, *, batch=None, data_extra=None):
    batch = {} if batch is None else dict(batch)
    data = {
        "predicted": predicted,
        "target": target,
    }
    if data_extra:
        data.update(data_extra)
    return harness.rlhf_logp_per_sample(None, batch, data, None)


def test_mage_dpo_reward_is_raw_squared_error():
    harness = _RewardHarness()
    predicted = torch.tensor([
        [[[1.0, 2.0]]],
        [[[3.0, 5.0]]],
    ])
    target = torch.tensor([
        [[[0.0, 0.0]]],
        [[[1.0, 1.0]]],
    ])

    score = _score(harness, predicted, target)

    expected = torch.tensor([
        -(1.0 ** 2 + 2.0 ** 2) / 2.0,
        -(2.0 ** 2 + 4.0 ** 2) / 2.0,
    ])
    torch.testing.assert_close(score, expected)
    assert score.dtype == torch.float32


def test_training_only_weights_do_not_change_mage_dpo_reward():
    harness = _RewardHarness()
    predicted = torch.tensor([[[[1.0, 2.0]]]])
    target = torch.zeros_like(predicted)

    baseline = _score(harness, predicted, target)
    weighted = _score(
        harness,
        predicted,
        target,
        batch={"loss_weight": torch.tensor([37.0])},
        data_extra={
            # Mage normal training can place sigma/timestep weighting here.
            # DPO reward must intentionally ignore it.
            "element_loss_weight": torch.full_like(predicted, 123.0),
            "timestep": torch.tensor([999]),
        },
    )

    torch.testing.assert_close(weighted, baseline)
    torch.testing.assert_close(baseline, torch.tensor([-2.5]))


def test_localized_dpo_weight_is_preserved_without_training_weight():
    harness = _RewardHarness()
    predicted = torch.tensor([[[[1.0, 2.0]]]])
    target = torch.zeros_like(predicted)

    score = _score(
        harness,
        predicted,
        target,
        batch={
            "dpo_mask": torch.tensor([[[[1.0, 0.0]]]]),
            "dpo_masked": torch.tensor([1.0]),
            "dpo_mask_weight": torch.tensor([3.0]),
            "loss_weight": torch.tensor([99.0]),
        },
        data_extra={
            "element_loss_weight": torch.full_like(predicted, 50.0),
        },
    )

    # First element gets explicit localized-DPO weight 3; second stays at 1.
    # Training element_loss_weight=50 and batch loss_weight=99 are ignored.
    expected = torch.tensor([-(3.0 * 1.0 ** 2 + 1.0 * 2.0 ** 2) / 2.0])
    torch.testing.assert_close(score, expected)


def test_mage_dpo_reward_keeps_policy_gradient():
    harness = _RewardHarness()
    predicted = torch.tensor([[[[2.0]]]], requires_grad=True)
    target = torch.tensor([[[[1.0]]]])

    score = _score(harness, predicted, target)
    score.sum().backward()

    # score = -(prediction-target)^2, so dscore/dprediction = -2*(p-t)
    torch.testing.assert_close(predicted.grad, torch.tensor([[[[-2.0]]]]))
