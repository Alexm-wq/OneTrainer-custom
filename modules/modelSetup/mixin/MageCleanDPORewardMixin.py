from __future__ import annotations

import torch
from torch import Tensor


class MageCleanDPORewardMixin:
    """Use a stable, training-scale-independent Mage DPO reward proxy.

    Mage's ordinary training loss intentionally applies dataset/sample weights,
    optional batch/gradient-accumulation loss scaling, and timestep weighting.
    Those factors belong to optimization bookkeeping; folding them into the
    policy/reference score makes the numerical DPO reward depend on unrelated
    training settings and can make reward magnitudes explode.

    Non-Linear Mage DPO objectives therefore score the raw squared flow
    prediction error per sample:

        score = -mean((prediction - target) ** 2)
        reward = score_policy - score_reference

    The explicit localized-DPO mask remains part of the preference score because
    it describes which image region the preference itself applies to. Normal
    training ``element_loss_weight`` (for example Mage sigma weighting) is
    deliberately ignored here.

    Linear-DPO is unaffected: BaseModelSetup._dpo_score_per_sample() dispatches
    Linear-DPO through rlhf_linear_error_per_sample() before this hook.
    """

    def _mage_dpo_localized_weight(
            self,
            batch: dict,
            predicted: Tensor,
    ) -> Tensor | None:
        raw_mask = batch.get("dpo_mask")
        if not isinstance(raw_mask, torch.Tensor):
            return None

        batch_size = int(predicted.shape[0])
        flags = self._dpo_per_sample_tensor(
            batch.get("dpo_masked"),
            batch_size,
            predicted.device,
            torch.float32,
            0.0,
        ).clamp(0.0, 1.0)
        multipliers = self._dpo_per_sample_tensor(
            batch.get("dpo_mask_weight"),
            batch_size,
            predicted.device,
            torch.float32,
            10.0,
        )

        active = flags > 0.0
        active_multipliers = multipliers[active]
        if not bool(torch.isfinite(active_multipliers).all().item()):
            raise ValueError("Localized DPO Mask Weight must be finite")
        if bool(torch.any(active_multipliers < 1.0).item()):
            raise ValueError("Localized DPO Mask Weight must be >= 1")

        # Inactive concepts can carry legacy metadata. Neutralize it before
        # arithmetic so unused NaN/Inf values cannot contaminate the score.
        multipliers = torch.where(
            active,
            multipliers,
            torch.ones_like(multipliers),
        )

        mask = self._resize_dpo_mask_like(raw_mask, predicted).to(
            device=predicted.device,
            dtype=torch.float32,
        )
        broadcast_shape = (batch_size,) + (1,) * (predicted.ndim - 1)
        flags = flags.reshape(broadcast_shape)
        multipliers = multipliers.reshape(broadcast_shape)
        return 1.0 + flags * (multipliers - 1.0) * mask

    def rlhf_logp_per_sample(
            self,
            model,
            batch: dict,
            data: dict,
            config,
    ) -> Tensor:
        predicted = data["predicted"]
        target = data["target"]
        if predicted.shape != target.shape:
            raise RuntimeError(
                "Mage DPO prediction/target shape mismatch: "
                f"{tuple(predicted.shape)} != {tuple(target.shape)}"
            )
        if predicted.ndim < 2:
            raise RuntimeError(
                "Mage DPO prediction must contain a batch dimension and at "
                "least one feature dimension"
            )

        # Keep activation storage in its native dtype, matching the generic DPO
        # path, and accumulate only the final reduction in fp32.
        error = (predicted - target).pow(2)

        localized_weight = self._mage_dpo_localized_weight(batch, predicted)
        if localized_weight is not None:
            error = error * localized_weight.to(
                device=error.device,
                dtype=error.dtype,
            )

        return -error.mean(
            dim=list(range(1, predicted.ndim)),
            dtype=torch.float32,
        )
