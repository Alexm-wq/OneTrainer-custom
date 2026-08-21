from __future__ import annotations

import math

import torch
from torch import Tensor

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DPOObjective import DPOObjective


class MageBalancedRejectV2Mixin:
    """Mage-specific stabilization for the Balanced Reject objective.

    Balanced Reject's rejected branch is intentionally one-sided. Its original
    target, however, is proportional to ``relu(chosen_reward)``. If chosen has
    not improved over the reference yet, the reject target collapses to zero
    and the preference objective has no mechanism of its own to acquire a
    positive margin.

    BR-v2 fixes that bootstrap dead-zone without introducing a symmetric DPO
    tug-of-war:

    * chosen supervision is boosted while the previous measured margin is below
      a small target;
    * the chosen reward used as the rejected suppression budget is smoothed by
      a per-pair EMA on the ordinary batched DPO path;
    * the boost remains part of the independent chosen supervised term, so
      hard-pair curriculum continues to scale only the rejected preference
      branch.

    Settings intentionally use getattr defaults for backwards compatibility:
      rlhf_dpo_balanced_margin_target       (default 0.03)
      rlhf_dpo_balanced_bootstrap_weight    (default 0.50)
      rlhf_dpo_balanced_budget_ema          (default 0.90)
    """

    _BRV2_MARGIN_TARGET = 0.03
    _BRV2_BOOTSTRAP_WEIGHT = 0.50
    _BRV2_BUDGET_EMA = 0.90

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._brv2_last_margin = 0.0
        self._brv2_budget_ema_by_pair: dict[str, float] = {}
        self._brv2_reference_chosen_by_pair: dict[str, Tensor] = {}
        self._brv2_active = False
        self._brv2_raw_chosen_reward_mean = 0.0
        self._brv2_ema_chosen_reward_mean = 0.0
        self._brv2_warned_streamed_budget_ema = False

    @staticmethod
    def _brv2_validate_settings(config: TrainConfig) -> tuple[float, float, float]:
        target = float(getattr(
            config,
            "rlhf_dpo_balanced_margin_target",
            MageBalancedRejectV2Mixin._BRV2_MARGIN_TARGET,
        ))
        bootstrap = float(getattr(
            config,
            "rlhf_dpo_balanced_bootstrap_weight",
            MageBalancedRejectV2Mixin._BRV2_BOOTSTRAP_WEIGHT,
        ))
        decay = float(getattr(
            config,
            "rlhf_dpo_balanced_budget_ema",
            MageBalancedRejectV2Mixin._BRV2_BUDGET_EMA,
        ))
        if not math.isfinite(target) or target <= 0.0:
            raise ValueError("Balanced Reject v2 Margin Target must be finite and > 0")
        if not math.isfinite(bootstrap) or bootstrap < 0.0:
            raise ValueError("Balanced Reject v2 Chosen Bootstrap must be finite and >= 0")
        if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
            raise ValueError("Balanced Reject v2 Budget EMA must satisfy 0 <= EMA < 1")
        return target, bootstrap, decay

    @staticmethod
    def _brv2_bootstrap_factor(margin: float, target: float) -> float:
        """Return [0,1] chosen-rescue strength for a detached margin."""
        if not math.isfinite(margin):
            return 1.0
        return max(0.0, min(1.0, (float(target) - float(margin)) / float(target)))

    @staticmethod
    def _brv2_update_ema(previous: float | None, current: float, decay: float) -> float:
        if previous is None:
            return float(current)
        return float(decay) * float(previous) + (1.0 - float(decay)) * float(current)

    @staticmethod
    def _brv2_replace_value_preserve_gradient(
            value: Tensor,
            target_value: Tensor,
    ) -> Tensor:
        """Change only the forward value; keep d(output)/d(value) == 1."""
        return value + (target_value.to(value) - value.detach()).detach()

    @staticmethod
    def _brv2_config_uses_balanced_reject(config: TrainConfig) -> bool:
        try:
            if DPOObjective(config.rlhf_dpo_objective) == DPOObjective.BALANCED_REJECT:
                return True
        except (AttributeError, ValueError):
            pass

        # Per-concept overrides are normally already materialized on the config.
        # If they are not, retain the base model-family behavior rather than
        # doing filesystem I/O from this lifecycle hook.
        for concept in getattr(config, "concepts", None) or []:
            concept_dict = concept.to_dict() if hasattr(concept, "to_dict") else concept
            if not isinstance(concept_dict, dict) or not concept_dict.get("enabled", True):
                continue
            raw = concept_dict.get("dpo_objective")
            try:
                if raw is not None and DPOObjective(raw) == DPOObjective.BALANCED_REJECT:
                    return True
            except ValueError:
                continue
        return False

    def rlhf_chosen_supervised_requires_separate_forward(
            self,
            config: TrainConfig,
    ) -> bool:
        # BR-v2's chosen rescue is a normal positive-training component. Keep it
        # out of the rejected DPO graph even when Self-Flow is disabled so the
        # no-momentum/curriculum post-normalization path cannot attenuate it.
        if self._brv2_config_uses_balanced_reject(config):
            return True
        return super().rlhf_chosen_supervised_requires_separate_forward(config)

    def rlhf_chosen_supervised_weight(
            self,
            config: TrainConfig,
            objective,
    ) -> float:
        objective = DPOObjective(objective)
        base = super().rlhf_chosen_supervised_weight(config, objective)
        if objective != DPOObjective.BALANCED_REJECT:
            return base

        target, bootstrap, _ = self._brv2_validate_settings(config)
        factor = self._brv2_bootstrap_factor(self._brv2_last_margin, target)
        return float(base) * (1.0 + bootstrap * factor)

    def _brv2_pair_key(self, batch: dict, index: int) -> str | None:
        try:
            return self._dpo_pair_identity(batch, index)
        except Exception:
            # Never let a missing/legacy identity mix EMA state between
            # unrelated pairs. The objective remains valid without smoothing.
            return None

    def rlhf_logp_per_sample(
            self,
            model,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        score = super().rlhf_logp_per_sample(model, batch, data, config)
        if not self._brv2_active:
            return score

        # The streamed custom-autograd path replays score evaluation during
        # backward. Stateful EMA mutation would make replay values differ from
        # forward values, so keep only the chosen bootstrap in streamed mode.
        if self._dpo_stream_active.get():
            if not self._brv2_warned_streamed_budget_ema:
                print(
                    "[Mage BR-v2] streamed DPO: chosen bootstrap active; "
                    "per-pair reject-budget EMA disabled for replay exactness"
                )
                self._brv2_warned_streamed_budget_ema = True
            return score

        if score.ndim != 1 or score.numel() == 0 or score.numel() % 2 != 0:
            return score

        half = int(score.numel() // 2)
        if self._dpo_reference_prediction():
            references: dict[str, Tensor] = {}
            for i in range(half):
                pair_key = self._brv2_pair_key(batch, i)
                if pair_key is not None:
                    references[pair_key] = score[i].detach().float().clone()
            self._brv2_reference_chosen_by_pair = references
            return score

        _, _, decay = self._brv2_validate_settings(config)
        adjusted = score.clone()
        raw_values: list[float] = []
        ema_values: list[float] = []

        for i in range(half):
            pair_key = self._brv2_pair_key(batch, i)
            if pair_key is None:
                continue
            reference = self._brv2_reference_chosen_by_pair.get(pair_key)
            if reference is None:
                continue
            reference = reference.to(device=score.device, dtype=score.dtype)
            raw_ratio_tensor = score[i].detach() - reference
            raw_ratio = float(raw_ratio_tensor.float().cpu().item())
            previous = self._brv2_budget_ema_by_pair.get(pair_key)
            ema_ratio = self._brv2_update_ema(previous, raw_ratio, decay)
            self._brv2_budget_ema_by_pair[pair_key] = ema_ratio

            target_score = reference + score.new_tensor(ema_ratio)
            adjusted[i] = self._brv2_replace_value_preserve_gradient(
                score[i],
                target_score,
            )
            raw_values.append(raw_ratio)
            ema_values.append(ema_ratio)

        if raw_values:
            self._brv2_raw_chosen_reward_mean = sum(raw_values) / len(raw_values)
            self._brv2_ema_chosen_reward_mean = sum(ema_values) / len(ema_values)

        return adjusted

    def calculate_dpo_loss(
            self,
            model,
            batch: dict,
            config: TrainConfig,
            train_progress,
            *,
            objective=None,
            reference_mode=None,
            reference_key=None,
            streamed: bool = False,
            external_chosen_supervised_loss_value: float | None = None,
    ) -> Tensor:
        effective_objective = DPOObjective(
            config.rlhf_dpo_objective if objective is None else objective
        )
        brv2 = effective_objective == DPOObjective.BALANCED_REJECT
        if brv2:
            self._brv2_validate_settings(config)
            self._brv2_active = True
            self._brv2_reference_chosen_by_pair.clear()
            self._brv2_raw_chosen_reward_mean = 0.0
            self._brv2_ema_chosen_reward_mean = 0.0

        try:
            result = super().calculate_dpo_loss(
                model,
                batch,
                config,
                train_progress,
                objective=objective,
                reference_mode=reference_mode,
                reference_key=reference_key,
                streamed=streamed,
                external_chosen_supervised_loss_value=external_chosen_supervised_loss_value,
            )
        finally:
            if brv2:
                self._brv2_active = False
                self._brv2_reference_chosen_by_pair.clear()

        if brv2:
            metrics = self.get_last_dpo_metrics()
            margin = float(metrics.get("reward_margin", 0.0))
            if math.isfinite(margin):
                self._brv2_last_margin = margin

            target, bootstrap, decay = self._brv2_validate_settings(config)
            factor = self._brv2_bootstrap_factor(self._brv2_last_margin, target)
            metrics.update({
                "balanced_v2_margin_target": target,
                "balanced_v2_bootstrap_factor_next": factor,
                "balanced_v2_bootstrap_extra_next": bootstrap * factor,
                "balanced_v2_budget_ema_decay": decay,
                "balanced_v2_raw_chosen_reward": self._brv2_raw_chosen_reward_mean,
                "balanced_v2_ema_chosen_reward": self._brv2_ema_chosen_reward_mean,
                "balanced_v2_budget_ema_active": float(not streamed),
            })
            self._last_dpo_metrics = metrics

        return result
