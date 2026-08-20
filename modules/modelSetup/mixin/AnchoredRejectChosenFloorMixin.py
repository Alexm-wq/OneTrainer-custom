from modules.util.enum.DPOObjective import DPOObjective

import torch
import torch.nn.functional as F


class AnchoredRejectChosenFloorMixin:
    """Normalize DPO reward scale and protect Anchored Reject chosen quality.

    Every DPO objective uses a symmetric relative-error reward instead of a raw
    absolute loss difference:

        reward = (E_ref - E_policy) / (0.5 * (E_ref + E_policy) + eps)

    The denominator is detached, so normalization only rescales the gradient;
    it never creates gradient through the scale estimate. With non-negative
    prediction errors the reward is naturally bounded near [-2, 2], preventing
    terminal-noise timesteps from dominating merely because their raw error
    scale is much larger.

    Anchored Reject additionally gets a one-sided chosen reward floor at zero.
    It contributes only while chosen_reward < 0 and becomes exactly inactive
    once the policy is at least as good as its reference on the chosen image.
    """

    _DPO_BRANCH_KEY = "_ot_normalized_dpo_branch"
    _DPO_NORMALIZATION_EPS = 1e-6

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dpo_normalization_reference_fast = None
        self._dpo_normalization_reference_stream = {}
        self._dpo_normalization_scale_sum = 0.0
        self._dpo_normalization_scale_count = 0

        self._anchored_floor_active = False
        self._anchored_floor_policy_score = None
        self._anchored_floor_last_value = 0.0
        self._anchored_floor_last_active_fraction = 0.0

    def _create_dpo_stream_batches(self, batch: dict):
        chosen, rejected, chosen_b = super()._create_dpo_stream_batches(batch)
        chosen = dict(chosen)
        rejected = dict(rejected)
        chosen[self._DPO_BRANCH_KEY] = "chosen"
        rejected[self._DPO_BRANCH_KEY] = "rejected"
        return chosen, rejected, chosen_b

    def calculate_dpo_loss(
            self,
            model,
            batch: dict,
            config,
            train_progress,
            *,
            objective=None,
            reference_mode=None,
            reference_key=None,
            streamed: bool = False,
            external_chosen_supervised_loss_value: float | None = None,
    ):
        resolved_objective = DPOObjective(
            config.rlhf_dpo_objective if objective is None else objective
        )

        self._dpo_normalization_reference_fast = None
        self._dpo_normalization_reference_stream = {}
        self._dpo_normalization_scale_sum = 0.0
        self._dpo_normalization_scale_count = 0

        self._anchored_floor_active = (
            resolved_objective == DPOObjective.ANCHORED_REJECT
        )
        self._anchored_floor_policy_score = None
        self._anchored_floor_last_value = 0.0
        self._anchored_floor_last_active_fraction = 0.0

        loss = super().calculate_dpo_loss(
            model,
            batch,
            config,
            train_progress,
            objective=objective,
            reference_mode=reference_mode,
            reference_key=reference_key,
            streamed=streamed,
            external_chosen_supervised_loss_value=(
                external_chosen_supervised_loss_value
            ),
        )

        metrics = getattr(self, "_last_dpo_metrics", None)
        if isinstance(metrics, dict):
            metrics["reward_normalization_active"] = 1.0
            metrics["reward_normalization_scale"] = (
                self._dpo_normalization_scale_sum
                / max(self._dpo_normalization_scale_count, 1)
            )
            if self._anchored_floor_active:
                metrics["anchored_chosen_floor_loss"] = float(
                    self._anchored_floor_last_value
                )
                metrics["anchored_chosen_floor_active_fraction"] = float(
                    self._anchored_floor_last_active_fraction
                )

        # Keep cached reference scores alive through streamed-DPO backward.
        # The custom streamed branch replays its score forward during backward.
        return loss

    def _reference_score_for_policy(self, branch, policy_score: torch.Tensor):
        if branch in {"chosen", "rejected"}:
            reference_score = self._dpo_normalization_reference_stream.get(branch)
            if not isinstance(reference_score, torch.Tensor):
                raise RuntimeError(
                    f"Normalized DPO is missing streamed {branch} reference score"
                )
        else:
            reference_score = self._dpo_normalization_reference_fast
            if not isinstance(reference_score, torch.Tensor):
                raise RuntimeError(
                    "Normalized DPO is missing batched reference score"
                )

        if reference_score.shape != policy_score.shape:
            raise RuntimeError(
                "Normalized DPO policy/reference score shape mismatch: "
                f"policy={tuple(policy_score.shape)}, "
                f"reference={tuple(reference_score.shape)}"
            )
        return reference_score

    def _dpo_score_per_sample(
            self,
            model,
            batch: dict,
            data: dict,
            config,
            objective,
    ):
        raw_score = super()._dpo_score_per_sample(
            model,
            batch,
            data,
            config,
            objective,
        )
        branch = batch.get(self._DPO_BRANCH_KEY)

        if self._dpo_reference_prediction():
            detached_reference = raw_score.detach()
            if branch in {"chosen", "rejected"}:
                self._dpo_normalization_reference_stream[branch] = (
                    detached_reference
                )
            else:
                self._dpo_normalization_reference_fast = detached_reference

            # BaseModelSetup later forms policy_score - reference_score. Returning
            # zero here lets the policy pass return the normalized reward itself.
            return torch.zeros_like(raw_score)

        reference_score = self._reference_score_for_policy(branch, raw_score)

        # Scores are negative losses/errors. abs(score) therefore recovers the
        # non-negative error magnitude while remaining robust to tiny numerical
        # sign noise or a model-specific score implementation.
        scale = 0.5 * (
            reference_score.detach().abs()
            + raw_score.detach().abs()
        )
        scale = scale.clamp_min(self._DPO_NORMALIZATION_EPS)

        normalized_reward = (
            raw_score - reference_score.detach()
        ) / scale

        self._dpo_normalization_scale_sum += float(
            scale.detach().float().mean().item()
        )
        self._dpo_normalization_scale_count += 1

        if self._anchored_floor_active:
            self._anchored_floor_policy_score = normalized_reward

        return normalized_reward

    def rlhf_policy_auxiliary_loss(
            self,
            model,
            batch: dict,
            data: dict,
            config,
    ):
        base_loss = super().rlhf_policy_auxiliary_loss(
            model,
            batch,
            data,
            config,
        )
        if not self._anchored_floor_active:
            return base_loss

        normalized_score = self._anchored_floor_policy_score
        if not isinstance(normalized_score, torch.Tensor):
            raise RuntimeError(
                "Anchored Reject chosen floor is missing normalized policy score"
            )

        branch = batch.get(self._DPO_BRANCH_KEY)
        if branch == "rejected":
            return base_loss

        if branch == "chosen":
            chosen_reward = normalized_score
        else:
            if normalized_score.ndim != 1 or int(normalized_score.shape[0]) % 2 != 0:
                raise RuntimeError(
                    "Anchored Reject batched normalized DPO score must be [2B]"
                )
            chosen_b = int(normalized_score.shape[0]) // 2
            chosen_reward = normalized_score[:chosen_b]

        violation = F.relu(-chosen_reward)
        huber_delta = max(
            float(getattr(config, "rlhf_dpo_anchored_huber_delta", 0.1)),
            1e-8,
        )
        floor_weight = max(
            float(getattr(
                config,
                "rlhf_dpo_anchored_chosen_floor_weight",
                1.0,
            )),
            0.0,
        )
        floor_loss = floor_weight * F.smooth_l1_loss(
            violation,
            torch.zeros_like(violation),
            beta=huber_delta,
            reduction="mean",
        )

        self._anchored_floor_last_value = float(
            floor_loss.detach().float().item()
        )
        self._anchored_floor_last_active_fraction = float(
            (violation.detach() > 0).float().mean().item()
        )

        if base_loss is None:
            return floor_loss
        return base_loss + floor_loss
