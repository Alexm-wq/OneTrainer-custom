from modules.util.enum.DPOObjective import DPOObjective

import torch
import torch.nn.functional as F


class AnchoredRejectChosenFloorMixin:
    """Add a one-sided chosen-reward floor to Anchored Reject.

    The floor is zero-target only: it contributes gradient while the policy's
    chosen reward is below its DPO reference and becomes exactly inactive once
    chosen_reward >= 0.  It is implemented through the existing policy-auxiliary
    hook so the large shared BaseModelSetup DPO implementation remains untouched.
    """

    _ANCHORED_BRANCH_KEY = "_ot_anchored_dpo_branch"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._anchored_floor_active = False
        self._anchored_floor_reference_fast = None
        self._anchored_floor_reference_stream = {}
        self._anchored_floor_policy_score = None
        self._anchored_floor_last_value = 0.0

    def _create_dpo_stream_batches(self, batch: dict):
        chosen, rejected, chosen_b = super()._create_dpo_stream_batches(batch)
        chosen = dict(chosen)
        rejected = dict(rejected)
        chosen[self._ANCHORED_BRANCH_KEY] = "chosen"
        rejected[self._ANCHORED_BRANCH_KEY] = "rejected"
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
        self._anchored_floor_active = (
            resolved_objective == DPOObjective.ANCHORED_REJECT
        )
        self._anchored_floor_reference_fast = None
        self._anchored_floor_reference_stream = {}
        self._anchored_floor_policy_score = None
        self._anchored_floor_last_value = 0.0

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
        if self._anchored_floor_active:
            metrics = getattr(self, "_last_dpo_metrics", None)
            if isinstance(metrics, dict):
                metrics["anchored_chosen_floor_loss"] = float(
                    self._anchored_floor_last_value
                )
        # Deliberately keep the state alive until the next DPO call. Streamed
        # DPO replays its branch graph during backward after this method returns.
        return loss

    def _dpo_score_per_sample(
            self,
            model,
            batch: dict,
            data: dict,
            config,
            objective,
    ):
        score = super()._dpo_score_per_sample(
            model,
            batch,
            data,
            config,
            objective,
        )
        if DPOObjective(objective) != DPOObjective.ANCHORED_REJECT:
            return score

        branch = batch.get(self._ANCHORED_BRANCH_KEY)
        if self._dpo_reference_prediction():
            if branch in {"chosen", "rejected"}:
                self._anchored_floor_reference_stream[branch] = score.detach()
            else:
                self._anchored_floor_reference_fast = score.detach()
        else:
            self._anchored_floor_policy_score = score
        return score

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

        policy_score = self._anchored_floor_policy_score
        if not isinstance(policy_score, torch.Tensor):
            raise RuntimeError(
                "Anchored Reject chosen floor is missing the policy score"
            )

        branch = batch.get(self._ANCHORED_BRANCH_KEY)
        if branch == "rejected":
            return base_loss

        if branch == "chosen":
            reference_score = self._anchored_floor_reference_stream.get("chosen")
            if not isinstance(reference_score, torch.Tensor):
                raise RuntimeError(
                    "Anchored Reject chosen floor is missing streamed chosen reference"
                )
            chosen_policy_score = policy_score
            chosen_reference_score = reference_score
        else:
            reference_score = self._anchored_floor_reference_fast
            if not isinstance(reference_score, torch.Tensor):
                raise RuntimeError(
                    "Anchored Reject chosen floor is missing batched reference"
                )
            if int(reference_score.numel()) != int(policy_score.numel()):
                raise RuntimeError(
                    "Anchored Reject chosen floor policy/reference size mismatch"
                )
            if int(reference_score.shape[0]) % 2 != 0:
                raise RuntimeError(
                    "Anchored Reject batched DPO score must have an even batch size"
                )
            chosen_b = int(reference_score.shape[0]) // 2
            chosen_policy_score = policy_score[:chosen_b]
            chosen_reference_score = reference_score[:chosen_b]

        chosen_reward = (
            chosen_policy_score - chosen_reference_score.detach()
        )
        violation = F.relu(-chosen_reward)
        huber_delta = max(
            float(getattr(config, "rlhf_dpo_anchored_huber_delta", 0.1)),
            1e-8,
        )
        floor_loss = F.smooth_l1_loss(
            violation,
            torch.zeros_like(violation),
            beta=huber_delta,
            reduction="mean",
        )
        self._anchored_floor_last_value = float(
            floor_loss.detach().float().item()
        )

        if base_loss is None:
            return floor_loss
        return base_loss + floor_loss
