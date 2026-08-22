"""Compatibility shim for BaseModelSetup with the Balanced Reject ordering rescue.

The full implementation is kept in _BaseModelSetupCore.py as the exact pre-patch
blob.  This shim patches only calculate_dpo_loss at import time, with strict
source assertions so upstream drift fails loudly instead of silently changing
the objective.
"""

import inspect

from modules.modelSetup._BaseModelSetupCore import BaseModelSetup as _CoreBaseModelSetup


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Balanced Reject ordering patch expected exactly one {label} marker, "
            f"found {count}"
        )
    return source.replace(old, new, 1)


def _build_calculate_dpo_loss():
    source = inspect.getsource(_CoreBaseModelSetup.calculate_dpo_loss)

    source = _replace_once(
        source,
        """        balanced_reject_pair_loss = None
        balanced_chosen_budget = None
""",
        """        balanced_reject_pair_loss = None
        balanced_chosen_budget = None
        balanced_ordering_violation = None
        balanced_ordering_pair_loss = None
""",
        "declaration",
    )

    source = _replace_once(
        source,
        """            balanced_huber_delta = max(
                float(getattr(
                    config, "rlhf_dpo_balanced_huber_delta", 0.1
                )),
                1e-8,
            )

            balanced_chosen_budget = F.relu(chosen_ratio.detach())
""",
        """            balanced_huber_delta = max(
                float(getattr(
                    config, "rlhf_dpo_balanced_huber_delta", 0.1
                )),
                1e-8,
            )
            balanced_ordering_weight = max(
                float(getattr(
                    config, "rlhf_dpo_balanced_ordering_weight", 0.5
                )),
                0.0,
            )
            balanced_ordering_margin = max(
                float(getattr(
                    config, "rlhf_dpo_balanced_ordering_margin", 0.02
                )),
                0.0,
            )

            balanced_chosen_budget = F.relu(chosen_ratio.detach())
""",
        "parameter",
    )

    source = _replace_once(
        source,
        """            raw_pair_total_loss = balanced_reject_pair_loss
""",
        """            # Rejected-only ordering rescue. The detached chosen reward
            # is a moving threshold, not a gradient target. While rejected is
            # above chosen - margin this supplies a constant, bounded downward
            # push on rejected. Once the requested margin is reached it shuts
            # off exactly.
            balanced_ordering_violation = F.relu(
                rejected_ratio
                - chosen_ratio.detach()
                + balanced_ordering_margin
            )
            balanced_ordering_pair_loss = (
                balanced_ordering_weight * balanced_ordering_violation
            )
            raw_pair_total_loss = (
                balanced_reject_pair_loss
                + balanced_ordering_pair_loss
            )
""",
        "loss",
    )

    source = _replace_once(
        source,
        """                "balanced_reject_violation": (
                    balanced_reject_violation.detach().mean().item()
                ),
                "balanced_target_satisfied": (
""",
        """                "balanced_reject_violation": (
                    balanced_reject_violation.detach().mean().item()
                ),
                "balanced_ordering_loss": (
                    balanced_ordering_pair_loss.detach().mean().item()
                ),
                "balanced_ordering_violation": (
                    balanced_ordering_violation.detach().mean().item()
                ),
                "balanced_ordering_active": (
                    (balanced_ordering_violation.detach() > 0)
                    .float()
                    .mean()
                    .item()
                ),
                "balanced_target_satisfied": (
""",
        "metric",
    )

    namespace = dict(_CoreBaseModelSetup.calculate_dpo_loss.__globals__)
    exec("class _Patched:\n" + source, namespace)
    return namespace["_Patched"].calculate_dpo_loss


_CoreBaseModelSetup.calculate_dpo_loss = _build_calculate_dpo_loss()
_CoreBaseModelSetup.__module__ = __name__
BaseModelSetup = _CoreBaseModelSetup

__all__ = ["BaseModelSetup"]
