from __future__ import annotations

import math

import torch

from .SinkSGD_adv import SinkSGD_adv as _BaseSinkSGD


class SinkSGD_adv(_BaseSinkSGD):
    """OneTrainer SinkSGD extension with a stateless DPO update path.

    Normal training still uses the vendored adv-optm implementation unchanged.
    DPO momentum bypass supplies an explicit gradient here so the preference
    update receives SinkSGD's normalization/parameter update without reading or
    modifying the normal momentum buffer.
    """

    @property
    def supports_dpo_momentum_bypass(self) -> bool:
        return True

    @torch.no_grad()
    def step_parameter_without_momentum(
            self,
            p: torch.Tensor,
            grad: torch.Tensor,
            group: dict,
            i: int | None = None,
            *,
            apply_weight_decay: bool = True,
            increment_state_step: bool = True,
            update_scale: float = 1.0,
    ) -> None:
        del i
        if grad is None:
            return
        if not math.isfinite(float(update_scale)) or float(update_scale) <= 0.0:
            raise ValueError(f"update_scale must be finite and > 0, got {update_scale!r}")

        # The base optimizer initializes every parameter at construction time,
        # but call its private initializer defensively for restored/new groups.
        init_state = getattr(self, "_SinkSGD_adv__init_state", None)
        if init_state is not None:
            init_state(p, group)

        state = self.state[p]
        bypass_group = dict(group)
        bypass_group["momentum"] = 0.0
        bypass_group["nesterov"] = False
        bypass_group["nesterov_coef"] = None
        bypass_group["snr_cond"] = False

        if not apply_weight_decay:
            bypass_group["weight_decay"] = 0.0
            bypass_group["centered_wd"] = 0.0

        # Use the same Sinkhorn/sign normalization and stochastic-rounding path
        # as a regular SinkSGD step, but never update/reconstruct momentum state.
        step_size = float(group["lr"]) * float(update_scale)
        self._step_parameter(
            p,
            grad,
            state,
            bypass_group,
            step_size,
            None,
            None,
        )

        if increment_state_step:
            state["step"] = int(state.get("step", 0)) + 1
