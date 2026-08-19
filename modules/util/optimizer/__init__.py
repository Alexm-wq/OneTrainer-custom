from __future__ import annotations

import math

import torch


def _install_sinksgd_dpo_momentum_bypass() -> None:
    """Add the OneTrainer DPO momentum-bypass parameter step to adv-optm SinkSGD.

    adv-optm 2.5.12 exposes ``step_parameter`` but no way to apply an isolated
    gradient without feeding it through the optimizer's momentum buffer.
    GenericTrainer intentionally keeps DPO gradients separate and expects an
    optimizer endpoint named ``step_parameter_without_momentum``.

    The bypass keeps SinkSGD's ordinary gradient transforms (OrthoGrad,
    Sinkhorn/sign normalization, spectral scaling and stochastic rounding), but
    executes them with momentum/Nesterov/SNR-momentum conditioning disabled.
    ``update_scale`` scales the *step size*, rather than the raw gradient,
    because SinkSGD normalizes the gradient magnitude. This preserves the DPO
    sample-fraction weighting used by GenericTrainer.
    """
    try:
        from adv_optm import SinkSGD_adv
    except ImportError:
        return

    if (
        bool(getattr(SinkSGD_adv, "supports_dpo_momentum_bypass", False))
        and hasattr(SinkSGD_adv, "step_parameter_without_momentum")
    ):
        return

    @torch.no_grad()
    def step_parameter_without_momentum(
            self,
            parameter: torch.Tensor,
            grad: torch.Tensor,
            group: dict,
            i: int | None = None,
            *,
            apply_weight_decay: bool = True,
            increment_state_step: bool = True,
            update_scale: float = 1.0,
    ) -> None:
        del i  # SinkSGD's per-parameter core does not use the group index.

        if grad is None:
            return

        scale = float(update_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                "SinkSGD DPO momentum-bypass update_scale must be finite and > 0, "
                f"got {update_scale!r}"
            )

        state = self.state[parameter]
        if "step" not in state:
            # Normally initialized by SinkSGD_adv.__init__/load_state_dict. Keep
            # this defensive for parameters added to an optimizer later.
            self.init_step()
            state = self.state[parameter]
        if "step" not in state:
            raise RuntimeError(
                "SinkSGD DPO momentum bypass could not initialize optimizer state"
            )

        # Do not mutate the live parameter group: the normal optimizer step must
        # keep using the user's configured momentum on subsequent updates.
        bypass_group = dict(group)
        bypass_group["momentum"] = 0.0
        bypass_group["nesterov"] = False
        bypass_group["nesterov_coef"] = None
        # SNR conditioning derives its preconditioner from the momentum state,
        # so it is deliberately disabled for a truly momentum-free DPO update.
        bypass_group["snr_cond"] = False
        # Call SinkSGD's eager per-parameter core directly. Compiling a second
        # mutable-group variant is unnecessary and risks sharing the normal
        # momentum graph/cache.
        bypass_group["compiled_optimizer"] = False

        if not apply_weight_decay:
            bypass_group["weight_decay"] = 0.0
            bypass_group["centered_wd"] = 0.0
        elif scale != 1.0:
            # SinkSGD's core uses the same step_size for the parameter update
            # and decoupled/centered decay. GenericTrainer scales only the DPO
            # update by its sample fraction, while weight decay must still be
            # applied exactly once at the full scheduled LR when this is the
            # parameter's only update in the optimizer step.
            bypass_group["weight_decay"] = (
                group.get("weight_decay", 0.0) / scale
            )
            bypass_group["centered_wd"] = (
                group.get("centered_wd", 0.0) / scale
            )

        step_size = group["lr"] * scale
        self._step_parameter(
            parameter,
            grad,
            state,
            bypass_group,
            step_size,
            None,
            None,
        )

        # If a normal gradient already updated this parameter, its ordinary
        # step_parameter() already incremented the state step. Otherwise the
        # isolated DPO update is the parameter's optimizer step for this window.
        if increment_state_step:
            state["step"] += 1

    SinkSGD_adv.step_parameter_without_momentum = (
        step_parameter_without_momentum
    )
    SinkSGD_adv.supports_dpo_momentum_bypass = True


_install_sinksgd_dpo_momentum_bypass()
