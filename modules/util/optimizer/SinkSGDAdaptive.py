"""SinkSGD with an adaptive learning rate and brakes against overfitting. Experimental prototype.

Update direction: the gradient is normalized by square-root Sinkhorn (alternating row and column
L2 normalization, as in adv-optm's SinkSGD), averaged by momentum, and normalized again. Every
entry of the direction then has an RMS of about 1, so the step size is set by the learning rate
alone. On top of that:

- Distance estimation (Prodigy): the step size is lr * d, where d estimates the distance to the
  solution from <g, x0 - x> over the accumulated gradients, with the L1 norm in the denominator
  that suits normalized (sign-like) updates. Unlike Prodigy, d may shrink again by ``d_shrink``
  once the estimate falls, e.g. when the gradients start to point back towards the start, but
  not below ``d_min_ratio`` times its maximum: a learning rate decay driven by stalled progress.
- Agreement mask: memorizing single images yields gradients that differ from sample to sample,
  while the features of a concept point the same way across samples. Each entry of the momentum
  of normalized gradients is compared with the momentum of their magnitudes; entries whose
  agreement is below ``agreement_threshold`` are damped. A layer that only sees noise therefore
  moves less as a whole. The momentum horizon should stay well below one epoch.
- Drift budget: the step shrinks linearly with the RMS distance of the trained weights from their
  start and stops at ``drift_budget``.
- Validation brake: ``observe_validation_gradient`` compares a held-out gradient with the current
  training direction. While the training direction stops helping held-out images, the step is
  damped and d stops growing.
"""
import math

from modules.util.bf16_stochastic_rounding import copy_stochastic_

import torch
from torch import Tensor


def _sinkhorn(update: Tensor, iterations: int) -> Tensor:
    """Square-root Sinkhorn normalization to an RMS of about 1 per entry, in place on a float32 copy."""
    if update.ndim < 2:
        return update.mul_(math.sqrt(update.numel()) / update.norm().clamp_min(1e-12))
    matrix = update.view(update.shape[0], -1)
    rows, cols = matrix.shape
    # Normalize the longer dimension first, as adv-optm does, for stability.
    dims = (0, 1) if rows > cols else (1, 0)
    for _ in range(iterations):
        for dim in dims:
            length = matrix.shape[dim]
            norm = matrix.norm(dim=dim, keepdim=True).clamp_min(1 / math.sqrt(length))
            matrix.mul_(math.sqrt(length) / norm)
    return update


class SinkSGDAdaptive(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr: float = 1.0,
            momentum: float = 0.9,
            sinkhorn_iterations: int = 5,
            weight_decay: float = 0.0,
            d0: float = 1e-6,
            d_coef: float = 0.1,
            growth_rate: float = 1.05,
            beta3: float = 0.999,
            d_shrink: float = 0.01,
            d_min_ratio: float = 0.1,
            agreement_threshold: float = 0.2,
            drift_budget: float = 0.0,
            validation_alignment_ref: float = 0.1,
            stochastic_rounding: bool = True,
    ):
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}.")
        if not 0.0 <= momentum < 1.0 or not 0.0 <= beta3 < 1.0:
            raise ValueError("momentum and beta3 must be in [0, 1).")
        if not 0.0 <= agreement_threshold < 1.0:
            raise ValueError("agreement_threshold must be in [0, 1).")
        if not 0.0 <= d_shrink <= 1.0 or not 0.0 <= d_min_ratio <= 1.0:
            raise ValueError("d_shrink and d_min_ratio must be in [0, 1].")
        if drift_budget < 0.0 or weight_decay < 0.0:
            raise ValueError("drift_budget and weight_decay must not be negative.")
        defaults = {
            "lr": lr, "momentum": momentum, "sinkhorn_iterations": sinkhorn_iterations,
            "weight_decay": weight_decay, "d": d0, "d0": d0, "d_max": d0, "d_coef": d_coef,
            "growth_rate": growth_rate, "beta3": beta3, "d_numerator": 0.0, "d_shrink": d_shrink,
            "d_min_ratio": d_min_ratio,
            "agreement_threshold": agreement_threshold, "drift_budget": drift_budget,
            "drift": 0.0, "brake": 1.0, "alignment": None,
            "validation_alignment_ref": validation_alignment_ref,
            # Reported as "effective_lr" * d by OneTrainer: the learning rate after drift budget and brake.
            "effective_lr": lr,
        }
        self.stochastic_rounding = stochastic_rounding
        super().__init__(params, defaults)

    @property
    def supports_fused_back_pass(self):
        # d is estimated over all parameters of a group before any of them is updated.
        return False

    @property
    def supports_memory_efficient_fp16(self):
        return False

    @property
    def supports_flat_params(self):
        return False

    def _init_state(self, p: Tensor, group: dict) -> dict:
        state = self.state[p]
        if "x0" not in state:
            state["x0"] = p.detach().clone()
            state["s"] = torch.zeros_like(p, dtype=torch.float32)
            state["momentum"] = torch.zeros_like(p, dtype=torch.float32)
            if group["agreement_threshold"] > 0:
                state["magnitude"] = torch.zeros_like(p, dtype=torch.float32)
        return state

    @torch.no_grad()
    def _update_d(self, group: dict):
        """Prodigy's distance estimate, with the step weights it uses (d / d0 * d * lr)."""
        d, d0, lr, beta3 = group["d"], group["d0"], group["lr"], group["beta3"]
        weight = d / d0 * d * lr
        # Sums stay on the device and are read once per group, not once per parameter.
        sums = None
        count = 0
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self._init_state(p, group)
            grad = p.grad.float()
            displacement = state["x0"].float() - p.float()
            state["s"].mul_(beta3).add_(grad, alpha=weight)
            terms = torch.stack((
                torch.dot(grad.flatten(), displacement.flatten()),
                state["s"].abs().sum(),
                displacement.square().sum(),
            ))
            sums = terms if sums is None else sums + terms.to(sums.device)
            count += p.numel()
        if sums is None:
            return
        dot, denominator, drift_squared = sums.tolist()
        numerator = group["d_numerator"] * beta3 + weight * dot
        group["d_numerator"] = numerator
        group["drift"] = math.sqrt(drift_squared / max(count, 1))
        if denominator <= 0.0:
            return
        estimate = group["d_coef"] * numerator / denominator
        group["d_max"] = max(group["d_max"], estimate)
        # While held-out gradients disagree with the training direction, d may not grow.
        growth_allowed = group["brake"] >= 1.0
        if estimate > d:
            if growth_allowed:
                group["d"] = min(estimate, d * group["growth_rate"])
        else:
            # Shrinking acts as a learning rate decay once progress stalls; the floor keeps training alive.
            floor = max(d0, group["d_min_ratio"] * group["d_max"])
            group["d"] = max(floor, d - group["d_shrink"] * (d - estimate))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self._update_d(group)
            scale = 1.0
            if group["drift_budget"] > 0:
                scale = max(0.0, 1.0 - group["drift"] / group["drift_budget"])
            scale *= group["brake"]
            group["effective_lr"] = group["lr"] * scale
            step_size = group["d"] * group["effective_lr"]
            beta, threshold = group["momentum"], group["agreement_threshold"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self._init_state(p, group)
                normalized = _sinkhorn(p.grad.float().clone(), group["sinkhorn_iterations"])
                state["momentum"].lerp_(normalized, 1 - beta)
                direction = _sinkhorn(state["momentum"].clone(), group["sinkhorn_iterations"])
                if threshold > 0:
                    state["magnitude"].lerp_(normalized.abs_(), 1 - beta)
                    agreement = state["momentum"].abs() / state["magnitude"].clamp_min(1e-12)
                    direction.mul_(((agreement - threshold) / (1 - threshold)).clamp_(0, 1))
                if step_size == 0:
                    continue
                if group["weight_decay"] > 0:
                    direction.add_(p.float(), alpha=group["weight_decay"])
                if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                    copy_stochastic_(p, p.float().add_(direction, alpha=-step_size))
                else:
                    p.add_(direction.to(p.dtype), alpha=-step_size)
        return loss

    @torch.no_grad()
    def observe_validation_gradient(self, gradients: dict[Tensor, Tensor]):
        """Update the validation brake from a held-out gradient per parameter.

        A first-order decrease of the held-out loss along the current training direction needs a
        positive cosine between both. The brake follows that cosine relative to
        ``validation_alignment_ref``; in high dimensions with noisy diffusion gradients, useful
        alignments are small numbers.
        """
        for group in self.param_groups:
            dot, train_norm, val_norm = 0.0, 0.0, 0.0
            for p in group["params"]:
                gradient = gradients.get(p)
                state = self.state.get(p)
                if gradient is None or not state or "momentum" not in state:
                    continue
                direction = state["momentum"]
                gradient = gradient.float()
                dot += torch.dot(direction.flatten(), gradient.flatten()).item()
                train_norm += direction.square().sum().item()
                val_norm += gradient.square().sum().item()
            if train_norm == 0.0 or val_norm == 0.0:
                continue
            alignment = dot / math.sqrt(train_norm * val_norm)
            previous = group["alignment"]
            group["alignment"] = alignment if previous is None else 0.5 * previous + 0.5 * alignment
            group["brake"] = min(1.0, max(0.0, group["alignment"] / group["validation_alignment_ref"]))

    def tensorboard_stats(self) -> dict[str, float]:
        stats = {}
        for index, group in enumerate(self.param_groups):
            stats[f"sinksgd_adaptive/d/{index}"] = group["d"]
            stats[f"sinksgd_adaptive/drift/{index}"] = group["drift"]
            stats[f"sinksgd_adaptive/brake/{index}"] = group["brake"]
            if group["alignment"] is not None:
                stats[f"sinksgd_adaptive/validation_alignment/{index}"] = group["alignment"]
        return stats
