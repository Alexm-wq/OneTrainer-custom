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
- Validation brake: ``observe_validation_gradient`` compares a held-out gradient (OneTrainer sums
  it over a few validation images spread over the validation concepts) with the current
  training direction. While the training direction stops helping held-out images, the step is
  damped; d stops growing once it does not help at all.
"""
import math

from modules.util.bf16_stochastic_rounding import copy_stochastic_
from modules.util.optimizer.split_sinkhorn import SINKHORN_CHUNKS

import torch
from torch import Tensor


def _sinkhorn_parts(update: Tensor, iterations: int, parts: int | None) -> Tensor:
    """Sinkhorn per stacked part of a fused layer (rows split into ``parts``), see split_sinkhorn."""
    if not parts or update.ndim < 2:
        return _sinkhorn(update, iterations)
    if update.shape[0] % parts:
        raise ValueError(f"Split Fused Layers: {update.shape[0]} rows do not split into {parts} parts.")
    for part in update.chunk(parts):
        _sinkhorn(part, iterations)
    return update


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
            d0: float = 1e-5,
            d_coef: float = 1.0,
            growth_rate: float = 1.05,
            beta3: float = 0.999,
            d_shrink: float = 0.01,
            d_min_ratio: float = 0.1,
            agreement_threshold: float = 0.2,
            drift_budget: float = 0.0,
            validation_alignment_ref: float = 0.1,
            stochastic_rounding: bool = True,
            slice_p: int = 11,
            state_precision: str = "bf16_sr",
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
        if slice_p < 1:
            raise ValueError(f"slice_p must be at least 1, got {slice_p}.")
        if state_precision not in ("bf16_sr", "fp32"):
            raise ValueError(f"SINKSGD_ADAPTIVE supports the state precisions bf16_sr and fp32, got {state_precision}.")
        defaults = {
            "lr": lr, "momentum": momentum, "sinkhorn_iterations": sinkhorn_iterations,
            "weight_decay": weight_decay, "d": d0, "d0": d0, "d_max": d0, "d_coef": d_coef,
            "growth_rate": growth_rate, "beta3": beta3, "d_numerator": 0.0, "d_shrink": d_shrink,
            "d_min_ratio": d_min_ratio,
            "agreement_threshold": agreement_threshold, "drift_budget": drift_budget,
            "drift": 0.0, "brake": 1.0, "alignment": None,
            "validation_alignment_ref": validation_alignment_ref,
            "slice_p": slice_p, "state_precision": state_precision,
            # Reported as "effective_lr" * d by OneTrainer: the learning rate after drift budget and brake.
            "effective_lr": lr,
        }
        self.stochastic_rounding = stochastic_rounding
        super().__init__(params, defaults)

    def load_state_dict(self, state_dict: dict) -> None:
        # torch casts loaded states to the parameter dtype; restore the configured state precisions.
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            dtype = torch.bfloat16 if group["state_precision"] == "bf16_sr" else torch.float32
            for p in group["params"]:
                state = self.state.get(p, {})
                for key in ("x0", "s"):
                    if key in state:
                        state[key] = state[key].float()
                for key in ("momentum", "magnitude"):
                    if key in state:
                        state[key] = state[key].to(dtype)

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

    @staticmethod
    def _slice(tensor: Tensor, group: dict) -> Tensor:
        """Every slice_p-th entry: the distance estimate is a ratio of sums, so a sample suffices (Prodigy)."""
        return tensor.reshape(-1)[::group["slice_p"]]

    def _init_state(self, p: Tensor, group: dict) -> dict:
        state = self.state[p]
        sliced = self._slice(p.detach(), group)
        if "x0" not in state or state["x0"].numel() != sliced.numel():
            state["x0"] = sliced.float().clone()
            state["s"] = torch.zeros_like(state["x0"])
        if "momentum" not in state:
            dtype = torch.bfloat16 if group["state_precision"] == "bf16_sr" else torch.float32
            state["momentum"] = torch.zeros_like(p, dtype=dtype)
            if group["agreement_threshold"] > 0:
                state["magnitude"] = torch.zeros_like(p, dtype=dtype)
        return state

    @staticmethod
    def _store(state_tensor: Tensor, value: Tensor):
        if state_tensor.dtype == torch.bfloat16:
            copy_stochastic_(state_tensor, value)
        elif value is not state_tensor:
            state_tensor.copy_(value)

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
            grad = self._slice(p.grad, group).float()
            displacement = state["x0"] - self._slice(p.detach(), group).float()
            state["s"].mul_(beta3).add_(grad, alpha=weight)
            terms = torch.stack((
                torch.dot(grad.flatten(), displacement.flatten()),
                state["s"].abs().sum(),
                displacement.square().sum(),
            ))
            sums = terms if sums is None else sums + terms.to(sums.device)
            count += grad.numel()
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
        # d may not grow while held-out gradients no longer agree with the training direction at all.
        # A small positive alignment still means progress: it damps the step, but must not freeze d
        # at a start value at which the weights barely move and the alignment cannot rise.
        growth_allowed = group["alignment"] is None or group["alignment"] > 0
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
                parts = getattr(p, SINKHORN_CHUNKS, None)
                normalized = _sinkhorn_parts(p.grad.float().clone(), group["sinkhorn_iterations"], parts)
                momentum = state["momentum"].float().lerp_(normalized, 1 - beta)
                self._store(state["momentum"], momentum)
                direction = _sinkhorn_parts(momentum.clone(), group["sinkhorn_iterations"], parts)
                if threshold > 0:
                    magnitude = state["magnitude"].float().lerp_(normalized.abs_(), 1 - beta)
                    self._store(state["magnitude"], magnitude)
                    # Not in place: with fp32 states, momentum and magnitude are the stored states themselves.
                    agreement = momentum.abs().div_(magnitude.clamp_min(1e-12))
                    direction.mul_(agreement.sub_(threshold).div_(1 - threshold).clamp_(0, 1))
                del normalized, momentum
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
                direction = state["momentum"].float()
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
