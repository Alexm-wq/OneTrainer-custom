"""SinkSGD_adv with an adaptive step size and brakes against overfitting (SINKSGD_ADAPTIVE). Experimental.

Builds on adv-optm's SinkSGD_adv, so its options behave the same: normalization then momentum,
Nesterov, SNR preconditioning, OrthoGrad, orthogonal Sinkhorn, spectral normalization, cautious,
geometric and centered weight decay, state precisions (including factored and int8) and stochastic
rounding. On top of that:

- Distance estimation (Prodigy): the step size is lr * d, where d estimates the distance to the
  solution from <g, x0 - x> over the accumulated gradients, with the L1 norm in the denominator
  that suits normalized (sign-like) updates. d may shrink again by ``d_shrink`` once the estimate
  falls, but not below ``d_min_ratio`` times its maximum.
- Momentum renormalization: with normalization then momentum, the momentum is normalized once
  more (``momentum_renorm``), as in the first version of this optimizer.
- Agreement mask: entries whose momentum is small against the momentum of their magnitudes, i.e.
  whose recent gradients disagree, are damped (``agreement_threshold``).
- T-LoRA awareness: T-LoRA leaves the LoRA ranks of other noise levels with exactly zero gradient.
  Such ranks are left out of the step: their momentum, agreement statistics and weights stay as
  they are, and the Sinkhorn normalization covers only the active ranks. Rarely active ranks thus
  keep a momentum of their own steps, and the step per active entry does not depend on how many
  ranks were active. Without T-LoRA every rank has a gradient and nothing changes.
- Split Fused Layers (parameters marked by split_sinkhorn): normalization per stacked part.
- Drift budget and validation brake (see observe_validation_gradient).
- Fused back pass: d is estimated over all parameters of a step. With the fused back pass the
  parameters are updated during the backward pass, so they use the d of the previous step.
"""
import math

from modules.util.optimizer.split_sinkhorn import SINKHORN_CHUNKS

import torch
from torch import Tensor

from adv_optm import SinkSGD_adv
from adv_optm.util import param_update
from adv_optm.util.centered_decay import dequantize_anchor
from adv_optm.util.factorization_util import _factorize_state, _reconstruct_state
from adv_optm.util.OrthoGrad import _orthogonalize_gradient
from adv_optm.util.scaled_optm import scale_update
from adv_optm.util.signed_util import get_signsgd_wd_target
from adv_optm.util.sinkhorn import apply_sr_sinkhorn, get_sinkhorn_wd_scaler
from adv_optm.util.state_util import get_state, init_state_tensor, set_state, upcast_grad_for_precision


class SinkSGDAdaptive(SinkSGD_adv):
    def __init__(
            self,
            params,
            lr: float = 1.0,
            momentum: float = 0.9,
            weight_decay: float = 0.0,
            sinkhorn_iterations: int = 5,
            orthogonal_sinkhorn: bool = False,
            normed_momentum: bool = True,
            momentum_renorm: bool = True,
            snr_cond: bool = False,
            nesterov: bool = False,
            nesterov_coef: float | None = None,
            geometric_wd: bool = False,
            cautious_wd: bool = False,
            stochastic_rounding: bool = True,
            orthogonal_gradient: str = "disabled",
            spectral_normalization: bool = False,
            centered_wd: float = 0.0,
            centered_wd_mode: str = "float8",
            state_precision: str = "bf16_sr",
            compiled_optimizer: bool = False,
            d0: float = 1e-5,
            d_coef: float = 1.0,
            growth_rate: float = 1.05,
            beta3: float = 0.999,
            d_shrink: float = 0.01,
            d_min_ratio: float = 0.1,
            agreement_threshold: float = 0.2,
            drift_budget: float = 0.0,
            validation_alignment_ref: float = 0.1,
            slice_p: int = 11,
            tlora_aware: bool = True,
    ):
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}.")
        if not 0.0 <= momentum < 1.0 or not 0.0 <= beta3 < 1.0:
            raise ValueError("momentum and beta3 must be in [0, 1).")
        if not 0.0 <= agreement_threshold < 1.0:
            raise ValueError("agreement_threshold must be in [0, 1).")
        if agreement_threshold > 0 and momentum == 0:
            raise ValueError("The agreement mask compares momenta; it needs momentum > 0.")
        if not 0.0 <= d_shrink <= 1.0 or not 0.0 <= d_min_ratio <= 1.0:
            raise ValueError("d_shrink and d_min_ratio must be in [0, 1].")
        if drift_budget < 0.0:
            raise ValueError("drift_budget must not be negative.")
        if slice_p < 1:
            raise ValueError(f"slice_p must be at least 1, got {slice_p}.")
        # Steps are taken by step_parameter below; SinkSGD_adv's own compiled step would bypass it.
        super().__init__(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay,
            sinkhorn_iterations=sinkhorn_iterations, orthogonal_sinkhorn=orthogonal_sinkhorn,
            normed_momentum=normed_momentum, snr_cond=snr_cond, nesterov=nesterov,
            nesterov_coef=nesterov_coef, geometric_wd=geometric_wd, cautious_wd=cautious_wd,
            stochastic_rounding=stochastic_rounding, orthogonal_gradient=orthogonal_gradient,
            spectral_normalization=spectral_normalization, centered_wd=centered_wd,
            centered_wd_mode=centered_wd_mode, state_precision=state_precision, compiled_optimizer=False,
        )
        adaptive = {
            "momentum_renorm": momentum_renorm, "compiled_optimizer": compiled_optimizer,
            "d": d0, "d0": d0, "d_max": d0, "d_coef": d_coef, "growth_rate": growth_rate,
            "beta3": beta3, "d_numerator": 0.0, "d_shrink": d_shrink, "d_min_ratio": d_min_ratio,
            "agreement_threshold": agreement_threshold, "drift_budget": drift_budget,
            "drift": 0.0, "brake": 1.0, "alignment": None,
            "validation_alignment_ref": validation_alignment_ref, "slice_p": slice_p,
            "tlora_aware": tlora_aware,
            # Reported as "effective_lr" * d by OneTrainer: the learning rate after drift budget and brake.
            "effective_lr": lr,
        }
        self.defaults.update(adaptive)
        for group in self.param_groups:
            for key, value in adaptive.items():
                group.setdefault(key, value)
            group["compiled_optimizer"] = compiled_optimizer
        self._normalize_fn = torch.compile(apply_sr_sinkhorn, dynamic=True) if compiled_optimizer else apply_sr_sinkhorn
        # Fused back pass: distance sums of the parameters already stepped, per group index.
        self._pending = {}
        self._stepped = set()

    @property
    def supports_fused_back_pass(self):
        return True

    # Distance estimation

    @staticmethod
    def _slice(tensor: Tensor, group: dict) -> Tensor:
        """Every slice_p-th entry: the distance estimate is a ratio of sums, so a sample suffices (Prodigy)."""
        return tensor.reshape(-1)[::group["slice_p"]]

    def _accumulate_distance(self, p: Tensor, group: dict, group_index: int):
        state = self.state[p]
        sliced = self._slice(p.detach(), group)
        if "d_x0" not in state or state["d_x0"].numel() != sliced.numel():
            state["d_x0"] = sliced.float().clone()
            state["d_s"] = torch.zeros_like(state["d_x0"])
        d, d0, lr = group["d"], group["d0"], group["lr"]
        weight = d / d0 * d * lr
        grad = self._slice(p.grad, group).float()
        displacement = state["d_x0"] - sliced.float()
        state["d_s"].mul_(group["beta3"]).add_(grad, alpha=weight)
        terms = torch.stack((
            torch.dot(grad, displacement), state["d_s"].abs().sum(), displacement.square().sum(),
        ))
        pending = self._pending.get(group_index)
        if pending is None:
            self._pending[group_index] = [terms, grad.numel(), weight]
        else:
            pending[0] = pending[0] + terms.to(pending[0].device)
            pending[1] += grad.numel()

    def _finish_distance(self):
        """Update d of every group from its accumulated sums; one device read per group."""
        for group_index, (sums, count, weight) in self._pending.items():
            group = self.param_groups[group_index]
            dot, denominator, drift_squared = sums.tolist()
            d, d0, beta3 = group["d"], group["d0"], group["beta3"]
            numerator = group["d_numerator"] * beta3 + weight * dot
            group["d_numerator"] = numerator
            group["drift"] = math.sqrt(drift_squared / max(count, 1))
            if denominator <= 0.0:
                continue
            estimate = group["d_coef"] * numerator / denominator
            group["d_max"] = max(group["d_max"], estimate)
            # d may not grow while held-out gradients no longer agree with the training direction at all.
            growth_allowed = group["alignment"] is None or group["alignment"] > 0
            if estimate > d:
                if growth_allowed:
                    group["d"] = min(estimate, d * group["growth_rate"])
            else:
                # Shrinking acts as a learning rate decay once progress stalls; the floor keeps training alive.
                floor = max(d0, group["d_min_ratio"] * group["d_max"])
                group["d"] = max(floor, d - group["d_shrink"] * (d - estimate))
        self._pending.clear()
        self._stepped.clear()

    @staticmethod
    def _step_size(group: dict) -> float:
        scale = 1.0
        if group["drift_budget"] > 0:
            scale = max(0.0, 1.0 - group["drift"] / group["drift_budget"])
        group["effective_lr"] = group["lr"] * scale * group["brake"]
        return group["d"] * group["effective_lr"]

    # Stepping

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        stepped = [(i, group, [p for p in group["params"] if p.grad is not None])
                   for i, group in enumerate(self.param_groups)]
        # Finishes the fused back pass of this step, or a scaler path that skipped step().
        self._finish_distance()
        if not any(params for _, _, params in stepped):
            return loss
        for group_index, group, params in stepped:
            for p in params:
                self._accumulate_distance(p, group, group_index)
        self._finish_distance()
        for _, group, params in stepped:
            step_size = self._step_size(group)
            for p in params:
                self._update_parameter(p, group, step_size)
        return loss

    @torch.no_grad()
    def step_parameter(self, p: Tensor, group: dict, i: int | None = None):
        """Fused back pass: step one parameter with the d of the previous step."""
        if p.grad is None:
            return
        if p in self._stepped:
            # A new step began without step() in between (e.g. through a gradient scaler).
            self._finish_distance()
        group_index = next(index for index, g in enumerate(self.param_groups) if g is group)
        self._accumulate_distance(p, group, group_index)
        self._update_parameter(p, group, self._step_size(group))
        self._stepped.add(p)

    def _normalize(self, update: Tensor, p: Tensor, group: dict, is_vector: bool, parts: int | None) -> Tensor:
        if is_vector:
            return update.sign_()
        iterations, ortho = group["sinkhorn_iterations"], group["orthogonal_sinkhorn"]
        if not parts:
            return self._normalize_fn(update, iters=iterations, p=p, ortho_project=ortho)
        if update.shape[0] % parts:
            raise ValueError(f"Split Fused Layers: {update.shape[0]} rows do not split into {parts} parts.")
        for update_part, p_part in zip(update.chunk(parts), p.chunk(parts), strict=True):
            update_part.copy_(self._normalize_fn(update_part, iters=iterations, p=p_part, ortho_project=ortho))
        return update

    @staticmethod
    def _rank_dim(p: Tensor) -> int | None:
        if getattr(p, "_is_lora_B", False):
            return 1
        if getattr(p, "_is_lora_A", False):
            return 0
        return None

    def _active_ranks(self, p: Tensor, grad: Tensor, group: dict) -> tuple[int, Tensor] | None:
        """(dim, indices) of the LoRA ranks with a gradient, or None if every rank has one."""
        dim = self._rank_dim(p) if group["tlora_aware"] else None
        if dim is None:
            return None
        active = grad.movedim(dim, 0).reshape(grad.shape[dim], -1).ne(0).any(1)
        if bool(active.all()):
            return None
        return dim, active.nonzero().flatten()

    def _momentum(self, p: Tensor, group: dict) -> Tensor | None:
        state = self.state.get(p)
        if not state or group["momentum"] == 0:
            return None
        if state.get("factored", False):
            d1, d2 = state["effective_shape"]
            return _reconstruct_state(
                (state["mu_b_nmf"], state["mv_b_nmf"], state["sign"], d2), signed=True, shifter=state["shifter"],
            ).view(p.shape)
        return get_state(state, "momentum_buffer", group["actual_state_precision"]).float()

    def _store_momentum(self, p: Tensor, group: dict, buf: Tensor):
        state = self.state[p]
        if state.get("factored", False):
            d1, d2 = state["effective_shape"]
            # The factorization works in place on its input; buf is still needed afterwards.
            for key, value in zip(("mu_b_nmf", "mv_b_nmf", "sign"),
                                  _factorize_state(buf.reshape(d1, d2).clone(), signed=True, shifter=state["shifter"]),
                                  strict=True):
                state[key].copy_(value)
        else:
            set_state(state, "momentum_buffer", buf, group["actual_state_precision"], None)

    def _update_parameter(self, p: Tensor, group: dict, step_size: float):
        state = self.state[p]
        self._SinkSGD_adv__init_state(p, group)
        grad = upcast_grad_for_precision(p.grad, state, group["state_precision"]).float()
        is_vector = grad.ndim < 2 or getattr(p, "_is_dora_scale", False) or getattr(p, "is_vector", False)
        grad = _orthogonalize_gradient(p, grad, group["orthogonal_gradient"])
        momentum, normed = group["momentum"], group["normed_momentum"]
        nesterov, snr_cond = group["nesterov"], group["snr_cond"]

        ranks = None if is_vector else self._active_ranks(p, grad, group)
        if ranks is None:
            def part(tensor):
                return tensor
        else:
            dim, index = ranks
            if index.numel() == 0:
                return

            def part(tensor):
                return tensor.index_select(dim, index)

        p_part = part(p.detach())
        # Read from p itself: detached views do not carry the mark.
        parts = getattr(p, SINKHORN_CHUNKS, None)
        x = self._normalize(part(grad).clone(), p_part, group, is_vector, parts) if normed else part(grad)

        vt_row = vt_col = denom = None
        buf_part = None
        if momentum != 0:
            buf = self._momentum(p, group)
            if snr_cond:
                if not is_vector:
                    squared = buf.view(buf.shape[0], -1).square()
                    vt_row = (1 - squared.mean(dim=-1)).clamp_min_(1e-30)
                    vt_col = (1 - squared.mean(dim=-2)).clamp_min_(1e-30)
                    del squared
                else:
                    denom = (1.0 - buf.square()).clamp_min_(1e-30).sqrt_()
            buf_part = part(buf).clone()
            if nesterov and normed:
                # Scale the normalized gradient by the empirical buffer magnitude (SNR recovery).
                normed_grad = buf_part.abs().mul_(x)
            buf_part.lerp_(x, 1 - momentum)
            if nesterov:
                coef = momentum if group["nesterov_coef"] is None else group["nesterov_coef"]
                update = normed_grad.lerp_(buf_part, coef) if normed else x.lerp(buf_part, coef)
            else:
                update = buf_part.clone()
            self._store_momentum(p, group, buf_part if ranks is None else buf.index_copy_(dim, index, buf_part))
        else:
            update = x.clone()

        if snr_cond:
            if not is_vector:
                rows = vt_row if ranks is None or dim != 0 else vt_row[index]
                cols = vt_col if ranks is None or dim != 1 else vt_col[index]
                update_2d = update.view(update.shape[0], -1)
                update_2d.mul_(rows.rsqrt().unsqueeze(1)).mul_(cols.rsqrt().unsqueeze(0))
                update = update_2d.atan_().view_as(update)
            else:
                update.atan2_(denom)

        if not normed or group["momentum_renorm"]:
            update = self._normalize(update, p_part, group, is_vector, parts)

        threshold = group["agreement_threshold"]
        if threshold > 0:
            # Factored momentum has no dense precision of its own; the magnitudes use bf16 there.
            precision = "bf16_sr" if state.get("factored", False) else group["actual_state_precision"]
            if "magnitude" not in state:
                init_state_tensor(state, "magnitude", p.shape, precision, p.device,
                                  torch.float32 if state.get("factored", False) else p.dtype, non_neg=True)
            magnitude = get_state(state, "magnitude", precision).float()
            magnitude_part = part(magnitude).clone().lerp_(x.abs(), 1 - momentum)
            set_state(state, "magnitude",
                      magnitude_part if ranks is None else magnitude.index_copy_(dim, index, magnitude_part),
                      precision, None, non_neg=True)
            agreement = buf_part.abs().div_(magnitude_part.clamp_min(1e-12))
            update.mul_(agreement.sub_(threshold).div_(1 - threshold).clamp_(0, 1))

        if ranks is not None:
            update = torch.zeros(p.shape, device=p.device, dtype=update.dtype).index_copy_(dim, index, update)

        wd_scaler = wd_target = cwd_target = None
        if group["geometric_wd"]:
            if group["weight_decay"] > 0:
                if not is_vector:
                    wd_scaler = get_sinkhorn_wd_scaler(p, row_denom=vt_row, col_denom=vt_col)
                else:
                    wd_target = get_signsgd_wd_target(p, denom=denom)
            if is_vector and group["centered_wd"] > 0 and "anchor_data" in state:
                anchor = dequantize_anchor(p, state, group, p.dtype)
                cwd_target = get_signsgd_wd_target(p.sub(anchor), denom=denom)
                del anchor

        if group["spectral_normalization"]:
            update = scale_update(p, update, step_size, state=state)
        else:
            update.mul_(step_size * (4 / math.pi) if snr_cond else step_size)

        param_update.apply_parameter_update(
            self, p, group, update, step_size, wd_scaler=wd_scaler, wd_target=wd_target, cwd_target=cwd_target,
        )
        state["step"] += 1

    # Validation brake

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
                direction = self._momentum(p, group) if gradient is not None else None
                if direction is None:
                    continue
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
