"""SinkSGD with a step size controlled by the measured slope and curvature (SINKSGD_CURVATURE).

Experimental. The update direction is SINKSGD_ADAPTIVE's, so SinkSGD_adv's options, T-LoRA
awareness and Split Fused Layers work alike; only the step size s differs. It is not set by hand:

- Slope: every step measures g_{t+1} . u_t, the gradient of the next batch along the previous
  step u_t = (x_t - x_{t+1}) / s_t. That batch is independent of the noise in u_t, so its
  average is the true slope left after the step: positive if the steps stop short, negative if
  they overshoot.
- Curvature: every ``curvature_interval`` steps the trainer computes the gradient of the same
  batch once more at the new weights; its change along the step gives u'Hu, the slope one step
  of size 1 removes. It turns the slope into a dimensionless ratio, slope / (s u'Hu).
- Control: s grows while that ratio exceeds ``slope_target`` and shrinks below it, by at most
  ``gain`` (in log space) per step. A Newton step on s would assume each step only affects itself;
  with momentum every step keeps acting for 1 / (1 - momentum) steps, and the step size then
  oscillates between its limits. Leaving some slope (``slope_target`` 0.25) is more
  conservative than stepping to the bottom of the local parabola; across noisy test problems it
  landed close to the best fixed step size on each, where 0 overshot on one of them.

While the loss falls steeply the step size rises; once the remaining slope is gradient noise it
falls. It starts at ``d0`` (it grows by at most ``growth_rate`` per step once ``min_probes``
curvature measurements exist) and stays between ``d_floor`` and ``d_ceiling``. The learning rate
multiplies the step; leave it at 1.0. All dot products use every ``slice_p``-th entry, as the
distance estimate of Prodigy does.
"""
import math

from modules.util.optimizer.SinkSGDAdaptive import SinkSGDAdaptive

import torch
from torch import Tensor


class SinkSGDCurvature(SinkSGDAdaptive):
    def __init__(
            self,
            params,
            lr: float = 1.0,
            d0: float = 1e-5,
            d_floor: float = 1e-6,
            d_ceiling: float = 1e-2,
            growth_rate: float = 1.1,
            decay_rate: float = 0.95,
            gain: float = 0.01,
            slope_target: float = 0.25,
            curvature_interval: int = 10,
            curvature_beta: float = 0.8,
            slope_beta: float = 0.95,
            min_probes: int = 3,
            momentum: float = 0.95,
            normed_momentum: bool = True,
            momentum_renorm: bool = True,
            nesterov: bool = True,
            agreement_threshold: float = 0.0,
            **kwargs,
    ):
        if not 0.0 < d_floor <= d0 <= d_ceiling:
            raise ValueError(f"Need 0 < D Floor <= Initial D <= D Ceiling, got {d_floor}, {d0}, {d_ceiling}.")
        if growth_rate < 1.0 or not 0.0 < decay_rate <= 1.0 or gain <= 0.0:
            raise ValueError("growth_rate must be at least 1, decay_rate in (0, 1] and gain positive.")
        if curvature_interval < 1 or min_probes < 1:
            raise ValueError("curvature_interval and min_probes must be at least 1.")
        if not 0.0 <= curvature_beta < 1.0 or not 0.0 <= slope_beta < 1.0:
            raise ValueError("curvature_beta and slope_beta must be in [0, 1).")
        super().__init__(
            params, lr=lr, d0=d0, d_floor=d_floor, growth_rate=growth_rate, momentum=momentum,
            normed_momentum=normed_momentum, momentum_renorm=momentum_renorm, nesterov=nesterov,
            agreement_threshold=agreement_threshold, **kwargs,
        )
        curvature = {
            "d_ceiling": d_ceiling, "decay_rate": decay_rate, "gain": gain, "slope_target": slope_target,
            "curvature_interval": curvature_interval, "curvature_beta": curvature_beta,
            "slope_beta": slope_beta, "min_probes": min_probes,
            # Bias-corrected averages: value sums and their weights.
            "slope": 0.0, "slope_weight": 0.0, "curvature": 0.0, "curvature_weight": 0.0,
            "probes": 0, "steps": 0, "last_step": 0.0,
        }
        self.defaults.update(curvature)
        for group in self.param_groups:
            for key, value in curvature.items():
                group.setdefault(key, value)
        self._probing = []

    @property
    def supports_fused_back_pass(self):
        # The curvature probe needs the gradients of all parameters at the stepped weights.
        return False

    def step_parameter(self, p: Tensor, group: dict, i: int | None = None):
        raise RuntimeError("SINKSGD_CURVATURE does not support the fused back pass.")

    # Measurements

    def _observe_slope(self, group: dict, params: list[Tensor]):
        """g_{t+1} . u_t from the weights before and after the last step."""
        if group["last_step"] <= 0.0:
            return
        sums = None
        for p in params:
            state = self.state[p]
            if "c_x" not in state:
                continue
            step = state["c_x"] - self._slice(p.detach(), group).float()
            term = torch.dot(self._slice(p.grad, group).float(), step) / group["last_step"]
            sums = term if sums is None else sums + term.to(sums.device)
        if sums is None:
            return
        beta = group["slope_beta"]
        group["slope"] = beta * group["slope"] + (1 - beta) * sums.item()
        group["slope_weight"] = beta * group["slope_weight"] + (1 - beta)

    def wants_curvature_probe(self) -> bool:
        """After step(): whether the trainer should compute the same batch's gradient once more."""
        return bool(self._probing)

    def cancel_curvature_probe(self):
        for group in self._probing:
            for p in group["params"]:
                self.state.get(p, {}).pop("c_g", None)
        self._probing = []

    @torch.no_grad()
    def finish_curvature_probe(self):
        """Curvature along the last step from the gradient of the same batch at the new weights."""
        for group in self._probing:
            sums = None
            for p in group["params"]:
                state = self.state.get(p)
                if not state or "c_g" not in state:
                    continue
                if p.grad is None:
                    state.pop("c_g")
                    continue
                # (g(x + Δ) - g(x)) . Δ = Δ'HΔ, with Δ = x_{t+1} - x_t = -s u.
                delta = self._slice(p.detach(), group).float() - state["c_x"]
                term = torch.dot(self._slice(p.grad, group).float() - state.pop("c_g"), delta)
                sums = term if sums is None else sums + term.to(sums.device)
            if sums is None:
                continue
            beta = group["curvature_beta"]
            group["curvature"] = beta * group["curvature"] + (1 - beta) * sums.item() / group["last_step"] ** 2
            group["curvature_weight"] = beta * group["curvature_weight"] + (1 - beta)
            group["probes"] += 1
        self._probing = []

    def _update_step_size(self, group: dict):
        if group["probes"] < group["min_probes"] or group["slope_weight"] == 0.0:
            return
        d = group["d"]
        slope = group["slope"] / group["slope_weight"]
        curvature = group["curvature"] / group["curvature_weight"]
        if curvature > 0.0:
            ratio = slope / (group["last_step"] * curvature) - group["slope_target"]
            target = d * math.exp(group["gain"] * math.tanh(ratio))
        else:
            # Negative curvature, e.g. at the saddle a LoRA starts in (up = 0): no parabola to step
            # to, so follow the sign of the remaining slope, as gently as the controller does.
            target = d * math.exp(group["gain"] if slope > 0.0 else -group["gain"])
        target = min(max(target, group["d_floor"]), group["d_ceiling"])
        group["d"] = min(max(target, d * group["decay_rate"]), d * group["growth_rate"])

    # Stepping

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.cancel_curvature_probe()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            self._observe_slope(group, params)
            self._update_step_size(group)
            scale = 1.0
            if group["drift_budget"] > 0:
                scale = max(0.0, 1.0 - group["drift"] / group["drift_budget"])
            group["effective_lr"] = group["lr"] * scale * group["brake"]
            step_size = group["d"] * group["effective_lr"]
            probe = group["steps"] % group["curvature_interval"] == 0 and step_size > 0.0
            drift_squared, count = None, 0
            for p in params:
                state = self.state[p]
                sliced = self._slice(p.detach(), group).float()
                if "d_x0" not in state:
                    state["d_x0"] = sliced.clone()
                term = (state["d_x0"] - sliced).square().sum()
                drift_squared = term if drift_squared is None else drift_squared + term.to(drift_squared.device)
                count += sliced.numel()
                # A float32 slice of float32 weights is a view: keep the values from before the step.
                state["c_x"] = sliced.clone()
                if probe:
                    state["c_g"] = self._slice(p.grad, group).float().clone()
                self._update_parameter(p, group, step_size)
            if group["drift_budget"] > 0:
                group["drift"] = math.sqrt(drift_squared.item() / max(count, 1))
            group["last_step"] = step_size
            group["steps"] += 1
            if probe:
                self._probing.append(group)
        return loss

    def tensorboard_stats(self) -> dict[str, float]:
        stats = {}
        for index, group in enumerate(self.param_groups):
            stats[f"sinksgd_curvature/step/{index}"] = group["d"]
            if group["slope_weight"] > 0:
                stats[f"sinksgd_curvature/slope/{index}"] = group["slope"] / group["slope_weight"]
            if group["curvature_weight"] > 0:
                stats[f"sinksgd_curvature/curvature/{index}"] = group["curvature"] / group["curvature_weight"]
            if group["alignment"] is not None:
                stats[f"sinksgd_curvature/validation_alignment/{index}"] = group["alignment"]
        return stats
