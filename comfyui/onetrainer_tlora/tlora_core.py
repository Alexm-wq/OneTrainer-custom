"""T-LoRA inference without ComfyUI imports, so OneTrainer's tests can check it against its training code.

A layer's LoRA runs beside the layer at every forward instead of being merged into the weight, and
only the leading LoRA ranks of the current noise level take part: as in OneTrainer and the official
T-LoRA, rank(t) = floor((1000 - t) * (rank - min_rank) / 1000) + min_rank for t = sigma * 1000.
"""
import json
import logging

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def file_min_rank(metadata: dict | None) -> int | None:
    """T-LoRA min rank from a OneTrainer LoRA's safetensors metadata, or None if it does not say."""
    metadata = metadata or {}
    try:
        config = json.loads(metadata["ot_config"])
    except (KeyError, ValueError):
        config = {}
    gradient_only = metadata.get("ot_tlora_gradient_only", str(config.get("tlora_gradient_only", False)))
    if gradient_only == "True":
        logging.warning("OneTrainer T-LoRA: trained in gradient-only mode, a plain LoRA; a rank mask would change it.")
        return None
    if "ot_tlora_min_rank" in metadata:
        return int(metadata["ot_tlora_min_rank"])
    if not config:
        return None
    if not config.get("tlora", False):
        logging.warning("OneTrainer T-LoRA: this LoRA was trained without T-LoRA; the rank mask will change it.")
        return None
    return int(config.get("tlora_min_rank", 1))


def rank_mask(sigmas: Tensor, rank: int, min_rank: int, batch: int, device, dtype) -> Tensor:
    """[batch, rank] mask of the LoRA ranks used at each sample's noise level."""
    times = sigmas.detach().float().reshape(-1).to(device) * 1000
    if times.numel() != batch:
        # Reference latents can run as their own batch; within one sampling step all sigmas agree.
        if not bool((times == times[0]).all()):
            raise ValueError(f"T-LoRA: {times.numel()} noise levels for a batch of {batch}.")
        times = times[:1].expand(batch)
    ranks = ((1000 - times) * (rank - min_rank) / 1000).floor().clamp(0, rank - min_rank) + min_rank
    return (torch.arange(rank, device=device) < ranks[:, None]).to(dtype)


class Schedule:
    """The noise level of the running diffusion-model call, shared by all masked layers of one LoRA."""

    def __init__(self, min_rank: int):
        self.min_rank = min_rank
        self.sigmas = None

    def mask(self, rank: int, batch: int, device, dtype) -> Tensor | None:
        # Outside a diffusion-model call (e.g. references prepared ahead at t = 0) all ranks are used.
        if self.sigmas is None:
            return None
        return rank_mask(self.sigmas, rank, self.min_rank, batch, device, dtype)

    def wrapper(self, executor, x, timestep, context, *args, **kwargs):
        """ComfyUI DIFFUSION_MODEL wrapper: the sampler's sigmas, or the timestep (sigma for MOSS Image)."""
        transformer_options = args[0] if args else kwargs.get("transformer_options", {})
        self.sigmas = transformer_options.get("sigmas", timestep)
        try:
            return executor(x, timestep, context, *args, **kwargs)
        finally:
            self.sigmas = None


class MaskedLoRA:
    """Adds the masked LoRA paths of one linear layer to its output.

    A LoRA of a fused projection (q/k/v into one qkv weight) adds to its slice of the output.
    """

    def __init__(self, module: nn.Module, schedule: Schedule):
        self.module = module
        self.schedule = schedule
        self.parts = []
        self.original_forward = None
        self.original_force_forward = False

    def add(self, up: Tensor, down: Tensor, scale: float, start: int | None = None, size: int | None = None):
        self.parts.append((up, down, scale, start, size))

    def inject(self, device=None):
        if self.original_forward is not None:
            return
        if device is not None:
            self.parts = [(up.to(device), down.to(device), scale, start, size)
                          for up, down, scale, start, size in self.parts]
        self.original_forward = self.module.forward
        # ComfyUI's fused linear+activation kernels skip forward(); this makes them call it.
        self.original_force_forward = getattr(self.module, "comfy_force_forward", False)
        self.module.comfy_force_forward = True
        self.module.forward = self.forward

    def eject(self):
        if self.original_forward is None:
            return
        self.module.forward = self.original_forward
        self.module.comfy_force_forward = self.original_force_forward
        self.original_forward = None

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        out = self.original_forward(x, *args, **kwargs)
        for up, down, scale, start, size in self.parts:
            hidden = F.linear(x, down.to(device=x.device, dtype=x.dtype))
            mask = self.schedule.mask(down.shape[0], hidden.shape[0], hidden.device, hidden.dtype)
            if mask is not None:
                hidden = hidden * mask.view(mask.shape[0], *[1] * (hidden.ndim - 2), -1)
            delta = F.linear(hidden, up.to(device=x.device, dtype=x.dtype)) * scale
            if start is None:
                out = out + delta.to(out.dtype)
            else:
                out = torch.cat((
                    out[..., :start], out[..., start:start + size] + delta.to(out.dtype), out[..., start + size:],
                ), -1)
        return out
