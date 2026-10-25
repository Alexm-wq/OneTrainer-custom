"""Model-independent parts of Self-Flow (Black Forest Labs, "Self-Supervised Flow Matching").

Ported from the FLUX.2 and Mage-Flow implementations on the mage-flow branches. A
model integration supplies a forward with per-image-token timesteps that can return
an intermediate image-token feature; everything else lives here.
"""
from collections.abc import Iterable
from contextlib import contextmanager, nullcontext

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SelfFlowProjector(nn.Module):
    """Small training-only MLP that maps student features into the teacher's space."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        bottleneck_dim = min(hidden_dim, max(256, hidden_dim // 4))
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, hidden_dim),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.net(hidden_states)


class SelfFlowEMA:
    """fp32 EMA of the trained parameters in CPU memory, used as the Self-Flow teacher.

    The teacher forward runs with the EMA weights by pointing each ``Parameter.data`` at
    a device copy of its EMA value and back afterwards: the student weights are neither
    copied nor rewritten, Parameter objects (optimizer references) stay the same, and the
    autograd version counter is untouched, so a graph that is still alive (mixed RLHF)
    stays valid. Only the EMA travels to the device, one transfer per tensor from pinned
    memory, and the update after each optimizer step runs on the device.

    ``layer_groups`` lists parameter indices per DiT block. Those are swapped in only
    around their own block (``layer_weights``), so a full fine-tune never holds a second
    copy of the whole model on the GPU. All other parameters swap for the whole teacher
    forward (``teacher_weights``).
    """

    # Page-locking much more host memory than this can starve the rest of the system.
    PIN_LIMIT_BYTES = 4 * 1024 ** 3

    def __init__(
            self,
            parameters: Iterable[nn.Parameter],
            decay: float,
            state_dict: dict | None = None,
            layer_groups: list[list[int]] | None = None,
    ):
        self.parameters = list(parameters)
        if not self.parameters:
            raise ValueError("Self-Flow requires at least one trainable parameter.")
        if not 0.0 <= decay < 1.0:
            raise ValueError("Self-Flow EMA decay must be in [0, 1).")

        self.decay = float(decay)
        self.optimization_steps = 0
        self._swapped: set[int] = set()
        self.layer_groups = [list(group) for group in layer_groups or []]
        in_layers = {index for group in self.layer_groups for index in group}
        self.global_group = [index for index in range(len(self.parameters)) if index not in in_layers]
        total_bytes = 4 * sum(parameter.numel() for parameter in self.parameters)
        self._pin = torch.cuda.is_available() and total_bytes <= self.PIN_LIMIT_BYTES
        self.ema_parameters = [self._host_float(parameter) for parameter in self.parameters]

        if state_dict is not None:
            self.load_state_dict(state_dict)

    def _host_float(self, tensor: Tensor) -> Tensor:
        host = torch.empty(tensor.shape, dtype=torch.float32, pin_memory=self._pin)
        host.copy_(tensor.detach())
        return host

    @contextmanager
    def _swap(self, indices: list[int]):
        if self._swapped.intersection(indices):
            raise RuntimeError("Nested Self-Flow EMA swaps are not supported.")
        parameters = [self.parameters[index] for index in indices]
        originals = [parameter.data for parameter in parameters]
        self._swapped.update(indices)
        try:
            for index, parameter in zip(indices, parameters, strict=True):
                parameter.data = self.ema_parameters[index].to(
                    device=parameter.device, dtype=parameter.dtype, non_blocking=True,
                )
            yield
        finally:
            for parameter, original in zip(parameters, originals, strict=True):
                parameter.data = original
            self._swapped.difference_update(indices)

    @contextmanager
    def teacher_weights(self, modules: Iterable[nn.Module] = ()):
        """EMA weights for every parameter outside ``layer_groups``; ``modules`` run in eval mode (LoRA dropout)."""
        modules = list(modules)
        training_states = [module.training for module in modules]
        try:
            for module in modules:
                module.eval()
            with self._swap(self.global_group):
                yield
        finally:
            for module, was_training in zip(modules, training_states, strict=True):
                module.train(was_training)

    def layer_weights(self, layer_index: int):
        if not self.layer_groups:
            return nullcontext()
        return self._swap(self.layer_groups[layer_index])

    @contextmanager
    def sampling_weights(self, modules: Iterable[nn.Module] = ()):
        """All EMA weights at once, for a native forward that has no per-block hook."""
        if not self.layer_groups:
            with self.teacher_weights(modules):
                yield
            return
        # A full fine-tune would need a second model copy on the GPU for a pointer swap.
        # Sampling is rare, so copy the EMA into the weights in place instead.
        active = [parameter.detach().to("cpu", copy=True) for parameter in self.parameters]
        try:
            with torch.no_grad():
                for parameter, ema in zip(self.parameters, self.ema_parameters, strict=True):
                    parameter.data.copy_(ema)
            yield
        finally:
            with torch.no_grad():
                for parameter, value in zip(self.parameters, active, strict=True):
                    parameter.data.copy_(value)

    @torch.no_grad()
    def update(self):
        if self._swapped:
            raise RuntimeError("Cannot update the Self-Flow EMA while its weights are swapped in.")

        weight = 1.0 - self.decay
        for group in [*self.layer_groups, self.global_group]:
            for index in group:
                parameter = self.parameters[index]
                ema = self.ema_parameters[index].to(parameter.device, non_blocking=True)
                ema.lerp_(parameter.detach().float(), weight)
                if ema is not self.ema_parameters[index]:
                    self.ema_parameters[index].copy_(ema, non_blocking=True)
        if self._pin:
            # Device-to-pinned copies are asynchronous; the host reads the EMA on save.
            torch.cuda.synchronize()
        self.optimization_steps += 1

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "optimization_steps": self.optimization_steps,
            "ema_parameters": [parameter.clone() for parameter in self.ema_parameters],
        }

    def load_state_dict(self, state_dict: dict):
        stored_parameters = state_dict.get("ema_parameters")
        if stored_parameters is None:
            raise RuntimeError("Self-Flow backup is missing EMA parameters.")
        if len(stored_parameters) != len(self.parameters):
            raise RuntimeError(
                "Self-Flow EMA parameter count mismatch: "
                f"backup={len(stored_parameters)}, active={len(self.parameters)}"
            )

        loaded_parameters = []
        for index, (stored, parameter) in enumerate(zip(stored_parameters, self.parameters, strict=True)):
            if stored.shape != parameter.shape:
                raise RuntimeError(
                    f"Self-Flow EMA parameter shape mismatch at index {index}: "
                    f"backup={tuple(stored.shape)}, active={tuple(parameter.shape)}"
                )
            loaded_parameters.append(self._host_float(stored))

        # The configured decay wins over the stored one, so it can be changed on resume.
        self.optimization_steps = int(state_dict.get("optimization_steps", 0))
        self.ema_parameters = loaded_parameters


def resolve_self_flow_layers(student_layer: int, teacher_layer: int, depth: int) -> tuple[int, int]:
    """Resolve -1 to the depth-aware defaults (student at ~1/3, teacher at ~5/6 depth)."""
    if depth < 2:
        raise RuntimeError("Self-Flow requires at least two transformer layers.")
    if student_layer < 0:
        student_layer = max(0, round((depth - 1) / 3))
    if teacher_layer < 0:
        teacher_layer = min(depth - 1, max(student_layer + 1, round((depth - 1) * 5 / 6)))
    if not 0 <= student_layer < depth:
        raise ValueError(f"Self-Flow student layer must be in [0, {depth - 1}], got {student_layer}.")
    if not 0 <= teacher_layer < depth:
        raise ValueError(f"Self-Flow teacher layer must be in [0, {depth - 1}], got {teacher_layer}.")
    if student_layer >= teacher_layer:
        raise ValueError("Self-Flow student layer must be earlier than the teacher layer.")
    return student_layer, teacher_layer


def validate_self_flow_config(config):
    if config.self_flow_structural_enabled and not config.self_flow_enabled:
        raise ValueError("Structural Self-Flow requires Self-Flow to be enabled.")
    if not config.self_flow_enabled:
        return
    if not 0.0 <= config.self_flow_mask_ratio <= 0.5:
        raise ValueError("Self-Flow mask ratio must be between 0 and 0.5.")
    if config.self_flow_rep_weight < 0.0:
        raise ValueError("Self-Flow representation weight must be non-negative.")
    if config.self_flow_structural_weight < 0.0:
        raise ValueError("Structural Self-Flow weight must be non-negative.")
    if not 2 <= config.self_flow_structural_tokens <= 2048:
        raise ValueError("Structural Self-Flow token count must be between 2 and 2048.")
    if not 0.0 <= config.self_flow_ema_decay < 1.0:
        raise ValueError("Self-Flow EMA decay must be in [0, 1).")


def stratified_token_indices(
        num_tokens: int,
        sample_count: int,
        device: torch.device | str,
        generator: torch.Generator | None = None,
) -> Tensor:
    """Draw one image-token position from each of ``sample_count`` evenly spaced raster bins.

    A small sample therefore still spans the whole image. The index set is shared
    across the batch.
    """
    if num_tokens < 2:
        raise ValueError("Structural Self-Flow requires at least two image tokens.")
    if sample_count < 2:
        raise ValueError("Structural Self-Flow token count must be at least 2.")

    sample_count = min(int(sample_count), int(num_tokens))
    if sample_count == num_tokens:
        return torch.arange(num_tokens, device=device, dtype=torch.long)

    bin_indices = torch.arange(sample_count, device=device, dtype=torch.long)
    left_edges = torch.div(bin_indices * num_tokens, sample_count, rounding_mode="floor")
    right_edges = torch.div((bin_indices + 1) * num_tokens, sample_count, rounding_mode="floor")
    widths = right_edges - left_edges

    if generator is None:
        offsets = torch.div(widths, 2, rounding_mode="floor")
    else:
        offsets = torch.floor(
            torch.rand(sample_count, device=generator.device, generator=generator).to(device) * widths
        ).to(dtype=torch.long)
    return left_edges + offsets


def representation_alignment_loss(projected_student_feature: Tensor, teacher_feature: Tensor) -> Tensor:
    """Per-sample 1 - mean token cosine similarity."""
    cosine = F.cosine_similarity(
        projected_student_feature.to(dtype=torch.float32),
        teacher_feature.detach().to(dtype=torch.float32),
        dim=-1,
    ).mean(dim=1)
    return 1.0 - cosine


def structural_alignment_loss(
        projected_student_feature: Tensor,
        teacher_feature: Tensor,
        sample_count: int,
        generator: torch.Generator | None = None,
) -> Tensor:
    """Per-sample off-diagonal Gram MSE between sampled student and teacher tokens."""
    if projected_student_feature.ndim != 3 or teacher_feature.ndim != 3:
        raise ValueError("Structural Self-Flow features must have shape [B, N, D].")
    if projected_student_feature.shape != teacher_feature.shape:
        raise ValueError(
            "Structural Self-Flow student/teacher feature shape mismatch: "
            f"student={tuple(projected_student_feature.shape)}, teacher={tuple(teacher_feature.shape)}"
        )

    token_indices = stratified_token_indices(
        num_tokens=projected_student_feature.shape[1],
        sample_count=sample_count,
        device=projected_student_feature.device,
        generator=generator,
    )
    student = F.normalize(
        projected_student_feature.index_select(1, token_indices).to(dtype=torch.float32), dim=-1,
    )
    teacher = F.normalize(
        teacher_feature.detach().index_select(1, token_indices).to(dtype=torch.float32), dim=-1,
    )

    squared_error = (student @ student.transpose(-1, -2) - teacher @ teacher.transpose(-1, -2)).square()
    # Self-similarity is always one after normalization; match only distinct tokens.
    off_diagonal_error = squared_error.sum(dim=(-2, -1)) - squared_error.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    sampled_tokens = token_indices.numel()
    return off_diagonal_error / float(sampled_tokens * (sampled_tokens - 1))
