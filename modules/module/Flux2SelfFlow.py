from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class Flux2SelfFlowForwardOutput:
    sample: Tensor | None
    feature: Tensor | None


class Flux2SelfFlowProjector(nn.Module):
    """Small training-only MLP used to align student and teacher features."""

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


class Flux2SelfFlowEMA:
    """CPU float32 EMA for the active FLUX.2 adapter parameters.

    The manager never replaces a ``Parameter`` object. Teacher inference swaps
    CPU EMA values into the existing storage and restores a CPU snapshot of the
    student values afterwards, keeping optimizer references and state intact.
    """

    def __init__(
            self,
            parameters: Iterable[nn.Parameter],
            decay: float,
            state_dict: dict | None = None,
    ):
        self.parameters = list(parameters)
        if not self.parameters:
            raise ValueError("Self-Flow requires at least one trainable FLUX.2 adapter parameter.")
        if not 0.0 <= decay < 1.0:
            raise ValueError("Self-Flow EMA decay must be in [0, 1).")

        self.decay = float(decay)
        self.optimization_steps = 0
        self._teacher_active = False
        self.ema_parameters = [self._cpu_copy(parameter) for parameter in self.parameters]
        self.student_parameters = [parameter.clone() for parameter in self.ema_parameters]

        if state_dict is not None:
            self.load_state_dict(state_dict)

    @staticmethod
    def _cpu_copy(parameter: nn.Parameter) -> Tensor:
        return parameter.detach().to(device="cpu", dtype=torch.float32).clone()

    @torch.no_grad()
    def _copy_to_active_parameters(self, source: list[Tensor]):
        if len(source) != len(self.parameters):
            raise RuntimeError("Self-Flow EMA parameter count no longer matches the active adapter.")
        for index, (stored, parameter) in enumerate(zip(source, self.parameters, strict=True)):
            if stored.shape != parameter.shape:
                raise RuntimeError(
                    "Self-Flow EMA parameter shape mismatch at index "
                    f"{index}: backup={tuple(stored.shape)}, active={tuple(parameter.shape)}"
                )
            parameter.copy_(stored, non_blocking=False)

    @contextmanager
    def teacher_parameters(self, adapter_modules: Iterable[nn.Module] = ()):
        if self._teacher_active:
            raise RuntimeError("Nested Self-Flow teacher parameter swaps are not supported.")

        adapter_modules = list(adapter_modules)
        training_states = [module.training for module in adapter_modules]
        self._teacher_active = True
        teacher_was_copied = False
        try:
            # Restore even if a device copy fails part-way through the list.
            teacher_was_copied = True
            self._copy_to_active_parameters(self.ema_parameters)
            for module in adapter_modules:
                module.eval()
            yield
        finally:
            try:
                if teacher_was_copied:
                    self._copy_to_active_parameters(self.student_parameters)
            finally:
                for module, was_training in zip(adapter_modules, training_states, strict=True):
                    module.train(was_training)
                self._teacher_active = False

    @contextmanager
    def sampling_parameters(
            self,
            use_teacher: bool,
            adapter_modules: Iterable[nn.Module] = (),
    ):
        """Temporarily select the student or EMA adapter for sampling."""
        if self._teacher_active:
            raise RuntimeError("Nested Self-Flow parameter swaps are not supported.")

        adapter_modules = list(adapter_modules)
        training_states = [module.training for module in adapter_modules]
        active_parameters = [self._cpu_copy(parameter) for parameter in self.parameters]
        selected_parameters = self.ema_parameters if use_teacher else self.student_parameters
        self._teacher_active = True
        selected_was_copied = False
        try:
            selected_was_copied = True
            self._copy_to_active_parameters(selected_parameters)
            for module in adapter_modules:
                module.eval()
            yield
        finally:
            try:
                if selected_was_copied:
                    self._copy_to_active_parameters(active_parameters)
            finally:
                for module, was_training in zip(adapter_modules, training_states, strict=True):
                    module.train(was_training)
                self._teacher_active = False

    @torch.no_grad()
    def update_after_optimizer_step(self):
        if self._teacher_active:
            raise RuntimeError("Cannot update Self-Flow EMA while teacher parameters are active.")

        one_minus_decay = 1.0 - self.decay
        new_student_parameters = []
        for ema_parameter, parameter in zip(self.ema_parameters, self.parameters, strict=True):
            student_parameter = self._cpu_copy(parameter)
            ema_parameter.mul_(self.decay).add_(student_parameter, alpha=one_minus_decay)
            new_student_parameters.append(student_parameter)

        self.student_parameters = new_student_parameters
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
                    "Self-Flow EMA parameter shape mismatch at index "
                    f"{index}: backup={tuple(stored.shape)}, active={tuple(parameter.shape)}"
                )
            loaded_parameters.append(stored.detach().to(device="cpu", dtype=torch.float32).clone())

        self.decay = float(state_dict.get("decay", self.decay))
        self.optimization_steps = int(state_dict.get("optimization_steps", 0))
        self.ema_parameters = loaded_parameters
        # An optimizer-boundary backup stores the current student in the LoRA
        # itself, so rebuilding this snapshot is exact and avoids duplicate data.
        self.student_parameters = [self._cpu_copy(parameter) for parameter in self.parameters]


def flux2_flow_sigma(timestep: Tensor, num_train_timesteps: int) -> Tensor:
    """Match OneTrainer's discrete FLUX.2 interpolation convention exactly."""
    return (timestep.to(dtype=torch.float32) + 1.0) / float(num_train_timesteps)


def flux2_interpolate_token_view(
        clean_tokens: Tensor,
        noise_tokens: Tensor,
        timestep: Tensor,
        num_train_timesteps: int,
) -> Tensor:
    sigma = flux2_flow_sigma(timestep, num_train_timesteps)
    while sigma.ndim < clean_tokens.ndim:
        sigma = sigma.unsqueeze(-1)
    output_dtype = clean_tokens.dtype
    return (
        noise_tokens.to(dtype=torch.float32) * sigma
        + clean_tokens.to(dtype=torch.float32) * (1.0 - sigma)
    ).to(dtype=output_dtype)


def flux2_token_weight_to_spatial(
        token_weight: Tensor,
        token_height: int,
        token_width: int,
) -> Tensor:
    if token_weight.ndim != 2 or token_weight.shape[1] != token_height * token_width:
        raise ValueError(
            "FLUX.2 token weight must have shape [B, token_height * token_width], got "
            f"{tuple(token_weight.shape)}"
        )
    weight = token_weight.reshape(token_weight.shape[0], 1, token_height, token_width)
    return weight.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)


def flux2_stratified_token_indices(
        num_tokens: int,
        sample_count: int,
        device: torch.device | str,
        generator: torch.Generator | None = None,
) -> Tensor:
    """Sample shared image-token positions from evenly spaced raster bins.

    One index is drawn from every bin, so a small sample still spans the whole
    image instead of clustering in a few regions. The returned one-dimensional
    index set is intentionally shared across the batch: chosen/rejected DPO
    branches therefore compare the same spatial positions.
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
            torch.rand(sample_count, device=device, generator=generator) * widths
        ).to(dtype=torch.long)
    return left_edges + offsets


def flux2_structural_alignment_loss(
        projected_student_feature: Tensor,
        teacher_feature: Tensor,
        sample_count: int,
        generator: torch.Generator | None = None,
) -> Tensor:
    """Per-sample off-diagonal Gram-MSE for sampled Self-Flow features."""
    if projected_student_feature.ndim != 3 or teacher_feature.ndim != 3:
        raise ValueError("Structural Self-Flow features must have shape [B, N, D].")
    if projected_student_feature.shape != teacher_feature.shape:
        raise ValueError(
            "Structural Self-Flow student/teacher feature shape mismatch: "
            f"student={tuple(projected_student_feature.shape)}, teacher={tuple(teacher_feature.shape)}"
        )

    token_indices = flux2_stratified_token_indices(
        num_tokens=projected_student_feature.shape[1],
        sample_count=sample_count,
        device=projected_student_feature.device,
        generator=generator,
    )
    student = F.normalize(
        projected_student_feature.index_select(1, token_indices).to(dtype=torch.float32),
        dim=-1,
    )
    teacher = F.normalize(
        teacher_feature.detach().index_select(1, token_indices).to(dtype=torch.float32),
        dim=-1,
    )

    student_relations = student @ student.transpose(-1, -2)
    teacher_relations = teacher @ teacher.transpose(-1, -2)
    squared_error = (student_relations - teacher_relations).square()

    # Self-similarity is always one after normalization and carries no spatial
    # information. Match only relations between distinct image tokens.
    off_diagonal_error = squared_error.sum(dim=(-2, -1)) - squared_error.diagonal(
        dim1=-2,
        dim2=-1,
    ).sum(dim=-1)
    sampled_tokens = token_indices.numel()
    return off_diagonal_error / float(sampled_tokens * (sampled_tokens - 1))


def _embed_timestep(
        transformer: nn.Module,
        timestep: Tensor,
        guidance: Tensor | None,
        dtype: torch.dtype,
) -> Tensor:
    original_shape = timestep.shape
    flat_timestep = timestep.reshape(-1).to(dtype=dtype) * 1000

    flat_guidance = None
    if guidance is not None:
        if guidance.ndim != 1 or guidance.shape[0] != original_shape[0]:
            raise ValueError("FLUX.2 guidance must have shape [B].")
        guidance_shape = (original_shape[0],) + (1,) * (len(original_shape) - 1)
        flat_guidance = guidance.reshape(guidance_shape).expand(original_shape).reshape(-1)
        flat_guidance = flat_guidance.to(dtype=dtype) * 1000

    embedding = transformer.time_guidance_embed(flat_timestep, flat_guidance)
    return embedding.reshape(*original_shape, embedding.shape[-1])


def _token_conditioned_output_norm(
        transformer: nn.Module,
        hidden_states: Tensor,
        image_temb: Tensor,
) -> Tensor:
    """AdaLayerNormContinuous with sequence-shaped conditioning.

    The pinned diffusers implementation chunks 2-D conditioning on dimension
    one. Self-Flow supplies [B, N, D], so the equivalent token-wise form must
    chunk the projected scale/shift on the feature dimension instead.
    """
    norm_out = transformer.norm_out
    embedding = norm_out.linear(norm_out.silu(image_temb).to(hidden_states.dtype))
    scale, shift = torch.chunk(embedding, 2, dim=-1)
    return norm_out.norm(hidden_states) * (1.0 + scale) + shift


def _call_self_flow_block(block: nn.Module, **kwargs):
    """Enter compiled Flux2 checkpoint blocks without Module.compile's _call_impl wrapper.

    PyTorch 2.12 can reject ``nn.Module.compile(fullgraph=True)`` calls when the
    module is entered manually from the Self-Flow traversal, reporting that no
    compiled frames were found. OneTrainer's non-offloaded compiled blocks are
    CheckpointLayer modules whose ``forward`` still contains the exact same
    checkpoint/original-block computation. Compile that bound forward directly
    and cache it for the Self-Flow path. Normal Flux2 keeps using its existing
    module-level compiled call path.
    """
    if getattr(block, "_compiled_call_impl", None) is None:
        return block(**kwargs)

    compiled_forward = getattr(block, "_ot_flux2_self_flow_compiled_forward", None)
    if compiled_forward is None:
        compiled_forward = torch.compile(block.forward, fullgraph=True)
        object.__setattr__(block, "_ot_flux2_self_flow_compiled_forward", compiled_forward)
    return compiled_forward(**kwargs)


def flux2_self_flow_forward(
        transformer: nn.Module,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        image_timestep: Tensor,
        text_timestep: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
        guidance: Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        capture_layer: int | None = None,
        stop_at_layer: int | None = None,
) -> Flux2SelfFlowForwardOutput:
    """Dedicated FLUX.2 training forward with per-image-token timesteps.

    ``image_timestep`` uses the same normalized input convention as the native
    diffusers forward (0..1). The ordinary transformer ``forward`` is not
    modified and remains the only path used by sampling and vanilla training.
    """
    batch_size, image_seq_len = hidden_states.shape[:2]
    text_seq_len = encoder_hidden_states.shape[1]

    if image_timestep.ndim == 1:
        image_timestep = image_timestep[:, None].expand(-1, image_seq_len)
    if image_timestep.shape != (batch_size, image_seq_len):
        raise ValueError(
            "Self-Flow image timesteps must have shape [B, N_image], got "
            f"{tuple(image_timestep.shape)}"
        )
    if text_timestep.shape != (batch_size,):
        raise ValueError(
            f"Self-Flow text timestep must have shape [B], got {tuple(text_timestep.shape)}"
        )

    num_single_layers = len(transformer.single_transformer_blocks)
    for name, layer in (("capture_layer", capture_layer), ("stop_at_layer", stop_at_layer)):
        if layer is not None and not 0 <= layer < num_single_layers:
            raise ValueError(f"{name}={layer} is outside the single-stream layer range [0, {num_single_layers - 1}].")
    if stop_at_layer is not None and capture_layer is not None:
        raise ValueError("Teacher early exit and student feature capture are separate forward modes.")

    dtype = hidden_states.dtype
    image_temb = _embed_timestep(transformer, image_timestep, guidance, dtype)
    text_temb = _embed_timestep(transformer, text_timestep, guidance, dtype)

    double_stream_mod_img = transformer.double_stream_modulation_img(image_temb)
    double_stream_mod_txt = transformer.double_stream_modulation_txt(text_temb)
    single_stream_mod_txt = transformer.single_stream_modulation(text_temb)
    single_stream_mod_img = transformer.single_stream_modulation(image_temb)
    single_stream_mod = torch.cat(
        [
            single_stream_mod_txt[:, None, :].expand(-1, text_seq_len, -1),
            single_stream_mod_img,
        ],
        dim=1,
    )

    hidden_states = transformer.x_embedder(hidden_states)
    encoder_hidden_states = transformer.context_embedder(encoder_hidden_states)

    if img_ids.ndim == 3:
        img_ids = img_ids[0]
    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]

    image_rotary_emb = transformer.pos_embed(img_ids)
    text_rotary_emb = transformer.pos_embed(txt_ids)
    concat_rotary_emb = (
        torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
        torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
    )

    for block in transformer.transformer_blocks:
        encoder_hidden_states, hidden_states = _call_self_flow_block(
            block,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb_mod_img=double_stream_mod_img,
            temb_mod_txt=double_stream_mod_txt,
            image_rotary_emb=concat_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    captured_feature = None

    for layer_index, block in enumerate(transformer.single_transformer_blocks):
        hidden_states = _call_self_flow_block(
            block,
            hidden_states=hidden_states,
            encoder_hidden_states=None,
            temb_mod=single_stream_mod,
            image_rotary_emb=concat_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )

        if layer_index == capture_layer:
            captured_feature = hidden_states[:, text_seq_len:, ...]
        if layer_index == stop_at_layer:
            return Flux2SelfFlowForwardOutput(
                sample=None,
                feature=hidden_states[:, text_seq_len:, ...],
            )

    if capture_layer is not None and captured_feature is None:
        raise RuntimeError("Self-Flow student forward did not capture the requested feature layer.")

    hidden_states = hidden_states[:, text_seq_len:, ...]
    hidden_states = _token_conditioned_output_norm(transformer, hidden_states, image_temb)
    output = transformer.proj_out(hidden_states)
    return Flux2SelfFlowForwardOutput(sample=output, feature=captured_feature)