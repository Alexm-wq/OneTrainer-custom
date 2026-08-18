from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MageFlowSelfFlowProjector(nn.Module):
    """Small training-only student->teacher feature projector."""

    def __init__(self, hidden_size: int):
        super().__init__()
        bottleneck = min(hidden_size, max(256, hidden_size // 4))
        self.net = nn.Sequential(
            nn.Linear(hidden_size, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, hidden_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MageFlowSelfFlowEMA:
    """VRAM-safe CPU float32 EMA for the active Mage policy parameters."""

    def __init__(self, modules: Iterable[nn.Module], decay: float = 0.9999, state_dict: dict | None = None):
        self.modules = [module for module in modules if module is not None]
        self.parameters = [p for module in self.modules for p in module.parameters() if p.requires_grad]
        if not self.parameters:
            raise ValueError("Mage Self-Flow requires at least one trainable policy parameter")
        if not 0.0 <= decay < 1.0:
            raise ValueError("Mage Self-Flow EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.optimization_steps = 0
        self._teacher_active = False
        self.ema_parameters = [self._cpu_copy(p) for p in self.parameters]
        self.student_parameters = [p.clone() for p in self.ema_parameters]
        if state_dict is not None:
            self.load_state_dict(state_dict)

    @staticmethod
    def _cpu_copy(parameter: nn.Parameter) -> Tensor:
        return parameter.detach().to(device="cpu", dtype=torch.float32).clone()

    @torch.no_grad()
    def _copy_to_active(self, source: list[Tensor]):
        if len(source) != len(self.parameters):
            raise RuntimeError("Mage Self-Flow EMA parameter count mismatch")
        for index, (stored, parameter) in enumerate(zip(source, self.parameters, strict=True)):
            if stored.shape != parameter.shape:
                raise RuntimeError(
                    f"Mage Self-Flow EMA shape mismatch at {index}: {tuple(stored.shape)} != {tuple(parameter.shape)}"
                )
            parameter.copy_(stored.to(device=parameter.device, dtype=parameter.dtype), non_blocking=False)

    @contextmanager
    def teacher_parameters(self, adapter_modules: Iterable[nn.Module] = ()):
        if self._teacher_active:
            raise RuntimeError("Nested Mage Self-Flow teacher swaps are not supported")
        modules = list(adapter_modules)
        states = [module.training for module in modules]
        self._teacher_active = True
        try:
            self._copy_to_active(self.ema_parameters)
            for module in modules:
                module.eval()
            yield
        finally:
            try:
                self._copy_to_active(self.student_parameters)
            finally:
                for module, state in zip(modules, states, strict=True):
                    module.train(state)
                self._teacher_active = False

    @contextmanager
    def sampling_parameters(self, use_teacher: bool, adapter_modules: Iterable[nn.Module] = ()):
        active = [self._cpu_copy(p) for p in self.parameters]
        selected = self.ema_parameters if use_teacher else self.student_parameters
        try:
            self._copy_to_active(selected)
            yield
        finally:
            self._copy_to_active(active)

    @torch.no_grad()
    def update_after_optimizer_step(self):
        if self._teacher_active:
            raise RuntimeError("Cannot update Mage Self-Flow EMA during a teacher swap")
        one_minus = 1.0 - self.decay
        new_students = []
        for ema, parameter in zip(self.ema_parameters, self.parameters, strict=True):
            student = self._cpu_copy(parameter)
            ema.mul_(self.decay).add_(student, alpha=one_minus)
            new_students.append(student)
        self.student_parameters = new_students
        self.optimization_steps += 1

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "optimization_steps": self.optimization_steps,
            "ema_parameters": [p.clone() for p in self.ema_parameters],
        }

    def load_state_dict(self, state: dict):
        stored = state.get("ema_parameters")
        if stored is None or len(stored) != len(self.parameters):
            raise RuntimeError("Mage Self-Flow checkpoint EMA parameter count mismatch")
        loaded = []
        for index, (source, parameter) in enumerate(zip(stored, self.parameters, strict=True)):
            if source.shape != parameter.shape:
                raise RuntimeError(f"Mage Self-Flow checkpoint EMA shape mismatch at parameter {index}")
            loaded.append(source.detach().to(device="cpu", dtype=torch.float32).clone())
        self.decay = float(state.get("decay", self.decay))
        self.optimization_steps = int(state.get("optimization_steps", 0))
        self.ema_parameters = loaded
        self.student_parameters = [self._cpu_copy(p) for p in self.parameters]


@dataclass
class MageFlowForwardResult:
    predicted: Tensor
    feature: Tensor | None


def _apply_modulation(block: nn.Module, x: Tensor, params: Tensor, cu_lens: Tensor | None):
    # A tokenwise Self-Flow modulation already has one shift/scale/gate vector
    # per packed image token and must not be repeat_interleaved again.
    if params.ndim == 3 and params.shape[:2] == x.shape[:2]:
        shift, scale, gate = params.chunk(3, dim=-1)
        return x * (1.0 + scale) + shift, gate
    return block._modulate(x, params, cu_lens)


def _split_block_forward(
        block: nn.Module,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        temb_img: Tensor,
        temb_txt: Tensor,
        image_rotary_emb: Tensor,
        txt_cu_lens: Tensor | None,
        img_cu_lens: Tensor | None,
        joint_attention_kwargs: dict | None = None,
) -> tuple[Tensor, Tensor]:
    img_mod1, img_mod2 = block.img_mod(temb_img).chunk(2, dim=-1)
    txt_mod1, txt_mod2 = block.txt_mod(temb_txt).chunk(2, dim=-1)

    img_normed = block.img_norm1(hidden_states)
    img_modulated, img_gate1 = _apply_modulation(block, img_normed, img_mod1, img_cu_lens)
    txt_normed = block.txt_norm1(encoder_hidden_states)
    txt_modulated, txt_gate1 = _apply_modulation(block, txt_normed, txt_mod1, txt_cu_lens)

    img_attn_output, txt_attn_output = block.attn(
        hidden_states=img_modulated,
        encoder_hidden_states=txt_modulated,
        image_rotary_emb=image_rotary_emb,
        txt_cu_lens=txt_cu_lens,
        img_cu_lens=img_cu_lens,
        **(joint_attention_kwargs or {}),
    )
    hidden_states = hidden_states + img_gate1 * img_attn_output
    encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

    img_normed2 = block.img_norm2(hidden_states)
    img_modulated2, img_gate2 = _apply_modulation(block, img_normed2, img_mod2, img_cu_lens)
    hidden_states = hidden_states + img_gate2 * block.img_mlp(img_modulated2)

    txt_normed2 = block.txt_norm2(encoder_hidden_states)
    txt_modulated2, txt_gate2 = _apply_modulation(block, txt_normed2, txt_mod2, txt_cu_lens)
    encoder_hidden_states = encoder_hidden_states + txt_gate2 * block.txt_mlp(txt_modulated2)

    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    return encoder_hidden_states, hidden_states


def _final_norm(transformer: nn.Module, hidden: Tensor, temb_img: Tensor, img_cu_lens: Tensor | None) -> Tensor:
    if temb_img.ndim == 3 and temb_img.shape[:2] == hidden.shape[:2]:
        norm = transformer.norm_out
        emb = norm.linear(norm.silu(temb_img).to(hidden.dtype))
        scale, shift = emb.chunk(2, dim=-1)
        return norm.norm(hidden) * (1.0 + scale) + shift
    return transformer.norm_out(hidden, temb_img, cu_seqlens=img_cu_lens)


def mage_flow_forward(
        transformer: nn.Module,
        img: Tensor,
        txt: Tensor,
        image_timesteps: Tensor,
        text_timesteps: Tensor | None,
        img_shapes,
        img_cu_seqlens: Tensor | None = None,
        txt_cu_seqlens: Tensor | None = None,
        capture_layer: int | None = None,
        stop_layer: int | None = None,
        attention_kwargs: dict | None = None,
) -> MageFlowForwardResult:
    """Official Mage dense/packed semantics plus per-image-token time conditioning."""
    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Mage training expects rank-3 image and text token tensors")

    ms_pe = transformer.pos_embed(img_shapes, device=img.device)
    hidden = transformer.img_in(img)
    text = transformer.txt_in(transformer.txt_norm(txt))

    if image_timesteps.ndim == 1:
        temb_img = transformer.time_text_embed(image_timesteps.to(hidden.dtype), hidden)
    elif image_timesteps.ndim == 2:
        flat = image_timesteps.reshape(-1).to(hidden.dtype)
        temb_img = transformer.time_text_embed(flat, hidden).reshape(*image_timesteps.shape, -1)
    else:
        raise ValueError(f"Unsupported Mage image timestep shape {tuple(image_timesteps.shape)}")

    if text_timesteps is None:
        if image_timesteps.ndim != 1:
            raise ValueError("Tokenwise Mage forward requires explicit homogeneous text timesteps")
        text_timesteps = image_timesteps
    temb_txt = transformer.time_text_embed(text_timesteps.to(hidden.dtype), hidden)

    feature = None
    kwargs = attention_kwargs or {}
    for index, block in enumerate(transformer.transformer_blocks):
        if transformer.training and getattr(transformer, "checkpoint", False):
            def block_forward(img_state: Tensor, txt_state: Tensor):
                return _split_block_forward(
                    block,
                    img_state,
                    txt_state,
                    temb_img,
                    temb_txt,
                    ms_pe,
                    txt_cu_seqlens,
                    img_cu_seqlens,
                    kwargs,
                )

            text, hidden = torch.utils.checkpoint.checkpoint(
                block_forward,
                hidden,
                text,
                use_reentrant=False,
            )
        else:
            text, hidden = _split_block_forward(
                block,
                hidden,
                text,
                temb_img,
                temb_txt,
                ms_pe,
                txt_cu_seqlens,
                img_cu_seqlens,
                kwargs,
            )

        if capture_layer is not None and index == capture_layer:
            feature = hidden
        if stop_layer is not None and index >= stop_layer:
            return MageFlowForwardResult(predicted=hidden, feature=feature if feature is not None else hidden)

    hidden = _final_norm(transformer, hidden, temb_img, img_cu_seqlens)
    predicted = transformer.proj_out(hidden)
    return MageFlowForwardResult(predicted=predicted, feature=feature)


def dual_timestep_view(clean: Tensor, noise: Tensor, sigma: Tensor) -> Tensor:
    while sigma.ndim < clean.ndim:
        sigma = sigma.unsqueeze(-1)
    dtype = clean.dtype
    return ((1.0 - sigma.float()) * clean.float() + sigma.float() * noise.float()).to(dtype=dtype)


def structural_alignment_loss(student: Tensor, teacher: Tensor, sample_count: int | None = None) -> Tensor:
    if student.ndim != 3 or teacher.ndim != 3 or student.shape != teacher.shape:
        raise ValueError("Mage structural Self-Flow expects matching [B,N,D] features")
    if sample_count is not None and student.shape[1] > sample_count:
        idx = torch.linspace(0, student.shape[1] - 1, sample_count, device=student.device).round().long()
        student = student.index_select(1, idx)
        teacher = teacher.index_select(1, idx)
    student = F.normalize(student.float(), dim=-1)
    teacher = F.normalize(teacher.detach().float(), dim=-1)
    student_rel = torch.bmm(student, student.transpose(1, 2))
    teacher_rel = torch.bmm(teacher, teacher.transpose(1, 2))
    sq = (student_rel - teacher_rel).square()
    n = sq.shape[-1]
    if n < 2:
        return sq.mean(dim=(1, 2))
    off = sq.sum(dim=(1, 2)) - sq.diagonal(dim1=1, dim2=2).sum(dim=1)
    return off / float(n * (n - 1))
