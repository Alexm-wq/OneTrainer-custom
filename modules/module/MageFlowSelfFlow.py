from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MageFlowSelfFlowProjector(nn.Module):
    """Small student->teacher feature projector used by Self-Flow."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MageFlowSelfFlowEMA:
    """CPU-backed EMA of the trainable policy modules.

    The implementation mirrors the existing Flux2 Self-Flow semantics: the
    shadow is independent from OneTrainer's normal sampling EMA and can be
    swapped in only for the no-grad teacher forward.
    """

    def __init__(self, modules: Iterable[nn.Module], decay: float = 0.9999):
        self.decay = float(decay)
        self.shadow: dict[str, Tensor] = {}
        self._capture(modules)

    @staticmethod
    def _named_parameters(modules: Iterable[nn.Module]):
        for module_index, module in enumerate(modules):
            if module is None:
                continue
            for name, parameter in module.named_parameters():
                yield f"{module_index}:{name}", parameter

    def _capture(self, modules: Iterable[nn.Module]):
        self.shadow = {
            name: p.detach().float().cpu().clone()
            for name, p in self._named_parameters(modules)
        }

    @torch.no_grad()
    def update(self, modules: Iterable[nn.Module]):
        one_minus = 1.0 - self.decay
        for name, parameter in self._named_parameters(modules):
            value = parameter.detach().float().cpu()
            if name not in self.shadow:
                self.shadow[name] = value.clone()
            else:
                self.shadow[name].mul_(self.decay).add_(value, alpha=one_minus)

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: dict):
        self.decay = float(state.get("decay", self.decay))
        self.shadow = {
            key: value.detach().float().cpu().clone()
            for key, value in state.get("shadow", {}).items()
        }

    @contextmanager
    def teacher_parameters(self, modules: Iterable[nn.Module]):
        modules = [m for m in modules if m is not None]
        originals: dict[str, Tensor] = {}
        try:
            with torch.no_grad():
                for name, parameter in self._named_parameters(modules):
                    if name not in self.shadow:
                        continue
                    originals[name] = parameter.detach().clone()
                    parameter.copy_(self.shadow[name].to(parameter.device, parameter.dtype))
            yield
        finally:
            with torch.no_grad():
                for name, parameter in self._named_parameters(modules):
                    if name in originals:
                        parameter.copy_(originals[name])


@dataclass
class MageFlowForwardResult:
    predicted: Tensor
    feature: Tensor | None


def _split_block_forward(
        block: nn.Module,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        temb_img: Tensor,
        temb_txt: Tensor,
        image_rotary_emb: Tensor,
        txt_cu_lens: Tensor | None = None,
        img_cu_lens: Tensor | None = None,
        joint_attention_kwargs: dict | None = None,
) -> tuple[Tensor, Tensor]:
    """Mage block forward with separate image/text timestep conditioning.

    Microsoft's current Mage block already type-annotates ``temb`` as a tuple
    and contains the intended tuple-unpack code commented out. Keeping this
    implementation in OneTrainer avoids globally monkey-patching Mage inference.
    Dense/bucketed training is intentionally used here; packed varlen inference
    remains on Microsoft's unmodified forward.
    """
    if img_cu_lens is not None or txt_cu_lens is not None:
        raise NotImplementedError("Tokenwise Mage Self-Flow currently uses dense OneTrainer buckets, not packed varlen batches")

    img_mod1, img_mod2 = block.img_mod(temb_img).chunk(2, dim=-1)
    txt_mod1, txt_mod2 = block.txt_mod(temb_txt).chunk(2, dim=-1)

    img_normed = block.img_norm1(hidden_states)
    img_modulated, img_gate1 = block._modulate(img_normed, img_mod1)
    txt_normed = block.txt_norm1(encoder_hidden_states)
    txt_modulated, txt_gate1 = block._modulate(txt_normed, txt_mod1)

    kwargs = joint_attention_kwargs or {}
    img_attn_output, txt_attn_output = block.attn(
        hidden_states=img_modulated,
        encoder_hidden_states=txt_modulated,
        image_rotary_emb=image_rotary_emb,
        txt_cu_lens=None,
        img_cu_lens=None,
        **kwargs,
    )
    hidden_states = hidden_states + img_gate1 * img_attn_output
    encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

    img_normed2 = block.img_norm2(hidden_states)
    img_modulated2, img_gate2 = block._modulate(img_normed2, img_mod2)
    hidden_states = hidden_states + img_gate2 * block.img_mlp(img_modulated2)

    txt_normed2 = block.txt_norm2(encoder_hidden_states)
    txt_modulated2, txt_gate2 = block._modulate(txt_normed2, txt_mod2)
    encoder_hidden_states = encoder_hidden_states + txt_gate2 * block.txt_mlp(txt_modulated2)

    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    return encoder_hidden_states, hidden_states


def mage_flow_forward(
        transformer: nn.Module,
        img: Tensor,
        txt: Tensor,
        image_timesteps: Tensor,
        text_timesteps: Tensor | None,
        img_shapes,
        capture_layer: int | None = None,
        stop_layer: int | None = None,
        attention_kwargs: dict | None = None,
) -> MageFlowForwardResult:
    """Training forward supporting one timestep per image token.

    ``image_timesteps`` may be [B] (ordinary Mage) or [B,N] (Self-Flow).
    Text keeps one homogeneous timestep per sample. When all image timesteps
    are identical this follows the same operations as the official Mage dense
    forward, which is covered by the smoke tests.
    """
    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Mage training expects [B,N,C] image and [B,T,C] text tensors")

    ms_pe = transformer.pos_embed(img_shapes, device=img.device)
    hidden = transformer.img_in(img)
    text = transformer.txt_in(transformer.txt_norm(txt))

    batch, image_tokens = hidden.shape[:2]
    if image_timesteps.ndim == 1:
        temb_img = transformer.time_text_embed(image_timesteps.to(hidden.dtype), hidden)
    elif image_timesteps.ndim == 2:
        flat = image_timesteps.reshape(-1).to(hidden.dtype)
        temb_img = transformer.time_text_embed(flat, hidden).reshape(batch, image_tokens, -1)
    else:
        raise ValueError(f"Unsupported Mage image timestep shape {tuple(image_timesteps.shape)}")

    if text_timesteps is None:
        text_timesteps = image_timesteps[:, 0] if image_timesteps.ndim == 2 else image_timesteps
    temb_txt = transformer.time_text_embed(text_timesteps.to(hidden.dtype), hidden)

    feature = None
    kwargs = attention_kwargs or {}
    for index, block in enumerate(transformer.transformer_blocks):
        text, hidden = _split_block_forward(
            block,
            hidden,
            text,
            temb_img,
            temb_txt,
            ms_pe,
            joint_attention_kwargs=kwargs,
        )
        if capture_layer is not None and index == capture_layer:
            feature = hidden
        if stop_layer is not None and index >= stop_layer:
            return MageFlowForwardResult(predicted=hidden, feature=feature if feature is not None else hidden)

    hidden = transformer.norm_out(hidden, temb_img, cu_seqlens=None)
    predicted = transformer.proj_out(hidden)
    return MageFlowForwardResult(predicted=predicted, feature=feature)


def dual_timestep_view(clean: Tensor, noise: Tensor, sigma: Tensor) -> Tensor:
    """Rectified-flow interpolation for scalar or tokenwise sigma."""
    while sigma.ndim < clean.ndim:
        sigma = sigma.unsqueeze(-1)
    return (1.0 - sigma) * clean + sigma * noise


def structural_alignment_loss(student: Tensor, teacher: Tensor) -> Tensor:
    """Per-sample relational structural loss, independent of hidden width."""
    student = F.normalize(student.float(), dim=-1)
    teacher = F.normalize(teacher.float(), dim=-1)
    student_rel = torch.bmm(student, student.transpose(1, 2))
    teacher_rel = torch.bmm(teacher, teacher.transpose(1, 2))
    return (student_rel - teacher_rel).square().mean(dim=(1, 2))
