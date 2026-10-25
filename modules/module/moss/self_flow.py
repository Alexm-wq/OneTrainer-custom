"""MOSS-Image training forward for Self-Flow: per-image-token timesteps and feature taps.

MOSSImageTransformer.forward stays the only path for sampling and ordinary training.
This forward reproduces it, but conditions every image token on its own timestep
(text tokens keep the sample's base timestep) and can return the image-token
hidden state after a given main DiT block, or stop there for the EMA teacher.
"""
from dataclasses import dataclass

from modules.module.moss.pipeline_moss_image import MOSSImageTransformer, _linear_input

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MOSSSelfFlowOutput:
    sample: Tensor | None
    feature: Tensor | None


def _time_modulation(transformer: MOSSImageTransformer, times: Tensor) -> tuple[Tensor, Tensor]:
    """Block modulation [M, 6, D] and output-head modulation [M, 2, D] for raw times 1..1000."""
    half = transformer.config.freq_dim // 2
    freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
    phase = times.double()[:, None] * freq
    embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
    time = transformer.time_embedding(_linear_input(embedding, transformer.time_embedding[0])).float()
    mod = transformer.shared_block_modulation(_linear_input(time, transformer.shared_block_modulation))
    head = transformer.video_head_modulation(_linear_input(time, transformer.video_head_modulation))
    return mod.view(times.shape[0], 6, -1), head.view(times.shape[0], 2, -1)


def _is_homogeneous(image_index: Tensor, text_index: Tensor) -> bool:
    return bool((image_index == text_index[:, None]).all())


def moss_self_flow_forward(
        transformer: MOSSImageTransformer,
        hidden_states: Tensor,
        image_timestep: Tensor,
        text_timestep: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor,
        capture_layer: int | None = None,
        stop_at_layer: int | None = None,
) -> MOSSSelfFlowOutput:
    """``image_timestep`` is [B] or [B, H*W] and ``text_timestep`` is [B], both raw times 1..1000."""
    if hidden_states.ndim != 4 or hidden_states.shape[1] != transformer.config.latent_channels:
        raise ValueError("hidden_states must be [B, latent_channels, H, W].")
    batch, _, height, width = hidden_states.shape
    image_tokens = height * width
    if image_timestep.ndim == 1:
        image_timestep = image_timestep[:, None].expand(-1, image_tokens)
    if image_timestep.shape != (batch, image_tokens):
        raise ValueError(f"Self-Flow image timesteps must be [B, H*W], got {tuple(image_timestep.shape)}.")
    if text_timestep.shape != (batch,):
        raise ValueError(f"Self-Flow text timesteps must be [B], got {tuple(text_timestep.shape)}.")
    depth = len(transformer.layers)
    for name, layer in (("capture_layer", capture_layer), ("stop_at_layer", stop_at_layer)):
        if layer is not None and not 0 <= layer < depth:
            raise ValueError(f"{name}={layer} is outside the MOSS block range [0, {depth - 1}].")
    if capture_layer is not None and stop_at_layer is not None:
        raise ValueError("Teacher early exit and student feature capture are separate forward modes.")
    mask = encoder_attention_mask
    if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch or mask.shape != encoder_hidden_states.shape[:2]:
        raise ValueError("encoder_hidden_states must be [B, S, text_dim] with a [B, S] mask.")

    # A sample's tokens use at most three distinct times, so embed each distinct time once.
    times = torch.cat((image_timestep.reshape(-1), text_timestep.reshape(-1))).to(device=hidden_states.device, dtype=torch.float32)
    unique_times, inverse = torch.unique(times, return_inverse=True)
    mod, head = _time_modulation(transformer, unique_times)
    image_index = inverse[:batch * image_tokens].view(batch, image_tokens)
    text_index = inverse[batch * image_tokens:]

    lengths = tuple(mask.bool().sum(1).tolist())
    if any(length == 0 for length in lengths):
        raise ValueError("Each sample needs at least one valid text token.")
    text_length = max(lengths)
    context = torch.stack([
        F.pad(row[valid.bool()], (0, 0, 0, text_length - length))
        for row, valid, length in zip(encoder_hidden_states, mask, lengths, strict=True)
    ])

    if _is_homogeneous(image_index, text_index):
        # Homogeneous teacher view: exactly the native per-sample modulation.
        block_mod = mod[text_index]
        head_mod = head[text_index][:, None]
    else:
        block_mod = torch.cat((
            mod[image_index],
            mod[text_index][:, None].expand(-1, text_length, -1, -1),
        ), 1)
        head_mod = head[image_index]

    video = hidden_states.flatten(2).transpose(1, 2)
    video = transformer.video_embedder(_linear_input(video, transformer.video_embedder).contiguous())
    rope = torch.cat([
        F.pad(transformer.rope(height, width, length, hidden_states.device), (0, 0, 0, text_length - length))
        for length in lengths
    ]).to(video.dtype)
    text = transformer.text_embedder(context, rope[:, image_tokens:], lengths)
    x = torch.cat((video, text), 1)
    joint_lengths = tuple(image_tokens + length for length in lengths)

    feature = None
    for layer_index, layer in enumerate(transformer.layers):
        x = layer(x, rope, block_mod, joint_lengths)
        if layer_index == capture_layer:
            feature = x[:, :image_tokens]
        if layer_index == stop_at_layer:
            return MOSSSelfFlowOutput(sample=None, feature=x[:, :image_tokens])

    x = transformer.final_norm_video(x[:, :image_tokens].to(transformer.final_norm_video.weight.dtype))
    x = x * (1 + head_mod[..., 1, :]) + head_mod[..., 0, :]
    x = transformer.final_linear_video(_linear_input(x, transformer.final_linear_video))
    sample = x.transpose(1, 2).reshape(batch, -1, height, width)
    return MOSSSelfFlowOutput(sample=sample, feature=feature)
