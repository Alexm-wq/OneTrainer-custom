# Copyright 2026 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FLUX.3 window masking; parameters and checkpoint namespaces remain upstream-compatible.

Forward bodies adapted from Diffusers b800f828f3951c4d206e154411bcbfa693cd322e.
Only the unused cache tail is masked. BFL's trained padding remains visible.
Explicit mask arguments survive checkpoint recomputation without shared mutable state.
"""
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import apply_lora_scale


def enable_flux3_window_attention(transformer):
    if getattr(transformer, "_ot_window_attention", False):
        return
    transformer.forward = MethodType(_transformer_forward, transformer)
    for block in [*transformer.txt_mode_blocks, *transformer.mode_blocks["image"], *transformer.mode_blocks["image_cond"]]:
        block.forward = MethodType(_mode_forward, block)
    for block in transformer.single_transformer_blocks:
        block.forward = MethodType(_single_forward, block)
    transformer._ot_window_attention = True


def _mode_forward(
    self,
    hidden_states: torch.Tensor,
    temb_mod: torch.Tensor,
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # modulation is split here rather than in Flux3Modulation, to avoid passing tuples into checkpoints https://github.com/huggingface/diffusers/issues/12776
    mod_shift, mod_scale, mod_gate = torch.chunk(temb_mod, 3, dim=-1)

    norm_hidden_states = self.norm(hidden_states)
    norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift

    attn_output = self.attn(hidden_states=norm_hidden_states, image_rotary_emb=image_rotary_emb, attention_mask=attention_mask)
    ff_output = self.ff(norm_hidden_states)

    return hidden_states + mod_gate * (attn_output + ff_output)


def _single_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb_mod: torch.Tensor,
    temb_mod_context: torch.Tensor,
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # modulation is split here rather than in Flux3Modulation, to avoid passing tuples into checkpoints https://github.com/huggingface/diffusers/issues/12776
    mod_shift, mod_scale, mod_gate = torch.chunk(temb_mod, 3, dim=-1)
    c_mod_shift, c_mod_scale, c_mod_gate = torch.chunk(temb_mod_context, 3, dim=-1)

    norm_hidden_states = self.norm(hidden_states)
    norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift

    norm_encoder_hidden_states = self.norm(encoder_hidden_states)
    norm_encoder_hidden_states = (1 + c_mod_scale) * norm_encoder_hidden_states + c_mod_shift

    combined = torch.cat([norm_hidden_states, norm_encoder_hidden_states], dim=1)

    attn_output = self.attn(hidden_states=combined, image_rotary_emb=image_rotary_emb, attention_mask=attention_mask)
    ff_output = self.ff(combined)
    output = attn_output + ff_output

    content_output, context_output = torch.split(
        output, [hidden_states.shape[1], encoder_hidden_states.shape[1]], dim=1
    )
    encoder_hidden_states = encoder_hidden_states + c_mod_gate * context_output
    hidden_states = hidden_states + mod_gate * content_output

    return encoder_hidden_states, hidden_states


@apply_lora_scale("attention_kwargs")
def _transformer_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    img_ids: torch.Tensor,
    txt_ids: torch.Tensor,
    image_cond_hidden_states: torch.Tensor | None = None,
    image_cond_ids: torch.Tensor | None = None,
    target_timestep: torch.Tensor | None = None,
    attention_kwargs: dict[str, Any] | None = None,
    return_dict: bool = True,
    text_window_mask: torch.Tensor | None = None,
) -> Transformer2DModelOutput | tuple[torch.Tensor]:
    """Pinned Diffusers forward with explicit masks for unused cache tails."""
    if image_cond_hidden_states is not None and image_cond_ids is None:
        raise ValueError("`image_cond_hidden_states` requires `image_cond_ids`.")
    if image_cond_hidden_states is None and image_cond_ids is not None:
        raise ValueError("`image_cond_ids` was passed without `image_cond_hidden_states`.")
    if (target_timestep is not None) != bool(self.config.flow_map):
        raise ValueError(
            "`target_timestep` is required by, and only by, a flow-map transformer (`flow_map=True`)."
        )

    batch_size, image_seq_len = hidden_states.shape[:2]
    dtype = hidden_states.dtype

    # 1. Timestep embeddings: the image tokens at `timestep`, the text context and the reference tokens at 0.
    #    Each is a per-token embedding (batch_size, seq_len, dim) since the modulation is per token.
    image_temb = self.time_proj(timestep.to(dtype) * 1000)
    image_temb = self.timestep_embedder(image_temb.to(dtype))[:, None, :]
    zero_temb = self.time_proj(torch.zeros(batch_size, device=hidden_states.device, dtype=dtype))
    zero_temb = self.timestep_embedder(zero_temb.to(dtype))[:, None, :]
    if target_timestep is not None:
        # flow map: the embedding of the timestep this step jumps to is added to every mode's embedding, so it
        # reaches the per-mode blocks, the joint blocks and the output layer
        target_temb = self.time_proj(target_timestep.to(dtype) * 1000)
        target_temb = self.target_timestep_embedder(target_temb.to(dtype))[:, None, :]
        image_temb = image_temb + target_temb
        zero_temb = zero_temb + target_temb
    image_temb = image_temb.expand(batch_size, image_seq_len, -1)
    zero_temb_act = F.silu(zero_temb)

    # 2. Text context: the two modulation sets, input projection and RoPE
    txt_temb_act = zero_temb_act.expand(batch_size, encoder_hidden_states.shape[1], -1)
    txt_early_mod = self.early_stream_modulations["txt"](txt_temb_act)
    txt_single_mod = self.single_stream_modulations["txt"](txt_temb_act)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)
    txt_rotary_emb = self.pos_embed(txt_ids)

    # 3. Image tokens
    image_temb_act = F.silu(image_temb)
    image_early_mod = self.early_stream_modulations["image"](image_temb_act)
    image_single_mod = self.single_stream_modulations["image"](image_temb_act)
    hidden_states = self.x_embedders["image"](hidden_states)
    image_rotary_emb = self.pos_embed(img_ids)

    # 4. Reference image tokens (optional)
    if image_cond_hidden_states is not None:
        cond_temb_act = zero_temb_act.expand(batch_size, image_cond_hidden_states.shape[1], -1)
        cond_early_mod = self.early_stream_modulations["image_cond"](cond_temb_act)
        cond_single_mod = self.single_stream_modulations["image_cond"](cond_temb_act)
        image_cond_hidden_states = self.x_embedders["image_cond"](image_cond_hidden_states)
        cond_rotary_emb = self.pos_embed(image_cond_ids)

    # 5. Per-mode transformer blocks (each mode self-attends within itself only)
    for block in self.txt_mode_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states = self._gradient_checkpointing_func(
                block, encoder_hidden_states, txt_early_mod, txt_rotary_emb, text_window_mask
            )
        else:
            encoder_hidden_states = block(encoder_hidden_states, txt_early_mod, txt_rotary_emb, text_window_mask)

    for block in self.mode_blocks["image"]:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                block, hidden_states, image_early_mod, image_rotary_emb
            )
        else:
            hidden_states = block(hidden_states, image_early_mod, image_rotary_emb)

    if image_cond_hidden_states is not None:
        for block in self.mode_blocks["image_cond"]:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                image_cond_hidden_states = self._gradient_checkpointing_func(
                    block, image_cond_hidden_states, cond_early_mod, cond_rotary_emb
                )
            else:
                image_cond_hidden_states = block(image_cond_hidden_states, cond_early_mod, cond_rotary_emb)

    # 6. Joint attention is permutation-equivariant. Put image/reference before
    # text so every visible key is a prefix, as required by flash_varlen.
    # RoPE follows the same permutation; outputs are split back by stream.
    if image_cond_hidden_states is not None:
        content_states = torch.cat([hidden_states, image_cond_hidden_states], dim=1)
        content_mod = torch.cat([image_single_mod, cond_single_mod], dim=1)
        joint_rotary_emb = (
            torch.cat([image_rotary_emb[0], cond_rotary_emb[0], txt_rotary_emb[0]], dim=0),
            torch.cat([image_rotary_emb[1], cond_rotary_emb[1], txt_rotary_emb[1]], dim=0),
        )
    else:
        content_states = hidden_states
        content_mod = image_single_mod
        joint_rotary_emb = (
            torch.cat([image_rotary_emb[0], txt_rotary_emb[0]], dim=0),
            torch.cat([image_rotary_emb[1], txt_rotary_emb[1]], dim=0),
        )

    joint_mask = None
    if text_window_mask is not None:
        joint_mask = torch.cat((torch.ones(
            batch_size, 1, 1, content_states.shape[1], dtype=torch.bool, device=hidden_states.device,
        ), text_window_mask), dim=-1)

    for block in self.single_transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, content_states = self._gradient_checkpointing_func(
                block, content_states, encoder_hidden_states, content_mod, txt_single_mod, joint_rotary_emb, joint_mask
            )
        else:
            encoder_hidden_states, content_states = block(
                hidden_states=content_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod=content_mod,
                temb_mod_context=txt_single_mod,
                image_rotary_emb=joint_rotary_emb,
                attention_mask=joint_mask,
            )

    # 7. Output layers: the image tokens sit first in the content sequence; the reference tokens produce no output
    hidden_states = content_states[:, :image_seq_len]
    hidden_states = self.norm_out(hidden_states, image_temb)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)
