from __future__ import annotations

from types import MethodType

import torch

import modules.module.MageFlowSelfFlow as mage_self_flow
from modules.util.checkpointing_util import enable_checkpointing


# Keep one immutable reference to the numerically validated Mage block body.
# BaseMageFlowSetup previously reassigned _split_block_forward when its custom
# compile experiment was active, so capture the original exactly once.
if hasattr(mage_self_flow, "_ot_uncompiled_split_block_forward"):
    _MAGE_SPLIT_BLOCK_IMPL = mage_self_flow._ot_uncompiled_split_block_forward
else:
    _MAGE_SPLIT_BLOCK_IMPL = mage_self_flow._split_block_forward


def _install_dual_timestep_block_forward(transformer) -> None:
    """Make native Mage blocks accept separate image/text timestep embeddings.

    FLUX.2 Self-Flow can call the transformer's already wrapped blocks directly.
    Mage's upstream block only accepts one ``temb`` for both streams, so this is
    the minimal architecture-specific bridge needed to let Mage use the exact
    same OneTrainer checkpoint/compile wrappers.

    No parameters or module hierarchy are changed here; only ``forward`` is
    replaced in-place, preserving LoRA names and checkpoint keys.
    """

    for block in transformer.transformer_blocks:
        if getattr(block, "_ot_mage_dual_timestep_forward", False):
            continue

        def forward(
                self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                temb: torch.Tensor | None = None,
                image_rotary_emb: torch.Tensor | None = None,
                txt_cu_lens: torch.Tensor | None = None,
                img_cu_lens: torch.Tensor | None = None,
                joint_attention_kwargs: dict | None = None,
                temb_img: torch.Tensor | None = None,
                temb_txt: torch.Tensor | None = None,
        ):
            if temb_img is None:
                temb_img = temb
            if temb_txt is None:
                temb_txt = temb
            if temb_img is None or temb_txt is None:
                raise ValueError("Mage block requires temb or both temb_img/temb_txt")

            return _MAGE_SPLIT_BLOCK_IMPL(
                self,
                hidden_states,
                encoder_hidden_states,
                temb_img,
                temb_txt,
                image_rotary_emb,
                txt_cu_lens,
                img_cu_lens,
                joint_attention_kwargs,
            )

        block.forward = MethodType(forward, block)
        block._ot_mage_dual_timestep_forward = True


def _restore_official_mage_attention(transformer) -> None:
    """Use Microsoft's Mage attention processor; only the backend selection is OT-controlled."""
    from mage_flow.models.modules.mage_layers import MageDoubleStreamAttnProcessor

    for block in transformer.transformer_blocks:
        block.attn.set_processor(MageDoubleStreamAttnProcessor())


def _install_self_flow_block_dispatch() -> None:
    """Route Mage Self-Flow through the same wrapped blocks as normal training.

    This mirrors Flux2SelfFlow: the Self-Flow forward calls ``block(...)`` and
    leaves checkpointing/torch.compile ownership to OneTrainer's CheckpointLayer
    instead of manually checkpointing or compiling a second block function.
    """

    def wrapped_block_forward(
            block,
            hidden_states,
            encoder_hidden_states,
            temb_img,
            temb_txt,
            image_rotary_emb,
            txt_cu_lens,
            img_cu_lens,
            joint_attention_kwargs=None,
    ):
        return block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=None,
            temb_img=temb_img,
            temb_txt=temb_txt,
            image_rotary_emb=image_rotary_emb,
            txt_cu_lens=txt_cu_lens,
            img_cu_lens=img_cu_lens,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    mage_self_flow._split_block_forward = wrapped_block_forward


def _install_native_packed_inputs(setup) -> None:
    """Stop building the custom OT routing object; use upstream Mage packing."""

    def packed_inputs(self, model, latent_tokens, text, text_mask):
        packed_img, img_cu = model.prepare_packed_images(latent_tokens)
        packed_txt, txt_cu = model.prepare_packed_text(text, text_mask)
        return packed_img, packed_txt, img_cu, txt_cu, latent_tokens.shape[0], {}

    setup._packed_inputs = MethodType(packed_inputs, setup)


def setup_mage_like_flux2(setup, model, config) -> None:
    """Apply OneTrainer's proven Flux2 checkpoint/compile lifecycle to Mage.

    BaseMageFlowSetup's experimental per-block ``block.compile(dynamic=True)``
    path is intentionally bypassed by the caller. This function then installs
    the smallest Mage-specific forward bridge and hands the transformer blocks
    to the same ``enable_checkpointing`` implementation used by Flux/Flux2.
    """

    _restore_official_mage_attention(model.transformer)
    _install_dual_timestep_block_forward(model.transformer)
    _install_self_flow_block_dispatch()
    _install_native_packed_inputs(setup)

    # Disable Mage upstream's internal checkpoint loop. OneTrainer's existing
    # CheckpointLayer now owns both checkpointing and compile, exactly as Flux2.
    model.transformer.checkpoint = False

    model.transformer_offload_conductor = enable_checkpointing(
        model.transformer,
        config,
        config.transformer,
        bool(config.compile),
        [
            (
                model.transformer.transformer_blocks,
                ["hidden_states", "encoder_hidden_states"],
            ),
        ],
    )

    print(
        "[Mage-Flow] transformer blocks use OneTrainer Flux2-style "
        f"CheckpointLayer (checkpoint={config.transformer.gradient_checkpointing}, "
        f"compile={bool(config.compile)})"
    )
