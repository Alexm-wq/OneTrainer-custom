from __future__ import annotations

from types import MethodType

import torch

import modules.module.MageFlowSelfFlow as mage_self_flow
from modules.module.MageFlowAttention import (
    build_packed_attention_routing,
    install_optimized_mage_attention,
)
from modules.util.checkpointing_util import (
    BaseCheckpointLayer,
    CheckpointLayer,
    _remove_checkpoint_keys,
)


# Keep one immutable reference to the numerically validated Mage block body.
if hasattr(mage_self_flow, "_ot_uncompiled_split_block_forward"):
    _MAGE_SPLIT_BLOCK_IMPL = mage_self_flow._ot_uncompiled_split_block_forward
else:
    _MAGE_SPLIT_BLOCK_IMPL = mage_self_flow._split_block_forward


def _install_dual_timestep_block_forward(transformer) -> None:
    """Make native Mage blocks accept separate image/text timestep embeddings."""

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


def _install_self_flow_block_dispatch() -> None:
    """Route every Mage Self-Flow/DPO block call through the OT wrapper.

    The previous implementation deliberately unwrapped no-grad reference and
    teacher calls back to the eager Mage block because FA4/CuTe could trigger a
    device-side illegal access through the compiled wrapper. The supported
    cuDNN/SDPA path does not have that failure, so no-grad Linear-DPO reference
    and Self-Flow teacher forwards now keep the same compiled CheckpointLayer as
    the trainable policy/student path.
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
    """Pack once, build routing once, and reuse it in every Mage block."""

    def packed_inputs(self, model, latent_tokens, text, text_mask):
        packed_img, img_cu = model.prepare_packed_images(latent_tokens)
        packed_txt, txt_cu = model.prepare_packed_text(text, text_mask)

        routing = build_packed_attention_routing(
            img_cu,
            txt_cu,
            img_token_count=int(packed_img.shape[1]),
            txt_token_count=int(packed_txt.shape[1]),
            # The optimized SDPA path does not use this scheduling hint. Keep a
            # host-known upper bound for the legacy flash-compatible fallback.
            max_joint_seqlen=int(latent_tokens.shape[1] + text.shape[1]),
        )
        attention_kwargs = {"ot_packed_routing": routing}
        return packed_img, packed_txt, img_cu, txt_cu, latent_tokens.shape[0], attention_kwargs

    setup._packed_inputs = MethodType(packed_inputs, setup)


def _wrap_blocks_with_ot_checkpoint_layer(transformer, config) -> None:
    """Use OT CheckpointLayer for grad and no-grad Mage forwards."""
    part = config.transformer
    if part.offloading_enabled():
        raise NotImplementedError(
            "Mage Flow's compiled CheckpointLayer path currently cannot combine "
            "with transformer layer/activation offloading. Disable transformer "
            "offloading; gradient checkpointing remains supported."
        )

    checkpointing = part.checkpointing_enabled()
    train_device = torch.device(config.train_device)
    compile_enabled = bool(config.compile)

    for index, block in enumerate(transformer.transformer_blocks):
        if isinstance(block, BaseCheckpointLayer):
            continue

        layer = CheckpointLayer(
            orig_module=block,
            orig_forward=None,
            train_device=train_device,
            checkpointing=checkpointing,
        )
        if compile_enabled:
            # The small complex-RoPE/packed-attention region is an explicit
            # eager boundary. Projections, modulation, residuals and MLPs remain
            # under Inductor in both grad and no-grad/reference graphs.
            layer.compile(fullgraph=False)
        transformer.transformer_blocks[index] = layer

    transformer._register_state_dict_hook(_remove_checkpoint_keys)


def setup_mage_like_flux2(setup, model, config) -> None:
    """Apply OneTrainer checkpoint/compile plus optimized packed attention."""

    _install_dual_timestep_block_forward(model.transformer)
    install_optimized_mage_attention(model.transformer)
    _install_self_flow_block_dispatch()
    _install_native_packed_inputs(setup)

    # Disable Mage upstream's internal checkpoint loop. OT's CheckpointLayer owns
    # checkpointing and compilation for ordinary, Self-Flow and DPO forwards.
    model.transformer.checkpoint = False
    model.transformer_offload_conductor = None
    _wrap_blocks_with_ot_checkpoint_layer(model.transformer, config)

    print(
        "[Mage-Flow] transformer blocks use OneTrainer CheckpointLayer "
        f"(checkpoint={config.transformer.checkpointing_enabled()}, "
        f"compile={bool(config.compile)}, no-grad-compiled={bool(config.compile)}, "
        "shared-packed-routing=on, grouped-SDPA=on)"
    )
