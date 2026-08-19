from __future__ import annotations

from types import MethodType

import torch

import modules.module.MageFlowSelfFlow as mage_self_flow
from modules.util.checkpointing_util import (
    BaseCheckpointLayer,
    CheckpointLayer,
    _remove_checkpoint_keys,
)


# Keep one immutable reference to the numerically validated Mage block body.
# BaseMageFlowSetup previously reassigned _split_block_forward when its custom
# compile experiment was active, so capture the original exactly once.
if hasattr(mage_self_flow, "_ot_uncompiled_split_block_forward"):
    _MAGE_SPLIT_BLOCK_IMPL = mage_self_flow._ot_uncompiled_split_block_forward
else:
    _MAGE_SPLIT_BLOCK_IMPL = mage_self_flow._split_block_forward


def _install_dual_timestep_block_forward(transformer) -> None:
    """Make native Mage blocks accept separate image/text timestep embeddings.

    FLUX.2 Self-Flow calls the transformer's already wrapped blocks directly.
    Mage's upstream block only accepts one ``temb`` for both streams, so this is
    the minimal architecture-specific bridge needed to let Mage use the same
    OneTrainer CheckpointLayer for normal and Self-Flow forwards.
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
    """Use Microsoft's Mage attention processor; only backend selection is OT-controlled."""
    from mage_flow.models.modules.mage_layers import MageDoubleStreamAttnProcessor

    for block in transformer.transformer_blocks:
        block.attn.set_processor(MageDoubleStreamAttnProcessor())


def _install_mage_compile_boundaries() -> None:
    """Keep upstream Mage RoPE and FA4/CuTe outside TorchDynamo.

    OneTrainer normally compiles checkpoint wrappers with ``fullgraph=True``.
    Mage cannot do that with FlashAttention4 because FA4's CuTe Python wrapper
    performs fake-mode/cache bookkeeping that Dynamo must not trace.  Mark only
    the two upstream Mage call sites that are compile-hostile as eager regions;
    QKV/output projections, modulation, norms, MLPs and W8A8 linears remain
    visible to Inductor.
    """
    import mage_flow.models.modules.mage_layers as mage_layers

    if getattr(mage_layers, "_ot_compile_boundaries_installed", False):
        return

    mage_layers.apply_rotary_emb_mageflow = torch.compiler.disable(
        mage_layers.apply_rotary_emb_mageflow,
        reason="Mage complex RoPE stays eager outside TorchDynamo",
    )
    mage_layers.flash_attn_varlen_func = torch.compiler.disable(
        mage_layers.flash_attn_varlen_func,
        reason="Mage FA4/CuTe varlen kernel stays eager outside TorchDynamo",
    )
    mage_layers._ot_compile_boundaries_installed = True


def _install_self_flow_block_dispatch() -> None:
    """Route Mage Self-Flow through the same wrapped blocks as normal training."""

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
    """Use Microsoft's native Mage packing rather than an OT attention processor."""

    def packed_inputs(self, model, latent_tokens, text, text_mask):
        packed_img, img_cu = model.prepare_packed_images(latent_tokens)
        packed_txt, txt_cu = model.prepare_packed_text(text, text_mask)
        return packed_img, packed_txt, img_cu, txt_cu, latent_tokens.shape[0], {}

    setup._packed_inputs = MethodType(packed_inputs, setup)


def _wrap_blocks_with_ot_checkpoint_layer(transformer, config) -> None:
    """Use OT's existing CheckpointLayer, allowing Mage FA4 graph breaks.

    This intentionally does not invent a second checkpoint/compile system.
    The wrapper class and checkpoint behavior are exactly OneTrainer's existing
    implementation.  The only Mage-specific difference is ``fullgraph=False``
    so the explicitly disabled upstream RoPE/FA4 calls can execute eagerly while
    the surrounding INT_W8A8 transformer math remains compiled.
    """
    part = config.transformer
    if part.offloading_enabled():
        raise NotImplementedError(
            "Mage Flow compile currently cannot combine transformer layer "
            "offloading with the FA4 eager graph boundary. Disable transformer "
            "layer offloading; gradient checkpointing remains supported."
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
            # Flux/Flux2 normally use fullgraph=True. Mage's FA4/CuTe call is a
            # legitimate external graph boundary, so Mage alone permits breaks.
            # Inductor still compiles the W8A8 projections/MLPs around it.
            layer.compile(fullgraph=False)
        transformer.transformer_blocks[index] = layer

    transformer._register_state_dict_hook(_remove_checkpoint_keys)


def setup_mage_like_flux2(setup, model, config) -> None:
    """Apply the established OneTrainer/Flux2 block lifecycle to Mage.

    Normal Mage and Self-Flow both call the same CheckpointLayer-wrapped block.
    Mage-specific glue is limited to dual timestep modulation plus the necessary
    eager boundary around Microsoft's RoPE and FA4/CuTe implementation.
    """

    _restore_official_mage_attention(model.transformer)
    _install_mage_compile_boundaries()
    _install_dual_timestep_block_forward(model.transformer)
    _install_self_flow_block_dispatch()
    _install_native_packed_inputs(setup)

    # Disable Mage upstream's internal checkpoint loop. OT's CheckpointLayer now
    # owns checkpointing for both ordinary and Self-Flow forwards.
    model.transformer.checkpoint = False
    model.transformer_offload_conductor = None
    _wrap_blocks_with_ot_checkpoint_layer(model.transformer, config)

    print(
        "[Mage-Flow] transformer blocks use OneTrainer CheckpointLayer "
        f"(checkpoint={config.transformer.checkpointing_enabled()}, "
        f"compile={bool(config.compile)}, fullgraph=False; Mage RoPE/FA4 eager)"
    )
