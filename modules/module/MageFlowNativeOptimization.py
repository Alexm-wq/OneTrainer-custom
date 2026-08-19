from __future__ import annotations

import inspect
import textwrap
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


def _patch_upstream_mage_attention_processor_for_compile() -> None:
    """Patch Mage's CUDA-scalar compile hazards without changing FA4 semantics.

    The pinned upstream Mage processor is retained verbatim except for:

    * ``joint_cu_lens[-1]`` being used as a torch.empty() shape. This CUDA
      scalar causes Dynamo's data-dependent-shape failure. The exact same total
      is available as the sum of the already-flattened text/image query lengths,
      which is a compile-safe SymInt.
    * ``joint_lens.max().item()`` normally synchronizes CUDA once in every
      transformer block. OneTrainer computes that exact maximum once while
      packing the batch and supplies it as a Python integer to every block.
      If the hint is absent, fall back to Mage's original exact calculation.

    Importantly, the value passed to FA4 is the exact maximum sequence length,
    not merely an upper bound. FA4/CuTe uses this value in kernel scheduling and
    should not be given a larger padded-text bound.
    """
    import mage_flow.models.modules.mage_layers as mage_layers

    if getattr(mage_layers, "_ot_compile_safe_processor_installed", False):
        return

    cls = mage_layers.MageDoubleStreamAttnProcessor
    source = textwrap.dedent(inspect.getsource(cls.__call__))

    old_total = "    total_tokens = joint_cu_lens[-1]\n"
    new_total = "    total_tokens = txt_query.shape[0] + img_query.shape[0]\n"
    old_max = "    max_seqlen = joint_lens.max().item()\n"
    new_max = (
        "    max_seqlen = kwargs.pop(\"ot_max_joint_seqlen\", None)\n"
        "    if max_seqlen is None:\n"
        "        max_seqlen = joint_lens.max().item()\n"
    )

    if source.count(old_total) != 1 or source.count(old_max) != 1:
        raise RuntimeError(
            "Pinned Mage attention source no longer matches the compile-safety patch; "
            "refusing to guess."
        )

    source = source.replace(old_total, new_total, 1).replace(old_max, new_max, 1)

    # Execute in Mage's real module globals so the patched method continues to
    # resolve the backend/RoPE symbols from that module. This is important when
    # the FA4 function is subsequently marked as a Dynamo eager boundary.
    namespace = mage_layers.__dict__
    exec(compile(source, mage_layers.__file__, "exec"), namespace)
    patched_call = namespace.pop("__call__")
    cls.__call__ = patched_call
    mage_layers._ot_compile_safe_processor_installed = True


def _restore_official_mage_attention(transformer) -> None:
    """Use Microsoft's Mage processor class, with only the compile-safe scalar patch above."""
    from mage_flow.models.modules.mage_layers import MageDoubleStreamAttnProcessor

    for block in transformer.transformer_blocks:
        block.attn.set_processor(MageDoubleStreamAttnProcessor())


def _install_mage_compile_boundaries() -> None:
    """Keep only upstream Mage complex RoPE and FA4/CuTe outside TorchDynamo."""
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
    """Route Mage Self-Flow through compiled blocks only when gradients are needed."""

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
        # Mage's FA4/CuTe eager boundary is stable in the ordinary eager block,
        # but running a no-grad/inference forward through a torch.compile'd
        # CheckpointLayer has produced device-side illegal accesses on the
        # Self-Flow teacher and Linear-DPO reference paths. Those paths do not
        # need checkpointing or Inductor at all, so unwrap the OT wrapper and
        # call the original Mage block directly. The trainable student/policy
        # path keeps gradients enabled and therefore still uses the compiled
        # CheckpointLayer below.
        dispatch_block = block
        if not torch.is_grad_enabled() and isinstance(block, BaseCheckpointLayer):
            eager_block = getattr(block, "checkpoint", None)
            if eager_block is not None:
                dispatch_block = eager_block

        return dispatch_block(
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
    """Use Microsoft's native packing and compute FA4's exact max sequence length once."""

    def packed_inputs(self, model, latent_tokens, text, text_mask):
        packed_img, img_cu = model.prepare_packed_images(latent_tokens)
        packed_txt, txt_cu = model.prepare_packed_text(text, text_mask)

        # Image token length is identical for every sample. Text is packed from
        # the validity mask, so the exact longest joint sequence is simply the
        # image-token length plus the maximum valid text length. Compute it once
        # here instead of synchronizing joint_lens.max().item() in every block.
        text_lengths = text_mask.to(dtype=torch.int32).sum(dim=1)
        max_text_seqlen = int(text_lengths.max().item()) if text_lengths.numel() else 0
        max_joint_seqlen = int(latent_tokens.shape[1]) + max_text_seqlen
        attention_kwargs = {"ot_max_joint_seqlen": max_joint_seqlen}
        return packed_img, packed_txt, img_cu, txt_cu, latent_tokens.shape[0], attention_kwargs

    setup._packed_inputs = MethodType(packed_inputs, setup)


def _wrap_blocks_with_ot_checkpoint_layer(transformer, config) -> None:
    """Use OT's existing CheckpointLayer while allowing the FA4 eager boundary."""
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
            # INT_W8A8 projections/MLPs and the ordinary block math stay under
            # Inductor; only the explicitly disabled RoPE/FA4 calls graph-break.
            layer.compile(fullgraph=False)
        transformer.transformer_blocks[index] = layer

    transformer._register_state_dict_hook(_remove_checkpoint_keys)


def setup_mage_like_flux2(setup, model, config) -> None:
    """Apply OneTrainer's established checkpoint/compile lifecycle to Mage."""

    _patch_upstream_mage_attention_processor_for_compile()
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
        f"compile={bool(config.compile)}, exact-FA4-varlen, no-grad=eager, Mage RoPE/FA4 eager)"
    )
