from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


def build_packed_attention_routing(
        img_cu_lens: Tensor,
        txt_cu_lens: Tensor,
        *,
        img_token_count: int,
        txt_token_count: int,
        max_joint_seqlen: int,
) -> dict[str, Any]:
    """Precompute Mage's packed joint-attention routing once per model forward.

    Upstream Mage rebuilds these indices in every transformer block. Its SDPA
    fallback additionally converts CUDA cu_seqlens to Python lists in every
    block, forcing a device synchronization each time. OneTrainer builds both
    the GPU routing and the tiny host-side sequence grouping once here instead.
    """
    img_lens = img_cu_lens[1:] - img_cu_lens[:-1]
    txt_lens = txt_cu_lens[1:] - txt_cu_lens[:-1]
    if img_lens.numel() != txt_lens.numel():
        raise ValueError(
            "Mage packed image/text batch mismatch: "
            f"images={img_lens.numel()} text={txt_lens.numel()}"
        )

    joint_lens = txt_lens + img_lens
    joint_cu_lens = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=joint_lens.device),
        torch.cumsum(joint_lens, dim=0, dtype=torch.int32),
    ])

    batch_size = int(img_lens.numel())
    sample_indices = torch.arange(batch_size, device=joint_lens.device)
    txt_sample_ids = torch.repeat_interleave(sample_indices, txt_lens)
    img_sample_ids = torch.repeat_interleave(sample_indices, img_lens)

    txt_intra_pos = torch.arange(txt_token_count, device=joint_lens.device) - txt_cu_lens[txt_sample_ids]
    img_intra_pos = torch.arange(img_token_count, device=joint_lens.device) - img_cu_lens[img_sample_ids]

    txt_dest_indices = joint_cu_lens[txt_sample_ids] + txt_intra_pos
    img_dest_indices = joint_cu_lens[img_sample_ids] + txt_lens[img_sample_ids] + img_intra_pos

    # One synchronization per complete model forward, rather than two .tolist()
    # synchronizations in every transformer block as in Mage's SDPA fallback.
    joint_cu_host = [int(value) for value in joint_cu_lens.detach().cpu().tolist()]
    groups_by_length: dict[int, list[tuple[int, int]]] = {}
    for sequence_index, (start, end) in enumerate(
            zip(joint_cu_host[:-1], joint_cu_host[1:], strict=True)
    ):
        length = end - start
        groups_by_length.setdefault(length, []).append((sequence_index, start))
    sdpa_groups = tuple(
        (length, tuple(entries))
        for length, entries in groups_by_length.items()
    )

    return {
        "joint_cu_lens": joint_cu_lens,
        "txt_dest_indices": txt_dest_indices,
        "img_dest_indices": img_dest_indices,
        "max_joint_seqlen": int(max_joint_seqlen),
        "total_joint_tokens": int(img_token_count + txt_token_count),
        "sequence_count": batch_size,
        "sdpa_groups": sdpa_groups,
    }


def _grouped_sdpa_attention(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        routing: dict[str, Any],
) -> Tensor:
    """Run packed Mage attention with one SDPA call per distinct sequence length.

    DPO chosen/rejected copies normally have the same prompt length, so the
    common batch-1 DPO pair becomes one cuDNN SDPA dispatch per block. Variable
    length batches remain exact by grouping only sequences with equal lengths.
    """
    sequence_count = int(routing["sequence_count"])
    sequence_outputs: list[Tensor | None] = [None] * sequence_count

    for length, entries in routing["sdpa_groups"]:
        q_batch = torch.stack(
            [query[start:start + length] for _, start in entries],
            dim=0,
        ).transpose(1, 2)
        k_batch = torch.stack(
            [key[start:start + length] for _, start in entries],
            dim=0,
        ).transpose(1, 2)
        v_batch = torch.stack(
            [value[start:start + length] for _, start in entries],
            dim=0,
        ).transpose(1, 2)

        out_batch = F.scaled_dot_product_attention(
            q_batch,
            k_batch,
            v_batch,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
        ).transpose(1, 2)

        for group_index, (sequence_index, _) in enumerate(entries):
            sequence_outputs[sequence_index] = out_batch[group_index]

    if any(output is None for output in sequence_outputs):
        raise RuntimeError("Mage grouped SDPA failed to produce every packed sequence")
    return torch.cat(sequence_outputs, dim=0)


@torch.compiler.disable(reason="Mage complex RoPE and packed attention stay eager inside compiled blocks")
def _eager_rotary_varlen_attention(
        img_query: Tensor,
        img_key: Tensor,
        img_value: Tensor,
        txt_query: Tensor,
        txt_key: Tensor,
        txt_value: Tensor,
        image_rotary_emb: Tensor | None,
        routing: dict[str, Any],
) -> tuple[Tensor, Tensor]:
    """Run only Mage's complex RoPE/packed-attention region eagerly.

    Q/K/V projections, output projections, modulation, norms, residuals and
    MLPs remain inside torch.compile. For the SDPA/cuDNN backend this bypasses
    Microsoft's per-block CUDA->host cu_seqlens conversion and Python
    per-sequence dispatch loop.
    """
    from mage_flow.models.modules import _attn_backend
    from mage_flow.models.modules.mage_layers import apply_rotary_emb_mageflow

    img_query = apply_rotary_emb_mageflow(img_query, image_rotary_emb)
    img_key = apply_rotary_emb_mageflow(img_key, image_rotary_emb)

    joint_cu_lens = routing["joint_cu_lens"]
    txt_dest_indices = routing["txt_dest_indices"]
    img_dest_indices = routing["img_dest_indices"]
    total_tokens = routing["total_joint_tokens"]
    max_seqlen = routing["max_joint_seqlen"]

    joint_query = torch.empty(
        (total_tokens, *txt_query.shape[1:]),
        dtype=txt_query.dtype,
        device=txt_query.device,
    )
    joint_key = torch.empty(
        (total_tokens, *txt_key.shape[1:]),
        dtype=txt_key.dtype,
        device=txt_key.device,
    )
    joint_value = torch.empty(
        (total_tokens, *txt_value.shape[1:]),
        dtype=txt_value.dtype,
        device=txt_value.device,
    )

    joint_query[txt_dest_indices] = txt_query
    joint_query[img_dest_indices] = img_query
    joint_key[txt_dest_indices] = txt_key
    joint_key[img_dest_indices] = img_key
    joint_value[txt_dest_indices] = txt_value
    joint_value[img_dest_indices] = img_value

    if getattr(_attn_backend, "_BACKEND", None) == "sdpa":
        joint_attn_output = _grouped_sdpa_attention(
            joint_query,
            joint_key,
            joint_value,
            routing,
        )
    else:
        joint_attn_output = _attn_backend.flash_attn_varlen_func(
            joint_query,
            joint_key,
            joint_value,
            cu_seqlens_q=joint_cu_lens,
            cu_seqlens_k=joint_cu_lens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            softmax_scale=None,
            causal=False,
        )

    txt_attn_output = joint_attn_output[txt_dest_indices]
    img_attn_output = joint_attn_output[img_dest_indices]
    return img_attn_output, txt_attn_output


class OneTrainerMageDoubleStreamAttnProcessor:
    """Drop-in Mage double-stream processor with one routing build per forward."""

    def __init__(self):
        self._cached_img_cu = None
        self._cached_txt_cu = None
        self._cached_img_tokens = None
        self._cached_txt_tokens = None
        self._cached_routing = None

    def _fallback_routing(
            self,
            img_cu_lens: Tensor,
            txt_cu_lens: Tensor,
            img_token_count: int,
            txt_token_count: int,
    ) -> dict[str, Any]:
        if (
            self._cached_routing is not None
            and self._cached_img_cu is img_cu_lens
            and self._cached_txt_cu is txt_cu_lens
            and self._cached_img_tokens == img_token_count
            and self._cached_txt_tokens == txt_token_count
        ):
            return self._cached_routing

        routing = build_packed_attention_routing(
            img_cu_lens,
            txt_cu_lens,
            img_token_count=img_token_count,
            txt_token_count=txt_token_count,
            max_joint_seqlen=img_token_count + txt_token_count,
        )
        self._cached_img_cu = img_cu_lens
        self._cached_txt_cu = txt_cu_lens
        self._cached_img_tokens = img_token_count
        self._cached_txt_tokens = txt_token_count
        self._cached_routing = routing
        return routing

    def __call__(
            self,
            attn,
            hidden_states: Tensor,
            img_cu_lens: Tensor,
            attention_mask: Tensor | None = None,
            encoder_hidden_states: Tensor | None = None,
            txt_cu_lens: Tensor | None = None,
            image_rotary_emb: Tensor | None = None,
            **kwargs,
    ) -> tuple[Tensor, Tensor]:
        if encoder_hidden_states is None:
            raise ValueError("Mage double-stream attention requires encoder_hidden_states")
        if img_cu_lens is None or txt_cu_lens is None:
            raise ValueError("Mage packed attention requires image/text cu_seqlens")

        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))
        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        if img_query.ndim == 4:
            img_query = img_query.flatten(0, 1)
            img_key = img_key.flatten(0, 1)
            img_value = img_value.flatten(0, 1)
        if txt_query.ndim == 4:
            txt_query = txt_query.flatten(0, 1)
            txt_key = txt_key.flatten(0, 1)
            txt_value = txt_value.flatten(0, 1)

        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        routing = kwargs.get("ot_packed_routing")
        if routing is None:
            routing = self._fallback_routing(
                img_cu_lens,
                txt_cu_lens,
                int(img_query.shape[0]),
                int(txt_query.shape[0]),
            )

        img_attn_output, txt_attn_output = _eager_rotary_varlen_attention(
            img_query,
            img_key,
            img_value,
            txt_query,
            txt_key,
            txt_value,
            image_rotary_emb,
            routing,
        )

        img_attn_output = img_attn_output.flatten(1, 2).to(img_query.dtype)
        txt_attn_output = txt_attn_output.flatten(1, 2).to(txt_query.dtype)

        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)
        txt_attn_output = attn.to_add_out(txt_attn_output)
        txt_attn_output = txt_attn_output.view(
            encoder_hidden_states.shape[0],
            encoder_hidden_states.shape[1],
            txt_attn_output.shape[-1],
        )
        return img_attn_output, txt_attn_output


def install_optimized_mage_attention(transformer) -> None:
    """Replace Mage's parameter-free processors with one shared OT processor."""
    shared = None
    for block in transformer.transformer_blocks:
        processor = block.attn.get_processor()
        if isinstance(processor, OneTrainerMageDoubleStreamAttnProcessor):
            shared = processor
            break
    if shared is None:
        shared = OneTrainerMageDoubleStreamAttnProcessor()

    installed = 0
    for block in transformer.transformer_blocks:
        if block.attn.get_processor() is shared:
            continue
        block.attn.set_processor(shared)
        installed += 1
    print(
        f"[Mage-Flow] optimized packed attention processors installed={installed} "
        "shared-routing=on grouped-SDPA=on"
    )


def configure_mage_attention_from_config(model, config) -> str:
    """Apply OneTrainer's Attention Mechanism selector to Mage's DiT backend."""
    from modules.util.enum.AttentionMechanism import AttentionMechanism
    from mage_flow.models.modules._attn_backend import set_attn_backend

    explicit = os.environ.get("OT_MAGE_ATTN_BACKEND", "").strip().lower()
    mechanism = config.attention_mechanism

    if explicit:
        selected = getattr(model, "mage_attention_backend", "sdpa")
        source = f"OT_MAGE_ATTN_BACKEND={explicit}"
    elif mechanism == AttentionMechanism.FLASH:
        try:
            from flash_attn.cute import flash_attn_varlen_func  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Mage Attention Mechanism=FLASH requires FlashAttention4, but "
                "flash_attn.cute is not importable in the active environment."
            ) from exc
        selected = "flash4"
        source = "Attention Mechanism=FLASH"
    else:
        selected = "sdpa"
        source = f"Attention Mechanism={mechanism}"

    if torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(
            mechanism == AttentionMechanism.CUDNN and selected == "sdpa"
        )

    set_attn_backend(selected)
    model.mage_attention_backend = selected
    state = getattr(model, "mage_attention_backend_state", None)
    if state is not None:
        state["dit"] = selected

    extra = " grouped-varlen=on" if selected == "sdpa" else ""
    print(f"[Mage-Flow] {source} -> DiT packed backend={selected}{extra}")
    return selected
