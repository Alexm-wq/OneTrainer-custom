"""Qwen3.5 prefill kernels for MOSS, with device-aware PyTorch fallbacks.

FLA's Triton convolution avoids a separate, Torch/CUDA-version-specific
causal-conv1d extension. No checkpoint names or prompt semantics are changed.
"""

import os
from functools import lru_cache

import torch
import torch.nn.functional as F

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5GatedDeltaNet,
    Qwen3_5RMSNormGated,
    torch_causal_conv1d_update,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)


@lru_cache(maxsize=1)
def _load_fla():
    from fla.modules.conv import causal_conv1d
    from fla.modules.fused_norm_gate import rms_norm_gated
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    return causal_conv1d, chunk_gated_delta_rule, rms_norm_gated


class _TextKernels:
    def __init__(self):
        self.disabled = os.environ.get("OT_MOSS_TEXT_KERNELS", "1") == "0"
        self.import_error = None
        self.reported = set()

    def _report(self, message):
        if message not in self.reported:
            print(f"[MOSS TE] {message}")
            self.reported.add(message)

    def operations(self, tensor):
        # FP8 is a weight-storage format in OneTrainer. Select kernels using
        # actual activation dtype after projection/autocast, not weight dtype.
        if self.disabled:
            reason = "disabled by OT_MOSS_TEXT_KERNELS=0"
        elif tensor.device.type != "cuda" or torch.version.hip is not None:
            reason = f"device={tensor.device.type}"
        elif tensor.dtype not in (torch.float16, torch.bfloat16):
            reason = f"compute={tensor.dtype}; fast kernels require FP16/BF16"
        else:
            if self.import_error is None:
                try:
                    ops = _load_fla()
                except (ImportError, OSError) as error:
                    self.import_error = str(error)
                else:
                    self._report(
                        "FLA/Triton selected: Gated DeltaNet, causal conv1d, gated RMSNorm "
                        f"(device={tensor.device}, compute={tensor.dtype}). "
                        "The first prompt may include kernel compilation."
                    )
                    return ops
            reason = f"FLA unavailable: {self.import_error}; update the CUDA environment"
        self._report(f"PyTorch text kernels ({reason}).")
        return None

    def conv(self, x, weight, bias=None, activation=None, seq_idx=None):
        if seq_idx is not None:
            raise ValueError("MOSS encodes separate unpadded prompts; packed seq_idx is unsupported.")
        if activation not in (None, "silu", "swish"):
            raise ValueError(f"Unsupported Qwen3.5 convolution activation: {activation}")
        ops = self.operations(x)
        if ops is not None:
            # Transformers/native causal-conv1d: B,C,T; FLA Triton: B,T,C.
            output, _ = ops[0](
                x=x.transpose(1, 2), weight=weight, bias=bias,
                activation=activation, backend="triton", output_final_state=False,
            )
            return output.transpose(1, 2)
        output = F.conv1d(
            x, weight.unsqueeze(1), bias,
            padding=weight.shape[-1] - 1, groups=x.shape[1],
        )[:, :, :x.shape[-1]]
        return F.silu(output) if activation in ("silu", "swish") else output

    def chunk(self, query, key, value, **kwargs):
        ops = self.operations(query)
        kernel = ops[1] if ops is not None else torch_chunk_gated_delta_rule
        # Keep log-decay, beta, Q/K normalization and state flags unchanged.
        # Runtime kernel errors deliberately propagate instead of silently
        # replacing a broken CUDA installation with the slow path.
        return kernel(query, key, value, **kwargs)


class _GatedRMSNorm(Qwen3_5RMSNormGated):
    def __init__(self, original, eps, kernels):
        # Retain the exact Parameter object, device, dtype and state_dict key.
        torch.nn.Module.__init__(self)
        self.weight = original.weight
        self.variance_epsilon = eps
        self.kernels = kernels
        self.train(original.training)

    def forward(self, hidden_states, gate=None):
        ops = self.kernels.operations(hidden_states)
        if ops is not None:
            return ops[2](
                hidden_states, gate, self.weight, None,
                activation="silu", eps=self.variance_epsilon,
            )
        return super().forward(hidden_states, gate)


def configure_moss_text_kernels(text_encoder):
    """Bind prefill kernels once, before offload/compile hooks are installed.

    MOSS always encodes whole prompts with use_cache=False. Autoregressive
    state updates retain the reference implementation; they are not involved
    in text-embedding caching.
    """
    kernels = _TextKernels()
    count = 0
    for module in text_encoder.modules():
        if not isinstance(module, Qwen3_5GatedDeltaNet):
            continue
        if isinstance(module.norm, _GatedRMSNorm):
            continue
        module.causal_conv1d_fn = kernels.conv
        module.chunk_gated_delta_rule = kernels.chunk
        module.causal_conv1d_update = torch_causal_conv1d_update
        module.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
        module.norm = _GatedRMSNorm(module.norm, module.layer_norm_epsilon, kernels)
        count += 1
    return count
