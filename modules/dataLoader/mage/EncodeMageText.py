from contextlib import ExitStack, nullcontext
import threading

import torch
from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

from modules.model.MageFlowModel import MAGE_PROMPT_CROP_START


# MGDS may request cache entries from multiple worker threads. Mage's packed
# attention backend is a process-global setting, so switching it per item must
# be serialized or another worker can observe the transient backend. More
# importantly, the CUDA-13 FA4 varlen kernel has proven unstable in the Qwen
# text-cache path (native segfault, no Python exception). Text encoding is a
# one-time cache operation, so use the slower but robust Mage SDPA varlen shim
# here while leaving the transformer on its selected backend for training.
_MAGE_TEXT_ATTENTION_LOCK = threading.Lock()


class EncodeMageText(PipelineModule, RandomAccessPipelineModule):
    """Run Microsoft's packed TextEncoder and cache a fixed-size padded result."""

    def __init__(
            self,
            tokens_name: str = "tokens",
            tokens_mask_name: str = "tokens_mask",
            hidden_state_out_name: str = "text_encoder_hidden_state",
            attention_mask_out_name: str = "text_encoder_attention_mask",
            text_encoder_wrapper=None,
            autocast_contexts=None,
            dtype: torch.dtype | None = None,
            max_output_length: int = 2048,
            restore_attention_backend: str = "sdpa",
    ):
        super().__init__()
        self.tokens_name = tokens_name
        self.tokens_mask_name = tokens_mask_name
        self.hidden_state_out_name = hidden_state_out_name
        self.attention_mask_out_name = attention_mask_out_name
        self.text_encoder_wrapper = text_encoder_wrapper
        self.autocast_contexts = autocast_contexts or [nullcontext()]
        self.dtype = dtype
        self.max_output_length = int(max_output_length)
        self.restore_attention_backend = str(restore_attention_backend)

    def length(self):
        return self._get_previous_length(self.tokens_name)

    def get_inputs(self):
        return [self.tokens_name, self.tokens_mask_name]

    def get_outputs(self):
        return [self.hidden_state_out_name, self.attention_mask_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None):
        tokens = self._get_previous_item(variation, self.tokens_name, index)
        mask = self._get_previous_item(variation, self.tokens_mask_name, index).bool()
        valid = tokens[mask]
        device = next(self.text_encoder_wrapper.parameters()).device
        valid = valid.to(device=device, non_blocking=True)
        cu = torch.tensor([0, valid.numel()], device=device, dtype=torch.int32)

        # The packed Mage TextEncoder always routes through
        # mage_flow.models.modules._attn_backend when cu_seqlens are supplied.
        # Force only this cache forward to the SDPA shim. The lock both avoids
        # concurrent forwards through one large Qwen module and makes the
        # process-global backend swap race-free.
        from mage_flow.models.modules._attn_backend import set_attn_backend
        with _MAGE_TEXT_ATTENTION_LOCK:
            set_attn_backend("sdpa")
            try:
                with torch.inference_mode(), ExitStack() as stack:
                    for context in self.autocast_contexts:
                        stack.enter_context(context if context is not None else nullcontext())
                    result = self.text_encoder_wrapper(
                        valid,
                        cu,
                        drop_idx_override=MAGE_PROMPT_CROP_START,
                    )
            finally:
                set_attn_backend(self.restore_attention_backend)

        hidden = result["txt"]
        valid_length = min(int(hidden.shape[0]), self.max_output_length)
        hidden = hidden[:valid_length]
        padded = torch.zeros(
            (self.max_output_length, hidden.shape[-1]),
            device=hidden.device,
            dtype=hidden.dtype if self.dtype is None else self.dtype,
        )
        padded[:valid_length] = hidden.to(dtype=padded.dtype)
        out_mask = torch.zeros(self.max_output_length, device=hidden.device, dtype=torch.bool)
        out_mask[:valid_length] = True
        return {
            self.hidden_state_out_name: padded.detach().cpu(),
            self.attention_mask_out_name: out_mask.detach().cpu(),
        }
