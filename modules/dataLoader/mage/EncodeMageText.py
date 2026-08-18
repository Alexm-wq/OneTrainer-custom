from contextlib import ExitStack, nullcontext

import torch
from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

from modules.model.MageFlowModel import MAGE_PROMPT_CROP_START


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

        with torch.inference_mode(), ExitStack() as stack:
            for context in self.autocast_contexts:
                stack.enter_context(context if context is not None else nullcontext())
            result = self.text_encoder_wrapper(valid, cu, drop_idx_override=MAGE_PROMPT_CROP_START)

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
