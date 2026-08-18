import torch
from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

from modules.model.MageFlowModel import MAGE_PROMPT_CROP_START, MAGE_PROMPT_TEMPLATE


class TokenizeMagePrompt(PipelineModule, RandomAccessPipelineModule):
    def __init__(
            self,
            in_name: str = "prompt",
            tokens_out_name: str = "tokens",
            mask_out_name: str = "tokens_mask",
            tokenizer=None,
            prompt_token_length: int = 2048,
    ):
        super().__init__()
        self.in_name = in_name
        self.tokens_out_name = tokens_out_name
        self.mask_out_name = mask_out_name
        self.tokenizer = tokenizer
        self.prompt_token_length = int(prompt_token_length)

    def length(self):
        return self._get_previous_length(self.in_name)

    def get_inputs(self):
        return [self.in_name]

    def get_outputs(self):
        return [self.tokens_out_name, self.mask_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None):
        prompt = str(self._get_previous_item(variation, self.in_name, index))
        # Keep the official Mage system/user/assistant template. The extra crop
        # budget ensures the post-crop conditioning can still reach the user
        # requested sequence length.
        encoded = self.tokenizer(
            MAGE_PROMPT_TEMPLATE.format(prompt),
            padding="max_length",
            truncation=True,
            max_length=self.prompt_token_length + MAGE_PROMPT_CROP_START,
            return_tensors="pt",
        )
        return {
            self.tokens_out_name: encoded.input_ids.squeeze(0).detach().cpu(),
            self.mask_out_name: encoded.attention_mask.squeeze(0).bool().detach().cpu(),
        }
