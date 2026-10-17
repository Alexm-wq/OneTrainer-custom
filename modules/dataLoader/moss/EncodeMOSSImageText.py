from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class EncodeMOSSImageText(PipelineModule, RandomAccessPipelineModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def length(self):
        return self._get_previous_length("tokens")

    def get_inputs(self):
        return ["tokens", "tokens_mask"]

    def get_outputs(self):
        return ["tokens", "tokens_mask", "text_encoder_hidden_state"]

    def get_item(self, variation, index, requested_name=None):
        tokens = self._get_previous_item(variation, "tokens", index)
        mask = self._get_previous_item(variation, "tokens_mask", index)
        hidden = self.model.encode_tokens(tokens.unsqueeze(0), mask.unsqueeze(0))[0]
        return {"tokens": tokens, "tokens_mask": mask, "text_encoder_hidden_state": hidden}
