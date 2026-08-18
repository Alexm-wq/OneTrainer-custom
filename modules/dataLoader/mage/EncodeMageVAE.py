from contextlib import ExitStack, nullcontext

import torch
from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class EncodeMageVAE(PipelineModule, RandomAccessPipelineModule):
    """Encode an image with MageVAE, whose ``encode`` returns a tensor directly."""

    def __init__(
            self,
            in_name: str,
            out_name: str,
            vae,
            autocast_contexts=None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_name = in_name
        self.out_name = out_name
        self.vae = vae
        self.autocast_contexts = autocast_contexts or [nullcontext()]
        self.dtype = dtype

    def length(self):
        return self._get_previous_length(self.in_name)

    def get_inputs(self):
        return [self.in_name]

    def get_outputs(self):
        return [self.out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None):
        image = self._get_previous_item(variation, self.in_name, index)
        parameter = next(self.vae.parameters())
        dtype = self.dtype or parameter.dtype
        image = image.unsqueeze(0).to(device=parameter.device, dtype=dtype, non_blocking=True)
        with torch.inference_mode(), ExitStack() as stack:
            for context in self.autocast_contexts:
                stack.enter_context(context if context is not None else nullcontext())
            latent = self.vae.encode(image)
        if not isinstance(latent, torch.Tensor):
            raise RuntimeError(f"MageVAE.encode() unexpectedly returned {type(latent).__name__}")
        return {self.out_name: latent.squeeze(0).detach()}
