from contextlib import nullcontext

import torch

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class EncodeDPORejectedOrDummyLatent(
    PipelineModule,
    RandomAccessPipelineModule,
):
    """Emits latent_image_rejected for homogeneous mixed RLHF.

    DPO rows VAE-encode image_rejected.
    Normal rows do not VAE-encode anything and emit zeros_like(latent_image).
    """

    def __init__(
            self,
            image_in_name: str = "image_rejected",
            latent_image_in_name: str = "latent_image",
            is_paired_in_name: str = "dpo_is_paired",
            latent_out_name: str = "latent_image_rejected",
            vae=None,
            autocast_contexts: list[torch.autocast | None] | None = None,
            dtype: torch.dtype | None = None,
            dummy_mode: str = "zeros",
    ):
        super().__init__()
        self.image_in_name = image_in_name
        self.latent_image_in_name = latent_image_in_name
        self.is_paired_in_name = is_paired_in_name
        self.latent_out_name = latent_out_name
        self.vae = vae
        self.autocast_contexts = [nullcontext()] if autocast_contexts is None else autocast_contexts
        self.dtype = dtype
        self.dummy_mode = dummy_mode

    def length(self) -> int:
        return self._get_previous_length(self.latent_image_in_name)

    def get_inputs(self) -> list[str]:
        return [self.latent_image_in_name, self.is_paired_in_name, self.image_in_name]

    def get_outputs(self) -> list[str]:
        return [self.latent_out_name]

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return False
            value = value.detach().cpu().flatten()[0].item()
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "dpo", "paired"}
        return bool(value)

    def _all_contexts(self):
        class _Stack:
            def __init__(self, contexts):
                self.contexts = contexts
                self.entered = []

            def __enter__(self):
                for ctx in self.contexts:
                    if ctx is None:
                        ctx = nullcontext()
                    self.entered.append(ctx)
                    ctx.__enter__()
                return self

            def __exit__(self, exc_type, exc, tb):
                for ctx in reversed(self.entered):
                    ctx.__exit__(exc_type, exc, tb)

        return _Stack(self.autocast_contexts)

    def _encode_rejected(self, image: torch.Tensor) -> torch.Tensor:
        image = image * 2.0 - 1.0

        try:
            vae_param = next(self.vae.parameters())
            vae_device = vae_param.device
            vae_dtype = vae_param.dtype
        except StopIteration:
            vae_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            vae_dtype = self.dtype if self.dtype is not None else image.dtype

        target_dtype = self.dtype if self.dtype is not None else vae_dtype
        image = image.to(device=vae_device, dtype=target_dtype, non_blocking=True)

        with torch.inference_mode():
            with self._all_contexts():
                vae_output = self.vae.encode(image.unsqueeze(0))

                if hasattr(vae_output, "latent_dist"):
                    latent = vae_output.latent_dist.mode()
                elif hasattr(vae_output, "latent"):
                    latent = vae_output.latent
                else:
                    raise RuntimeError("VAE output has neither latent_dist nor latent")

        return latent.squeeze(dim=0).detach()

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        latent_image = self._get_previous_item(variation, self.latent_image_in_name, index)
        is_paired = self._as_bool(self._get_previous_item(variation, self.is_paired_in_name, index))

        if not is_paired:
            dummy = latent_image if self.dummy_mode == "copy" else torch.zeros_like(latent_image)
            return {self.latent_out_name: dummy}

        if self.vae is None:
            raise RuntimeError("EncodeDPORejectedOrDummyLatent needs a VAE for paired DPO rows")

        image_rejected = self._get_previous_item(variation, self.image_in_name, index)
        if image_rejected is None:
            raise RuntimeError("DPO row has no image_rejected tensor")

        return {self.latent_out_name: self._encode_rejected(image_rejected)}
