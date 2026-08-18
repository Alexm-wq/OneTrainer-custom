import torch
import torch.nn.functional as F

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class PrepareDPOLocalizedMask(
    PipelineModule,
    RandomAccessPipelineModule,
):
    """Resize an augmented image-space DPO mask to the chosen latent grid."""

    def __init__(
            self,
            mask_in_name: str = "dpo_mask_image",
            latent_in_name: str = "latent_image",
            mask_out_name: str = "dpo_mask",
    ):
        super().__init__()
        self.mask_in_name = mask_in_name
        self.latent_in_name = latent_in_name
        self.mask_out_name = mask_out_name

    def length(self) -> int:
        return self._get_previous_length(self.latent_in_name)

    def get_inputs(self) -> list[str]:
        return [self.mask_in_name, self.latent_in_name]

    def get_outputs(self) -> list[str]:
        return [self.mask_out_name]

    @staticmethod
    def _resize(mask: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim < 2:
            raise RuntimeError(
                "Localized DPO expected a latent with channel and spatial "
                f"dimensions, got {tuple(latent.shape)}"
            )

        spatial_dims = latent.ndim - 1
        target_size = tuple(int(x) for x in latent.shape[1:])

        # A static image mask can be expanded into a video mask when the model
        # uses a temporal latent dimension.
        while mask.ndim < latent.ndim:
            mask = mask.unsqueeze(1)
        while mask.ndim > latent.ndim and int(mask.shape[1]) == 1:
            mask = mask.squeeze(1)

        if mask.ndim != latent.ndim:
            raise RuntimeError(
                "Localized DPO mask/latent rank mismatch after normalization: "
                f"mask={tuple(mask.shape)}, latent={tuple(latent.shape)}"
            )

        mask = mask.float().unsqueeze(0)
        if tuple(mask.shape[2:]) != target_size:
            mode = {
                1: "linear",
                2: "bilinear",
                3: "trilinear",
            }.get(spatial_dims)
            if mode is None:
                raise RuntimeError(
                    "Localized DPO supports one-, two-, or three-dimensional "
                    f"latent grids, got {spatial_dims} dimensions"
                )
            mask = F.interpolate(
                mask,
                size=target_size,
                mode=mode,
                align_corners=False,
            )

        return mask.squeeze(0).clamp_(0.0, 1.0)

    def get_item(
            self,
            variation: int,
            index: int,
            requested_name: str = None,
    ) -> dict:
        mask = self._get_previous_item(
            variation,
            self.mask_in_name,
            index,
        )
        latent = self._get_previous_item(
            variation,
            self.latent_in_name,
            index,
        )
        return {self.mask_out_name: self._resize(mask, latent)}
