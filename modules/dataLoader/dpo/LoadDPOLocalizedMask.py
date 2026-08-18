import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from mgds.crypto import open_source_binary
from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class LoadDPOLocalizedMask(
    PipelineModule,
    RandomAccessPipelineModule,
):
    """Load the chosen-image mask for concepts using localized DPO.

    The existing OneTrainer ``<stem>-masklabel.png`` convention is reused.
    Non-localized and non-DPO rows receive a zero mask so mixed datasets keep a
    homogeneous schema without touching an optional mask file.
    """

    def __init__(
            self,
            path_in_name: str = "image_path",
            image_in_name: str = "image",
            is_paired_in_name: str = "dpo_is_paired",
            concept_in_name: str = "concept",
            mask_out_name: str = "dpo_mask_image",
            mask_path_out_name: str = "dpo_mask_path",
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.path_in_name = path_in_name
        self.image_in_name = image_in_name
        self.is_paired_in_name = is_paired_in_name
        self.concept_in_name = concept_in_name
        self.mask_out_name = mask_out_name
        self.mask_path_out_name = mask_path_out_name
        self.dtype = dtype

    def length(self) -> int:
        return self._get_previous_length(self.image_in_name)

    def get_inputs(self) -> list[str]:
        return [
            self.path_in_name,
            self.image_in_name,
            self.is_paired_in_name,
            self.concept_in_name,
        ]

    def get_outputs(self) -> list[str]:
        return [self.mask_out_name, self.mask_path_out_name]

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return False
            value = value.detach().cpu().flatten()[0].item()
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        if isinstance(value, str):
            return value.strip().lower() in {
                "1", "true", "yes", "y", "dpo", "paired",
            }
        return bool(value)

    @staticmethod
    def _mask_path(image_path: str) -> str:
        stem, _ = os.path.splitext(str(image_path))
        return f"{stem}-masklabel.png"

    def _empty_mask(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim < 3:
            raise RuntimeError(
                "Localized DPO expected an image tensor with at least CHW "
                f"dimensions, got {tuple(image.shape)}"
            )
        return torch.zeros(
            (1, *image.shape[-2:]),
            device=image.device,
            dtype=self.dtype if self.dtype is not None else image.dtype,
        )

    def _load_mask(
            self,
            mask_path: str,
            image: torch.Tensor,
    ) -> torch.Tensor:
        if not os.path.isfile(mask_path):
            raise RuntimeError(
                "Localized DPO requires one chosen-image mask per pair. "
                f"Missing mask: {mask_path}"
            )

        with open_source_binary(str(Path(mask_path))) as source:
            with Image.open(source) as mask_image:
                mask_image = ImageOps.exif_transpose(mask_image).convert("L")
                array = np.asarray(mask_image, dtype=np.float32)

        expected_hw = tuple(int(x) for x in image.shape[-2:])
        if tuple(array.shape) != expected_hw:
            raise RuntimeError(
                "Localized DPO mask size must match its chosen image before "
                f"augmentation: mask={tuple(array.shape)}, "
                f"image={expected_hw}, path={mask_path}"
            )

        array = array / 255.0
        if not np.isfinite(array).all():
            raise RuntimeError(
                f"Localized DPO mask contains non-finite values: {mask_path}"
            )
        if array.size == 0 or float(array.max()) <= 0.0:
            raise RuntimeError(
                f"Localized DPO mask has no selected pixels: {mask_path}"
            )

        mask = torch.from_numpy(array).unsqueeze(0)
        if self.dtype is not None:
            mask = mask.to(dtype=self.dtype)
        return mask

    def get_item(
            self,
            variation: int,
            index: int,
            requested_name: str = None,
    ) -> dict:
        image_path = str(
            self._get_previous_item(variation, self.path_in_name, index)
        )
        image = self._get_previous_item(
            variation,
            self.image_in_name,
            index,
        )
        concept = self._get_previous_item(
            variation,
            self.concept_in_name,
            index,
        )
        paired = self._as_bool(
            self._get_previous_item(
                variation,
                self.is_paired_in_name,
                index,
            )
        )
        localized = paired and self._as_bool(concept.get("dpo_masked", False))

        if not localized:
            # Keep every encryption/cache source path valid without requiring a
            # file that the row does not use.
            return {
                self.mask_out_name: self._empty_mask(image),
                self.mask_path_out_name: image_path,
            }

        multiplier = float(concept.get("dpo_mask_weight", 10.0))
        if not math.isfinite(multiplier) or multiplier < 1.0:
            raise ValueError(
                "Localized DPO Mask Weight must be finite and >= 1, "
                f"got {multiplier} for concept "
                f"{concept.get('name') or concept.get('path')}"
            )

        mask_path = self._mask_path(image_path)
        return {
            self.mask_out_name: self._load_mask(mask_path, image),
            self.mask_path_out_name: mask_path,
        }
