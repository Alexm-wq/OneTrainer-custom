from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class LoadDPORejectedImageOrDummy(
    PipelineModule,
    RandomAccessPipelineModule,
):
    """Conditional rejected-image loader for homogeneous mixed RLHF.

    DPO rows load image_path_rejected from disk.
    Normal rows do not touch disk and emit zeros_like(image).
    """

    def __init__(
            self,
            path_in_name: str = "image_path_rejected",
            image_in_name: str = "image",
            is_paired_in_name: str = "dpo_is_paired",
            image_out_name: str = "image_rejected",
            range_min: float = 0.0,
            range_max: float = 1.0,
            channels: int = 3,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.path_in_name = path_in_name
        self.image_in_name = image_in_name
        self.is_paired_in_name = is_paired_in_name
        self.image_out_name = image_out_name
        self.range_min = range_min
        self.range_max = range_max
        self.channels = channels
        self.dtype = dtype

    def length(self) -> int:
        return self._get_previous_length(self.image_in_name)

    def get_inputs(self) -> list[str]:
        return [self.image_in_name, self.is_paired_in_name, self.path_in_name]

    def get_outputs(self) -> list[str]:
        return [self.image_out_name]

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

    def _load_image(self, path: str) -> torch.Tensor:
        path = str(Path(path))
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            if self.channels == 1:
                img = img.convert("L")
                arr = np.asarray(img, dtype=np.float32)[None, :, :]
            elif self.channels == 4:
                img = img.convert("RGBA")
                arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)
            else:
                img = img.convert("RGB")
                arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)

        arr = arr / 255.0
        arr = arr * (self.range_max - self.range_min) + self.range_min
        t = torch.from_numpy(arr)
        if self.dtype is not None:
            t = t.to(dtype=self.dtype)
        return t

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        image = self._get_previous_item(variation, self.image_in_name, index)
        is_paired = self._as_bool(self._get_previous_item(variation, self.is_paired_in_name, index))

        if not is_paired:
            return {self.image_out_name: torch.zeros_like(image)}

        path = self._get_previous_item(variation, self.path_in_name, index)
        return {self.image_out_name: self._load_image(path)}
