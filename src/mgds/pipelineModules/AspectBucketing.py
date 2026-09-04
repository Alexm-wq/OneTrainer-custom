import itertools
import math
from random import Random
from typing import Any

import numpy as np

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule
from mgds.pipelineModules.MultiResolutionVariation import decode_multi_resolution_variation


class AspectBucketing(
    PipelineModule,
    RandomAccessPipelineModule,
):
    # Expanded custom OneTrainer bucket table. Both orientations are generated
    # below, so these are the >=1 side of the aspect-ratio set.
    all_possible_input_aspects = [
        (1.0, 1.0),
        (1.0, 1.05),
        (1.0, 1.10),
        (1.0, 1.15),
        (1.0, 1.20),
        (1.0, 1.25),
        (1.0, 1.3333333333),
        (1.0, 1.40),
        (1.0, 1.50),
        (1.0, 1.60),
        (1.0, 1.6666666667),
        (1.0, 1.75),
        (1.0, 1.7777777778),
        (1.0, 1.85),
        (1.0, 2.0),
        (1.0, 2.2),
        (1.0, 2.4),
        (1.0, 2.5),
        (1.0, 2.75),
        (1.0, 3.0),
        (1.0, 3.5),
        (1.0, 4.0),
    ]

    def __init__(
            self,
            quantization: int,
            resolution_in_name: str,
            target_resolution_in_name: str,
            enable_target_resolutions_override_in_name: str,
            target_resolutions_override_in_name: str,
            target_frames_in_name: str,
            frame_dim_enabled: bool,
            scale_resolution_out_name: str,
            crop_resolution_out_name: str,
            possible_resolutions_out_name: str,
            resolution_variants_out_name: str | None = None,
    ):
        super().__init__()
        self.quantization = quantization
        self.resolution_in_name = resolution_in_name
        self.target_resolutions_in_name = target_resolution_in_name
        self.enable_target_resolutions_override_in_name = enable_target_resolutions_override_in_name
        self.target_resolutions_override_in_name = target_resolutions_override_in_name
        self.target_frames_in_name = target_frames_in_name
        self.frame_dim_enabled = frame_dim_enabled
        self.scale_resolution_out_name = scale_resolution_out_name
        self.crop_resolution_out_name = crop_resolution_out_name
        self.possible_resolutions_out_name = possible_resolutions_out_name
        self.resolution_variants_out_name = resolution_variants_out_name
        self.bucket_resolutions: dict[int, list[tuple[int, int]]] = {}
        self.bucket_aspects: dict[int, np.ndarray] = {}
        self.flattened_possible_resolutions: list[Any] = []

    def length(self) -> int:
        return self._get_previous_length(self.resolution_in_name)

    def get_inputs(self) -> list[str]:
        return [name for name in [
            self.resolution_in_name,
            self.target_resolutions_in_name,
            self.enable_target_resolutions_override_in_name,
            self.target_resolutions_override_in_name,
            self.target_frames_in_name,
        ] if name is not None]

    def get_outputs(self) -> list[str]:
        outputs = [
            self.scale_resolution_out_name,
            self.crop_resolution_out_name,
            self.possible_resolutions_out_name,
        ]
        if self.resolution_variants_out_name is not None:
            outputs.append(self.resolution_variants_out_name)
        return outputs

    @staticmethod
    def _parse_resolution_spec(value: int | str) -> list[int | tuple[int, int]]:
        if isinstance(value, int):
            return [value]
        text = str(value).strip()
        if 'x' in text and ',' not in text:
            width, height = (int(part.strip()) for part in text.lower().split('x', 1))
            return [(height, width)]
        return [int(part.strip()) for part in text.split(',') if part.strip()]

    def __quantize_resolution(self, resolution: tuple[float, float], quantization: int) -> tuple[int, int]:
        return (
            round(resolution[0] / quantization) * quantization,
            round(resolution[1] / quantization) * quantization,
        )

    def __create_automatic_buckets(
            self,
            target_resolutions: list[int],
    ) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[float]]]:
        possible_resolutions: dict[int, list[tuple[int, int]]] = {}
        possible_aspects: dict[int, list[float]] = {}
        for target_resolution in target_resolutions:
            new_resolutions = [(
                h / math.sqrt(h * w) * target_resolution,
                w / math.sqrt(h * w) * target_resolution,
            ) for h, w in self.all_possible_input_aspects]
            new_resolutions += [(w, h) for h, w in new_resolutions]
            new_resolutions = [
                self.__quantize_resolution(resolution, self.quantization)
                for resolution in new_resolutions
            ]
            new_resolutions = sorted(set(new_resolutions))
            possible_resolutions[target_resolution] = new_resolutions
            possible_aspects[target_resolution] = [h / w for h, w in new_resolutions]
        return possible_resolutions, possible_aspects

    def __get_bucket(self, h: int, w: int, target_resolution: int) -> tuple[int, int]:
        aspect = h / w
        bucket_index = int(np.argmin(abs(self.bucket_aspects[target_resolution] - aspect)))
        return self.bucket_resolutions[target_resolution][bucket_index]

    def get_meta(self, variation: int, name: str) -> Any:
        if name == self.possible_resolutions_out_name:
            return self.flattened_possible_resolutions
        return None

    def start(self, variation: int):
        possible_target_resolutions: set[int] = set()
        possible_fixed_resolutions: set[tuple[int, int]] = set()
        possible_frames = {1}

        for index in range(self._get_previous_length(self.target_resolutions_in_name)):
            for descriptor in self._parse_resolution_spec(
                    self._get_previous_item(variation, self.target_resolutions_in_name, index)):
                if isinstance(descriptor, tuple):
                    possible_fixed_resolutions.add(self.__quantize_resolution(descriptor, self.quantization))
                else:
                    possible_target_resolutions.add(descriptor)

        if self.target_resolutions_override_in_name is not None:
            for index in range(self._get_previous_length(self.target_resolutions_override_in_name)):
                for descriptor in self._parse_resolution_spec(
                        self._get_previous_item(variation, self.target_resolutions_override_in_name, index)):
                    if isinstance(descriptor, tuple):
                        possible_fixed_resolutions.add(self.__quantize_resolution(descriptor, self.quantization))
                    else:
                        possible_target_resolutions.add(descriptor)

        if self.target_frames_in_name is not None:
            for index in range(self._get_previous_length(self.target_frames_in_name)):
                possible_frames.add(int(self._get_previous_item(variation, self.target_frames_in_name, index)))

        self.bucket_resolutions, raw_aspects = self.__create_automatic_buckets(
            sorted(possible_target_resolutions)
        )
        self.bucket_aspects = {key: np.asarray(value) for key, value in raw_aspects.items()}
        flattened = set(sum(self.bucket_resolutions.values(), [])) | possible_fixed_resolutions
        self.flattened_possible_resolutions = sorted(flattened)
        if self.frame_dim_enabled:
            self.flattened_possible_resolutions = [
                (frames, *resolution)
                for frames, resolution in itertools.product(
                    sorted(possible_frames), self.flattened_possible_resolutions
                )
            ]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        _, resolution_index = decode_multi_resolution_variation(variation)
        rand = self._get_rand(variation, index)
        resolution = self._get_previous_item(variation, self.resolution_in_name, index)
        spec = self._get_previous_item(variation, self.target_resolutions_in_name, index)

        if self.enable_target_resolutions_override_in_name is not None:
            enabled = self._get_previous_item(
                variation, self.enable_target_resolutions_override_in_name, index
            )
            if enabled:
                spec = self._get_previous_item(
                    variation, self.target_resolutions_override_in_name, index
                )

        descriptors = self._parse_resolution_spec(spec)
        if not descriptors:
            raise RuntimeError(f"No target resolutions configured for dataset index {index}")
        if resolution_index is None:
            descriptor = rand.choice(descriptors)
        else:
            if resolution_index >= len(descriptors):
                raise RuntimeError(
                    f"Resolution variant index {resolution_index} is out of range for {descriptors}"
                )
            descriptor = descriptors[resolution_index]

        if isinstance(descriptor, tuple):
            target_resolution = self.__quantize_resolution(descriptor, self.quantization)
        else:
            target_resolution = self.__get_bucket(resolution[-2], resolution[-1], descriptor)

        aspect = resolution[-2] / resolution[-1]
        target_aspect = target_resolution[-2] / target_resolution[-1]
        if aspect > target_aspect:
            scale = target_resolution[-1] / resolution[-1]
            scale_resolution = (
                *resolution[:-2],
                round(resolution[-2] * scale),
                target_resolution[-1],
            )
        else:
            scale = target_resolution[-2] / resolution[-2]
            scale_resolution = (
                *resolution[:-2],
                target_resolution[-2],
                round(resolution[-1] * scale),
            )

        crop_resolution = (*resolution[:-2], *target_resolution)
        result = {
            self.scale_resolution_out_name: scale_resolution,
            self.crop_resolution_out_name: crop_resolution,
        }
        if self.resolution_variants_out_name is not None:
            result[self.resolution_variants_out_name] = descriptors
        return result
