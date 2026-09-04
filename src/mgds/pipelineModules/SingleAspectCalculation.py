from typing import Any

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule
from mgds.pipelineModules.MultiResolutionVariation import decode_multi_resolution_variation


class SingleAspectCalculation(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            resolution_in_name: str,
            target_resolution_in_name: str,
            enable_target_resolutions_override_in_name: str,
            target_resolutions_override_in_name: str,
            scale_resolution_out_name: str,
            crop_resolution_out_name: str,
            possible_resolutions_out_name: str,
            resolution_variants_out_name: str | None = None,
    ):
        super().__init__()
        self.resolution_in_name = resolution_in_name
        self.target_resolutions_in_name = target_resolution_in_name
        self.enable_target_resolutions_override_in_name = enable_target_resolutions_override_in_name
        self.target_resolutions_override_in_name = target_resolutions_override_in_name
        self.scale_resolution_out_name = scale_resolution_out_name
        self.crop_resolution_out_name = crop_resolution_out_name
        self.possible_resolutions_out_name = possible_resolutions_out_name
        self.resolution_variants_out_name = resolution_variants_out_name
        self.possible_target_resolutions: list[int] = []

    def length(self) -> int:
        return self._get_previous_length(self.resolution_in_name)

    def get_inputs(self) -> list[str]:
        return [name for name in [
            self.resolution_in_name,
            self.target_resolutions_in_name,
            self.enable_target_resolutions_override_in_name,
            self.target_resolutions_override_in_name,
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
    def _parse_resolutions(value: int | str) -> list[int]:
        if isinstance(value, int):
            return [value]
        return [int(part.strip()) for part in str(value).split(',') if part.strip()]

    def get_meta(self, variation: int, name: str) -> Any:
        if name == self.possible_resolutions_out_name:
            return [(value, value) for value in self.possible_target_resolutions]
        return None

    def start(self, variation: int):
        possible: set[int] = set()
        for index in range(self._get_previous_length(self.target_resolutions_in_name)):
            possible.update(self._parse_resolutions(
                self._get_previous_item(variation, self.target_resolutions_in_name, index)
            ))
        if self.target_resolutions_override_in_name is not None:
            for index in range(self._get_previous_length(self.target_resolutions_override_in_name)):
                possible.update(self._parse_resolutions(
                    self._get_previous_item(variation, self.target_resolutions_override_in_name, index)
                ))
        self.possible_target_resolutions = sorted(possible)

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

        descriptors = self._parse_resolutions(spec)
        if not descriptors:
            raise RuntimeError(f"No target resolutions configured for dataset index {index}")
        if resolution_index is None:
            selected = rand.choice(descriptors)
        else:
            if resolution_index >= len(descriptors):
                raise RuntimeError(
                    f"Resolution variant index {resolution_index} is out of range for {descriptors}"
                )
            selected = descriptors[resolution_index]

        target_resolution = (selected, selected)
        aspect = resolution[0] / resolution[1]
        target_aspect = target_resolution[0] / target_resolution[1]
        if aspect > target_aspect:
            scale = target_resolution[1] / resolution[1]
            scale_resolution = (round(resolution[0] * scale), target_resolution[1])
        else:
            scale = target_resolution[0] / resolution[0]
            scale_resolution = (target_resolution[0], round(resolution[1] * scale))

        result = {
            self.scale_resolution_out_name: scale_resolution,
            self.crop_resolution_out_name: target_resolution,
        }
        if self.resolution_variants_out_name is not None:
            result[self.resolution_variants_out_name] = descriptors
        return result
