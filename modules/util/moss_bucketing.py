"""Use the training bucket/crop modules directly for the annotation tool."""
from types import SimpleNamespace

from modules.dataLoader.cache.AspectBucketing import AspectBucketing
from modules.dataLoader.cache.MultiResolutionVariation import encode_multi_resolution_variation
from modules.dataLoader.moss.MOSSLayout import MOSSScaleCrop
from modules.util.moss_layout import empty_layout, parse_layout

from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch
from torchvision.transforms.functional import pil_to_tensor, to_pil_image


class _Source(PipelineModule, RandomAccessPipelineModule):
    def __init__(self, values):
        super().__init__()
        self.values = values

    def get_inputs(self):
        return []

    def get_outputs(self):
        return list(self.values)

    def length(self):
        return 1

    def get_item(self, variation, index, requested_name=None):
        return self.values


def bucket_image(image, layout=None, resolution="1024", *, variant=0, crop_jitter=False, seed=42):
    """Return the exact MOSS MGDS resize/crop result, geometry and transformed layout.

    Center crop is the annotation default. Jitter uses the same MGDS RNG rule;
    its concrete offset is returned, so the exported annotation is unambiguous.
    """
    descriptors = AspectBucketing._parse_resolution_spec(resolution)
    if not descriptors or any(
        any(v < 32 for v in value) if isinstance(value, tuple) else value < 32
        for value in descriptors
    ):
        raise ValueError("Choose a MOSS resolution of at least 32 pixels per side.")
    if not 0 <= variant < len(descriptors):
        raise ValueError(f"Resolution variant must be between 0 and {len(descriptors) - 1}.")
    image = image.convert("RGB")
    layout = parse_layout(layout or empty_layout(*image.size), *image.size, match_image=True)
    source = _Source({
        "image": pil_to_tensor(image).float() / 255,
        "moss_layout": layout,
        "original_resolution": (image.height, image.width),
        "resolution": resolution,
        "jitter": crop_jitter,
    })
    bucket = AspectBucketing(
        quantization=16, resolution_in_name="original_resolution", target_resolution_in_name="resolution",
        enable_target_resolutions_override_in_name=None, target_resolutions_override_in_name=None,
        target_frames_in_name=None, frame_dim_enabled=False,
        scale_resolution_out_name="scale_resolution", crop_resolution_out_name="crop_resolution",
        possible_resolutions_out_name="possible_resolutions", resolution_variants_out_name="resolution_variants",
    )
    crop = MOSSScaleCrop(
        names=["image"], scale_resolution_in_name="scale_resolution", crop_resolution_in_name="crop_resolution",
        enable_crop_jitter_in_name="jitter", crop_offset_out_name="crop_offset",
    )
    pipeline = SimpleNamespace(modules=[source, bucket, crop], device=torch.device("cpu"))
    state = PipelineState(1)
    try:
        for index, module in enumerate(pipeline.modules):
            module.init(pipeline, seed, index, state)
        bucket.start(0)
        variation = encode_multi_resolution_variation(0, variant)
        result = crop.get_item(variation, 0)
        geometry = bucket.get_item(variation, 0)
        return {
            "image": to_pil_image(result["image"].clamp(0, 1)),
            "layout": result["moss_layout"],
            "crop_offset": result["crop_offset"],
            "scale_resolution": geometry["scale_resolution"],
            "crop_resolution": geometry["crop_resolution"],
        }
    finally:
        state.executor.shutdown(wait=True)
