"""Five-point face landmarks of the final training crop, for the identity reward."""
from modules.util.identity_reward import FaceLandmarkDetector

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch


class MOSSFaceLandmarks(PipelineModule, RandomAccessPipelineModule):
    """Detects on the image after every crop, flip and rotation, so the points match its latent.

    Rows without a usable face get NaN landmarks and are skipped by the reward.
    """

    def __init__(self, detector: FaceLandmarkDetector, image_in_name="image", landmarks_out_name="face_landmarks"):
        super().__init__()
        self.detector = detector
        self.image_in_name = image_in_name
        self.landmarks_out_name = landmarks_out_name

    def length(self):
        return self._get_previous_length(self.image_in_name)

    def get_inputs(self):
        return [self.image_in_name]

    def get_outputs(self):
        return [self.landmarks_out_name]

    def get_item(self, variation, index, requested_name=None):
        image = self._get_previous_item(variation, self.image_in_name, index)
        pixels = (image.detach().float().clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        landmarks = self.detector(pixels)
        result = torch.full((5, 2), float("nan")) if landmarks is None else torch.from_numpy(landmarks)
        return {self.landmarks_out_name: result}
