from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.ModelType import ModelType


class SampleFrameController:
    def __init__(
            self,
            sample: SampleConfig,
            model_type: ModelType,
            self_flow_ema_sampling: bool = False,
    ):
        self.sample = sample
        self.model_type = model_type
        self.self_flow_ema_sampling = self_flow_ema_sampling

    def is_flow_matching(self) -> bool:
        return self.model_type.is_flow_matching()

    def is_inpainting_model(self) -> bool:
        return self.model_type.has_conditioning_image_input()

    def is_video_model(self) -> bool:
        return self.model_type.is_video_model()

    def supports_negative_prompt(self) -> bool:
        return self.model_type.supports_negative_prompt()

    def supports_self_flow_ema_sampling(self) -> bool:
        return self.self_flow_ema_sampling
