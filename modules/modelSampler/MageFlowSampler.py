from collections.abc import Callable

from modules.model.MageFlowModel import MageFlowModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util import factory
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.torch_util import torch_gc

import torch


@factory.register(BaseModelSampler, ModelType.MAGE_FLOW)
class MageFlowSampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: MageFlowModel,
            model_type: ModelType,
    ):
        super().__init__(train_device, temp_device)
        self.model = model
        self.model_type = model_type

    @torch.no_grad()
    def _sample(self, sample_config: SampleConfig, on_update_progress) -> ModelSamplerOutput:
        try:
            from mage_flow.pipeline import generate_images
        except ImportError as exc:
            raise RuntimeError("Mage sampling requires Microsoft's official mage_flow package") from exc

        official = getattr(self.model, "official_model", None)
        if official is None:
            raise RuntimeError("Mage official model wrapper was not retained by the loader")

        self.model.to(self.train_device)
        seed = -1 if sample_config.random_seed else int(sample_config.seed)
        height = self.quantize_resolution(sample_config.height, 16)
        width = self.quantize_resolution(sample_config.width, 16)
        try:
            images = generate_images(
                official,
                prompts=[sample_config.prompt],
                neg_prompts=[sample_config.negative_prompt or " "],
                seeds=[seed],
                steps=int(sample_config.diffusion_steps),
                cfg=float(sample_config.cfg_scale),
                heights=[height],
                widths=[width],
                device=str(self.train_device),
                prompt_template="mage-flow",
                static_shift=None,
                batch_cfg=True,
            )
            on_update_progress(int(sample_config.diffusion_steps), int(sample_config.diffusion_steps))
            return ModelSamplerOutput(FileType.IMAGE, images[0])
        finally:
            self.model.to(self.temp_device)
            torch_gc()

    def sample(
            self,
            sample_config: SampleConfig,
            destination: str,
            image_format: ImageFormat | None = None,
            video_format: VideoFormat | None = None,
            audio_format: AudioFormat | None = None,
            on_sample: Callable[[ModelSamplerOutput], None] = lambda _: None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ):
        output = self._sample(sample_config, on_update_progress)
        self.save_sampler_output(output, destination, image_format, video_format, audio_format)
        on_sample(output)
