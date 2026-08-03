import copy
from collections.abc import Callable

from modules.model.PRXPixelModel import PRXPixelModel
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


@factory.register(BaseModelSampler, ModelType.PRX_PIXEL)
class PRXPixelSampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: PRXPixelModel,
            model_type: ModelType,
    ):
        super().__init__(train_device, temp_device)
        self.model = model
        self.model_type = model_type
        self.pipeline = model.create_pipeline()

    @torch.no_grad()
    def __sample_base(
            self,
            prompt: str,
            negative_prompt: str,
            height: int,
            width: int,
            seed: int,
            random_seed: bool,
            diffusion_steps: int,
            cfg_scale: float,
            on_update_progress: Callable[[int, int], None],
    ) -> ModelSamplerOutput:
        with self.model.autocast_context:
            generator = torch.Generator(device=self.train_device)
            if random_seed:
                generator.seed()
            else:
                generator.manual_seed(seed)

            self.pipeline.scheduler = copy.deepcopy(self.model.noise_scheduler)
            self.model.text_encoder_to(self.train_device)

            batch_size = 2 if cfg_scale > 1.0 else 1
            prompt_embedding, prompt_mask = self.model.encode_text(
                train_device=self.train_device,
                batch_size=batch_size,
                text=[prompt, negative_prompt] if cfg_scale > 1.0 else prompt,
            )
            if cfg_scale > 1.0:
                positive_embedding, negative_embedding = prompt_embedding.chunk(2)
                positive_mask, negative_mask = prompt_mask.chunk(2)
            else:
                positive_embedding = prompt_embedding
                positive_mask = prompt_mask
                negative_embedding = None
                negative_mask = None

            self.model.text_encoder_to(self.temp_device)
            torch_gc()
            self.model.transformer_to(self.train_device)

            def step_callback(_pipeline, step, _timestep, callback_kwargs):
                on_update_progress(step + 1, diffusion_steps)
                return callback_kwargs

            try:
                output = self.pipeline(
                    prompt=None,
                    negative_prompt=None,
                    height=height,
                    width=width,
                    num_inference_steps=diffusion_steps,
                    guidance_scale=cfg_scale,
                    generator=generator,
                    prompt_embeds=positive_embedding,
                    negative_prompt_embeds=negative_embedding,
                    prompt_attention_mask=positive_mask,
                    negative_prompt_attention_mask=negative_mask,
                    output_type="pil",
                    use_resolution_binning=False,
                    callback_on_step_end=step_callback,
                )
            finally:
                self.model.transformer_to(self.temp_device)
                torch_gc()

            return ModelSamplerOutput(file_type=FileType.IMAGE, data=output.images[0])

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
        patch_size = int(getattr(self.model.transformer.config, "patch_size", 16))
        sampler_output = self.__sample_base(
            prompt=sample_config.prompt,
            negative_prompt=sample_config.negative_prompt,
            height=self.quantize_resolution(sample_config.height, patch_size),
            width=self.quantize_resolution(sample_config.width, patch_size),
            seed=sample_config.seed,
            random_seed=sample_config.random_seed,
            diffusion_steps=sample_config.diffusion_steps,
            cfg_scale=sample_config.cfg_scale,
            on_update_progress=on_update_progress,
        )
        self.save_sampler_output(
            sampler_output, destination, image_format, video_format, audio_format,
        )
        on_sample(sampler_output)
