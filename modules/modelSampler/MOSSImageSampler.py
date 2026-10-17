import copy

from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.module.moss.pipeline_moss_image import _validate_skip_layers
from modules.util import factory
from modules.util.enum.FileType import FileType
from modules.util.enum.ModelType import ModelType
from modules.util.moss_layout import sample_layout_prompt
from modules.util.torch_util import torch_gc

import torch

from tqdm import tqdm


@factory.register(BaseModelSampler, ModelType.MOSS_IMAGE)
class MOSSImageSampler(BaseModelSampler):
    def __init__(self, train_device, temp_device, model, model_type):
        super().__init__(train_device, temp_device)
        self.model = model
        self.model_type = model_type
        self.pipeline = model.create_pipeline()

    @torch.no_grad()
    def sample(
            self, sample_config, destination, image_format=None, video_format=None, audio_format=None,
            on_sample=lambda _: None, on_update_progress=lambda _i, _n: None,
    ):
        model, config = self.model, sample_config
        skip = tuple(int(item.strip()) for item in config.moss_uncond_skip_layers.split(",") if item.strip())
        _validate_skip_layers(skip, len(model.transformer.layers), "moss_uncond_skip_layers")
        width, height = self.quantize_resolution(config.width, 16), self.quantize_resolution(config.height, 16)
        prompt = sample_layout_prompt(config, width, height)
        scheduler = copy.deepcopy(model.noise_scheduler)
        generator = torch.Generator(device="cpu")
        if config.random_seed:
            generator.seed()
        else:
            generator.manual_seed(config.seed)

        model.text_encoder_to(self.train_device)
        try:
            positive, positive_mask = model.encode_text(
                train_device=self.train_device, text=prompt,
                text_encoder_sequence_length=config.text_encoder_1_sequence_length,
            )
            if config.cfg_scale != 1.0:
                negative, negative_mask = model.encode_text(
                    train_device=self.train_device, text=config.negative_prompt,
                    text_encoder_sequence_length=config.text_encoder_1_sequence_length,
                )
        finally:
            model.text_encoder_to(self.temp_device)
            torch_gc()

        latents = torch.randn(
            (1, model.transformer.config.latent_channels, height // 16, width // 16),
            generator=generator, dtype=torch.float32,
        ).to(self.train_device)
        scheduler.set_timesteps(config.diffusion_steps, device=self.train_device)
        model.transformer_to(self.train_device)
        try:
            for i, timestep in enumerate(tqdm(scheduler.timesteps, desc="sampling")):
                with model.autocast_context:
                    inputs = latents.to(model.train_dtype.torch_dtype())
                    cond = model.transformer(
                        inputs, timestep, positive.to(model.train_dtype.torch_dtype()), positive_mask,
                    ).sample.float()
                    if config.cfg_scale != 1.0:
                        uncond = model.transformer(
                            inputs, timestep, negative.to(model.train_dtype.torch_dtype()), negative_mask,
                            skip_layers=skip,
                        ).sample.float()
                        cond = uncond + config.cfg_scale * (cond - uncond)
                # Keep CFG and integration in fp32, as in the upstream pipeline.
                latents = scheduler.step(cond, timestep, latents, return_dict=False)[0].float()
                on_update_progress(i + 1, len(scheduler.timesteps))
        finally:
            model.transformer_to(self.temp_device)
            torch_gc()

        model.vae_to(self.train_device)
        try:
            with torch.autocast(self.train_device.type, enabled=False):
                decoded = model.vae.decode(
                    model.unpatchify_latents(model.unscale_latents(latents.float())), return_dict=False,
                )[0]
            image = self.pipeline.image_processor.postprocess(decoded, output_type="pil")[0]
        finally:
            model.vae_to(self.temp_device)
            torch_gc()
        output = ModelSamplerOutput(file_type=FileType.IMAGE, data=image)
        self.save_sampler_output(output, destination, image_format, video_format, audio_format)
        on_sample(output)
