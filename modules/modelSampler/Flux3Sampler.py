import math

from modules.model.Flux3Model import SAMPLING_SHIFT
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util import factory
from modules.util.enum.FileType import FileType
from modules.util.enum.ModelType import ModelType
from modules.util.torch_util import torch_gc

import torch

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm


def sampling_sigmas(steps, shift=SAMPLING_SHIFT, device="cpu"):
    if steps < 1 or not math.isfinite(shift) or shift <= 0:
        raise ValueError("FLUX.3 sampling needs positive steps and timestep shift.")
    t = torch.linspace(1, 0, steps + 1, device=device, dtype=torch.float32)
    return shift * t / (1 + (shift - 1) * t)


def prepare_reference(image):
    # BFL single-reference canvas: aspect-preserving fit, then reflected padding.
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    scale = min(1.0, math.sqrt(4 * 1024 * 1024 / (width * height)), 4096 / max(width, height))
    while True:
        fit = max(1, round(width * scale)), max(1, round(height * scale))
        canvas = tuple(min(4096, max(64, -(-side // 16) * 16)) for side in fit)
        if canvas[0] * canvas[1] <= 4 * 1024 * 1024:
            break
        scale *= 0.995
    image = image.resize(fit, Image.Resampling.LANCZOS) if fit != image.size else image
    left, top = (canvas[0] - fit[0]) // 2, (canvas[1] - fit[1]) // 2
    pad = ((top, canvas[1] - fit[1] - top), (left, canvas[0] - fit[0] - left), (0, 0))
    array = np.pad(np.asarray(image), pad, mode="reflect" if min(fit) > 16 else "symmetric")
    return torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1


@factory.register(BaseModelSampler, ModelType.FLUX_3_KLEIN)
class Flux3Sampler(BaseModelSampler):
    def __init__(self, train_device, temp_device, model, model_type):
        super().__init__(train_device, temp_device)
        self.model, self.model_type = model, model_type
        self.pipeline = model.create_pipeline()

    @torch.no_grad()
    def sample(
            self, sample_config, destination, image_format=None, video_format=None, audio_format=None,
            on_sample=lambda _: None, on_update_progress=lambda _i, _n: None,
    ):
        model, config = self.model, sample_config
        width, height = self.quantize_resolution(config.width, 16), self.quantize_resolution(config.height, 16)
        schedule = sampling_sigmas(config.diffusion_steps, config.flux3_timestep_shift, self.train_device)
        flow_map = bool(model.transformer.config.flow_map)
        if flow_map and config.cfg_scale != 1.0:
            raise ValueError("Distilled FLUX.3 is a flow map: set CFG to 1 (recommended 16 steps).")
        generator = torch.Generator(device="cpu")
        generator.seed() if config.random_seed else generator.manual_seed(config.seed)

        model.text_encoder_to(self.train_device)
        try:
            positive, positive_mask = model.encode_text(
                train_device=self.train_device, text=config.prompt,
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

        reference = None
        if config.base_image_path:
            model.vae_to(self.train_device)
            try:
                with Image.open(config.base_image_path) as image:
                    tensor = prepare_reference(image).to(self.train_device)
                with torch.autocast(self.train_device.type, enabled=False):
                    reference = model.scale_latents(model.patchify_latents(model.vae.encode(tensor).latent_dist.mode()))
            finally:
                model.vae_to(self.temp_device)
                torch_gc()

        # Match BFL's CPU RNG layout (tokens, channels), bf16-rounded initial
        # noise and fp32 state/CFG/Euler integration.
        packed = torch.randn(
            (1, (height // 16) * (width // 16), model.transformer.config.in_channels), generator=generator,
        ).to(torch.bfloat16).float().to(self.train_device)
        latents = model.unpack_latents(packed, height // 16, width // 16)
        model.transformer_to(self.train_device)
        try:
            for i in tqdm(range(config.diffusion_steps), desc="sampling"):
                timestep = schedule[i].reshape(1)
                target = schedule[i + 1].reshape(1) if flow_map else None
                with model.autocast_context:
                    inputs = latents.to(model.train_dtype.torch_dtype())
                    cond = model.predict_velocity(
                        inputs, timestep, positive, positive_mask, target_timestep=target, reference=reference,
                    ).float()
                    if config.cfg_scale != 1.0:
                        uncond = model.predict_velocity(
                            inputs, timestep, negative, negative_mask, reference=reference,
                        ).float()
                        cond = uncond + config.cfg_scale * (cond - uncond)
                latents = latents + (schedule[i + 1] - schedule[i]) * cond
                on_update_progress(i + 1, config.diffusion_steps)
        finally:
            model.transformer_to(self.temp_device)
            torch_gc()

        model.vae_to(self.train_device)
        try:
            with torch.autocast(self.train_device.type, enabled=False):
                decoded = model.vae.decode(
                    model.unpatchify_latents(model.unscale_latents(latents.float())), return_dict=False,
                )[0]
            # BFL converts to uint8 by truncation, whereas generic postprocess rounds.
            array = ((decoded[0].float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray(array)
        finally:
            model.vae_to(self.temp_device)
            torch_gc()
        output = ModelSamplerOutput(file_type=FileType.IMAGE, data=image)
        self.save_sampler_output(output, destination, image_format, video_format, audio_format)
        on_sample(output)
