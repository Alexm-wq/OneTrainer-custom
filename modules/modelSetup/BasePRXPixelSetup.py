from abc import ABCMeta
from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.PRXPixelModel import PRXPixelModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.modelSetup.mixin.ModelSetupText2ImageMixin import ModelSetupText2ImageMixin
from modules.util.checkpointing_util import (
    enable_checkpointing_for_prx_transformer,
    enable_checkpointing_for_qwen3vl_encoder_layers,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.quantization_util import quantize_layers
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor


class BasePRXPixelSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    ModelSetupText2ImageMixin,
    metaclass=ABCMeta,
):
    LAYER_PRESETS = {
        "attn-mlp": ["attn", "gate_proj", "up_proj", "down_proj"],
        "attn-only": ["attn"],
        "blocks": ["blocks"],
        "full": [],
    }

    def setup_optimizations(self, model: PRXPixelModel, config: TrainConfig):
        model.transformer_offload_conductor = enable_checkpointing_for_prx_transformer(
            model.transformer, config, config.transformer,
        )
        model.text_encoder_offload_conductor = enable_checkpointing_for_qwen3vl_encoder_layers(
            model.text_encoder, config, config.text_encoder,
        )

        model.autocast_context, model.train_dtype = create_autocast_context(
            self.train_device, config.train_dtype, config.enable_autocast_cache,
        )
        model.text_encoder_autocast_context, model.text_encoder_train_dtype = disable_fp16_autocast_context(
            self.train_device,
            config.train_dtype,
            config.fallback_train_dtype,
            config.enable_autocast_cache,
        )

        quantize_layers(model.text_encoder, self.train_device, model.text_encoder_train_dtype, config)
        quantize_layers(model.transformer, self.train_device, model.train_dtype, config)
        self._set_attention_backend(model.transformer, config.attention_mechanism, mask=True)

    def predict(
            self,
            model: PRXPixelModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        with model.autocast_context:
            batch_seed = 0 if deterministic else train_progress.global_step * multi.world_size() + multi.rank()
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(batch_seed)
            rand = Random(batch_seed)

            text_encoder_output, text_attention_mask = model.encode_text(
                train_device=self.train_device,
                batch_size=batch["latent_image"].shape[0],
                rand=rand,
                tokens=batch.get("tokens"),
                tokens_mask=batch.get("tokens_mask"),
                text_encoder_output=batch.get("text_encoder_hidden_state"),
                text_encoder_dropout_probability=(
                    0.0 if self._dpo_conditioning_locked() else config.text_encoder.dropout_probability
                ) if not deterministic else None,
            )
            if config.cep_gamma > 0 and not deterministic and not self._dpo_conditioning_locked():
                text_encoder_output = self._apply_conditional_embedding_perturbation(
                    text_encoder_output, config.cep_gamma, generator,
                )

            pixel_image = batch["latent_image"]
            pixel_noise = self._create_noise(pixel_image, config, generator) * 2.0
            num_train_timesteps = model.noise_scheduler.config["num_train_timesteps"]
            timestep = self._get_timestep_discrete(
                num_train_timesteps,
                deterministic,
                generator,
                pixel_image.shape[0],
                config,
                shift=config.timestep_shift,
            )
            noisy_pixel_image, sigma = self._add_noise_discrete(
                pixel_image,
                pixel_noise,
                timestep,
                model.noise_scheduler.timesteps,
            )

            if torch.all(text_attention_mask):
                text_attention_mask = None

            predicted_image = model.transformer(
                hidden_states=noisy_pixel_image.to(dtype=model.train_dtype.torch_dtype()),
                timestep=timestep.to(dtype=torch.float32) / float(num_train_timesteps),
                encoder_hidden_states=text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                attention_mask=text_attention_mask,
                return_dict=False,
            )[0]

            model_output_data = {
                "loss_type": "target",
                "timestep": timestep,
                "predicted": predicted_image,
                "target": pixel_image,
            }

            if config.debug_mode:
                with torch.no_grad():
                    directory = config.debug_dir + "/training_batches"
                    self._save_tokens("6-prompt", batch["tokens"], model.tokenizer, config, train_progress)
                    self._save_image(pixel_noise, directory, "1-noise", train_progress.global_step)
                    self._save_image(noisy_pixel_image, directory, "2-noisy-image", train_progress.global_step)
                    self._save_image(predicted_image, directory, "3-predicted-image", train_progress.global_step)
                    self._save_image(pixel_image, directory, "4-target-image", train_progress.global_step)

        return model_output_data

    def rlhf_logp_per_sample(
            self,
            model: PRXPixelModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        return -self._flow_matching_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
            sigmas=model.noise_scheduler.sigmas,
        )

    def calculate_loss(
            self,
            model: PRXPixelModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        return self._flow_matching_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
            sigmas=model.noise_scheduler.sigmas,
        ).mean()

    def prepare_text_caching(self, model: PRXPixelModel, config: TrainConfig):
        model.to(self.temp_device)
        if not config.train_text_encoder_or_embedding():
            model.text_encoder_to(self.train_device)
        model.eval()
        torch_gc()
