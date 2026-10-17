from random import Random

import modules.util.multi_gpu_util as multi
from modules.modelSetup.BaseFlux2Setup import BaseFlux2Setup
from modules.util.checkpointing_util import enable_checkpointing
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.DataType import DataType

import torch


class BaseMOSSImageSetup(BaseFlux2Setup):
    LAYER_PRESETS = {
        "attn-mlp": ["layers."],
        "full": [],
    }

    def setup_optimizations(self, model, config):
        if config.text_encoder.train:
            raise ValueError("MOSS trains the DiT only; keep the Qwen3.5 text encoder frozen.")
        if config.train_dtype not in (DataType.FLOAT_32, DataType.BFLOAT_16):
            raise ValueError("MOSS training supports FLOAT_32 or BFLOAT_16.")
        if config.transformer.offloading_enabled() or config.text_encoder.offloading_enabled():
            raise ValueError(
                "MOSS's per-sample variable-length forward does not support layer/activation offloading yet. "
                "Set both offload fractions to 0 and activation offloading off; use text/image caching."
            )
        enable_checkpointing(
            model.transformer, config, config.transformer, config.compile,
            [(model.transformer.text_embedder.blocks, ["x"]), (model.transformer.layers, ["x"])],
            offload_enabled=False,
        )
        model.autocast_context, model.train_dtype = create_autocast_context(
            self.train_device, config.train_dtype, config.enable_autocast_cache,
        )
        model.text_encoder_autocast_context, model.text_encoder_train_dtype = disable_fp16_autocast_context(
            self.train_device, config.train_dtype, config.fallback_train_dtype, config.enable_autocast_cache,
        )
        model.vae.to(dtype=torch.float32)
        model.vae_autocast_context = torch.autocast(self.train_device.type, enabled=False)

    def prepare_text_caching(self, model, config):
        super().prepare_text_caching(model, config)
        if config.text_encoder.dropout_probability:
            model.prepare_empty_text(config.text_encoder_sequence_length)

    def predict(self, model, batch, config, train_progress, *, deterministic=False):
        seed = 0 if deterministic else train_progress.global_step * multi.world_size() + multi.rank()
        if self._dpo_validation_seed is not None:
            seed = self._dpo_validation_seed
        generator = torch.Generator(device=self.train_device).manual_seed(seed)
        with model.autocast_context:
            hidden, mask = model.encode_text(
                train_device=self.train_device, rand=Random(seed),
                tokens=batch.get("tokens"), tokens_mask=batch.get("tokens_mask"),
                text_encoder_output=batch.get("text_encoder_hidden_state"),
                text_encoder_sequence_length=config.text_encoder_sequence_length,
                text_encoder_dropout_probability=(
                    config.text_encoder.dropout_probability
                    if not deterministic and self._dpo_paired_half is None else 0.0
                ),
            )
            if config.cep_gamma > 0 and not deterministic and self._dpo_paired_half is None:
                hidden = self._apply_conditional_embedding_perturbation(hidden, config.cep_gamma, generator)
            clean = model.scale_latents(model.patchify_latents(batch["latent_image"].float()))
            noise = self._create_noise(clean, config, generator)
            total = model.noise_scheduler.config.num_train_timesteps
            index = self._get_timestep_discrete(
                total, deterministic, generator, clean.shape[0], config,
                shift=(model.noise_scheduler.config.training_shift
                       if config.dynamic_timestep_shifting else config.timestep_shift),
            )
            # Index 0 denotes sigma 1/1000. The model consumes raw time 1..1000,
            # whereas loss weighting consumes the zero-based grid index.
            sigma = (index.float() + 1) / total
            noisy = clean * (1 - sigma[:, None, None, None]) + noise * sigma[:, None, None, None]
            predicted = model.transformer(
                hidden_states=noisy.to(model.train_dtype.torch_dtype()),
                timestep=index.float() + 1,
                encoder_hidden_states=hidden.to(model.train_dtype.torch_dtype()),
                encoder_attention_mask=mask,
            ).sample
        return {
            "loss_type": "target", "timestep": index,
            "predicted": model.unpatchify_latents(predicted),
            "target": model.unpatchify_latents(noise - clean),
        }
