from random import Random

import modules.util.multi_gpu_util as multi
from modules.modelSetup.BaseFlux2Setup import BaseFlux2Setup
from modules.util.checkpointing_util import enable_checkpointing
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.DataType import DataType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.quantization_util import quantize_layers

import torch


class BaseMOSSImageSetup(BaseFlux2Setup):
    LAYER_PRESETS = {
        "attn-mlp": ["layers."],
        "attention": {"patterns": [r"^layers\.\d+\.proj\.(q|k|v|o)(\.|$)"], "regex": True},
        "mlp": {"patterns": [r"^layers\.\d+\.proj\.ffn_"], "regex": True},
        "text-refiner": ["text_embedder."],
        "full": [],
    }

    def setup_optimizations(self, model, config):
        if config.text_encoder.train:
            raise ValueError("MOSS trains the DiT only; keep the Qwen3.5 text encoder frozen.")
        if config.train_dtype not in (DataType.FLOAT_32, DataType.BFLOAT_16, DataType.FLOAT_16, DataType.TFLOAT_32):
            raise ValueError("Choose a floating-point training dtype; quantization applies to model weights.")
        if config.training_method == TrainingMethod.FINE_TUNE and config.transformer.weight_dtype.is_quantized():
            raise ValueError("Quantized MOSS base weights require LoRA/PEFT training. Use floating weights for full fine-tuning.")
        model.transformer_offload_conductor = enable_checkpointing(
            model.transformer, config, config.transformer, config.compile,
            [(model.transformer.text_embedder.blocks, ["x"]), (model.transformer.layers, ["x"])],
        )
        # Qwen3.5 has recurrent GDN blocks: keep each prompt unpadded and only
        # offload its language layers, in their actual execution order.
        if hasattr(model.text_encoder, "language_model"):
            model.text_encoder_offload_conductor = enable_checkpointing(
                model.text_encoder, config, config.text_encoder, False,
                [(model.text_encoder.language_model.layers, [])],
            )
        model.autocast_context, model.train_dtype = create_autocast_context(
            self.train_device, config.train_dtype, config.enable_autocast_cache,
        )
        model.text_encoder_autocast_context, model.text_encoder_train_dtype = disable_fp16_autocast_context(
            self.train_device, config.train_dtype, config.fallback_train_dtype, config.enable_autocast_cache,
        )
        if config.text_encoder.weight_dtype.is_quantized():
            # The loader stores a quantized encoder's remaining tensors in the fallback dtype.
            # Encoding runs outside the DiT autocast, so its projections must return that dtype too.
            model.text_encoder_train_dtype = config.fallback_train_dtype
        model.vae.to(dtype=torch.float32)
        model.vae_autocast_context = torch.autocast(self.train_device.type, enabled=False)
        quantize_layers(model.text_encoder, self.train_device, model.text_encoder_train_dtype, config)
        quantize_layers(model.transformer, self.train_device, model.train_dtype, config)
        self._set_attention_backend(model.transformer, config.attention_mechanism, mask=True)

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
                    if not deterministic and not self._dpo_conditioning_locked() else 0.0
                ),
            )
            if config.cep_gamma > 0 and not deterministic and not self._dpo_conditioning_locked():
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
