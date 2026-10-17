from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.Flux3Model import TRAINING_SHIFT
from modules.modelSetup.BaseFlux2Setup import BaseFlux2Setup
from modules.util.checkpointing_util import enable_checkpointing, enable_checkpointing_for_qwen3vl_encoder_layers
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.DataType import DataType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.quantization_util import quantize_layers

import torch


class BaseFlux3Setup(BaseFlux2Setup):
    LAYER_PRESETS = {
        "attn-mlp": ["mode_blocks.image.", "txt_mode_blocks.", "single_transformer_blocks."],
        "attention": {"patterns": [r"^(mode_blocks\.image|txt_mode_blocks|single_transformer_blocks)\.\d+\.attn\."], "regex": True},
        "mlp": {"patterns": [r"^(mode_blocks\.image|txt_mode_blocks|single_transformer_blocks)\.\d+\.ff\."], "regex": True},
        "image": ["mode_blocks.image.", "single_transformer_blocks."],
        "text-refiner": ["txt_mode_blocks."],
        "full": [],
    }

    def setup_optimizations(self, model, config):
        if config.text_encoder.train:
            raise ValueError("FLUX.3 trains the DiT only; keep the Qwen3-VL text encoder frozen.")
        if model.transformer.config.flow_map:
            raise ValueError("Train black-forest-labs/flux-3-image-klein-base. The distilled flow-map requires a different objective.")
        model.text_length(config.text_encoder_sequence_length)
        if config.train_dtype not in (DataType.FLOAT_32, DataType.BFLOAT_16, DataType.FLOAT_16, DataType.TFLOAT_32):
            raise ValueError("Choose a floating-point training dtype; quantization applies to model weights.")
        if config.training_method == TrainingMethod.FINE_TUNE and config.transformer.weight_dtype.is_quantized():
            raise ValueError("Quantized FLUX.3 base weights require LoRA/PEFT training. Use floating weights for full fine-tuning.")
        model.transformer_offload_conductor = enable_checkpointing(
            model.transformer, config, config.transformer, config.compile,
            [(model.transformer.txt_mode_blocks, ["hidden_states"]),
             (model.transformer.mode_blocks["image"], ["hidden_states"]),
             (model.transformer.single_transformer_blocks, ["hidden_states", "encoder_hidden_states"])],
        )
        # Reference-only blocks are absent from text-to-image forwards. Keep them
        # outside the conductor's ordered schedule; sampling may still use them.
        enable_checkpointing(
            model.transformer, config, config.transformer, config.compile,
            [(model.transformer.mode_blocks["image_cond"], ["hidden_states"])], offload_enabled=False,
        )
        model.text_encoder_offload_conductor = enable_checkpointing_for_qwen3vl_encoder_layers(
            model.text_encoder, config, config.text_encoder,
        )
        model.autocast_context, model.train_dtype = create_autocast_context(
            self.train_device, config.train_dtype, config.enable_autocast_cache,
        )
        model.text_encoder_autocast_context, model.text_encoder_train_dtype = disable_fp16_autocast_context(
            self.train_device, config.train_dtype, config.fallback_train_dtype, config.enable_autocast_cache,
        )
        model.vae.to(dtype=torch.float32)
        model.vae_autocast_context = torch.autocast(self.train_device.type, enabled=False)
        quantize_layers(model.text_encoder, self.train_device, model.text_encoder_train_dtype, config)
        quantize_layers(model.transformer, self.train_device, model.train_dtype, config)
        self._set_attention_backend(model.transformer, config.attention_mechanism, mask=False)

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
                shift=(TRAINING_SHIFT
                       if config.dynamic_timestep_shifting else config.timestep_shift),
            )
            # The loss index and actual sigma must describe the same noise level.
            sigma = (index.float() + 1) / total
            noisy = clean * (1 - sigma[:, None, None, None]) + noise * sigma[:, None, None, None]
            predicted = model.predict_velocity(
                noisy.to(model.train_dtype.torch_dtype()), sigma, hidden, mask,
            )
        return {
            "loss_type": "target", "timestep": index,
            "predicted": model.unpatchify_latents(predicted),
            "target": model.unpatchify_latents(noise - clean),
        }
