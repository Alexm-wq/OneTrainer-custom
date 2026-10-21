import re
from random import Random

import modules.util.multi_gpu_util as multi
from modules.modelSetup.BaseFlux2Setup import BaseFlux2Setup
from modules.util.checkpointing_util import enable_checkpointing
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.ConceptType import ConceptType
from modules.util.enum.DataType import DataType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.identity_reward import identity_losses, load_arcface, resolve_model_file
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
            hidden, mask, dropped = model.encode_text(
                train_device=self.train_device, rand=Random(seed), return_dropped=True,
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
        output = {
            "loss_type": "target", "timestep": index,
            "predicted": model.unpatchify_latents(predicted),
            "target": model.unpatchify_latents(noise - clean),
        }
        if not deterministic and config.identity_reward:
            rows = self._identity_reward_rows(batch, config, train_progress, sigma, dropped)
            if rows:
                output["identity_rows"] = rows
        return output

    def _identity_reward_rows(self, batch, config, train_progress, sigma, dropped):
        """Rows of an ordinary supervised forward whose face the identity reward may score.

        DPO pairs, their streamed replays, prior predictions and validation never qualify, and
        neither do rows that lost their caption to dropout or lack a trigger word: the reward
        only teaches the face in samples whose prompt names the character.
        """
        if (
                not torch.is_grad_enabled() or self._dpo_conditioning_locked()
                or "face_landmarks" not in batch
                or train_progress.global_step % max(config.identity_reward_interval, 1) != 0
        ):
            return None
        triggers = [word.strip() for word in config.identity_reward_trigger.split(",") if word.strip()]
        patterns = [re.compile(rf"(?<!\w){re.escape(word)}(?!\w)", re.IGNORECASE) for word in triggers]
        landmarks = batch["face_landmarks"]
        concept_types = batch.get("concept_type")
        prompts = batch.get("prompt")
        rows = []
        for row in range(sigma.shape[0]):
            concept_type = concept_types[row] if concept_types is not None else ConceptType.STANDARD
            if isinstance(concept_type, bytes):
                concept_type = concept_type.decode("utf-8", errors="ignore")
            if (
                    dropped[row] or float(sigma[row]) > config.identity_reward_max_sigma
                    or str(concept_type) != str(ConceptType.STANDARD)
                    or not torch.isfinite(landmarks[row]).all()
                    or (patterns and (prompts is None or not any(p.search(prompts[row]) for p in patterns)))
            ):
                continue
            rows.append(row)
        return rows

    def _prepare_identity_reward(self, model, config):
        if not config.identity_reward or getattr(self, "_arcface", None) is not None:
            return
        if not config.identity_reward_arcface_path or not config.identity_reward_detector_path:
            raise ValueError("The identity reward needs an ArcFace backbone and a YuNet face detector model.")
        self._arcface = load_arcface(resolve_model_file(config.identity_reward_arcface_path)).to(self.train_device)
        self._identity_metrics = [0.0, 0]
        if getattr(model.vae, "_supports_gradient_checkpointing", False):
            # Only matters for the reward's decode; the VAE runs without gradients everywhere else.
            model.vae.enable_gradient_checkpointing()

    def calculate_loss(self, model, batch, data, config):
        loss = super().calculate_loss(model, batch, data, config)
        rows = data.get("identity_rows")
        if rows:
            loss = loss + self._identity_reward_loss(model, batch, data, config, rows)
        return loss

    def _identity_reward_loss(self, model, batch, data, config, rows):
        """ArcFace distance between the face in the x0 prediction and in the training image.

        x0 = x_t - sigma * v. With the target v = noise - clean, that is the clean latent plus
        sigma * (target - predicted), unscaled from the BN-normalized 2x2 patches into the VAE latent.
        """
        self._prepare_identity_reward(model, config)
        index = torch.tensor(rows, device=data["predicted"].device)
        sigma = (data["timestep"].index_select(0, index).float() + 1) / model.noise_scheduler.config.num_train_timesteps
        error = model.patchify_latents((data["target"].index_select(0, index) - data["predicted"].index_select(0, index)).float())
        std = torch.sqrt(model.vae.bn.running_var + model.vae.config.batch_norm_eps).view(1, -1, 1, 1).to(error)
        clean = batch["latent_image"].index_select(0, index.to(batch["latent_image"].device)).to(error)
        prediction = clean + model.unpatchify_latents(error * std) * sigma.view(-1, 1, 1, 1)
        # The VAE decodes in bf16 where the DiT trains in bf16; FLUX.2's decoder overflows in fp16.
        decode_autocast = torch.autocast(
            self.train_device.type, dtype=torch.bfloat16,
            enabled=self.train_device.type == "cuda" and config.train_dtype == DataType.BFLOAT_16,
        )

        def decode(latent):
            with decode_autocast:
                return model.vae.decode(latent, return_dict=False)[0]

        def embed(faces):
            with torch.autocast(self.train_device.type, enabled=False):
                return self._arcface(faces.float())

        losses, valid = identity_losses(
            prediction, clean, batch["face_landmarks"][rows], decode, embed, config.identity_reward_max_crop,
        )
        if not valid.any():
            return losses.sum() * 0
        self._identity_metrics[0] += float(losses.detach()[valid.to(losses.device)].sum())
        self._identity_metrics[1] += int(valid.sum())
        weights = batch["loss_weight"].index_select(0, index.to(batch["loss_weight"].device)).to(losses)
        scale = config.loss_scaler.get_scale(config.batch_size, config.gradient_accumulation_steps)
        # Matches the supervised loss, which averages its rows over the whole batch.
        return config.identity_reward_weight * scale * (losses * weights).sum() / data["predicted"].shape[0]

    def report_to_tensorboard(self, model, config, scheduler, tensorboard):
        super().report_to_tensorboard(model, config, scheduler, tensorboard)
        metrics = getattr(self, "_identity_metrics", None)
        if metrics and metrics[1]:
            step = model.train_progress.global_step
            tensorboard.add_scalar("loss/identity", metrics[0] / metrics[1], step)
            tensorboard.add_scalar("identity/faces", metrics[1], step)
            metrics[:] = [0.0, 0]
