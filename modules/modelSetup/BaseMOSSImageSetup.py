import re
import time
from random import Random

import modules.util.multi_gpu_util as multi
from modules.modelSetup.BaseFlux2Setup import BaseFlux2Setup
from modules.module.moss.self_flow import moss_self_flow_forward
from modules.module.SelfFlow import (
    SelfFlowEMA,
    SelfFlowProjector,
    representation_alignment_loss,
    resolve_self_flow_layers,
    structural_alignment_loss,
    validate_self_flow_config,
)
from modules.util.checkpointing_util import enable_checkpointing
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.ConceptType import ConceptType
from modules.util.enum.DataType import DataType
from modules.util.enum.LossWeight import LossWeight
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

    def __init__(self, train_device, temp_device, debug_mode):
        super().__init__(train_device, temp_device, debug_mode)
        # Self-Flow metrics since the last TensorBoard report: name -> (sum, count)
        self._self_flow_metrics = {}
        self._self_flow_pending_timers = []

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
        if self._self_flow_forward(config, deterministic):
            return self._predict_self_flow(model, batch, config, train_progress)
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

    def _self_flow_forward(self, config, deterministic):
        """Self-Flow augments the ordinary supervised forward only.

        DPO preference scores (policy, reference and streamed replay), validation and prior
        predictions keep the native single-timestep forward, so preference log-likelihoods
        and validation losses stay comparable to training without Self-Flow.
        """
        return (
            config.self_flow_enabled and not deterministic
            and torch.is_grad_enabled() and not self._dpo_conditioning_locked()
        )

    def _validate_self_flow(self, model, config):
        validate_self_flow_config(config)
        if not config.self_flow_enabled:
            return
        if config.masked_training and config.masked_prior_preservation_weight > 0:
            raise NotImplementedError("Self-Flow cannot be combined with masked prior preservation.")
        if config.multi_gpu or multi.world_size() > 1:
            raise NotImplementedError("The Self-Flow CPU EMA teacher supports single-GPU training only.")
        if not config.transformer.train:
            raise ValueError("Self-Flow requires training the MOSS transformer.")
        if config.training_method == TrainingMethod.FINE_TUNE and config.transformer.offload_fraction > 0:
            raise NotImplementedError(
                "Self-Flow full fine-tuning swaps EMA weights into the transformer, which layer offloading "
                "moves between devices. Use LoRA, or disable layer offloading (activation offloading works)."
            )
        model.self_flow_student_layer, model.self_flow_teacher_layer = resolve_self_flow_layers(
            config.self_flow_student_layer, config.self_flow_teacher_layer, len(model.transformer.layers),
        )
        structural = f"on ({config.self_flow_structural_tokens} tokens)" if config.self_flow_structural_enabled else "off"
        print(
            f"[Self-Flow] student block {model.self_flow_student_layer}, EMA teacher block "
            f"{model.self_flow_teacher_layer}, mask ratio {config.self_flow_mask_ratio:g}, structural alignment {structural}"
        )

    def _create_self_flow_projector(self, model, config, dtype):
        if not config.self_flow_enabled:
            return
        model.self_flow_projector = SelfFlowProjector(model.transformer.config.dim).to(dtype=dtype)
        saved = model.self_flow_state_dict
        if saved is not None and saved.get("projector") is not None:
            model.self_flow_projector.load_state_dict(saved["projector"], strict=True)

    def _create_self_flow_ema(self, model, config, parameters, layer_groups=None):
        if not config.self_flow_enabled:
            return
        saved = model.self_flow_state_dict
        saved_ema = saved.get("ema") if saved is not None else None
        if saved_ema is None:
            print("[Self-Flow] initializing the EMA teacher from the current weights.")
        if config.training_method == TrainingMethod.FINE_TUNE:
            count = sum(parameter.numel() for parameter in parameters)
            print(f"[Self-Flow] the EMA teacher keeps an fp32 copy of {count / 1e9:.2f}B parameters in CPU RAM.")
        model.self_flow_ema = SelfFlowEMA(parameters, config.self_flow_ema_decay, saved_ema, layer_groups)
        model.self_flow_state_dict = None

    def _record_self_flow_metric(self, name, value):
        # Device tensors are summed on the device and read once per report, so logging never syncs a step.
        if isinstance(value, torch.Tensor):
            value = value.detach().float().mean()
        total, count = self._self_flow_metrics.get(name, (0.0, 0))
        self._self_flow_metrics[name] = (total + value, count + 1)

    def _predict_self_flow(self, model, batch, config, train_progress):
        """Dual-timestep training forward (Self-Flow, Black Forest Labs).

        Each image token is noised to one of two sampled timesteps; text tokens keep the
        sample's base timestep. The EMA teacher sees every token at the cleaner of the two
        timesteps, and the student's projected intermediate feature is aligned to the
        teacher's deeper feature. The velocity target is unchanged.
        """
        if model.self_flow_ema is None or model.self_flow_projector is None:
            raise RuntimeError("Self-Flow is enabled, but its EMA teacher and projector were not initialized.")
        seed = train_progress.global_step * multi.world_size() + multi.rank()
        generator = torch.Generator(device=self.train_device).manual_seed(seed)
        dtype = model.train_dtype.torch_dtype()
        with model.autocast_context:
            hidden, mask, dropped = model.encode_text(
                train_device=self.train_device, rand=Random(seed), return_dropped=True,
                tokens=batch.get("tokens"), tokens_mask=batch.get("tokens_mask"),
                text_encoder_output=batch.get("text_encoder_hidden_state"),
                text_encoder_sequence_length=config.text_encoder_sequence_length,
                text_encoder_dropout_probability=config.text_encoder.dropout_probability,
            )
            if config.cep_gamma > 0:
                hidden = self._apply_conditional_embedding_perturbation(hidden, config.cep_gamma, generator)
            hidden = hidden.to(dtype)
            clean = model.scale_latents(model.patchify_latents(batch["latent_image"].float()))
            noise = self._create_noise(clean, config, generator)
            batch_size, _, height, width = clean.shape
            total = model.noise_scheduler.config.num_train_timesteps
            shift = (model.noise_scheduler.config.training_shift
                     if config.dynamic_timestep_shifting else config.timestep_shift)
            index = self._get_timestep_discrete(total, False, generator, batch_size, config, shift=shift)
            second_index = self._get_timestep_discrete(total, False, generator, batch_size, config, shift=shift)
            token_mask = torch.rand(
                (batch_size, height * width), generator=generator, device=self.train_device,
            ) < config.self_flow_mask_ratio
            token_index = torch.where(token_mask, second_index[:, None], index[:, None])
            # sigma = (index + 1) / T, so the smaller index is the cleaner view.
            clean_index = torch.minimum(index, second_index)
            token_sigma = ((token_index.float() + 1) / total).view(batch_size, 1, height, width)
            clean_sigma = ((clean_index.float() + 1) / total).view(batch_size, 1, 1, 1)

            teacher_start = self._self_flow_timer()
            ema = model.self_flow_ema
            with ema.teacher_weights(model.self_flow_adapter_modules()), torch.no_grad():
                teacher_feature = moss_self_flow_forward(
                    model.transformer, (clean * (1 - clean_sigma) + noise * clean_sigma).to(dtype),
                    image_timestep=clean_index.float() + 1, text_timestep=clean_index.float() + 1,
                    encoder_hidden_states=hidden, encoder_attention_mask=mask,
                    stop_at_layer=model.self_flow_teacher_layer, layer_weights=ema.layer_weights,
                ).feature
            self._record_self_flow_time("performance/self_flow_teacher_ms", teacher_start)
            if config.self_flow_teacher_target_offload:
                teacher_feature = teacher_feature.to("cpu")

            student = moss_self_flow_forward(
                model.transformer, (clean * (1 - token_sigma) + noise * token_sigma).to(dtype),
                image_timestep=token_index.float() + 1, text_timestep=index.float() + 1,
                encoder_hidden_states=hidden, encoder_attention_mask=mask,
                capture_layer=model.self_flow_student_layer,
            )
            teacher_feature = teacher_feature.to(device=student.feature.device, dtype=student.feature.dtype)
            projected = model.self_flow_projector(student.feature)
            representation = representation_alignment_loss(projected, teacher_feature)
            structural = structural_alignment_loss(
                projected, teacher_feature, config.self_flow_structural_tokens, generator,
            ) if config.self_flow_structural_enabled else None

        output = {
            "loss_type": "target", "timestep": index,
            "self_flow_token_timestep": token_index,
            "predicted": model.unpatchify_latents(student.sample),
            "target": model.unpatchify_latents(noise - clean),
            "self_flow_representation_loss": representation,
        }
        if structural is not None:
            output["self_flow_structural_loss"] = structural
        if config.loss_weight_fn == LossWeight.SIGMA:
            # Weight every patch token by its own sigma, in the unpatchified latent layout.
            output["element_loss_weight"] = token_sigma.repeat_interleave(2, -2).repeat_interleave(2, -1)

        self._record_self_flow_metric("self_flow/t_mean", (index.float() + 1) / total)
        self._record_self_flow_metric("self_flow/s_mean", (second_index.float() + 1) / total)
        self._record_self_flow_metric("self_flow/clean_t_mean", (clean_index.float() + 1) / total)
        self._record_self_flow_metric("self_flow/mask_ratio_actual", token_mask)

        if config.identity_reward:
            rows = self._identity_reward_rows(batch, config, train_progress, token_sigma.flatten(1), dropped)
            if rows:
                output["identity_rows"] = rows
        return output

    def _self_flow_loss(self, batch, data, config, generation_loss):
        if "prior_target" in data:
            raise NotImplementedError(
                "Self-Flow cannot be combined with prior prediction concepts or masked prior preservation: "
                "their target comes from the native single-timestep forward."
            )
        representation = data["self_flow_representation_loss"]
        per_sample = config.self_flow_rep_weight * representation
        structural = data.get("self_flow_structural_loss")
        if structural is not None:
            per_sample = per_sample + config.self_flow_structural_weight * structural
        # Scale and weight like the generation loss, which averages its rows over the batch.
        weights = batch["loss_weight"].to(per_sample)
        scale = config.loss_scaler.get_scale(config.batch_size, config.gradient_accumulation_steps)
        loss = scale * (per_sample * weights).mean()

        self._record_self_flow_metric("loss/generation", generation_loss)
        self._record_self_flow_metric("loss/self_flow_rep", representation)
        self._record_self_flow_metric("self_flow/cosine_similarity", 1.0 - representation.detach())
        if structural is not None:
            self._record_self_flow_metric("loss/self_flow_structural", structural)
        return loss

    def _self_flow_timer(self):
        if self.train_device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def _record_self_flow_time(self, name, start):
        if isinstance(start, float):
            self._record_self_flow_metric(name, (time.perf_counter() - start) * 1000.0)
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        # Resolved at the next report, so timing never blocks the training stream.
        self._self_flow_pending_timers.append((name, start, end))

    def _report_self_flow(self, model, config, tensorboard):
        if not config.self_flow_enabled:
            return
        for name, start, end in self._self_flow_pending_timers:
            end.synchronize()
            self._record_self_flow_metric(name, start.elapsed_time(end))
        self._self_flow_pending_timers.clear()
        step = model.train_progress.global_step
        for name, (total, count) in self._self_flow_metrics.items():
            tensorboard.add_scalar(name, float(total) / count, step)
        self._self_flow_metrics.clear()
        if self.train_device.type == "cuda":
            tensorboard.add_scalar(
                "performance/peak_vram_allocated_mb", torch.cuda.max_memory_allocated(self.train_device) / 2 ** 20, step,
            )

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
                    dropped[row] or float(sigma[row].max()) > config.identity_reward_max_sigma
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
        if "self_flow_representation_loss" in data:
            loss = loss + self._self_flow_loss(batch, data, config, loss)
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
        error = model.patchify_latents((data["target"].index_select(0, index) - data["predicted"].index_select(0, index)).float())
        # Self-Flow noises every 2x2 patch token to its own sigma; the native forward uses one per sample.
        timestep = data.get("self_flow_token_timestep", data["timestep"]).index_select(0, index)
        sigma = (timestep.float() + 1) / model.noise_scheduler.config.num_train_timesteps
        sigma = sigma.view(-1, 1, *error.shape[2:]) if sigma.ndim == 2 else sigma.view(-1, 1, 1, 1)
        std = torch.sqrt(model.vae.bn.running_var + model.vae.config.batch_norm_eps).view(1, -1, 1, 1).to(error)
        clean = batch["latent_image"].index_select(0, index.to(batch["latent_image"].device)).to(error)
        prediction = clean + model.unpatchify_latents(error * std * sigma.to(error))
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
        self._report_self_flow(model, config, tensorboard)
        metrics = getattr(self, "_identity_metrics", None)
        if metrics and metrics[1]:
            step = model.train_progress.global_step
            tensorboard.add_scalar("loss/identity", metrics[0] / metrics[1], step)
            tensorboard.add_scalar("identity/faces", metrics[1], step)
            metrics[:] = [0.0, 0]
