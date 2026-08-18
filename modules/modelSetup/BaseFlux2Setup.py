from abc import ABCMeta
from random import Random
import time

import modules.util.multi_gpu_util as multi
from modules.model.Flux2Model import Flux2Model
from modules.model.FluxModel import FluxModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupEmbeddingMixin import ModelSetupEmbeddingMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.module.Flux2SelfFlow import (
    flux2_flow_sigma,
    flux2_interpolate_token_view,
    flux2_self_flow_forward,
    flux2_structural_alignment_loss,
    flux2_token_weight_to_spatial,
)
from modules.util.checkpointing_util import (
    enable_checkpointing_for_flux2_transformer,
    enable_checkpointing_for_mistral_encoder_layers,
    enable_checkpointing_for_qwen3_encoder_layers,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.LossWeight import LossWeight
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.quantization_util import quantize_layers
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
import torch.nn.functional as F
from torch import Tensor


class BaseFlux2Setup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    ModelSetupEmbeddingMixin,
    metaclass=ABCMeta
):
    LAYER_PRESETS = {
        "blocks": ["transformer_block"],
        "full": [],
    }

    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super().__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )
        self._self_flow_metric_sums: dict[str, float] = {}
        self._self_flow_metric_counts: dict[str, int] = {}
        self._self_flow_cuda_timers: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        self._self_flow_student_timer = None

    def _record_self_flow_metric(self, name: str, value: float):
        self._self_flow_metric_sums[name] = self._self_flow_metric_sums.get(name, 0.0) + float(value)
        self._self_flow_metric_counts[name] = self._self_flow_metric_counts.get(name, 0) + 1

    def _start_self_flow_timer(self):
        if self.train_device.type == "cuda" and torch.cuda.is_available():
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return ("cuda", event)
        return ("cpu", time.perf_counter())

    def _finish_self_flow_timer(self, name: str, timer):
        if timer is None:
            return
        timer_type, start = timer
        if timer_type == "cuda":
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._self_flow_cuda_timers.setdefault(name, []).append((start, end))
        else:
            self._record_self_flow_metric(name, (time.perf_counter() - start) * 1000.0)

    def _resolve_self_flow_cuda_timers(self):
        for name, timers in self._self_flow_cuda_timers.items():
            for start, end in timers:
                end.synchronize()
                self._record_self_flow_metric(name, start.elapsed_time(end))
        self._self_flow_cuda_timers.clear()

    def setup_optimizations(
            self,
            model: Flux2Model,
            config: TrainConfig,
    ):
        model.transformer_offload_conductor = enable_checkpointing_for_flux2_transformer(model.transformer, config, config.transformer)
        if model.is_dev():
            model.text_encoder_offload_conductor = enable_checkpointing_for_mistral_encoder_layers(model.text_encoder, config, config.text_encoder)
        else:
            model.text_encoder_offload_conductor = enable_checkpointing_for_qwen3_encoder_layers(model.text_encoder, config, config.text_encoder)

        model.autocast_context, model.train_dtype = create_autocast_context(
            self.train_device, config.train_dtype, config.enable_autocast_cache)

        model.text_encoder_autocast_context, model.text_encoder_train_dtype = \
            disable_fp16_autocast_context(
                self.train_device,
                config.train_dtype,
                config.fallback_train_dtype,
                config.enable_autocast_cache,
            )

        quantize_layers(model.text_encoder, self.train_device, model.text_encoder_train_dtype, config)
        quantize_layers(model.vae, self.train_device, model.train_dtype, config)
        quantize_layers(model.transformer, self.train_device, model.train_dtype, config)

        self._set_attention_backend(model.transformer, config.attention_mechanism, mask=False)

    def __predict_self_flow(
            self,
            model: Flux2Model,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            deterministic: bool,
    ) -> dict:
        if model.self_flow_ema is None or model.self_flow_projector is None:
            raise RuntimeError("Self-Flow is enabled but its EMA/projector state was not initialized.")
        if model.self_flow_student_layer is None or model.self_flow_teacher_layer is None:
            raise RuntimeError("Self-Flow layer selection was not initialized.")

        dpo_reference_forward = self._dpo_reference_prediction()
        dpo_policy_forward = self._dpo_conditioning_locked() and not dpo_reference_forward
        training_pass = torch.is_grad_enabled() and not dpo_reference_forward
        if (
            training_pass
            and
            self.train_device.type == "cuda"
            and torch.cuda.is_available()
            and not self._self_flow_metric_sums
            and not self._self_flow_cuda_timers
        ):
            torch.cuda.reset_peak_memory_stats(self.train_device)

        with model.autocast_context:
            batch_seed = 0 if deterministic else train_progress.global_step * multi.world_size() + multi.rank()
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(batch_seed)
            rand = Random(batch_seed)

            text_encoder_output = model.encode_text(
                train_device=self.train_device,
                batch_size=batch['latent_image'].shape[0],
                rand=rand,
                tokens=batch.get("tokens"),
                tokens_mask=batch.get("tokens_mask"),
                text_encoder_sequence_length=config.text_encoder_sequence_length,
                text_encoder_output=batch.get('text_encoder_hidden_state'),
                text_encoder_dropout_probability=(
                    0.0 if self._dpo_conditioning_locked()
                    else config.text_encoder.dropout_probability
                ) if not deterministic else None,
            )
            if config.cep_gamma > 0 and not deterministic and not self._dpo_conditioning_locked():
                text_encoder_output = self._apply_conditional_embedding_perturbation(
                    text_encoder_output, config.cep_gamma, generator
                )

            latent_image = model.patchify_latents(batch['latent_image'].float())
            latent_height = latent_image.shape[-2]
            latent_width = latent_image.shape[-1]
            scaled_latent_image = model.scale_latents(latent_image)
            latent_noise = self._create_noise(scaled_latent_image, config, generator)

            batch_size = scaled_latent_image.shape[0]
            num_train_timesteps = model.noise_scheduler.config['num_train_timesteps']
            shift = model.calculate_timestep_shift(latent_height, latent_width)
            timestep_shift = shift if config.dynamic_timestep_shifting else config.timestep_shift

            timestep = self._get_timestep_discrete(
                num_train_timesteps,
                deterministic,
                generator,
                batch_size,
                config,
                shift=timestep_shift,
            )
            second_timestep = self._get_timestep_discrete(
                num_train_timesteps,
                deterministic,
                generator,
                batch_size,
                config,
                shift=timestep_shift,
            )
            if timestep.shape[0] == 1 and batch_size > 1:
                timestep = timestep.expand(batch_size)
            if second_timestep.shape[0] == 1 and batch_size > 1:
                second_timestep = second_timestep.expand(batch_size)

            clean_tokens = model.pack_latents(scaled_latent_image)
            noise_tokens = model.pack_latents(latent_noise)
            image_seq_len = clean_tokens.shape[1]
            token_mask = torch.rand(
                (batch_size, image_seq_len),
                generator=generator,
                device=clean_tokens.device,
            ) < config.self_flow_mask_ratio
            # Batched DPO is laid out [chosen; rejected]. Mirror the mask just
            # like OneTrainer already mirrors noise and sampled timesteps so a
            # preference pair and its reference use identical corruption.
            token_mask = self._apply_dpo_paired_rng(token_mask)
            token_timestep = torch.where(
                token_mask,
                second_timestep[:, None],
                timestep[:, None],
            )

            # OneTrainer uses sigma=(t+1)/T, so the lower discrete timestep is
            # the cleaner view under the actual interpolation convention.
            clean_timestep = torch.minimum(timestep, second_timestep)
            student_tokens = flux2_interpolate_token_view(
                clean_tokens,
                noise_tokens,
                token_timestep,
                num_train_timesteps,
            )
            teacher_tokens = flux2_interpolate_token_view(
                clean_tokens,
                noise_tokens,
                clean_timestep,
                num_train_timesteps,
            )

            guidance = None
            if model.transformer.config.guidance_embeds:
                guidance = torch.tensor(
                    [config.transformer.guidance_scale],
                    device=self.train_device,
                    dtype=model.train_dtype.torch_dtype(),
                ).expand(batch_size)

            text_ids = model.prepare_text_ids(text_encoder_output)
            image_ids = model.prepare_latent_image_ids(scaled_latent_image)

            teacher_feature = None
            if not dpo_reference_forward:
                # Memory-critical order: inference-only EMA teacher first,
                # restore the policy/student adapter, then build the student
                # autograd graph. A DPO reference forward deliberately skips
                # this swap: the currently active adapter is already the fixed
                # DPO reference and must not be overwritten by the policy EMA.
                with model.self_flow_ema.teacher_parameters(model.self_flow_adapter_modules()):
                    teacher_timer = self._start_self_flow_timer() if training_pass else None
                    with torch.inference_mode():
                        teacher_output = flux2_self_flow_forward(
                            transformer=model.transformer,
                            hidden_states=teacher_tokens.to(dtype=model.train_dtype.torch_dtype()),
                            encoder_hidden_states=text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                            image_timestep=clean_timestep / 1000,
                            text_timestep=clean_timestep / 1000,
                            guidance=guidance,
                            txt_ids=text_ids,
                            img_ids=image_ids,
                            joint_attention_kwargs=None,
                            stop_at_layer=model.self_flow_teacher_layer,
                        )
                    self._finish_self_flow_timer("performance/teacher_forward_ms", teacher_timer)
                    if teacher_output.feature is None:
                        raise RuntimeError("Self-Flow EMA teacher did not return its target feature.")
                    # Tensors created by inference_mode retain inference-tensor
                    # semantics after leaving the context. Materialize a normal,
                    # detached tensor before the student loss may save it.
                    teacher_feature = teacher_output.feature.detach().clone()
                    del teacher_output

                if config.self_flow_teacher_target_offload:
                    teacher_feature = teacher_feature.to(device="cpu")

                if training_pass:
                    if self._self_flow_student_timer is not None:
                        raise RuntimeError("A previous Self-Flow student timer was not completed by backward.")
                    self._self_flow_student_timer = self._start_self_flow_timer()

            student_output = flux2_self_flow_forward(
                transformer=model.transformer,
                hidden_states=student_tokens.to(dtype=model.train_dtype.torch_dtype()),
                encoder_hidden_states=text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                image_timestep=token_timestep / 1000,
                # FLUX.2 adaptation: the double-stream text tokens retain the
                # majority/base timestep while image tokens receive tau_i.
                text_timestep=timestep / 1000,
                guidance=guidance,
                txt_ids=text_ids,
                img_ids=image_ids,
                joint_attention_kwargs=None,
                # DPO reference logp needs the dual-timestep prediction only;
                # representation alignment is a policy-side auxiliary.
                capture_layer=None if dpo_reference_forward else model.self_flow_student_layer,
            )

            representation_loss_per_sample = None
            cosine_similarity_per_sample = None
            structural_loss_per_sample = None
            if not dpo_reference_forward:
                student_feature = student_output.feature
                if student_feature is None or teacher_feature is None:
                    raise RuntimeError("Self-Flow policy forward is missing student/teacher features.")
                teacher_feature = teacher_feature.to(
                    device=student_feature.device,
                    dtype=student_feature.dtype,
                    non_blocking=False,
                )
                if teacher_feature.shape != student_feature.shape:
                    raise RuntimeError(
                        "Self-Flow student/teacher feature shape mismatch: "
                        f"student={tuple(student_feature.shape)}, teacher={tuple(teacher_feature.shape)}"
                    )

                projected_student = model.self_flow_projector(student_feature)
                cosine_similarity_per_sample = F.cosine_similarity(
                    projected_student.to(dtype=torch.float32),
                    teacher_feature.to(dtype=torch.float32),
                    dim=-1,
                ).mean(dim=1)
                representation_loss_per_sample = 1.0 - cosine_similarity_per_sample
                if config.self_flow_structural_enabled:
                    structural_timer = self._start_self_flow_timer() if training_pass else None
                    structural_loss_per_sample = flux2_structural_alignment_loss(
                        projected_student_feature=projected_student,
                        teacher_feature=teacher_feature,
                        sample_count=config.self_flow_structural_tokens,
                        generator=generator,
                    )
                    self._finish_self_flow_timer("performance/structural_loss_ms", structural_timer)

            packed_predicted_flow = student_output.sample
            if packed_predicted_flow is None:
                raise RuntimeError("Self-Flow prediction forward did not return a flow sample.")
            predicted_flow = model.unpack_latents(
                packed_predicted_flow,
                latent_height,
                latent_width,
            )
            flow = latent_noise - scaled_latent_image
            model_output_data = {
                'loss_type': 'target',
                'timestep': token_timestep,
                'predicted': model.unpatchify_latents(predicted_flow),
                'target': model.unpatchify_latents(flow),
                'self_flow_training_pass': training_pass,
                'self_flow_dpo_policy': dpo_policy_forward,
            }
            if representation_loss_per_sample is not None:
                model_output_data['self_flow_representation_loss_per_sample'] = representation_loss_per_sample
                model_output_data['self_flow_cosine_similarity_per_sample'] = cosine_similarity_per_sample.detach()
            if structural_loss_per_sample is not None:
                model_output_data['self_flow_structural_loss_per_sample'] = structural_loss_per_sample

            if config.loss_weight_fn == LossWeight.SIGMA:
                token_loss_weight = flux2_flow_sigma(token_timestep, num_train_timesteps)
                model_output_data['element_loss_weight'] = flux2_token_weight_to_spatial(
                    token_loss_weight,
                    latent_height,
                    latent_width,
                )

            if training_pass:
                self._record_self_flow_metric("self_flow/t_mean", timestep.float().mean().item() / 1000.0)
                self._record_self_flow_metric("self_flow/s_mean", second_timestep.float().mean().item() / 1000.0)
                self._record_self_flow_metric("self_flow/clean_t_mean", clean_timestep.float().mean().item() / 1000.0)
                self._record_self_flow_metric("self_flow/mask_ratio_actual", token_mask.float().mean().item())

            if config.debug_mode:
                with torch.no_grad():
                    student_noisy_latent = model.unpack_latents(student_tokens, latent_height, latent_width)
                    sigma_grid = flux2_flow_sigma(token_timestep, num_train_timesteps).reshape(
                        batch_size, 1, latent_height, latent_width
                    )
                    predicted_scaled_latent_image = student_noisy_latent - predicted_flow * sigma_grid
                    self._save_tokens("7-prompt", batch['tokens'], model.tokenizer, config, train_progress)
                    self._save_latent("1-noise", latent_noise, config, train_progress)
                    self._save_latent("2-noisy_image", student_noisy_latent, config, train_progress)
                    self._save_latent("3-predicted_flow", predicted_flow, config, train_progress)
                    self._save_latent("4-flow", flow, config, train_progress)
                    self._save_latent("5-predicted_image", predicted_scaled_latent_image, config, train_progress)
                    self._save_latent("6-image", scaled_latent_image, config, train_progress)

        return model_output_data

    def predict(
            self,
            model: Flux2Model,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        # Full Self-Flow DPO experiment: when Self-Flow is enabled, normal,
        # chosen/rejected policy, and DPO reference forwards all use the same
        # dual-timestep Self-Flow prediction pipeline. __predict_self_flow()
        # distinguishes policy vs. reference internally: the fixed DPO reference
        # receives the identical dual-timestep corruption but skips the policy
        # EMA teacher/projector auxiliaries, while the trainable policy receives
        # representation/structural auxiliaries.
        if config.self_flow_enabled:
            return self.__predict_self_flow(model, batch, config, train_progress, deterministic)

        with model.autocast_context:
            batch_seed = 0 if deterministic else train_progress.global_step * multi.world_size() + multi.rank()
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(batch_seed)
            rand = Random(batch_seed)

            text_encoder_output = model.encode_text(
                train_device=self.train_device,
                batch_size=batch['latent_image'].shape[0],
                rand=rand,
                tokens=batch.get("tokens"),
                tokens_mask=batch.get("tokens_mask"),
                text_encoder_sequence_length=config.text_encoder_sequence_length,
                text_encoder_output=batch.get('text_encoder_hidden_state'),
                text_encoder_dropout_probability=(0.0 if self._dpo_conditioning_locked() else config.text_encoder.dropout_probability) if not deterministic else None,
            )
            if config.cep_gamma > 0 and not deterministic and not self._dpo_conditioning_locked():
                text_encoder_output = self._apply_conditional_embedding_perturbation(
                    text_encoder_output, config.cep_gamma, generator
                )

            latent_image = model.patchify_latents(batch['latent_image'].float())
            latent_height = latent_image.shape[-2]
            latent_width = latent_image.shape[-1]
            scaled_latent_image = model.scale_latents(latent_image)

            latent_noise = self._create_noise(scaled_latent_image, config, generator)

            shift = model.calculate_timestep_shift(latent_height, latent_width)
            timestep = self._get_timestep_discrete(
                model.noise_scheduler.config['num_train_timesteps'],
                deterministic,
                generator,
                scaled_latent_image.shape[0],
                config,
                shift = shift if config.dynamic_timestep_shifting else config.timestep_shift,
            )

            scaled_noisy_latent_image, sigma = self._add_noise_discrete(
                scaled_latent_image,
                latent_noise,
                timestep,
                model.noise_scheduler.timesteps,
            )
            latent_input = scaled_noisy_latent_image

            if model.transformer.config.guidance_embeds:
                guidance = torch.tensor([config.transformer.guidance_scale], device=self.train_device, dtype=model.train_dtype.torch_dtype())
                guidance = guidance.expand(latent_input.shape[0])
            else:
                guidance = None

            text_ids = model.prepare_text_ids(text_encoder_output)
            image_ids = model.prepare_latent_image_ids(latent_input)
            packed_latent_input = model.pack_latents(latent_input)

            packed_predicted_flow = model.transformer(
                hidden_states=packed_latent_input.to(dtype=model.train_dtype.torch_dtype()),
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states=text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                txt_ids=text_ids,
                img_ids=image_ids,
                joint_attention_kwargs=None,
                return_dict=True
            ).sample

            predicted_flow = model.unpack_latents(
                packed_predicted_flow,
                latent_input.shape[2],
                latent_input.shape[3],
            )

            flow = latent_noise - scaled_latent_image
            model_output_data = {
                'loss_type': 'target',
                'timestep': timestep,
                #unpatchify, to make the shape match the mask shape of masked training:
                'predicted': model.unpatchify_latents(predicted_flow),
                'target': model.unpatchify_latents(flow),
            }
            if config.debug_mode:
                with torch.no_grad():
                    predicted_scaled_latent_image = scaled_noisy_latent_image - predicted_flow * sigma
                    self._save_tokens("7-prompt", batch['tokens'], model.tokenizer, config, train_progress)
                    self._save_latent("1-noise", latent_noise, config, train_progress)
                    self._save_latent("2-noisy_image", scaled_noisy_latent_image, config, train_progress)
                    self._save_latent("3-predicted_flow", predicted_flow, config, train_progress)
                    self._save_latent("4-flow", flow, config, train_progress)
                    self._save_latent("5-predicted_image", predicted_scaled_latent_image, config, train_progress)
                    self._save_latent("6-image", scaled_latent_image, config, train_progress)

        return model_output_data

    def rlhf_logp_per_sample(
            self,
            model: Flux2Model,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        # Flux2 / Flux Klein native DPO likelihood proxy.
        #
        # Normal Flux2 training uses _flow_matching_losses(...).mean().
        # DPO needs the same native loss path before the batch mean, so the
        # preference objective uses one negative per-sample loss as its logp
        # proxy instead of BaseModelSetup's raw-MSE fallback.
        #
        # This keeps DPO in Flux2's own units: configured MSE/MAE/log-cosh/
        # Huber stack, mask handling, loss_scaler, per-sample loss_weight, and
        # sigma/timestep weighting all match normal Flux2 training.
        return -self._flow_matching_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
            sigmas=model.noise_scheduler.sigmas,
        )

    @staticmethod
    def rlhf_chosen_supervised_weight(
            config: TrainConfig,
            objective,
    ) -> float:
        # Restore the chosen-only positive Self-Flow supervision in addition
        # to the full chosen+rejected Self-Flow DPO path.
        base_weight = BaseModelSetup.rlhf_chosen_supervised_weight(
            config,
            objective,
        )
        if bool(config.self_flow_enabled):
            return 0.25 * base_weight
        return base_weight

    def rlhf_chosen_supervised_requires_separate_forward(
            self,
            config: TrainConfig,
    ) -> bool:
        # Run the additional chosen-positive term through the ordinary
        # Self-Flow training path before constructing the DPO graph.
        return bool(config.self_flow_enabled)

    def rlhf_mixed_normal_dpo_requires_sequential_backward(
            self,
            config: TrainConfig,
    ) -> bool:
        # Normal samples may still use Self-Flow. If a loader batch contains
        # both normal and DPO items, finish the normal Self-Flow graph before
        # constructing DPO. This prevents EMA teacher parameter swaps and the
        # two large activation graphs from overlapping, while both gradients
        # still accumulate before the same optimizer step.
        return bool(config.self_flow_enabled)

    def rlhf_policy_auxiliary_loss(
            self,
            model: Flux2Model,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor | None:
        if not config.self_flow_enabled or not data.get('self_flow_dpo_policy', False):
            return None

        representation_loss_per_sample = data.get('self_flow_representation_loss_per_sample')
        if representation_loss_per_sample is None:
            raise RuntimeError("Self-Flow DPO policy forward did not return representation loss.")
        representation_loss = representation_loss_per_sample.mean()
        auxiliary_loss = config.self_flow_rep_weight * representation_loss

        structural_loss = None
        if config.self_flow_structural_enabled:
            structural_loss_per_sample = data.get('self_flow_structural_loss_per_sample')
            if structural_loss_per_sample is None:
                raise RuntimeError("Structural Self-Flow DPO policy forward did not return structural loss.")
            structural_loss = structural_loss_per_sample.mean()
            auxiliary_loss = auxiliary_loss + config.self_flow_structural_weight * structural_loss

        if data.get('self_flow_training_pass', False):
            self._record_self_flow_metric("loss/self_flow_rep", representation_loss.detach().item())
            if structural_loss is not None:
                self._record_self_flow_metric("loss/self_flow_structural", structural_loss.detach().item())
            self._record_self_flow_metric(
                "self_flow/cosine_similarity",
                data['self_flow_cosine_similarity_per_sample'].detach().mean().item(),
            )
        return auxiliary_loss

    def calculate_loss(
            self,
            model: Flux2Model,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        generation_loss = self._flow_matching_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
            sigmas=model.noise_scheduler.sigmas,
        ).mean()

        if (
            not config.self_flow_enabled
            or data.get('self_flow_bypassed_for_dpo', False)
        ):
            return generation_loss

        # DPO policy representation/structural terms are added exactly once by
        # rlhf_policy_auxiliary_loss(). If a caller asks calculate_loss() for a
        # DPO-policy output, return only generation loss to avoid double-counting
        # those auxiliaries.
        if data.get('self_flow_dpo_policy', False):
            if data.get('self_flow_training_pass', False):
                self._record_self_flow_metric("loss/generation", generation_loss.detach().item())
            return generation_loss

        representation_loss_per_sample = data.get('self_flow_representation_loss_per_sample')
        if representation_loss_per_sample is None:
            raise RuntimeError("Self-Flow prediction did not return a representation loss.")
        representation_loss = representation_loss_per_sample.mean()
        total_loss = generation_loss + config.self_flow_rep_weight * representation_loss

        structural_loss = None
        if config.self_flow_structural_enabled:
            structural_loss_per_sample = data.get('self_flow_structural_loss_per_sample')
            if structural_loss_per_sample is None:
                raise RuntimeError("Structural Self-Flow prediction did not return structural loss.")
            structural_loss = structural_loss_per_sample.mean()
            total_loss = total_loss + config.self_flow_structural_weight * structural_loss

        if data.get('self_flow_training_pass', False):
            self._record_self_flow_metric("loss/generation", generation_loss.detach().item())
            self._record_self_flow_metric("loss/self_flow_rep", representation_loss.detach().item())
            if structural_loss is not None:
                self._record_self_flow_metric("loss/self_flow_structural", structural_loss.detach().item())
            self._record_self_flow_metric("loss/total", total_loss.detach().item())
            self._record_self_flow_metric(
                "self_flow/cosine_similarity",
                data['self_flow_cosine_similarity_per_sample'].detach().mean().item(),
            )
        return total_loss

    def after_backward(
            self,
            model: Flux2Model,
            config: TrainConfig,
            train_progress: TrainProgress,
    ):
        if config.self_flow_enabled and self._self_flow_student_timer is not None:
            self._finish_self_flow_timer(
                "performance/student_forward_backward_ms",
                self._self_flow_student_timer,
            )
            self._self_flow_student_timer = None

    def after_streamed_dpo_branch_backward(
            self,
            model: Flux2Model,
            config: TrainConfig,
            train_progress: TrainProgress,
    ):
        # Each streamed branch owns a complete replay/backward before the next
        # branch starts, so close its timer here rather than at the outer DPO
        # loss.backward() boundary.
        self.after_backward(model, config, train_progress)

    def report_to_tensorboard(
            self,
            model: Flux2Model,
            config: TrainConfig,
            scheduler,
            tensorboard,
    ):
        super().report_to_tensorboard(model, config, scheduler, tensorboard)
        if not config.self_flow_enabled:
            return

        self._resolve_self_flow_cuda_timers()
        for name, total in self._self_flow_metric_sums.items():
            count = self._self_flow_metric_counts.get(name, 0)
            if count > 0:
                tensorboard.add_scalar(name, total / count, model.train_progress.global_step)

        if self.train_device.type == "cuda" and torch.cuda.is_available():
            tensorboard.add_scalar(
                "performance/peak_vram_allocated_mb",
                torch.cuda.max_memory_allocated(self.train_device) / (1024 ** 2),
                model.train_progress.global_step,
            )
            tensorboard.add_scalar(
                "performance/peak_vram_reserved_mb",
                torch.cuda.max_memory_reserved(self.train_device) / (1024 ** 2),
                model.train_progress.global_step,
            )

        self._self_flow_metric_sums.clear()
        self._self_flow_metric_counts.clear()

    def prepare_text_caching(self, model: FluxModel, config: TrainConfig):
        model.to(self.temp_device)
        model.text_encoder_to(self.train_device)
        model.eval()
        torch_gc()
