from __future__ import annotations

from abc import ABCMeta
from random import Random
import time

import modules.util.multi_gpu_util as multi
from modules.model.MageFlowModel import MageFlowModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.modelSetup.mixin.ModelSetupText2ImageMixin import ModelSetupText2ImageMixin
from modules.module.MageFlowAttention import (
    build_packed_attention_routing,
    install_optimized_mage_attention,
)
from modules.module.MageFlowSelfFlow import mage_flow_forward, structural_alignment_loss
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.LossWeight import LossWeight
from modules.util.quantization_util import quantize_layers
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
import torch.nn.functional as F
from torch import Tensor


class BaseMageFlowSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    ModelSetupText2ImageMixin,
    metaclass=ABCMeta,
):
    LAYER_PRESETS = {
        "attn-mlp": ["attn", "img_mlp", "txt_mlp"],
        "attn-only": ["attn"],
        "blocks": ["transformer_blocks"],
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

    def setup_optimizations(self, model: MageFlowModel, config: TrainConfig):
        model.transformer.checkpoint = bool(config.transformer.gradient_checkpointing)
        if model.text_encoder is not None and config.text_encoder.gradient_checkpointing:
            enable = getattr(model.text_encoder, "gradient_checkpointing_enable", None)
            if enable is not None:
                enable()

        model.autocast_context, model.train_dtype = create_autocast_context(
            self.train_device, config.train_dtype, config.enable_autocast_cache
        )
        model.text_encoder_autocast_context, model.text_encoder_train_dtype = disable_fp16_autocast_context(
            self.train_device,
            config.train_dtype,
            config.fallback_train_dtype,
            config.enable_autocast_cache,
        )

        quantize_layers(model.text_encoder, self.train_device, model.text_encoder_train_dtype, config)
        quantize_layers(model.vae, self.train_device, model.train_dtype, config)
        quantize_layers(model.transformer, self.train_device, model.train_dtype, config)

        # Upstream Mage rebuilds packed routing and synchronizes max sequence
        # length in every block. Install the OneTrainer processor before the
        # first forward so routing is built once per packed model invocation.
        install_optimized_mage_attention(model.transformer)

        # Mage previously ignored OneTrainer's Compile toggle entirely. Compile
        # the native block forward in-place and also compile the dedicated
        # dual-timestep Self-Flow block function. Do not require fullgraph: the
        # selected FlashAttention/CuTe backend is an external custom kernel and
        # is allowed to remain a graph boundary.
        import modules.module.MageFlowSelfFlow as mage_self_flow
        if not hasattr(mage_self_flow, "_ot_uncompiled_split_block_forward"):
            mage_self_flow._ot_uncompiled_split_block_forward = mage_self_flow._split_block_forward
        if config.compile:
            for block in model.transformer.transformer_blocks:
                block.compile(dynamic=True)
            mage_self_flow._split_block_forward = torch.compile(
                mage_self_flow._ot_uncompiled_split_block_forward,
                dynamic=True,
            )
            print(
                f"[Mage-Flow] torch.compile enabled for "
                f"{len(model.transformer.transformer_blocks)} transformer blocks + Self-Flow block path"
            )
        else:
            mage_self_flow._split_block_forward = mage_self_flow._ot_uncompiled_split_block_forward

    @staticmethod
    def _flat_sigma(sigma: Tensor, batch_size: int) -> Tensor:
        return sigma.reshape(batch_size, -1)[:, 0]

    @staticmethod
    def _image_shapes(batch_size: int, height: int, width: int):
        return [[(1, int(height), int(width)) for _ in range(batch_size)]]

    def _encode_conditioning(
            self,
            model: MageFlowModel,
            batch: dict,
            config: TrainConfig,
            rand: Random,
            generator: torch.Generator,
            deterministic: bool,
    ) -> tuple[Tensor, Tensor]:
        cached = batch.get("text_encoder_hidden_state")
        cached_mask = batch.get("text_encoder_attention_mask")
        output, mask = model.encode_text(
            train_device=self.train_device,
            batch_size=batch['latent_image'].shape[0],
            rand=rand,
            tokens=batch.get("tokens"),
            tokens_mask=cached_mask if cached is not None else batch.get("tokens_mask"),
            text_encoder_output=cached,
            text_encoder_dropout_probability=(
                0.0 if self._dpo_conditioning_locked() else config.text_encoder.dropout_probability
            ) if not deterministic else None,
        )
        if config.cep_gamma > 0 and not deterministic and not self._dpo_conditioning_locked():
            output = self._apply_conditional_embedding_perturbation(output, config.cep_gamma, generator)
        return output, mask

    def _packed_inputs(self, model: MageFlowModel, latent_tokens: Tensor, text: Tensor, text_mask: Tensor):
        packed_img, img_cu = model.prepare_packed_images(latent_tokens)
        packed_txt, txt_cu = model.prepare_packed_text(text, text_mask)
        routing = build_packed_attention_routing(
            img_cu,
            txt_cu,
            img_token_count=int(packed_img.shape[1]),
            txt_token_count=int(packed_txt.shape[1]),
            # Every sample has the same image-token length. The padded text
            # width is a host-known upper bound for the longest valid prompt.
            max_joint_seqlen=int(latent_tokens.shape[1] + text.shape[1]),
        )
        attention_kwargs = {"ot_packed_routing": routing}
        return packed_img, packed_txt, img_cu, txt_cu, latent_tokens.shape[0], attention_kwargs

    def _predict_normal(
            self,
            model: MageFlowModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            deterministic: bool,
    ) -> dict:
        with model.autocast_context:
            batch_seed = 0 if deterministic else train_progress.global_step * multi.world_size() + multi.rank()
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(batch_seed)
            rand = Random(batch_seed)

            text, text_mask = self._encode_conditioning(model, batch, config, rand, generator, deterministic)
            latent = batch['latent_image'].float()
            noise = self._create_noise(latent, config, generator)
            batch_size, _, height, width = latent.shape
            timestep = self._get_timestep_discrete(
                model.noise_scheduler.config['num_train_timesteps'],
                deterministic,
                generator,
                batch_size,
                config,
                shift=config.timestep_shift,
            )
            noisy, sigma = self._add_noise_discrete(latent, noise, timestep, model.noise_scheduler.timesteps)
            sigma_flat = self._flat_sigma(sigma, batch_size)

            tokens = model.pack_latents(noisy)
            packed_img, packed_txt, img_cu, txt_cu, _, attention_kwargs = self._packed_inputs(
                model, tokens, text, text_mask
            )
            packed_pred = model.transformer(
                img=packed_img.to(dtype=model.train_dtype.torch_dtype()),
                txt=packed_txt.to(dtype=model.train_dtype.torch_dtype()),
                timesteps=sigma_flat.to(dtype=model.train_dtype.torch_dtype()),
                img_shapes=self._image_shapes(batch_size, height, width),
                img_cu_seqlens=img_cu,
                txt_cu_seqlens=txt_cu,
                attention_kwargs=attention_kwargs,
            )
            pred_tokens = model.unprepare_packed_images(packed_pred, batch_size)
            predicted = model.unpack_latents(pred_tokens, height, width)
            data = {
                'loss_type': 'target',
                'timestep': timestep,
                'predicted': predicted,
                'target': noise - latent,
            }
            if config.loss_weight_fn == LossWeight.SIGMA:
                data['element_loss_weight'] = sigma
            return data

    def _predict_self_flow(
            self,
            model: MageFlowModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            deterministic: bool,
    ) -> dict:
        if model.self_flow_ema is None or model.self_flow_projector is None:
            raise RuntimeError("Mage Self-Flow EMA/projector was not initialized")
        if model.self_flow_student_layer is None or model.self_flow_teacher_layer is None:
            raise RuntimeError("Mage Self-Flow layers were not initialized")

        dpo_reference = self._dpo_reference_prediction()
        dpo_policy = self._dpo_conditioning_locked() and not dpo_reference
        training_pass = torch.is_grad_enabled() and not dpo_reference
        if (
            training_pass
            and self.train_device.type == "cuda"
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
            text, text_mask = self._encode_conditioning(model, batch, config, rand, generator, deterministic)

            clean = batch['latent_image'].float()
            noise = self._create_noise(clean, config, generator)
            batch_size, _, height, width = clean.shape
            num_steps = model.noise_scheduler.config['num_train_timesteps']

            timestep = self._get_timestep_discrete(
                num_steps, deterministic, generator, batch_size, config, shift=config.timestep_shift
            )
            second = self._get_timestep_discrete(
                num_steps, deterministic, generator, batch_size, config, shift=config.timestep_shift
            )
            if timestep.shape[0] == 1 and batch_size > 1:
                timestep = timestep.expand(batch_size)
            if second.shape[0] == 1 and batch_size > 1:
                second = second.expand(batch_size)

            noisy_t, sigma_t = self._add_noise_discrete(clean, noise, timestep, model.noise_scheduler.timesteps)
            noisy_s, sigma_s = self._add_noise_discrete(clean, noise, second, model.noise_scheduler.timesteps)
            sigma_t = self._flat_sigma(sigma_t, batch_size)
            sigma_s = self._flat_sigma(sigma_s, batch_size)

            t_tokens = model.pack_latents(noisy_t)
            s_tokens = model.pack_latents(noisy_s)
            image_tokens = t_tokens.shape[1]
            token_mask = torch.rand(
                (batch_size, image_tokens), generator=generator, device=clean.device
            ) < config.self_flow_mask_ratio
            token_mask = self._apply_dpo_paired_rng(token_mask)
            token_timestep = torch.where(token_mask, second[:, None], timestep[:, None])
            token_sigma = torch.where(token_mask, sigma_s[:, None], sigma_t[:, None])
            student_tokens = torch.where(token_mask[..., None], s_tokens, t_tokens)

            choose_t = timestep <= second
            clean_sigma = torch.where(choose_t, sigma_t, sigma_s)
            teacher_tokens = torch.where(choose_t[:, None, None], t_tokens, s_tokens)

            packed_student, packed_txt, img_cu, txt_cu, _, attention_kwargs = self._packed_inputs(
                model, student_tokens, text, text_mask
            )
            # Teacher image tokens have identical packed shape/cu_seqlens. Do
            # not repack text or rebuild routing for the second forward.
            packed_teacher, _ = model.prepare_packed_images(teacher_tokens)
            shapes = self._image_shapes(batch_size, height, width)

            teacher_feature = None
            if not dpo_reference:
                with model.self_flow_ema.teacher_parameters(model.self_flow_adapter_modules()):
                    teacher_timer = self._start_self_flow_timer() if training_pass else None
                    with torch.inference_mode():
                        teacher_out = mage_flow_forward(
                            transformer=model.transformer,
                            img=packed_teacher.to(dtype=model.train_dtype.torch_dtype()),
                            txt=packed_txt.to(dtype=model.train_dtype.torch_dtype()),
                            image_timesteps=clean_sigma.to(dtype=model.train_dtype.torch_dtype()),
                            text_timesteps=clean_sigma.to(dtype=model.train_dtype.torch_dtype()),
                            img_shapes=shapes,
                            img_cu_seqlens=img_cu,
                            txt_cu_seqlens=txt_cu,
                            stop_layer=model.self_flow_teacher_layer,
                            attention_kwargs=attention_kwargs,
                        )
                    self._finish_self_flow_timer("performance/teacher_forward_ms", teacher_timer)
                    if teacher_out.feature is None:
                        raise RuntimeError("Mage Self-Flow teacher feature is missing")
                    teacher_feature = teacher_out.feature.detach().clone()
                if config.self_flow_teacher_target_offload:
                    teacher_feature = teacher_feature.cpu()

                if training_pass:
                    if self._self_flow_student_timer is not None:
                        raise RuntimeError("A previous Mage Self-Flow student timer was not completed by backward")
                    self._self_flow_student_timer = self._start_self_flow_timer()

            student_out = mage_flow_forward(
                transformer=model.transformer,
                img=packed_student.to(dtype=model.train_dtype.torch_dtype()),
                txt=packed_txt.to(dtype=model.train_dtype.torch_dtype()),
                image_timesteps=token_sigma.reshape(1, -1).to(dtype=model.train_dtype.torch_dtype()),
                text_timesteps=sigma_t.to(dtype=model.train_dtype.torch_dtype()),
                img_shapes=shapes,
                img_cu_seqlens=img_cu,
                txt_cu_seqlens=txt_cu,
                capture_layer=None if dpo_reference else model.self_flow_student_layer,
                attention_kwargs=attention_kwargs,
            )

            representation_loss = None
            cosine = None
            structural = None
            if not dpo_reference:
                if student_out.feature is None:
                    raise RuntimeError("Mage Self-Flow student feature is missing")
                student_feature = model.unprepare_packed_images(student_out.feature, batch_size)
                teacher_feature = model.unprepare_packed_images(
                    teacher_feature.to(student_feature.device, student_feature.dtype), batch_size
                )
                projected = model.self_flow_projector(student_feature)
                cosine = F.cosine_similarity(projected.float(), teacher_feature.float(), dim=-1).mean(dim=1)
                representation_loss = 1.0 - cosine
                if config.self_flow_structural_enabled:
                    structural_timer = self._start_self_flow_timer() if training_pass else None
                    structural = structural_alignment_loss(
                        projected, teacher_feature, sample_count=config.self_flow_structural_tokens
                    )
                    self._finish_self_flow_timer("performance/structural_loss_ms", structural_timer)

            pred_tokens = model.unprepare_packed_images(student_out.predicted, batch_size)
            predicted = model.unpack_latents(pred_tokens, height, width)
            data = {
                'loss_type': 'target',
                'timestep': token_timestep,
                'predicted': predicted,
                'target': noise - clean,
                'self_flow_training_pass': training_pass,
                'self_flow_dpo_policy': dpo_policy,
            }
            if representation_loss is not None:
                data['self_flow_representation_loss_per_sample'] = representation_loss
                data['self_flow_cosine_similarity_per_sample'] = cosine.detach()
            if structural is not None:
                data['self_flow_structural_loss_per_sample'] = structural
            if config.loss_weight_fn == LossWeight.SIGMA:
                data['element_loss_weight'] = token_sigma.reshape(batch_size, 1, height, width)

            if training_pass:
                self._record_self_flow_metric("self_flow/t_mean", sigma_t.detach().float().mean().item())
                self._record_self_flow_metric("self_flow/s_mean", sigma_s.detach().float().mean().item())
                self._record_self_flow_metric("self_flow/clean_t_mean", clean_sigma.detach().float().mean().item())
                self._record_self_flow_metric("self_flow/mask_ratio_actual", token_mask.detach().float().mean().item())
            return data

    def predict(
            self,
            model: MageFlowModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        if config.self_flow_enabled:
            return self._predict_self_flow(model, batch, config, train_progress, deterministic)
        return self._predict_normal(model, batch, config, train_progress, deterministic)

    def rlhf_logp_per_sample(
            self,
            model: MageFlowModel,
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

    @staticmethod
    def rlhf_chosen_supervised_weight(config: TrainConfig, objective) -> float:
        weight = BaseModelSetup.rlhf_chosen_supervised_weight(config, objective)
        return 0.25 * weight if bool(config.self_flow_enabled) else weight

    def rlhf_chosen_supervised_requires_separate_forward(self, config: TrainConfig) -> bool:
        return bool(config.self_flow_enabled)

    def rlhf_mixed_normal_dpo_requires_sequential_backward(self, config: TrainConfig) -> bool:
        return bool(config.self_flow_enabled)

    def rlhf_policy_auxiliary_loss(
            self,
            model: MageFlowModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor | None:
        if not config.self_flow_enabled or not data.get('self_flow_dpo_policy', False):
            return None
        rep_per_sample = data.get('self_flow_representation_loss_per_sample')
        if rep_per_sample is None:
            raise RuntimeError("Mage Self-Flow DPO policy forward is missing representation loss")
        rep = rep_per_sample.mean()
        result = config.self_flow_rep_weight * rep
        structural = None
        if config.self_flow_structural_enabled:
            structural_per_sample = data.get('self_flow_structural_loss_per_sample')
            if structural_per_sample is None:
                raise RuntimeError("Mage structural Self-Flow DPO forward is missing structural loss")
            structural = structural_per_sample.mean()
            result = result + config.self_flow_structural_weight * structural
        if data.get('self_flow_training_pass', False):
            self._record_self_flow_metric("loss/self_flow_rep", rep.detach().item())
            if structural is not None:
                self._record_self_flow_metric("loss/self_flow_structural", structural.detach().item())
            self._record_self_flow_metric(
                "self_flow/cosine_similarity",
                data['self_flow_cosine_similarity_per_sample'].detach().mean().item(),
            )
        return result

    def calculate_loss(
            self,
            model: MageFlowModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        generation = self._flow_matching_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
            sigmas=model.noise_scheduler.sigmas,
        ).mean()
        if not config.self_flow_enabled or data.get('self_flow_bypassed_for_dpo', False):
            return generation
        if data.get('self_flow_dpo_policy', False):
            if data.get('self_flow_training_pass', False):
                self._record_self_flow_metric("loss/generation", generation.detach().item())
            return generation

        rep_per_sample = data.get('self_flow_representation_loss_per_sample')
        if rep_per_sample is None:
            raise RuntimeError("Mage Self-Flow prediction is missing representation loss")
        rep = rep_per_sample.mean()
        total = generation + config.self_flow_rep_weight * rep
        structural = None
        if config.self_flow_structural_enabled:
            structural_per_sample = data.get('self_flow_structural_loss_per_sample')
            if structural_per_sample is None:
                raise RuntimeError("Mage structural Self-Flow prediction is missing structural loss")
            structural = structural_per_sample.mean()
            total = total + config.self_flow_structural_weight * structural

        if data.get('self_flow_training_pass', False):
            self._record_self_flow_metric("loss/generation", generation.detach().item())
            self._record_self_flow_metric("loss/self_flow_rep", rep.detach().item())
            if structural is not None:
                self._record_self_flow_metric("loss/self_flow_structural", structural.detach().item())
            self._record_self_flow_metric("loss/total", total.detach().item())
            self._record_self_flow_metric(
                "self_flow/cosine_similarity",
                data['self_flow_cosine_similarity_per_sample'].detach().mean().item(),
            )
        return total

    def after_backward(
            self,
            model: MageFlowModel,
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
            model: MageFlowModel,
            config: TrainConfig,
            train_progress: TrainProgress,
    ):
        self.after_backward(model, config, train_progress)

    def report_to_tensorboard(
            self,
            model: MageFlowModel,
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

    def prepare_text_caching(self, model: MageFlowModel, config: TrainConfig):
        model.to(self.temp_device)
        model.text_encoder_to(self.train_device)
        model.eval()
        torch_gc()

    def prepare_image_caching(self, model: MageFlowModel, config: TrainConfig):
        model.to(self.temp_device)
        model.vae_to(self.train_device)
        model.eval()
        torch_gc()