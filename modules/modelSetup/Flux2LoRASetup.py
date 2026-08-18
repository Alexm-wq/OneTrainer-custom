from modules.model.Flux2Model import Flux2Model
import modules.util.multi_gpu_util as multi
from modules.modelSetup.BaseFlux2Setup import BaseFlux2Setup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.module.Flux2SelfFlow import Flux2SelfFlowEMA, Flux2SelfFlowProjector
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress

import torch


@factory.register(BaseModelSetup, ModelType.FLUX_2, TrainingMethod.LORA)
class Flux2LoRASetup(
    BaseFlux2Setup,
):
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

    def create_parameters(
            self,
            model: Flux2Model,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        self._create_model_part_parameters(parameter_group_collection, "transformer", model.transformer_lora, config.transformer)
        if config.self_flow_enabled and model.self_flow_projector is not None:
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name="self_flow_projector",
                display_name="self_flow_projector",
                parameters=model.self_flow_projector.parameters(),
                learning_rate=config.transformer.learning_rate,
            ))
        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: Flux2Model,
            config: TrainConfig,
    ):
        model.text_encoder.requires_grad_(False)
        model.transformer.requires_grad_(False)
        model.vae.requires_grad_(False)

        self._setup_model_part_requires_grad("transformer", model.transformer_lora, config.transformer, model.train_progress)
        if model.self_flow_projector is not None:
            model.self_flow_projector.requires_grad_(config.self_flow_enabled and config.transformer.train)

    @staticmethod
    def __validate_and_resolve_self_flow(model: Flux2Model, config: TrainConfig):
        if config.self_flow_structural_enabled and not config.self_flow_enabled:
            raise ValueError("Structural Self-Flow requires Self-Flow to be enabled.")
        if not config.self_flow_enabled:
            return
        if config.masked_prior_preservation_weight > 0:
            raise NotImplementedError("Self-Flow cannot be combined with masked prior preservation yet.")
        if config.custom_conditioning_image:
            raise NotImplementedError("Self-Flow currently supports standard FLUX.2 samples, not edit/reference-image training.")
        if config.multi_gpu or multi.world_size() > 1:
            raise NotImplementedError("Self-Flow CPU adapter EMA currently supports single-GPU training only.")
        if not config.transformer.train:
            raise ValueError("Self-Flow requires FLUX.2 transformer LoRA training to be enabled.")
        if not model.is_klein():
            raise NotImplementedError("Self-Flow currently targets FLUX.2 Klein Base LoRA training only.")
        if bool(model.transformer.config.guidance_embeds):
            raise NotImplementedError("Self-Flow currently targets a FLUX.2 Klein Base checkpoint, not a distilled checkpoint.")
        if not 0.0 <= config.self_flow_mask_ratio <= 0.5:
            raise ValueError("Self-Flow mask ratio must be between 0 and 0.5.")
        if config.self_flow_rep_weight < 0.0:
            raise ValueError("Self-Flow representation weight must be non-negative.")
        if config.self_flow_structural_weight < 0.0:
            raise ValueError("Structural Self-Flow weight must be non-negative.")
        if config.self_flow_structural_tokens < 2:
            raise ValueError("Structural Self-Flow token count must be at least 2.")
        if config.self_flow_structural_tokens > 2048:
            raise ValueError("Structural Self-Flow token count must not exceed 2048.")
        if not 0.0 <= config.self_flow_ema_decay < 1.0:
            raise ValueError("Self-Flow EMA decay must be in [0, 1).")

        depth = len(model.transformer.single_transformer_blocks)
        if depth < 2:
            raise RuntimeError("Self-Flow requires at least two FLUX.2 single-stream transformer layers.")

        student_layer = int(config.self_flow_student_layer)
        teacher_layer = int(config.self_flow_teacher_layer)
        if student_layer < 0:
            student_layer = max(0, round((depth - 1) / 3))
        if teacher_layer < 0:
            teacher_layer = min(depth - 1, max(student_layer + 1, round((depth - 1) * 5 / 6)))

        if not 0 <= student_layer < depth:
            raise ValueError(f"Self-Flow student layer must be in [0, {depth - 1}], got {student_layer}.")
        if not 0 <= teacher_layer < depth:
            raise ValueError(f"Self-Flow teacher layer must be in [0, {depth - 1}], got {teacher_layer}.")
        if student_layer >= teacher_layer:
            raise ValueError("Self-Flow student layer must be earlier than the teacher layer.")

        model.self_flow_student_layer = student_layer
        model.self_flow_teacher_layer = teacher_layer
        structural_status = "disabled"
        if config.self_flow_structural_enabled:
            structural_status = f"enabled ({config.self_flow_structural_tokens} tokens)"
        print(
            "[Self-Flow] enabled: "
            f"student single-stream layer {student_layer}, teacher layer {teacher_layer}, "
            f"mask ratio {config.self_flow_mask_ratio:.3f}, "
            f"structural alignment {structural_status}"
        )

    def setup_model(
            self,
            model: Flux2Model,
            config: TrainConfig,
    ):
        self.__validate_and_resolve_self_flow(model, config)

        model.transformer_lora = LoRAModuleWrapper(
            model.transformer, "transformer", config, config.layer_filter.split(","),
            fusion_spec=model.fusion_groups(), fuse=config.output_model_format.needs_qkv_fusion(),
        )

        if model.lora_state_dict:
            model.transformer_lora.load_state_dict(model.lora_state_dict)
            model.lora_state_dict = None

        model.transformer_lora.set_dropout(config.dropout_probability)
        model.transformer_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        model.transformer_lora.hook_to_module()

        if config.self_flow_enabled:
            adapter_parameters = model.transformer_lora.parameters()
            projector_dtype = adapter_parameters[0].dtype
            projector_device = adapter_parameters[0].device
            model.self_flow_projector = Flux2SelfFlowProjector(model.transformer.inner_dim).to(
                device=projector_device,
                dtype=projector_dtype,
            )

            saved_self_flow_state = model.self_flow_state_dict
            if saved_self_flow_state is not None and saved_self_flow_state.get("projector") is not None:
                model.self_flow_projector.load_state_dict(saved_self_flow_state["projector"], strict=True)
            elif saved_self_flow_state is not None:
                print("[Self-Flow] backup has no projector state; initializing a new projector.")

            self.__setup_requires_grad(model, config)
            saved_ema_state = saved_self_flow_state.get("ema") if saved_self_flow_state is not None else None
            if saved_ema_state is None:
                print("[Self-Flow] no EMA teacher state found; initializing EMA from the loaded student adapter.")
            model.self_flow_ema = Flux2SelfFlowEMA(
                parameters=adapter_parameters,
                decay=config.self_flow_ema_decay,
                state_dict=saved_ema_state,
            )
            model.self_flow_state_dict = None

        params = self.create_parameters(model, config)
        self.__setup_requires_grad(model, config)
        init_model_parameters(model, params, self.train_device)

    def setup_train_device(
            self,
            model: Flux2Model,
            config: TrainConfig,
    ):
        vae_on_train_device = not config.image_caching
        text_encoder_on_train_device = not config.text_caching

        model.text_encoder_to(self.train_device if text_encoder_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.transformer_to(self.train_device)

        model.text_encoder.eval()
        model.vae.eval()

        if config.transformer.train:
            model.transformer.train()
            if model.self_flow_projector is not None:
                model.self_flow_projector.train()
        else:
            model.transformer.eval()
            if model.self_flow_projector is not None:
                model.self_flow_projector.eval()

    def after_optimizer_step(
            self,
            model: Flux2Model,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        self.__setup_requires_grad(model, config)
        if config.self_flow_enabled:
            if model.self_flow_ema is None:
                raise RuntimeError("Self-Flow EMA manager was not initialized.")
            model.self_flow_ema.update_after_optimizer_step()
