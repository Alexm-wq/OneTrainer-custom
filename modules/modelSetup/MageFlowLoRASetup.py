import modules.util.multi_gpu_util as multi
from modules.model.MageFlowModel import MageFlowModel
from modules.modelSetup.BaseMageFlowSetup import BaseMageFlowSetup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.module.MageFlowSelfFlow import MageFlowSelfFlowEMA, MageFlowSelfFlowProjector
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress

import torch


@factory.register(BaseModelSetup, ModelType.MAGE_FLOW, TrainingMethod.LORA)
class MageFlowLoRASetup(BaseMageFlowSetup):
    def __init__(self, train_device: torch.device, temp_device: torch.device, debug_mode: bool):
        super().__init__(train_device=train_device, temp_device=temp_device, debug_mode=debug_mode)

    @staticmethod
    def _validate_self_flow(model: MageFlowModel, config: TrainConfig):
        if config.self_flow_structural_enabled and not config.self_flow_enabled:
            raise ValueError("Structural Self-Flow requires Self-Flow")
        if not config.self_flow_enabled:
            return
        if config.multi_gpu or multi.world_size() > 1:
            raise NotImplementedError("Mage Self-Flow CPU EMA currently supports single-GPU training only")
        if not config.transformer.train:
            raise ValueError("Mage Self-Flow requires transformer LoRA training")
        if not 0.0 <= config.self_flow_mask_ratio <= 0.5:
            raise ValueError("Self-Flow mask ratio must be between 0 and 0.5")
        if config.self_flow_rep_weight < 0.0 or config.self_flow_structural_weight < 0.0:
            raise ValueError("Self-Flow loss weights must be non-negative")
        if not 0.0 <= config.self_flow_ema_decay < 1.0:
            raise ValueError("Self-Flow EMA decay must be in [0,1)")
        if config.self_flow_structural_tokens < 2:
            raise ValueError("Structural Self-Flow requires at least two sampled tokens")

        depth = len(model.transformer.transformer_blocks)
        if depth < 2:
            raise RuntimeError("Mage Self-Flow requires at least two transformer blocks")
        student = int(config.self_flow_student_layer)
        teacher = int(config.self_flow_teacher_layer)
        if student < 0:
            student = max(0, round((depth - 1) * 0.30))
        if teacher < 0:
            teacher = min(depth - 1, max(student + 1, round((depth - 1) * 0.70)))
        if not 0 <= student < teacher < depth:
            raise ValueError(f"Mage Self-Flow layers must satisfy 0 <= student < teacher < {depth}")
        model.self_flow_student_layer = student
        model.self_flow_teacher_layer = teacher
        print(
            f"[Mage Self-Flow] enabled: student block {student}, teacher block {teacher}, "
            f"mask={config.self_flow_mask_ratio:.3f}, structural={config.self_flow_structural_enabled}"
        )

    def create_parameters(self, model: MageFlowModel, config: TrainConfig) -> NamedParameterGroupCollection:
        groups = NamedParameterGroupCollection()
        self._create_model_part_parameters(groups, "transformer", model.transformer_lora, config.transformer)
        if config.self_flow_enabled and model.self_flow_projector is not None:
            groups.add_group(NamedParameterGroup(
                unique_name="self_flow_projector",
                display_name="self_flow_projector",
                parameters=model.self_flow_projector.parameters(),
                learning_rate=config.transformer.learning_rate,
            ))
        return groups

    def _setup_requires_grad(self, model: MageFlowModel, config: TrainConfig):
        model.text_encoder.requires_grad_(False)
        model.transformer.requires_grad_(False)
        model.vae.requires_grad_(False)
        self._setup_model_part_requires_grad(
            "transformer", model.transformer_lora, config.transformer, model.train_progress
        )
        if model.self_flow_projector is not None:
            model.self_flow_projector.requires_grad_(config.self_flow_enabled and config.transformer.train)

    def setup_model(self, model: MageFlowModel, config: TrainConfig):
        self._validate_self_flow(model, config)
        model.transformer_lora = LoRAModuleWrapper(
            model.transformer,
            "transformer",
            config,
            config.layer_filter.split(","),
        )
        if model.lora_state_dict:
            model.transformer_lora.load_state_dict(model.lora_state_dict)
            model.lora_state_dict = None
        model.transformer_lora.set_dropout(config.dropout_probability)
        model.transformer_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        model.transformer_lora.hook_to_module()

        if config.self_flow_enabled:
            self._setup_requires_grad(model, config)
            first = next((p for p in model.transformer_lora.parameters() if p.requires_grad), None)
            if first is None:
                raise RuntimeError("Mage LoRA has no trainable parameters for Self-Flow")
            model.self_flow_projector = MageFlowSelfFlowProjector(model.transformer.inner_dim).to(
                device=first.device, dtype=first.dtype
            )
            saved = model.self_flow_state_dict
            if saved is not None and saved.get("projector") is not None:
                model.self_flow_projector.load_state_dict(saved["projector"], strict=True)
            self._setup_requires_grad(model, config)
            model.self_flow_ema = MageFlowSelfFlowEMA(
                model.self_flow_adapter_modules(),
                decay=config.self_flow_ema_decay,
                state_dict=saved.get("ema") if saved is not None else None,
            )
            model.self_flow_state_dict = None

        groups = self.create_parameters(model, config)
        self._setup_requires_grad(model, config)
        init_model_parameters(model, groups, self.train_device)

    def setup_train_device(self, model: MageFlowModel, config: TrainConfig):
        model.text_encoder_to(self.train_device if not config.text_caching else self.temp_device)
        model.vae_to(self.train_device if not config.image_caching else self.temp_device)
        model.transformer_to(self.train_device)
        model.text_encoder.eval()
        model.vae.eval()
        model.transformer.train(config.transformer.train)
        if model.self_flow_projector is not None:
            model.self_flow_projector.train(config.transformer.train and config.self_flow_enabled)

    def after_optimizer_step(self, model: MageFlowModel, config: TrainConfig, train_progress: TrainProgress):
        self._setup_requires_grad(model, config)
        if config.self_flow_enabled:
            if model.self_flow_ema is None:
                raise RuntimeError("Mage Self-Flow EMA was not initialized")
            model.self_flow_ema.update_after_optimizer_step()
