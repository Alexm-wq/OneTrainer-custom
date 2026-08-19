import modules.util.multi_gpu_util as multi
from modules.model.MageFlowModel import MageFlowModel
from modules.modelSetup.BaseMageFlowSetup import BaseMageFlowSetup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.module.MageFlowAttention import configure_mage_attention_from_config
from modules.module.MageFlowEMAStorage import create_mage_self_flow_ema, log_mage_runtime_devices
from modules.module.MageFlowNativeOptimization import setup_mage_like_flux2
from modules.module.MageFlowSelfFlow import MageFlowSelfFlowProjector
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModuleFilter import ModuleFilter
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress

import torch


@factory.register(BaseModelSetup, ModelType.MAGE_FLOW, TrainingMethod.FINE_TUNE)
class MageFlowFineTuneSetup(BaseMageFlowSetup):
    def __init__(self, train_device: torch.device, temp_device: torch.device, debug_mode: bool):
        super().__init__(train_device=train_device, temp_device=temp_device, debug_mode=debug_mode)

    @staticmethod
    def _image_shapes(batch_size: int, height: int, width: int):
        return MageFlowModel.image_shapes(batch_size, height, width)

    def setup_optimizations(self, model: MageFlowModel, config: TrainConfig):
        requested_compile = bool(config.compile)
        config.compile = False
        try:
            super().setup_optimizations(model, config)
        finally:
            config.compile = requested_compile

        setup_mage_like_flux2(self, model, config)

    @staticmethod
    def _validate_self_flow(model: MageFlowModel, config: TrainConfig):
        if config.self_flow_structural_enabled and not config.self_flow_enabled:
            raise ValueError("Structural Self-Flow requires Self-Flow")
        if not config.self_flow_enabled:
            return
        if config.multi_gpu or multi.world_size() > 1:
            raise NotImplementedError("Mage Self-Flow EMA currently supports single-GPU training only")
        if not config.transformer.train:
            raise ValueError("Mage Self-Flow requires transformer training")
        if not 0.0 <= config.self_flow_mask_ratio <= 0.5:
            raise ValueError("Self-Flow mask ratio must be between 0 and 0.5")
        if config.self_flow_rep_weight < 0.0 or config.self_flow_structural_weight < 0.0:
            raise ValueError("Self-Flow loss weights must be non-negative")
        if not 0.0 <= config.self_flow_ema_decay < 1.0:
            raise ValueError("Self-Flow EMA decay must be in [0,1)")
        depth = len(model.transformer.transformer_blocks)
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
            "[Mage Self-Flow] full-finetune EMA enabled. This keeps a complete "
            "FP32 EMA of the trainable transformer and can consume substantial "
            "CPU RAM or VRAM depending on the selected EMA storage."
        )

    def create_parameters(self, model: MageFlowModel, config: TrainConfig) -> NamedParameterGroupCollection:
        groups = NamedParameterGroupCollection()
        self._create_model_part_parameters(
            groups,
            "transformer",
            model.transformer,
            config.transformer,
            freeze=ModuleFilter.create(config),
            debug=config.debug_mode,
        )
        if config.self_flow_enabled and model.self_flow_projector is not None:
            groups.add_group(NamedParameterGroup(
                unique_name="self_flow_projector",
                display_name="self_flow_projector",
                parameters=model.self_flow_projector.parameters(),
                learning_rate=config.transformer.learning_rate,
            ))
        return groups

    def _setup_requires_grad(self, model: MageFlowModel, config: TrainConfig):
        self._setup_model_part_requires_grad("transformer", model.transformer, config.transformer, model.train_progress)
        model.vae.requires_grad_(False)
        model.text_encoder.requires_grad_(False)
        if model.self_flow_projector is not None:
            model.self_flow_projector.requires_grad_(config.self_flow_enabled and config.transformer.train)

    def setup_model(self, model: MageFlowModel, config: TrainConfig):
        configure_mage_attention_from_config(model, config)
        self._validate_self_flow(model, config)
        self._setup_requires_grad(model, config)

        if config.self_flow_enabled:
            first = next((p for p in model.transformer.parameters() if p.requires_grad), None)
            if first is None:
                raise RuntimeError("Mage full fine-tune has no trainable transformer parameters")
            model.self_flow_projector = MageFlowSelfFlowProjector(model.transformer.inner_dim).to(
                device=first.device, dtype=first.dtype
            )
            saved = model.self_flow_state_dict
            if saved is not None and saved.get("projector") is not None:
                model.self_flow_projector.load_state_dict(saved["projector"], strict=True)
            self._setup_requires_grad(model, config)
            model.self_flow_ema = create_mage_self_flow_ema(
                model.self_flow_adapter_modules(),
                config,
                first,
                state_dict=saved.get("ema") if saved is not None else None,
            )
            model.self_flow_state_dict = None

        groups = self.create_parameters(model, config)
        self._setup_requires_grad(model, config)
        init_model_parameters(model, groups, self.train_device)
        log_mage_runtime_devices(
            self,
            model,
            config,
            phase="model setup complete",
        )

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
