from modules.model.MOSSImageModel import MOSSImageModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.BaseMOSSImageSetup import BaseMOSSImageSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress

import torch


@factory.register(BaseModelSetup, ModelType.MOSS_IMAGE, TrainingMethod.LORA)
class MOSSImageLoRASetup(
    BaseMOSSImageSetup,
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
            model: MOSSImageModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        self._create_model_part_parameters(parameter_group_collection, "transformer", model.transformer_lora, config.transformer)
        if model.self_flow_projector is not None:
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name="self_flow_projector",
                parameters=model.self_flow_projector.parameters(),
                learning_rate=config.transformer.learning_rate,
            ))
        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: MOSSImageModel,
            config: TrainConfig,
    ):
        model.text_encoder.requires_grad_(False)
        model.transformer.requires_grad_(False)
        model.vae.requires_grad_(False)

        self._setup_model_part_requires_grad("transformer", model.transformer_lora, config.transformer, model.train_progress)
        if model.self_flow_projector is not None:
            model.self_flow_projector.requires_grad_(True)

    def setup_model(
            self,
            model: MOSSImageModel,
            config: TrainConfig,
    ):
        self._validate_self_flow(model, config)

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

        self._create_self_flow_projector(model, config, config.lora_weight_dtype.torch_dtype())
        params = self.create_parameters(model, config)
        self.__setup_requires_grad(model, config)
        self._create_self_flow_ema(model, config, model.transformer_lora.parameters())
        init_model_parameters(model, params, self.train_device)

    def setup_train_device(
            self,
            model: MOSSImageModel,
            config: TrainConfig,
    ):
        if config.text_encoder.dropout_probability and (config.text_encoder_sequence_length or 4096) not in model.empty_text:
            # Cache hits do not invoke prepare_text_caching, so initialize dropout
            # conditioning here as well, before moving the DiT onto the GPU.
            model.transformer_to(self.temp_device)
            model.vae_to(self.temp_device)
            model.text_encoder_to(self.train_device)
            model.prepare_empty_text(config.text_encoder_sequence_length)

        # The identity reward decodes face crops of the x0 prediction every few steps.
        vae_on_train_device = not config.image_caching or config.identity_reward
        text_encoder_on_train_device = not config.text_caching

        model.text_encoder_to(self.train_device if text_encoder_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.transformer_to(self.train_device)

        model.text_encoder.eval()
        model.vae.eval()

        if config.transformer.train:
            model.transformer.train()
        else:
            model.transformer.eval()
        if model.self_flow_projector is not None:
            model.self_flow_projector.train()

        self._prepare_identity_reward(model, config)

    def after_optimizer_step(
            self,
            model: MOSSImageModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        self.__setup_requires_grad(model, config)
        if model.self_flow_ema is not None:
            model.self_flow_ema.update()
