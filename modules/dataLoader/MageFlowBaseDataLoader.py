from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.dataLoader.mage.EncodeMageText import EncodeMageText
from modules.dataLoader.mage.EncodeMageVAE import EncodeMageVAE
from modules.dataLoader.mage.TokenizeMagePrompt import TokenizeMagePrompt
from modules.model.BaseModel import BaseModel
from modules.model.MageFlowModel import MageFlowModel
from modules.modelSetup.BaseMageFlowSetup import BaseMageFlowSetup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.TrainProgress import TrainProgress

from mgds.pipelineModules.RescaleImageChannels import RescaleImageChannels
from mgds.pipelineModules.ScaleImage import ScaleImage


@factory.register(BaseDataLoader, ModelType.MAGE_FLOW)
class MageFlowBaseDataLoader(BaseDataLoader, DataLoaderText2ImageMixin):
    """Native Mage latent/text caching with OneTrainer's normal DPO augmentation."""

    def _prompt_length(self, config: TrainConfig) -> int:
        length = getattr(config, "text_encoder_sequence_length", None)
        return int(length if length is not None else 2048)

    def _preparation_modules(self, config: TrainConfig, model: MageFlowModel):
        prompt_length = self._prompt_length(config)
        modules = [
            RescaleImageChannels(
                image_in_name="image",
                image_out_name="image",
                in_range_min=0,
                in_range_max=1,
                out_range_min=-1,
                out_range_max=1,
            ),
            EncodeMageVAE(
                in_name="image",
                out_name="latent_image",
                vae=model.vae,
                autocast_contexts=[model.autocast_context],
                dtype=model.train_dtype.torch_dtype(),
            ),
        ]
        if config.masked_training or config.model_type.has_mask_input():
            modules.append(ScaleImage(in_name="mask", out_name="latent_mask", factor=1.0 / 16.0))

        modules.append(TokenizeMagePrompt(
            in_name="prompt",
            tokens_out_name="tokens",
            mask_out_name="tokens_mask",
            tokenizer=model.tokenizer,
            prompt_token_length=prompt_length,
        ))

        if not config.train_text_encoder_or_embedding():
            modules.append(EncodeMageText(
                tokens_name="tokens",
                tokens_mask_name="tokens_mask",
                hidden_state_out_name="text_encoder_hidden_state",
                attention_mask_out_name="text_encoder_attention_mask",
                text_encoder_wrapper=model.text_encoder_wrapper,
                autocast_contexts=[model.text_encoder_autocast_context],
                dtype=model.text_encoder_train_dtype.torch_dtype(),
                max_output_length=prompt_length,
                restore_attention_backend=getattr(
                    model,
                    "mage_attention_backend",
                    "sdpa",
                ),
            ))
        return modules

    def _cache_modules(self, config: TrainConfig, model: MageFlowModel, model_setup: BaseMageFlowSetup):
        image_split_names = ["latent_image", "original_resolution", "crop_offset"]
        if config.masked_training or config.model_type.has_mask_input():
            image_split_names.append("latent_mask")
        image_aggregate_names = ["crop_resolution", "image_path"]

        text_split_names = []
        sort_names = image_aggregate_names + image_split_names + [
            "prompt", "tokens", "tokens_mask", "text_encoder_hidden_state",
            "text_encoder_attention_mask", "concept",
        ]
        if not config.train_text_encoder_or_embedding():
            text_split_names += [
                "tokens", "tokens_mask", "text_encoder_hidden_state", "text_encoder_attention_mask"
            ]

        return self._cache_modules_from_names(
            model,
            model_setup,
            image_split_names=image_split_names,
            image_aggregate_names=image_aggregate_names,
            text_split_names=text_split_names,
            sort_names=sort_names,
            config=config,
            text_caching=config.text_caching and not config.train_text_encoder_or_embedding(),
        )

    def _output_modules(self, config: TrainConfig, model: MageFlowModel, model_setup: BaseMageFlowSetup):
        output_names = [
            "image_path", "latent_image", "prompt", "tokens", "tokens_mask",
            "original_resolution", "crop_resolution", "crop_offset",
        ]
        if config.masked_training or config.model_type.has_mask_input():
            output_names.append("latent_mask")
        if not config.train_text_encoder_or_embedding():
            output_names += ["text_encoder_hidden_state", "text_encoder_attention_mask"]

        return self._output_modules_from_out_names(
            model,
            model_setup,
            output_names=output_names,
            config=config,
            use_conditioning_image=False,
            vae=model.vae,
            autocast_context=[model.autocast_context],
            train_dtype=model.train_dtype,
        )

    def _debug_modules(self, config: TrainConfig, model: MageFlowModel):
        # MageVAE's interface is not Diffusers' AutoencoderKL interface. Debug
        # decoding is intentionally left to the sampler rather than routing the
        # direct-tensor VAE through MGDS DecodeVAE.
        return []

    def _create_dataset(
            self,
            config: TrainConfig,
            model: BaseModel,
            model_setup: BaseModelSetup,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        return DataLoaderText2ImageMixin._create_dataset(
            self,
            config,
            model,
            model_setup,
            train_progress,
            is_validation,
            aspect_bucketing_quantization=16,
            allow_video_files=False,
            vae_frame_dim=False,
        )
