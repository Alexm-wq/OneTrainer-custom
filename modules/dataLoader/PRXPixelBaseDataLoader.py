import os

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.model.BaseModel import BaseModel
from modules.model.PRXPixelModel import PROMPT_MAX_LENGTH, PRXPixelModel, clean_prompt
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.BasePRXPixelSetup import BasePRXPixelSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.TrainProgress import TrainProgress

from mgds.pipelineModules.DecodeTokens import DecodeTokens
from mgds.pipelineModules.MapData import MapData
from mgds.pipelineModules.EncodeQwenText import EncodeQwenText
from mgds.pipelineModules.PadMaskedTokens import PadMaskedTokens
from mgds.pipelineModules.PruneMaskedTokens import PruneMaskedTokens
from mgds.pipelineModules.RescaleImageChannels import RescaleImageChannels
from mgds.pipelineModules.SaveImage import SaveImage
from mgds.pipelineModules.SaveText import SaveText
from mgds.pipelineModules.ScaleImage import ScaleImage
from mgds.pipelineModules.Tokenize import Tokenize


@factory.register(BaseDataLoader, ModelType.PRX_PIXEL)
class PRXPixelBaseDataLoader(BaseDataLoader, DataLoaderText2ImageMixin):
    @staticmethod
    def _no_image_encoder():
        pass

    def _preparation_modules(self, config: TrainConfig, model: PRXPixelModel):
        encode_image = RescaleImageChannels(
            image_in_name="image",
            image_out_name="latent_image",
            in_range_min=0,
            in_range_max=1,
            out_range_min=-1,
            out_range_max=1,
        )
        clean_prompt_module = MapData(
            in_name="prompt",
            out_name="prompt",
            map_fn=clean_prompt,
        )
        tokenize_prompt = Tokenize(
            in_name="prompt",
            tokens_out_name="tokens",
            mask_out_name="tokens_mask",
            tokenizer=model.tokenizer,
            max_token_length=PROMPT_MAX_LENGTH,
        )
        encode_prompt = EncodeQwenText(
            tokens_name="tokens",
            tokens_attention_mask_in_name="tokens_mask",
            hidden_state_out_name="text_encoder_hidden_state",
            tokens_attention_mask_out_name="tokens_mask",
            text_encoder=model.text_encoder,
            hidden_state_output_index=-1,
            autocast_contexts=[model.text_encoder_autocast_context],
            dtype=model.text_encoder_train_dtype.torch_dtype(),
        )
        prune_masked_tokens = PruneMaskedTokens(
            tokens_name="tokens",
            tokens_mask_name="tokens_mask",
            hidden_state_name="text_encoder_hidden_state",
        )

        modules = [encode_image]
        if config.masked_training:
            modules.append(ScaleImage(in_name="mask", out_name="latent_mask", factor=1.0))
        modules.append(clean_prompt_module)
        modules.append(tokenize_prompt)

        if not config.train_text_encoder_or_embedding():
            modules.append(encode_prompt)
        if config.text_caching and not config.train_text_encoder_or_embedding():
            modules.append(prune_masked_tokens)
        return modules

    def _cache_modules(
            self,
            config: TrainConfig,
            model: PRXPixelModel,
            model_setup: BasePRXPixelSetup,
    ):
        image_split_names = ["latent_image", "original_resolution", "crop_offset"]
        if config.masked_training:
            image_split_names.append("latent_mask")

        image_aggregate_names = ["crop_resolution", "image_path"]
        text_split_names = []
        sort_names = image_aggregate_names + image_split_names + [
            "prompt", "tokens", "tokens_mask", "text_encoder_hidden_state", "concept",
        ]
        if not config.train_text_encoder_or_embedding():
            text_split_names += ["tokens", "tokens_mask", "text_encoder_hidden_state"]

        return self._cache_modules_from_names(
            model,
            model_setup,
            image_split_names=image_split_names,
            image_aggregate_names=image_aggregate_names,
            text_split_names=text_split_names,
            sort_names=sort_names,
            config=config,
            text_caching=config.text_caching and not config.train_text_encoder_or_embedding(),
            before_cache_image_fun=self._no_image_encoder,
        )

    def _output_modules(
            self,
            config: TrainConfig,
            model: PRXPixelModel,
            model_setup: BasePRXPixelSetup,
    ):
        output_names = [
            "image_path", "latent_image", "prompt", "tokens", "tokens_mask",
            "original_resolution", "crop_resolution", "crop_offset",
        ]
        if config.masked_training:
            output_names.append("latent_mask")
        if not config.train_text_encoder_or_embedding():
            output_names.append("text_encoder_hidden_state")

        modules = self._output_modules_from_out_names(
            model,
            model_setup,
            output_names=output_names,
            config=config,
            before_cache_image_fun=self._no_image_encoder,
            use_conditioning_image=False,
            vae=None,
            autocast_context=[model.autocast_context],
            train_dtype=model.train_dtype,
        )
        if config.text_caching and not config.train_text_encoder_or_embedding():
            modules = [PadMaskedTokens(
                tokens_name="tokens",
                tokens_mask_name="tokens_mask",
                hidden_state_name="text_encoder_hidden_state",
                max_length=PROMPT_MAX_LENGTH,
            )] + modules
        return modules

    def _debug_modules(self, config: TrainConfig, model: PRXPixelModel):
        debug_dir = os.path.join(config.debug_dir, "dataloader")
        save_image = SaveImage(
            image_in_name="latent_image",
            original_path_in_name="image_path",
            path=debug_dir,
            in_range_min=-1,
            in_range_max=1,
        )
        decode_prompt = DecodeTokens(
            in_name="tokens",
            out_name="decoded_prompt",
            tokenizer=model.tokenizer,
        )
        save_prompt = SaveText(
            text_in_name="decoded_prompt",
            original_path_in_name="image_path",
            path=debug_dir,
        )
        return [save_image, decode_prompt, save_prompt]

    def _create_dataset(
            self,
            config: TrainConfig,
            model: BaseModel,
            model_setup: BaseModelSetup,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        patch_size = int(getattr(model.transformer.config, "patch_size", 16))
        return DataLoaderText2ImageMixin._create_dataset(
            self,
            config,
            model,
            model_setup,
            train_progress,
            is_validation,
            aspect_bucketing_quantization=patch_size,
            allow_video_files=False,
            supports_inpainting=False,
        )
