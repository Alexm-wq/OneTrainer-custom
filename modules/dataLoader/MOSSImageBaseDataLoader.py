import copy
import hashlib
import json
import os

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.model.MOSSImageModel import MOSSImageModel
from modules.dataLoader.moss.EncodeMOSSImageText import EncodeMOSSImageText
from modules.modelSetup.BaseMOSSImageSetup import BaseMOSSImageSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.TrainProgress import TrainProgress

from mgds.pipelineModules.DecodeTokens import DecodeTokens
from mgds.pipelineModules.DecodeVAE import DecodeVAE
from mgds.pipelineModules.EncodeVAE import EncodeVAE
from mgds.pipelineModules.RescaleImageChannels import RescaleImageChannels
from mgds.pipelineModules.SampleVAEDistribution import SampleVAEDistribution
from mgds.pipelineModules.SaveImage import SaveImage
from mgds.pipelineModules.SaveText import SaveText
from mgds.pipelineModules.ScaleImage import ScaleImage
from mgds.pipelineModules.Tokenize import Tokenize


@factory.register(BaseDataLoader, ModelType.MOSS_IMAGE)
class MOSSImageBaseDataLoader(
    BaseDataLoader,
    DataLoaderText2ImageMixin,
):
    def _preparation_modules(self, config: TrainConfig, model: MOSSImageModel):
        rescale_image = RescaleImageChannels(image_in_name='image', image_out_name='image', in_range_min=0, in_range_max=1, out_range_min=-1, out_range_max=1)
        encode_image = EncodeVAE(in_name='image', out_name='latent_image_distribution', vae=model.vae, autocast_contexts=[model.vae_autocast_context], dtype=model.vae_train_dtype.torch_dtype())
        image_sample = SampleVAEDistribution(in_name='latent_image_distribution', out_name='latent_image', mode='mean')
        downscale_mask = ScaleImage(in_name='mask', out_name='latent_mask', factor=0.125)
        tokenize_prompt = Tokenize(
            in_name='prompt', tokens_out_name='tokens', mask_out_name='tokens_mask',
            tokenizer=model.tokenizer, max_token_length=config.text_encoder_sequence_length or 4096,
            apply_chat_template=lambda caption: [{"role": "user", "content": caption}],
            apply_chat_template_kwargs={'add_generation_prompt': True, 'enable_thinking': True},
        )
        encode_prompt = EncodeMOSSImageText(model)

        modules = [rescale_image, encode_image, image_sample]
        if config.masked_training or config.model_type.has_mask_input():
            modules.append(downscale_mask)

        modules += [tokenize_prompt, encode_prompt]
        return modules

    def _cache_modules(self, config: TrainConfig, model: MOSSImageModel, model_setup: BaseMOSSImageSetup):
        # Separate MOSS from other model families and invalidate conditioning caches
        # when the base/encoder precision/context length or VAE override changes.
        config = copy.copy(config)
        identity = {
            "version": 1, "base": config.base_model_name, "vae": config.vae.model_name,
            "text_dtype": str(config.text_encoder.weight_dtype),
            "sequence_length": config.text_encoder_sequence_length or 4096,
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        config.cache_dir = os.path.join(config.cache_dir, "moss-image-" + key)
        image_split_names = ['latent_image', 'original_resolution', 'crop_offset']

        if config.masked_training or config.model_type.has_mask_input():
            image_split_names.append('latent_mask')

        image_aggregate_names = ['crop_resolution', 'image_path']

        text_split_names = []

        sort_names = image_aggregate_names + image_split_names + [
            'prompt', 'tokens', 'tokens_mask', 'text_encoder_hidden_state',
            'concept'
        ]

        text_split_names += ['tokens', 'tokens_mask', 'text_encoder_hidden_state']

        return self._cache_modules_from_names(
            model, model_setup,
            image_split_names=image_split_names,
            image_aggregate_names=image_aggregate_names,
            text_split_names=text_split_names,
            sort_names=sort_names,
            config=config,
            text_caching=config.text_caching,
        )

    def _output_modules(self, config: TrainConfig, model: MOSSImageModel, model_setup: BaseMOSSImageSetup):
        output_names = [
            'image_path', 'latent_image',
            'prompt',
            'tokens',
            'tokens_mask',
            'original_resolution', 'crop_resolution', 'crop_offset',
        ]

        if config.masked_training or config.model_type.has_mask_input():
            output_names.append('latent_mask')

        output_names.append('text_encoder_hidden_state')

        return self._output_modules_from_out_names(
            model, model_setup,
            output_names=output_names,
            config=config,
            use_conditioning_image=False,
            vae=model.vae,
            autocast_context=[model.vae_autocast_context],
            train_dtype=model.vae_train_dtype,
        )

    def _debug_modules(self, config: TrainConfig, model: MOSSImageModel):
        debug_dir = os.path.join(config.debug_dir, "dataloader")

        def before_save_fun():
            model.vae_to(self.train_device)

        decode_image = DecodeVAE(in_name='latent_image', out_name='decoded_image', vae=model.vae, autocast_contexts=[model.vae_autocast_context], dtype=model.vae_train_dtype.torch_dtype())
        upscale_mask = ScaleImage(in_name='latent_mask', out_name='decoded_mask', factor=8)
        decode_prompt = DecodeTokens(in_name='tokens', out_name='decoded_prompt', tokenizer=model.tokenizer)
        save_image = SaveImage(image_in_name='decoded_image', original_path_in_name='image_path', path=debug_dir, in_range_min=-1, in_range_max=1, before_save_fun=before_save_fun)
        # SaveImage(image_in_name='latent_mask', original_path_in_name='image_path', path=debug_dir, in_range_min=0, in_range_max=1, before_save_fun=before_save_fun)
        save_mask = SaveImage(image_in_name='decoded_mask', original_path_in_name='image_path', path=debug_dir, in_range_min=0, in_range_max=1, before_save_fun=before_save_fun)
        save_prompt = SaveText(text_in_name='decoded_prompt', original_path_in_name='image_path', path=debug_dir, before_save_fun=before_save_fun)

        # These modules don't really work, since they are inserted after a sorting operation that does not include this data
        # SaveImage(image_in_name='mask', original_path_in_name='image_path', path=debug_dir, in_range_min=0, in_range_max=1),
        # SaveImage(image_in_name='image', original_path_in_name='image_path', path=debug_dir, in_range_min=-1, in_range_max=1),

        modules = []

        modules.append(decode_image)
        modules.append(save_image)

        if config.masked_training or config.model_type.has_mask_input():
            modules.append(upscale_mask)
            modules.append(save_mask)

        modules.append(decode_prompt)
        modules.append(save_prompt)

        return modules

    def _create_dataset(
            self,
            config: TrainConfig,
            model: MOSSImageModel,
            model_setup: BaseMOSSImageSetup,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        return DataLoaderText2ImageMixin._create_dataset(self,
            config, model, model_setup, train_progress, is_validation,
            aspect_bucketing_quantization=16,
        )
