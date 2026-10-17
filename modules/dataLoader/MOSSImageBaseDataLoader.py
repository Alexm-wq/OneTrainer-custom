import copy
import hashlib
import json
import os

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.moss.MOSSImageCache import MOSSImageCache
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.dataLoader.moss.MOSSConditioningCache import MOSSConditioningCache
from modules.dataLoader.moss.MOSSLayout import (
    LoadMOSSLayout,
    MOSSLayoutInventory,
    MOSSLayoutPrompt,
    MOSSMaskRotateCrop,
    MOSSNormalPrompt,
    MOSSRandomFlip,
    MOSSRandomRotate,
    MOSSRestoreConcept,
    MOSSScaleCrop,
    MOSSTextAugmentation,
)
from modules.model.MOSSImageModel import MOSSImageModel
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
from mgds.pipelineModules.VariationSorting import VariationSorting


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
        modules = [rescale_image, encode_image, image_sample]
        if config.masked_training or config.model_type.has_mask_input():
            modules.append(downscale_mask)

        return modules

    def _enumerate_input_modules(self, config, allow_videos=False):
        modules = super()._enumerate_input_modules(config, allow_videos)
        self.layout_inventory = MOSSLayoutInventory()
        return [*modules, self.layout_inventory]

    def _load_input_modules(self, config, train_dtype, vae_frame_dim=False):
        modules = super()._load_input_modules(config, train_dtype, vae_frame_dim)
        # Captions are resolved after cached image row/geometry selection.
        text_outputs = {"sample_prompts", "concept_prompts", "filename_prompt", "prompts", "prompt"}
        return [module for module in modules if not text_outputs.intersection(module.get_outputs())] + [LoadMOSSLayout()]

    @staticmethod
    def _layout_transform(module, cls):
        parameters = {
            MOSSScaleCrop: ("names", "scale_resolution_in_name", "crop_resolution_in_name",
                            "enable_crop_jitter_in_name", "crop_offset_out_name"),
            MOSSRandomFlip: ("names", "enabled_in_name", "fixed_enabled_in_name"),
            MOSSRandomRotate: ("names", "enabled_in_name", "fixed_enabled_in_name", "max_angle_in_name"),
            MOSSMaskRotateCrop: ("mask_name", "additional_names", "enabled_in_name", "min_size",
                                 "min_padding_percent", "max_padding_percent", "max_rotate_angle"),
        }
        return cls(**{name: getattr(module, name) for name in parameters[cls]})

    def _mask_augmentation_modules(self, config):
        return [
            self._layout_transform(m, MOSSMaskRotateCrop) if type(m).__name__ == "RandomMaskRotateCrop" else m
            for m in super()._mask_augmentation_modules(config)
        ]

    def _crop_modules(self, config):
        return [self._layout_transform(m, MOSSScaleCrop) for m in super()._crop_modules(config)]

    def _augmentation_modules(self, config):
        modules = super()._augmentation_modules(config)
        result = []
        for module in modules:
            name = type(module).__name__
            if name in ("DropTags", "CapitalizeTags", "ShuffleTags"):
                continue
            if name == "RandomFlip":
                module = self._layout_transform(module, MOSSRandomFlip)
            elif name == "RandomRotate":
                module = self._layout_transform(module, MOSSRandomRotate)
            result.append(module)
        return result

    def _cache_modules(self, config: TrainConfig, model: MOSSImageModel, model_setup: BaseMOSSImageSetup):
        config = copy.copy(config)
        identity = {
            "version": 2, "base": config.base_model_name, "vae": config.vae.model_name,
            "text_encoder": config.text_encoder.model_name,
            "text_dtype": str(config.text_encoder.weight_dtype),
            "train_dtype": str(config.train_dtype), "fallback_dtype": str(config.fallback_train_dtype),
            "encoder_revision": getattr(getattr(model.text_encoder, "config", None), "_commit_hash", None),
            "sequence_length": config.text_encoder_sequence_length or 4096,
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        config.cache_dir = os.path.join(config.cache_dir, "moss-layout-" + key)
        image_split = ["latent_image", "original_resolution", "crop_offset"]
        if config.masked_training or config.model_type.has_mask_input():
            image_split.append("latent_mask")
        # Layout and source index belong to the SAME resolution/augmentation as
        # the latent. No separate row-based text cache may select a different one.
        image_aggregate = ["crop_resolution", "image_path", "moss_layout", "moss_source_index", "moss_source_key"]
        if config.rlhf_enabled:
            image_split.append("latent_image_rejected")
            image_aggregate += [
                "dpo_is_paired", "dpo_pair_key", "dpo_cache_mode", "image_path_rejected", "scale_resolution",
            ]
        groups = [
            "concept.path", "concept.seed", "concept.include_subdirectories", "concept.image",
            "concept.dpo_chosen_pattern", "concept.dpo_rejected_pattern",
            "concept._dpo_split", "concept._dpo_validation_percentage", "concept._dpo_validation_seed",
            "concept._image_cache_fingerprint", "concept._moss_layout_fingerprint",
            "settings.target_resolution", "settings.target_frames",
        ]

        def prepare_vae():
            from modules.util.torch_util import torch_gc
            model.to(self.temp_device)
            model.vae_to(self.train_device)
            model.eval()
            torch_gc()

        if config.image_caching:
            modules = [MOSSImageCache(
                inventory=self.layout_inventory,
                cache_dir=os.path.join(config.cache_dir, "image"),
                split_names=image_split, aggregate_names=image_aggregate,
                variations_in_name="concept.image_variations",
                balancing_in_name="concept.balancing", balancing_strategy_in_name="concept.balancing_strategy",
                variations_group_in_name=groups, group_enabled_in_name="concept.enabled",
                before_cache_fun=prepare_vae,
            )]
        else:
            modules = [VariationSorting(
                names=image_split + image_aggregate,
                balancing_in_name="concept.balancing", balancing_strategy_in_name="concept.balancing_strategy",
                variations_group_in_name=groups, group_enabled_in_name="concept.enabled",
            )]
        modules += [MOSSRestoreConcept(self.layout_inventory), MOSSNormalPrompt(config.text_caching)]
        text_augmentations = [
            module for module in super()._augmentation_modules(config)
            if type(module).__name__ in ("DropTags", "CapitalizeTags", "ShuffleTags")
        ]
        modules += [MOSSTextAugmentation(module, config.text_caching) for module in text_augmentations]
        modules += [
            MOSSLayoutPrompt(),
            MOSSConditioningCache(
                model, config.text_encoder_sequence_length, os.path.join(config.cache_dir, "text"),
                identity, lambda: model_setup.prepare_text_caching(model, config), enabled=config.text_caching,
                prepare_empty=bool(config.text_encoder.dropout_probability),
            ),
        ]
        return modules

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
