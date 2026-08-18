import json
import os
import re
from abc import ABCMeta, abstractmethod
from collections.abc import Callable

import modules.util.multi_gpu_util as multi
from modules.model.BaseModel import BaseModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupText2ImageMixin import ModelSetupText2ImageMixin
from modules.util import path_util
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.CacheEncryptionScope import CacheEncryptionScope
from modules.util.enum.ConceptType import ConceptType
from modules.util.enum.DataType import DataType
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.pipelineModules.AspectBatchSorting import AspectBatchSorting
from mgds.pipelineModules.AspectBucketing import AspectBucketing
from mgds.pipelineModules.CalcAspect import CalcAspect
from mgds.pipelineModules.CapitalizeTags import CapitalizeTags
from mgds.pipelineModules.CollectPaths import CollectPaths
from mgds.pipelineModules.DiskCache import DiskCache
from mgds.pipelineModules.MultiResolutionDiskCache import MultiResolutionDiskCache
from mgds.pipelineModules.DistributedSampler import DistributedSampler
from mgds.pipelineModules.DownloadHuggingfaceDatasets import DownloadHuggingfaceDatasets
from mgds.pipelineModules.DropTags import DropTags
from mgds.pipelineModules.GenerateImageLike import GenerateImageLike
from mgds.pipelineModules.GenerateMaskedConditioningImage import GenerateMaskedConditioningImage
from mgds.pipelineModules.GetFilename import GetFilename
from mgds.pipelineModules.ImageToVideo import ImageToVideo
from mgds.pipelineModules.InlineAspectBatchSorting import InlineAspectBatchSorting
from mgds.pipelineModules.InlineDistributedSampler import InlineDistributedSampler
from mgds.pipelineModules.LoadImage import LoadImage
from mgds.pipelineModules.LoadMultipleTexts import LoadMultipleTexts
from mgds.pipelineModules.LoadVideo import LoadVideo
from mgds.pipelineModules.ModifyPath import ModifyPath
from mgds.pipelineModules.RandomBrightness import RandomBrightness
from mgds.pipelineModules.RandomCircularMaskShrink import RandomCircularMaskShrink
from mgds.pipelineModules.RandomContrast import RandomContrast
from mgds.pipelineModules.RandomFlip import RandomFlip
from mgds.pipelineModules.RandomHue import RandomHue
from mgds.pipelineModules.RandomLatentMaskRemove import RandomLatentMaskRemove
from mgds.pipelineModules.RandomMaskRotateCrop import RandomMaskRotateCrop
from mgds.pipelineModules.RandomRotate import RandomRotate
from mgds.pipelineModules.RandomSaturation import RandomSaturation
from mgds.pipelineModules.ScaleCropImage import ScaleCropImage
from mgds.pipelineModules.SelectFirstInput import SelectFirstInput
from mgds.pipelineModules.SelectInput import SelectInput
from mgds.pipelineModules.SelectRandomText import SelectRandomText
from mgds.pipelineModules.ShuffleTags import ShuffleTags
from mgds.pipelineModules.SingleAspectCalculation import SingleAspectCalculation
from mgds.pipelineModules.VariationSorting import VariationSorting

import torch

from diffusers import AutoencoderKL
from modules.dataLoader.dpo.AdaptiveDPODataset import AdaptiveDPODataset
from modules.dataLoader.dpo.DeriveDPORejectedPath import DeriveDPORejectedPath
from modules.dataLoader.dpo.FilterDPOChosenPaths import FilterDPOChosenPaths
from modules.dataLoader.dpo.LoadDPORejectedImageOrDummy import LoadDPORejectedImageOrDummy
from modules.dataLoader.dpo.EncodeDPORejectedOrDummyLatent import EncodeDPORejectedOrDummyLatent
from modules.dataLoader.dpo.LoadDPOLocalizedMask import LoadDPOLocalizedMask
from modules.dataLoader.dpo.PrepareDPOLocalizedMask import PrepareDPOLocalizedMask


class DataLoaderText2ImageMixin(metaclass=ABCMeta):
    @staticmethod
    def _localized_dpo_enabled(config: TrainConfig) -> bool:
        if not bool(getattr(config, "rlhf_enabled", False)):
            return False

        concepts = getattr(config, "concepts", None)
        if concepts is None:
            try:
                with open(
                        config.concept_file_name,
                        "r",
                        encoding="utf-8",
                ) as handle:
                    concepts = [
                        ConceptConfig.default_values().from_dict(item)
                        for item in json.load(handle)
                    ]
            except (OSError, json.JSONDecodeError, TypeError):
                concepts = []

        for concept in concepts:
            concept_dict = (
                concept.to_dict()
                if hasattr(concept, "to_dict")
                else concept
            )
            if not bool(concept_dict.get("enabled", True)):
                continue
            if not bool(concept_dict.get("dpo_masked", False)):
                continue
            if (
                concept_dict.get("dpo_chosen_pattern", "")
                or concept_dict.get("dpo_rejected_pattern", "")
            ):
                return True
        return False

    @staticmethod
    def _cache_only_concepts(config: TrainConfig) -> list[dict]:
        concepts = config.concepts
        if concepts is None:
            try:
                with open(
                        config.concept_file_name,
                        "r",
                        encoding="utf-8",
                ) as handle:
                    concepts = [
                        ConceptConfig.default_values().from_dict(item)
                        for item in json.load(handle)
                    ]
            except (OSError, json.JSONDecodeError, TypeError):
                # Cache-only can run from self-describing manifests with no
                # concept file at all. Missing concepts merely disable live
                # per-concept overrides; they never invalidate the cache.
                concepts = []
        concept_dicts = [
            concept.to_dict() if hasattr(concept, "to_dict") else concept
            for concept in concepts
        ]
        is_validation = bool(
            getattr(config, "_cache_only_is_validation", False)
        )
        return [
            concept
            for concept in concept_dicts
            if (
                ConceptType(concept["type"]) == ConceptType.VALIDATION
            ) == is_validation
        ]

    def _enumerate_input_modules(self, config: TrainConfig, allow_videos: bool = False) -> list:
        supported_extensions = set()
        supported_extensions |= path_util.supported_image_extensions()

        if allow_videos:
            supported_extensions |= path_util.supported_video_extensions()

        download_datasets = DownloadHuggingfaceDatasets(
            concept_in_name='concept', path_in_name='path', enabled_in_name='enabled',
            concept_out_name='concept',
        )

        collect_paths = CollectPaths(
            concept_in_name='concept', path_in_name='path', include_subdirectories_in_name='concept.include_subdirectories', enabled_in_name='enabled',
            path_out_name='image_path', concept_out_name='concept',
            extensions=supported_extensions, include_postfix=None, exclude_postfix=['-masklabel','-condlabel']
        )

        mask_path = ModifyPath(in_name='image_path', out_name='mask_path', postfix='-masklabel', extension='.png')
        cond_path = ModifyPath(in_name='image_path', out_name='cond_path', postfix='-condlabel', extension='.png')
        sample_prompt_path = ModifyPath(in_name='image_path', out_name='sample_prompt_path', postfix='', extension='.txt')

        modules = [download_datasets, collect_paths, FilterDPOChosenPaths(), sample_prompt_path]

        if config.rlhf_enabled:
            modules.append(DeriveDPORejectedPath())

        if config.masked_training:
            modules.append(mask_path)
        if config.custom_conditioning_image:
            modules.append(cond_path)

        return modules

    def _load_input_modules(
            self,
            config: TrainConfig,
            train_dtype: DataType,
            vae_frame_dim: bool = False,
    ) -> list:
        load_image = LoadImage(path_in_name='image_path', image_out_name='image', range_min=0, range_max=1, supported_extensions=path_util.supported_image_extensions(), dtype=train_dtype.torch_dtype())
        load_video = LoadVideo(path_in_name='image_path', target_frame_count_in_name='settings.target_frames', video_out_name='image', range_min=0, range_max=1, target_frame_rate=24, supported_extensions=path_util.supported_video_extensions(), dtype=train_dtype.torch_dtype())
        image_to_video = ImageToVideo(in_name='image', out_name='image')

        generate_mask = GenerateImageLike(image_in_name='image', image_out_name='mask', color=255, range_min=0, range_max=1)
        load_mask = LoadImage(path_in_name='mask_path', image_out_name='mask', range_min=0, range_max=1, channels=1, supported_extensions={".png"}, dtype=train_dtype.torch_dtype())
        mask_to_video = ImageToVideo(in_name='mask', out_name='mask')

        load_cond_image = LoadImage(path_in_name='cond_path', image_out_name='custom_conditioning_image', range_min=0, range_max=1, supported_extensions=path_util.supported_image_extensions(), dtype=train_dtype.torch_dtype())

        load_sample_prompts = LoadMultipleTexts(path_in_name='sample_prompt_path', texts_out_name='sample_prompts')
        load_concept_prompts = LoadMultipleTexts(path_in_name='concept.text.prompt_path', texts_out_name='concept_prompts')
        filename_prompt = GetFilename(path_in_name='image_path', filename_out_name='filename_prompt', include_extension=False)
        select_prompt_input = SelectInput(setting_name='concept.text.prompt_source', out_name='prompts', setting_to_in_name_map={
            'sample': 'sample_prompts',
            'concept': 'concept_prompts',
            'filename': 'filename_prompt',
        }, default_in_name='sample_prompts')
        select_random_text = SelectRandomText(texts_in_name='prompts', text_out_name='prompt')

        modules = [load_image, load_video]

        if config.rlhf_enabled:
            modules.append(LoadDPORejectedImageOrDummy(
                path_in_name='image_path_rejected', image_in_name='image', is_paired_in_name='dpo_is_paired',
                image_out_name='image_rejected', range_min=0, range_max=1,
                dtype=train_dtype.torch_dtype(),
            ))
            if self._localized_dpo_enabled(config):
                modules.append(LoadDPOLocalizedMask(
                    path_in_name="image_path",
                    image_in_name="image",
                    is_paired_in_name="dpo_is_paired",
                    concept_in_name="concept",
                    mask_out_name="dpo_mask_image",
                    mask_path_out_name="dpo_mask_path",
                    dtype=train_dtype.torch_dtype(),
                ))

        if vae_frame_dim:
            modules.append(image_to_video)
            if config.rlhf_enabled:
                modules.append(ImageToVideo(in_name='image_rejected', out_name='image_rejected'))
                if self._localized_dpo_enabled(config):
                    modules.append(ImageToVideo(
                        in_name="dpo_mask_image",
                        out_name="dpo_mask_image",
                    ))

        modules.extend([load_sample_prompts, load_concept_prompts, filename_prompt, select_prompt_input, select_random_text])

        if config.masked_training:
            modules.append(generate_mask)
            modules.append(load_mask)
        elif config.model_type.has_mask_input():
            modules.append(generate_mask)

        if config.custom_conditioning_image:
            modules.append(load_cond_image)

        if vae_frame_dim:
            modules.append(mask_to_video)

        return modules

    def _mask_augmentation_modules(self, config: TrainConfig) -> list:
        inputs = ['image']
        if config.rlhf_enabled and 'image_rejected' not in inputs:
            inputs.append('image_rejected')
        if self._localized_dpo_enabled(config):
            inputs.append("dpo_mask_image")

        lowest_resolution = min([int(x.strip()) for x in re.split(r'\D', config.resolution) if x.strip() != ''])
        circular_mask_shrink = RandomCircularMaskShrink(mask_name='mask', shrink_probability=1.0, shrink_factor_min=0.2, shrink_factor_max=1.0, enabled_in_name='concept.image.enable_random_circular_mask_shrink')
        random_mask_rotate_crop = RandomMaskRotateCrop(mask_name='mask', additional_names=inputs, min_size=lowest_resolution, min_padding_percent=10, max_padding_percent=30, max_rotate_angle=20, enabled_in_name='concept.image.enable_random_mask_rotate_crop')

        modules = []

        if config.masked_training or config.model_type.has_mask_input():
            modules.append(circular_mask_shrink)

        if config.masked_training or config.model_type.has_mask_input():
            modules.append(random_mask_rotate_crop)

        return modules

    def _aspect_bucketing_in(self, config: TrainConfig, aspect_bucketing_quantization: int, frame_dim_enabled:bool=False):
        calc_aspect = CalcAspect(image_in_name='image', resolution_out_name='original_resolution')

        aspect_bucketing_quantization = AspectBucketing(
            quantization=aspect_bucketing_quantization,
            resolution_in_name='original_resolution',
            target_resolution_in_name='settings.target_resolution',
            enable_target_resolutions_override_in_name='concept.image.enable_resolution_override',
            target_resolutions_override_in_name='concept.image.resolution_override',
            target_frames_in_name='settings.target_frames',
            frame_dim_enabled=frame_dim_enabled,
            scale_resolution_out_name='scale_resolution',
            crop_resolution_out_name='crop_resolution',
            possible_resolutions_out_name='possible_resolutions',
            resolution_variants_out_name='resolution_variants',
        )

        single_aspect_calculation = SingleAspectCalculation(
            resolution_in_name='original_resolution',
            target_resolution_in_name='settings.target_resolution',
            enable_target_resolutions_override_in_name='concept.image.enable_resolution_override',
            target_resolutions_override_in_name='concept.image.resolution_override',
            scale_resolution_out_name='scale_resolution',
            crop_resolution_out_name='crop_resolution',
            possible_resolutions_out_name='possible_resolutions',
            resolution_variants_out_name='resolution_variants',
        )

        modules = [calc_aspect]

        if config.aspect_ratio_bucketing:
            modules.append(aspect_bucketing_quantization)
        else:
            modules.append(single_aspect_calculation)

        return modules

    def _crop_modules(self, config: TrainConfig):
        inputs = ['image']
        if config.rlhf_enabled and 'image_rejected' not in inputs:
            inputs.append('image_rejected')
        if self._localized_dpo_enabled(config):
            inputs.append("dpo_mask_image")

        if config.masked_training or config.model_type.has_mask_input():
            inputs.append('mask')

        if config.model_type.has_depth_input():
            inputs.append('depth')

        if config.custom_conditioning_image:
            inputs.append('custom_conditioning_image')

        scale_crop = ScaleCropImage(names=inputs, scale_resolution_in_name='scale_resolution', crop_resolution_in_name='crop_resolution', enable_crop_jitter_in_name='concept.image.enable_crop_jitter', crop_offset_out_name='crop_offset')

        modules = [scale_crop]

        return modules

    def _augmentation_modules(self, config: TrainConfig):
        inputs = ['image']
        image_inputs = ['image']
        if config.rlhf_enabled:
            if 'image_rejected' not in inputs:
                inputs.append('image_rejected')
            if 'image_rejected' not in image_inputs:
                image_inputs.append('image_rejected')
            if self._localized_dpo_enabled(config):
                inputs.append("dpo_mask_image")

        if config.masked_training or config.model_type.has_mask_input():
            inputs.append('mask')

        if config.model_type.has_depth_input():
            inputs.append('depth')

        if config.custom_conditioning_image:
            inputs.append('custom_conditioning_image')
            image_inputs.append('custom_conditioning_image')

        # image augmentations
        random_flip = RandomFlip(names=inputs, enabled_in_name='concept.image.enable_random_flip', fixed_enabled_in_name='concept.image.enable_fixed_flip')
        random_rotate = RandomRotate(names=inputs, enabled_in_name='concept.image.enable_random_rotate', fixed_enabled_in_name='concept.image.enable_fixed_rotate', max_angle_in_name='concept.image.random_rotate_max_angle')
        random_brightness = RandomBrightness(names=image_inputs, enabled_in_name='concept.image.enable_random_brightness', fixed_enabled_in_name='concept.image.enable_fixed_brightness', max_strength_in_name='concept.image.random_brightness_max_strength')
        random_contrast = RandomContrast(names=image_inputs, enabled_in_name='concept.image.enable_random_contrast', fixed_enabled_in_name='concept.image.enable_fixed_contrast', max_strength_in_name='concept.image.random_contrast_max_strength')
        random_saturation = RandomSaturation(names=image_inputs, enabled_in_name='concept.image.enable_random_saturation', fixed_enabled_in_name='concept.image.enable_fixed_saturation', max_strength_in_name='concept.image.random_saturation_max_strength')
        random_hue = RandomHue(names=image_inputs, enabled_in_name='concept.image.enable_random_hue', fixed_enabled_in_name='concept.image.enable_fixed_hue', max_strength_in_name='concept.image.random_hue_max_strength')

        # text augmentations
        drop_tags = DropTags(text_in_name='prompt', enabled_in_name='concept.text.tag_dropout_enable', probability_in_name='concept.text.tag_dropout_probability', dropout_mode_in_name='concept.text.tag_dropout_mode',
                             special_tags_in_name='concept.text.tag_dropout_special_tags', special_tag_mode_in_name='concept.text.tag_dropout_special_tags_mode', delimiter_in_name='concept.text.tag_delimiter',
                             keep_tags_count_in_name='concept.text.keep_tags_count', text_out_name='prompt', regex_enabled_in_name='concept.text.tag_dropout_special_tags_regex')
        caps_randomize = CapitalizeTags(text_in_name='prompt', enabled_in_name='concept.text.caps_randomize_enable', probability_in_name='concept.text.caps_randomize_probability',
                                        capitalize_mode_in_name='concept.text.caps_randomize_mode', delimiter_in_name='concept.text.tag_delimiter', convert_lowercase_in_name='concept.text.caps_randomize_lowercase', text_out_name='prompt')
        shuffle_tags = ShuffleTags(text_in_name='prompt', enabled_in_name='concept.text.enable_tag_shuffling', delimiter_in_name='concept.text.tag_delimiter', keep_tags_count_in_name='concept.text.keep_tags_count', text_out_name='prompt')

        modules = [
            random_flip,
            random_rotate,
            random_brightness,
            random_contrast,
            random_saturation,
            random_hue,
            drop_tags,
            caps_randomize,
            shuffle_tags,
        ]

        return modules

    def _inpainting_modules(self, config: TrainConfig):
        conditioning_image = GenerateMaskedConditioningImage(image_in_name='image', mask_in_name='mask', image_out_name='conditioning_image', image_range_min=0, image_range_max=1)
        select_conditioning_image = SelectFirstInput(in_names=['custom_conditioning_image', 'conditioning_image'], out_name='conditioning_image')

        modules = []

        if config.model_type.has_conditioning_image_input():
            modules.append(conditioning_image)
            modules.append(select_conditioning_image)

        return modules

    def _output_modules_from_out_names(
            self,
            model: BaseModel,
            model_setup: ModelSetupText2ImageMixin,
            output_names: list[str | tuple[str, str]],
            config: TrainConfig,
            before_cache_image_fun: Callable[[], None] | None = None,
            use_conditioning_image: bool = False,
            vae: AutoencoderKL | None = None,
            autocast_context: list[torch.autocast | None] = None,
            train_dtype: DataType | None = None,
    ):
        if config.rlhf_enabled:
            output_names = output_names + [
                'latent_image_rejected',
                'dpo_is_paired',
                'dpo_pair_key',
                'dpo_cache_mode',
                'image_path_rejected',
                'crop_resolution',
                'concept.image.resolution_override',
                'concept.image.enable_resolution_override',
            ]
            if self._localized_dpo_enabled(config):
                output_names = output_names + [
                    "dpo_mask",
                    "dpo_mask_path",
                ]

        if before_cache_image_fun is None:
            def prepare_vae():
                model.to(self.temp_device)
                model.vae_to(self.train_device)
                model.eval()
                torch_gc()
            before_cache_image_fun = prepare_vae

        sort_names = output_names + ['concept']
        if config.rlhf_enabled:
            for _rlhf_concept_key in ['concept.path', 'concept.image.enable_resolution_override', 'concept.image.resolution_override']:
                if _rlhf_concept_key not in sort_names:
                    sort_names.append(_rlhf_concept_key)

        output_names = output_names + [
            ('concept.loss_weight', 'loss_weight'),
            ('concept.type', 'concept_type'),
        ]
        if config.rlhf_enabled:
            output_names.append((
                'concept.dpo_objective',
                'dpo_objective',
            ))
            output_names.append((
                'concept.dpo_reference_mode',
                'dpo_reference_mode',
            ))
            output_names.append((
                'concept.dpo_streamed',
                'dpo_streamed',
            ))
            output_names.append((
                'concept.dpo_masked',
                'dpo_masked',
            ))
            output_names.append((
                'concept.dpo_mask_weight',
                'dpo_mask_weight',
            ))
            # The concept seed is stable across saves and does not depend on
            # the source path.  It therefore identifies the frozen adapter
            # snapshot belonging to this concept without putting that live
            # selector into either the image or text cache.
            output_names.append((
                'concept.seed',
                'dpo_reference_key',
            ))

        if config.validation:
            output_names.append(('concept.name', 'concept_name'))
            output_names.append(('concept.path', 'concept_path'))
            output_names.append(('concept.seed', 'concept_seed'))

        mask_remove = RandomLatentMaskRemove(
            latent_mask_name='latent_mask', latent_conditioning_image_name='latent_conditioning_image' if use_conditioning_image else None,
            replace_probability=config.unmasked_probability, vae=vae,
            possible_resolutions_in_name='possible_resolutions',
            autocast_contexts=autocast_context, dtype=train_dtype.torch_dtype(),
            before_cache_fun=before_cache_image_fun,
        )

        world_size = multi.world_size() if config.multi_gpu else 1  #world_size can be 1 for validation dataloader, even if multi.world_size() returns > 1
        if config.image_caching:
            batch_sorting = AspectBatchSorting(resolution_in_name='crop_resolution', names=sort_names, batch_size=config.batch_size * world_size)
            distributed_sampler = DistributedSampler(names=sort_names, world_size=world_size, rank=multi.rank())
        else:
            batch_sorting = InlineAspectBatchSorting(resolution_in_name='crop_resolution', names=sort_names, batch_size=config.batch_size * world_size)
            distributed_sampler = InlineDistributedSampler(names=sort_names, world_size=world_size, rank=multi.rank())

        output = OutputPipelineModule(names=output_names)

        modules = []

        if config.model_type.has_mask_input():
            modules.append(mask_remove)

        if (
            config.rlhf_enabled
            and bool(getattr(config, "rlhf_dpo_adaptive_dataset", False))
        ):
            # This random-access remap sits after cache/variation construction
            # but before aspect sorting. Epoch start indexes only cheap metadata.
            # The keep/replace draw itself happens live when the candidate is
            # actually requested for training, before its expensive payload is
            # fetched; replacement stays inside the same resolution bucket.
            adaptive_names = list(dict.fromkeys(
                name.split(".", 1)[0]
                for name in sort_names
            ))
            modules.append(AdaptiveDPODataset(
                names=adaptive_names,
                ema_decay=float(getattr(
                    config,
                    "rlhf_dpo_adaptive_dataset_ema",
                    0.8,
                )),
                min_observations=int(getattr(
                    config,
                    "rlhf_dpo_adaptive_dataset_min_observations",
                    3,
                )),
                min_keep_probability=float(getattr(
                    config,
                    "rlhf_dpo_adaptive_dataset_min_keep_probability",
                    0.25,
                )),
                replacement_power=float(getattr(
                    config,
                    "rlhf_dpo_adaptive_dataset_replacement_power",
                    2.0,
                )),
                default_objective=str(config.rlhf_dpo_objective),
            ))

        modules.append(batch_sorting)
        if world_size > 1:
            modules.append(distributed_sampler)

        modules.append(output)

        return modules

    def _cache_modules_from_names(
            self,
            model: BaseModel,
            model_setup: ModelSetupText2ImageMixin,
            image_split_names: list[str],
            image_aggregate_names: list[str],
            text_split_names: list[str],
            sort_names: list[str],
            config: TrainConfig,
            text_caching: bool,
            before_cache_image_fun: Callable[[], None] | None = None,
    ):
        image_cache_dir = os.path.join(config.cache_dir, "image")
        text_cache_dir = os.path.join(config.cache_dir, "text")

        if before_cache_image_fun is None:
            def prepare_vae():
                model.to(self.temp_device)
                model.vae_to(self.train_device)
                model.eval()
                torch_gc()
            before_cache_image_fun = prepare_vae

        def before_cache_text_fun():
            model_setup.prepare_text_caching(model, config)

        if config.rlhf_enabled:
            if 'latent_image_rejected' not in image_split_names:
                image_split_names = image_split_names + ['latent_image_rejected']
            for _dpo_name in [
                'dpo_is_paired',
                'dpo_pair_key',
                'dpo_cache_mode',
                'image_path_rejected',
                'crop_resolution',
            ]:
                if _dpo_name not in image_aggregate_names:
                    image_aggregate_names = image_aggregate_names + [_dpo_name]
                if _dpo_name not in sort_names:
                    sort_names = sort_names + [_dpo_name]

            if self._localized_dpo_enabled(config):
                if "dpo_mask" not in image_split_names:
                    image_split_names = image_split_names + ["dpo_mask"]
                if "dpo_mask_path" not in image_aggregate_names:
                    image_aggregate_names = image_aggregate_names + [
                        "dpo_mask_path"
                    ]
                for _localized_name in ("dpo_mask", "dpo_mask_path"):
                    if _localized_name not in sort_names:
                        sort_names = sort_names + [_localized_name]

        cache_only_concepts = (
            self._cache_only_concepts(config)
            if config.use_cache_only
            else None
        )
        if config.use_cache_only:
            # Existing cache manifests contain every tensor needed for
            # training, but older manifests intentionally omitted the prompt
            # string and concept object. The cache module reconstructs the
            # concept from the current config and supplies an empty display
            # prompt; cached tokens/hidden states remain the training input.
            for metadata_name in ["prompt", "concept"]:
                if metadata_name not in image_aggregate_names:
                    image_aggregate_names = image_aggregate_names + [metadata_name]

        image_encryption_source_names = ['image_path']
        if config.rlhf_enabled:
            image_encryption_source_names.append('image_path_rejected')
            if self._localized_dpo_enabled(config):
                image_encryption_source_names.append('dpo_mask_path')
        if config.masked_training:
            image_encryption_source_names.append('mask_path')
        if config.custom_conditioning_image:
            image_encryption_source_names.append('cond_path')

        encrypt_all_cache_files = (
            config.cache_encryption_scope == CacheEncryptionScope.ALL
        )
        # In cache-only mode both cache modules negotiate one on-disk layout.
        # This shared object makes the image cache authoritative and prevents
        # the text cache from independently reinterpreting concept balancing.
        cache_only_layout = {} if config.use_cache_only else None

        image_variation_groups = [
            'concept.path',
            'concept.seed',
            'concept.include_subdirectories',
            'concept.image',
            'concept.dpo_chosen_pattern',
            'concept.dpo_rejected_pattern',
        ]
        text_variation_groups = [
            'concept.path',
            'concept.seed',
            'concept.include_subdirectories',
            'concept.text',
            'concept.dpo_chosen_pattern',
            'concept.dpo_rejected_pattern',
        ]
        if self._localized_dpo_enabled(config):
            # Enabling/disabling localization changes the image-cache schema.
            # The numeric multiplier is live concept metadata and does not
            # alter either cached pixels/latents or cached text, so changing
            # it must not force an unnecessary cache rebuild.
            image_variation_groups.append('concept.dpo_masked')

        image_disk_cache = MultiResolutionDiskCache(
            cache_dir=image_cache_dir,
            split_names=image_split_names,
            aggregate_names=image_aggregate_names,
            resolution_variants_in_name='resolution_variants',
            selection_key_in_names=['dpo_pair_key', 'image_path'],
            variations_in_name='concept.image_variations',
            balancing_in_name='concept.balancing',
            balancing_strategy_in_name='concept.balancing_strategy',
            variations_group_in_name=image_variation_groups,
            group_enabled_in_name='concept.enabled',
            before_cache_fun=before_cache_image_fun,
            encrypted=config.cache_encryption_enabled,
            encryption_context="image",
            encrypt_all=encrypt_all_cache_files,
            encryption_source_path_in_name=image_encryption_source_names,
            cache_only=config.use_cache_only,
            cache_only_concepts=cache_only_concepts,
            cache_only_layout=cache_only_layout,
        )

        text_disk_cache = DiskCache(cache_dir=text_cache_dir, split_names=text_split_names, aggregate_names=[], variations_in_name='concept.text_variations', balancing_in_name='concept.balancing', balancing_strategy_in_name='concept.balancing_strategy',
                                    variations_group_in_name=text_variation_groups, group_enabled_in_name='concept.enabled', before_cache_fun=before_cache_text_fun,
                                    encrypted=config.cache_encryption_enabled, encryption_context="text",
                                    encrypt_all=encrypt_all_cache_files,
                                    encryption_source_path_in_name=['sample_prompt_path', 'concept.text.prompt_path'],
                                    cache_only=config.use_cache_only,
                                    cache_only_concepts=cache_only_concepts,
                                    cache_only_layout=cache_only_layout)

        modules = []

        if config.image_caching:
            modules.append(image_disk_cache)

            sort_names = [x for x in sort_names if x not in image_aggregate_names]
            sort_names = [x for x in sort_names if x not in image_split_names]

        if text_caching:
            modules.append(text_disk_cache)
            sort_names = [x for x in sort_names if x not in text_split_names]

        if len(sort_names) > 0:
            if config.use_cache_only:
                raise RuntimeError(
                    "Use Cache Only cannot provide these uncached pipeline "
                    f"values: {', '.join(sort_names)}"
                )
            variation_sorting = VariationSorting(names=sort_names, balancing_in_name='concept.balancing', balancing_strategy_in_name='concept.balancing_strategy',
                                                 variations_group_in_name=text_variation_groups, group_enabled_in_name='concept.enabled')

            modules.append(variation_sorting)

        return modules


    def _create_dataset(
            self,
            config: TrainConfig,
            model: BaseModel,
            model_setup: ModelSetupText2ImageMixin,
            train_progress: TrainProgress,
            is_validation: bool,
            aspect_bucketing_quantization: int,
            frame_dim_enabled: bool=False,
            allow_video_files: bool=False,
            vae_frame_dim: bool=False,
            supports_inpainting: bool=True, #TODO many models probably don't support inpainting, but this has been enabled in most dataloaders before refactoring, too
    ):
        if config.use_cache_only:
            if not config.image_caching or not config.text_caching:
                raise RuntimeError(
                    "Use Cache Only requires both Image Caching and Text "
                    "Caching to be enabled."
                )
            if config.train_text_encoder_or_embedding():
                raise RuntimeError(
                    "Use Cache Only cannot train a text encoder or embedding; "
                    "the source captions are intentionally unavailable. Use a "
                    "fully cached LoRA/fine-tune configuration."
                )

            # Cache constructors need the same train/validation concept subset
            # that MGDS will use after it builds the pipeline.
            config._cache_only_is_validation = is_validation
            cache_modules = self._cache_modules(config, model, model_setup)
            output_modules = self._output_modules(config, model, model_setup)
            return self._create_mgds(
                config,
                [cache_modules, output_modules],
                train_progress,
                is_validation,
            )

        enumerate_input = self._enumerate_input_modules(config, allow_videos=allow_video_files)
        load_input = self._load_input_modules(config, model.train_dtype, vae_frame_dim=vae_frame_dim)
        mask_augmentation = self._mask_augmentation_modules(config)
        aspect_bucketing_in = self._aspect_bucketing_in(config, aspect_bucketing_quantization, frame_dim_enabled)
        crop_modules = self._crop_modules(config)
        augmentation_modules = self._augmentation_modules(config)
        if supports_inpainting:
            inpainting_modules = self._inpainting_modules(config)
        preparation_modules = self._preparation_modules(config, model)
        if self._localized_dpo_enabled(config):
            preparation_modules = preparation_modules + [
                PrepareDPOLocalizedMask(
                    mask_in_name="dpo_mask_image",
                    latent_in_name="latent_image",
                    mask_out_name="dpo_mask",
                )
            ]
        if config.rlhf_enabled:
            rejected_vae_contexts = [model.autocast_context]
            rejected_vae_autocast_context = getattr(
                model,
                'vae_autocast_context',
                None,
            )
            if rejected_vae_autocast_context is not None:
                rejected_vae_contexts.append(rejected_vae_autocast_context)
            rejected_vae_dtype = getattr(
                model,
                'vae_train_dtype',
                model.train_dtype,
            ).torch_dtype()
            preparation_modules = preparation_modules + [EncodeDPORejectedOrDummyLatent(
                image_in_name='image_rejected',
                latent_image_in_name='latent_image',
                is_paired_in_name='dpo_is_paired',
                latent_out_name='latent_image_rejected',
                vae=getattr(model, "vae", None),
                autocast_contexts=rejected_vae_contexts,
                dtype=rejected_vae_dtype,
                dummy_mode='zeros',
            )]
        cache_modules = self._cache_modules(config, model, model_setup)
        output_modules = self._output_modules(config, model, model_setup)

        debug_modules = self._debug_modules(config, model)

        return self._create_mgds(
            config,
            [
                enumerate_input,
                load_input,
                mask_augmentation,
                aspect_bucketing_in,
                crop_modules,
                augmentation_modules
            ] + ([inpainting_modules] if supports_inpainting else []) + [
                preparation_modules,
                cache_modules,
                output_modules,

                debug_modules if config.debug_mode else None,
                # inserted before output_modules, which contains a sorting operation
            ],
            train_progress,
            is_validation
        )

    @abstractmethod
    def _preparation_modules(self, config: TrainConfig, model: BaseModel):
        pass

    @abstractmethod
    def _cache_modules(self, config: TrainConfig, model: BaseModel, model_setup: BaseModelSetup):
        pass

    @abstractmethod
    def _output_modules(self, config: TrainConfig, model: BaseModel, model_setup: BaseModelSetup):
        pass

    @abstractmethod
    def _debug_modules(self, config: TrainConfig, model: BaseModel):
        pass
