import json
from abc import ABCMeta

from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ConceptType import ConceptType
from modules.util.TrainProgress import TrainProgress

from mgds.MGDS import MGDS
from mgds.PipelineModule import PipelineState

import torch


class DataLoaderMgdsMixin(metaclass=ABCMeta):

    def _create_mgds(
            self,
            config: TrainConfig,
            definition: list,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        concepts = config.concepts
        if concepts is None:
            with open(config.concept_file_name, 'r') as f:
                concepts = [ConceptConfig.default_values().from_dict(c) for c in json.load(f)]

        # Select the split before augmentation, resolution expansion and caching.
        selected = []
        holdout = config.rlhf_enabled and config.rlhf_dpo_validation and config.rlhf_dpo_validation_percentage > 0
        for concept in concepts:
            explicit_validation = ConceptType(concept.type) == ConceptType.VALIDATION
            paired = bool(concept.dpo_chosen_pattern and concept.dpo_rejected_pattern)
            auto_split = holdout and paired and not explicit_validation
            if explicit_validation != is_validation and not (is_validation and auto_split):
                continue
            item = concept.to_dict()
            item["_dpo_split"] = ("validation" if is_validation else "train") if auto_split else ""
            item["_dpo_validation_percentage"] = config.rlhf_dpo_validation_percentage if auto_split else 0.0
            item["_dpo_validation_seed"] = config.rlhf_dpo_validation_seed if auto_split else 0
            if is_validation and config.rlhf_enabled:
                item["image_variations"] = item["text_variations"] = 1
                item["balancing"] = 1.0
                item["balancing_strategy"] = "REPEATS"
                for key in item["image"]:
                    if key == "enable_crop_jitter" or key.startswith("enable_random_"):
                        item["image"][key] = False
                for key in ("enable_tag_shuffling", "tag_dropout_enable", "caps_randomize_enable"):
                    item["text"][key] = False
            selected.append(item)
        concepts = selected

        settings = {
            "target_resolution": config.resolution,
            "target_frames": config.frames,
        }

        # Just defaults for now.
        ds = MGDS(
            torch.device(config.train_device),
            concepts,
            settings,
            definition,
            batch_size=config.batch_size, #local batch size
            state=PipelineState(config.caching_threads),
            initial_epoch=train_progress.epoch,
            initial_epoch_sample=train_progress.epoch_sample,
        )

        return ds
