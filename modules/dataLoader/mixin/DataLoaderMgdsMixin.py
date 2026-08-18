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
            try:
                with open(config.concept_file_name, 'r') as f:
                    concepts = [
                        ConceptConfig.default_values().from_dict(c)
                        for c in json.load(f)
                    ]
            except (OSError, json.JSONDecodeError, TypeError):
                if not config.use_cache_only:
                    raise
                # Self-describing cache manifests supply the runtime concept
                # defaults. MGDS itself does not need an upstream concept row
                # when cache-only modules have no inputs.
                concepts = []

        # choose all validation concepts, or none of them, depending on is_validation
        concepts = [
            concept
            for concept in concepts
            if (
                ConceptType(
                    concept.get("type", ConceptType.STANDARD)
                    if isinstance(concept, dict)
                    else concept.type
                ) == ConceptType.VALIDATION
            ) == is_validation
        ]

        # convert before passing to MGDS
        concepts = [
            c if isinstance(c, dict) else c.to_dict()
            for c in concepts
        ]

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
