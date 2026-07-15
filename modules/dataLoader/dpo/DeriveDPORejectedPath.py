import os

from modules.util.dpo_curation_util import resolve_aspect_ratio
from modules.util.dpo_pattern_util import build_rejected_index, match_chosen, resolve_rejected, validate_dpo_patterns

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class DeriveDPORejectedPath(
    PipelineModule,
    RandomAccessPipelineModule,
):
    """Mixed RLHF behavior:
    - Normal rows emit image_path_rejected = image_path so the global schema is homogeneous.
    - DPO rows emit the actual rejected/loser image path.
    - dpo_is_paired tells the trainer which rows use DPO loss.
    """

    def __init__(
            self,
            path_in_name: str = 'image_path',
            concept_in_name: str = 'concept',
            rejected_path_out_name: str = 'image_path_rejected',
            dpo_is_paired_out_name: str = 'dpo_is_paired',
            dpo_pair_key_out_name: str = 'dpo_pair_key',
            dpo_cache_mode_out_name: str = 'dpo_cache_mode',
    ):
        super().__init__()
        self.path_in_name = path_in_name
        self.concept_in_name = concept_in_name
        self.rejected_path_out_name = rejected_path_out_name
        self.dpo_is_paired_out_name = dpo_is_paired_out_name
        self.dpo_pair_key_out_name = dpo_pair_key_out_name
        self.dpo_cache_mode_out_name = dpo_cache_mode_out_name

        self._rejected_paths: list[str] = []
        self._is_paired: list[bool] = []
        self._pair_keys: list[str] = []
        self._cache_modes: list[str] = []

    def length(self) -> int:
        return self._get_previous_length(self.path_in_name)

    def get_inputs(self) -> list[str]:
        return [self.path_in_name, self.concept_in_name]

    def get_outputs(self) -> list[str]:
        return [
            self.rejected_path_out_name,
            self.dpo_is_paired_out_name,
            self.dpo_pair_key_out_name,
            self.dpo_cache_mode_out_name,
        ]

    def start(self, variation: int):
        self._rejected_paths = []
        self._is_paired = []
        self._pair_keys = []
        self._cache_modes = []

        index_cache: dict[str, dict[str, list[str]]] = {}
        missing: list[str] = []
        aspect_mismatches: list[str] = []
        normal_count = 0
        dpo_count = 0

        for index in range(self._get_previous_length(self.path_in_name)):
            concept = self._get_previous_item(variation, self.concept_in_name, index)
            image_path = self._get_previous_item(variation, self.path_in_name, index)

            chosen_pattern = concept.get('dpo_chosen_pattern', '') or ''
            rejected_pattern = concept.get('dpo_rejected_pattern', '') or ''

            # Normal concept. Use same image as dummy rejected path, but mark row as non-DPO.
            if not chosen_pattern and not rejected_pattern:
                self._rejected_paths.append(image_path)
                self._is_paired.append(False)
                self._pair_keys.append('')
                self._cache_modes.append('normal')
                normal_count += 1
                continue

            validate_dpo_patterns(chosen_pattern, rejected_pattern)

            concept_path = concept['path']
            stem = match_chosen(chosen_pattern, concept_path, image_path)

            if stem is None:
                missing.append(
                    f"Chosen image did not match chosen pattern: image='{image_path}', pattern='{chosen_pattern}'"
                )
                self._rejected_paths.append(image_path)
                self._is_paired.append(False)
                self._pair_keys.append('')
                self._cache_modes.append('invalid')
                continue

            if concept_path not in index_cache:
                index_cache[concept_path] = build_rejected_index(
                    concept_path, concept.get('include_subdirectories', False)
                )

            try:
                rejected_path = resolve_rejected(
                    rejected_pattern, concept_path, stem,
                    os.path.splitext(image_path)[1], index_cache[concept_path],
                )
            except FileNotFoundError as e:
                missing.append(str(e))
                self._rejected_paths.append(image_path)
                self._is_paired.append(False)
                self._pair_keys.append(stem)
                self._cache_modes.append('missing')
                continue

            self._rejected_paths.append(rejected_path)
            self._is_paired.append(True)
            self._pair_keys.append(stem)
            self._cache_modes.append('dpo')
            dpo_count += 1

            if resolve_aspect_ratio("", image_path) != resolve_aspect_ratio("", rejected_path):
                aspect_mismatches.append(f"{image_path} vs {rejected_path}")

        if missing:
            raise RuntimeError(
                f"RLHF DPO: {len(missing)} chosen images have no rejected match. "
                "First errors: " + " | ".join(missing[:10])
            )

        if aspect_mismatches:
            print(
                f"WARNING: {len(aspect_mismatches)} DPO pairs land in different aspect buckets; "
                "the rejected image will be scaled and cropped to the chosen image's bucket. "
                "First pairs: " + " | ".join(aspect_mismatches[:10])
            )

        print(
            f"[OT-MIXED-RLHF] DeriveDPORejectedPath normal_rows={normal_count}, dpo_rows={dpo_count}"
        )

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return {
            self.rejected_path_out_name: self._rejected_paths[index],
            self.dpo_is_paired_out_name: self._is_paired[index],
            self.dpo_pair_key_out_name: self._pair_keys[index],
            self.dpo_cache_mode_out_name: self._cache_modes[index],
        }
