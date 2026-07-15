from modules.util.dpo_pattern_util import match_chosen, validate_dpo_patterns

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class FilterDPOChosenPaths(
    PipelineModule,
    RandomAccessPipelineModule,
):
    """Mixed RLHF behavior:
    - Concepts without DPO patterns are normal concepts and pass through untouched.
    - Concepts with DPO patterns keep only chosen/winner images.
    - Rejected/loser images are removed from the chosen stream and loaded later.
    """

    def __init__(self, path_in_name: str = 'image_path', concept_in_name: str = 'concept'):
        super().__init__()
        self.path_in_name = path_in_name
        self.concept_in_name = concept_in_name
        self._kept: list[int] = []

    def length(self) -> int:
        return len(self._kept)

    def get_inputs(self) -> list[str]:
        return [self.path_in_name, self.concept_in_name]

    def get_outputs(self) -> list[str]:
        return [self.path_in_name, self.concept_in_name]

    def start(self, variation: int):
        self._kept = []
        kept_per_dpo_concept: dict[str, int] = {}
        normal_count = 0
        dpo_count = 0

        total = self._get_previous_length(self.path_in_name)
        for index in range(total):
            concept = self._get_previous_item(variation, self.concept_in_name, index)
            chosen_pattern = concept.get('dpo_chosen_pattern', '') or ''
            rejected_pattern = concept.get('dpo_rejected_pattern', '') or ''

            # Normal concept: no DPO patterns. Keep it.
            if not chosen_pattern and not rejected_pattern:
                self._kept.append(index)
                normal_count += 1
                continue

            validate_dpo_patterns(chosen_pattern, rejected_pattern)

            if '/' in chosen_pattern.replace('\\', '/') and not concept.get('include_subdirectories', False):
                raise RuntimeError(
                    f"DPO concept '{concept.get('name') or concept.get('path')}' uses the pattern "
                    f"'{chosen_pattern}' but 'Include Subdirectories' is disabled, so its winner images "
                    "are never collected."
                )

            concept_path = concept['path']
            kept_per_dpo_concept.setdefault(concept_path, 0)
            image_path = self._get_previous_item(variation, self.path_in_name, index)

            if match_chosen(chosen_pattern, concept_path, image_path) is not None:
                self._kept.append(index)
                kept_per_dpo_concept[concept_path] += 1
                dpo_count += 1

        empty = [path for path, count in kept_per_dpo_concept.items() if count == 0]
        if empty:
            raise RuntimeError(
                "No images matched the DPO chosen pattern for: " + ", ".join(sorted(empty))
            )

        print(
            f"[OT-MIXED-RLHF] FilterDPOChosenPaths kept normal_rows={normal_count}, "
            f"dpo_chosen_rows={dpo_count}, total_kept={len(self._kept)}"
        )

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        previous_index = self._kept[index]
        return {
            self.path_in_name: self._get_previous_item(variation, self.path_in_name, previous_index),
            self.concept_in_name: self._get_previous_item(variation, self.concept_in_name, previous_index),
        }
