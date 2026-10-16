import hashlib
import json
import math
import os


def preference_pair_key(concept_path: str, stem: str) -> str:
    """Namespace relative pair IDs; the same pair keeps its ID at every resolution."""
    root = os.path.normcase(os.path.realpath(concept_path))
    return json.dumps([root, stem.replace("\\", "/")], ensure_ascii=False, separators=(",", ":"))


def preference_seed(pair_key: str, seed: int, draw: int = 0) -> int:
    payload = json.dumps([pair_key, int(seed), int(draw)], separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def is_validation_pair(pair_key: str, percentage: float, seed: int) -> bool:
    # Hash selection is independent of file order, repetitions and resolution.
    return preference_seed(pair_key, seed) / (2**31) < percentage / 100.0


def validate_rlhf_config(config):
    if not config.rlhf_enabled:
        return
    for name in ("rlhf_normal_weight", "rlhf_dpo_weight", "rlhf_chosen_weight"):
        value = getattr(config, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not any(getattr(config, name) > 0 for name in
               ("rlhf_normal_weight", "rlhf_dpo_weight", "rlhf_chosen_weight")):
        raise ValueError("At least one mixed RLHF weight must be positive")
    if not math.isfinite(config.rlhf_dpo_beta) or config.rlhf_dpo_beta <= 0:
        raise ValueError("DPO beta must be finite and positive")
    if not 0 <= config.rlhf_dpo_label_smoothing < 0.5:
        raise ValueError("DPO label smoothing must be in [0, 0.5)")
    if not 0 <= config.rlhf_dpo_validation_percentage < 100:
        raise ValueError("DPO validation percentage must be in [0, 100)")
    if config.rlhf_dpo_validation_samples < 1 or config.rlhf_dpo_patience_value < 1:
        raise ValueError("DPO validation samples and patience must be at least one")
    if config.rlhf_dpo_hard_pair_curriculum and not 0 < config.rlhf_dpo_hard_pair_curriculum_min_weight <= 1:
        raise ValueError("Curriculum minimum weight must be in (0, 1] so new pairs can learn")
