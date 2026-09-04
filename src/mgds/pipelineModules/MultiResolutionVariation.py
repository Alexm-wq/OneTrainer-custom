"""Encode a resolution selector into an MGDS variation without changing RNG seeds.

Python hashes integers modulo ``2**61 - 1`` on the 64-bit runtimes used by
OneTrainer. Adding a multiple of that modulus therefore keeps ``hash(variation)``
identical to the ordinary/base variation. MGDS modules that seed their random
state from the variation consequently use the same augmentation decisions for
all cached resolution variants.
"""

HASH_MODULUS = (1 << 61) - 1


def encode_multi_resolution_variation(base_variation: int, resolution_index: int) -> int:
    if isinstance(base_variation, bool) or not isinstance(base_variation, int) or base_variation < 0:
        raise ValueError(f"base_variation must be a non-negative int, got {base_variation!r}")
    if base_variation >= HASH_MODULUS:
        raise ValueError("base_variation is too large for multi-resolution encoding")
    if isinstance(resolution_index, bool) or not isinstance(resolution_index, int) or resolution_index < 0:
        raise ValueError(f"resolution_index must be a non-negative int, got {resolution_index!r}")
    return base_variation + (resolution_index + 1) * HASH_MODULUS


def decode_multi_resolution_variation(variation: int) -> tuple[int, int | None]:
    if isinstance(variation, bool) or not isinstance(variation, int):
        raise ValueError(f"variation must be an int, got {variation!r}")
    if variation < 0:
        raise ValueError(f"variation must be non-negative, got {variation}")
    if variation < HASH_MODULUS:
        return variation, None
    quotient, base_variation = divmod(variation, HASH_MODULUS)
    return base_variation, quotient - 1
