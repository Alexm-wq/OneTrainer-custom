from __future__ import annotations

import warnings
from collections.abc import Callable

from mgds.pipelineModules.DiskCache import DiskCache


class MultiResolutionDiskCache(DiskCache):
    """Compatibility bridge for OneTrainer's extended MGDS cache API.

    The preview branch vendors upstream MGDS, but OneTrainer still imports the
    extended cache class that existed in the custom MGDS checkout.  Keep the UI
    and ordinary cache path operational instead of failing during module import.

    This bridge intentionally does not pretend to implement encrypted or true
    multi-resolution cache storage.  Those modes fail or warn explicitly rather
    than silently weakening encryption or changing training semantics.
    """

    def __init__(
            self,
            cache_dir: str,
            split_names: list[str] | None = None,
            aggregate_names: list[str] | None = None,
            resolution_variants_in_name: str | None = None,
            selection_key_in_names: list[str] | None = None,
            variations_in_name: str | None = None,
            balancing_in_name: str | None = None,
            balancing_strategy_in_name: str | None = None,
            variations_group_in_name: str | list[str] | None = None,
            group_enabled_in_name: str | None = None,
            before_cache_fun: Callable[[], None] | None = None,
            encrypted: bool = False,
            encryption_context: str | None = None,
            encrypt_all: bool = False,
            encryption_source_path_in_name: str | None = None,
    ) -> None:
        if encrypted:
            raise RuntimeError(
                "Encrypted cache storage requires the original custom MGDS "
                "MultiResolutionDiskCache implementation; refusing to write an "
                "unencrypted fallback cache."
            )

        if resolution_variants_in_name is not None:
            warnings.warn(
                "The vendored MGDS package is missing the original custom "
                "MultiResolutionDiskCache implementation. Falling back to the "
                "standard DiskCache for this legacy VAE-finetune path.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.resolution_variants_in_name = resolution_variants_in_name
        self.selection_key_in_names = selection_key_in_names or []
        self.encryption_context = encryption_context
        self.encrypt_all = encrypt_all
        self.encryption_source_path_in_name = encryption_source_path_in_name

        super().__init__(
            cache_dir=cache_dir,
            split_names=split_names,
            aggregate_names=aggregate_names,
            variations_in_name=variations_in_name,
            balancing_in_name=balancing_in_name,
            balancing_strategy_in_name=balancing_strategy_in_name,
            variations_group_in_name=variations_group_in_name,
            group_enabled_in_name=group_enabled_in_name,
            before_cache_fun=before_cache_fun,
        )
