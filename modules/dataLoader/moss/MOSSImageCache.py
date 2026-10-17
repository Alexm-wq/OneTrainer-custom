"""Reuse existing layout cache payloads when replacing global row identities."""
import copy
import hashlib
import json
import os
from pathlib import Path

from modules.dataLoader.cache.MultiResolutionDiskCache import MultiResolutionDiskCache


class MOSSImageCache(MultiResolutionDiskCache):
    """The normal multi-resolution cache, with a metadata-only legacy upgrade."""

    def __init__(self, *args, inventory, **kwargs):
        super().__init__(*args, **kwargs)
        self.inventory = inventory
        self._upgraded = False

    def _upgrade_existing_groups(self):
        root = Path(self.cache_dir)
        if not root.is_dir():
            return
        candidates = sorted(path for path in root.iterdir() if path.is_dir())
        for group, indices in self.group_indices.items():
            rows = [self.inventory.rows[index] for index in indices]
            if not rows:
                continue
            current_values = [self._get_previous_item(0, name, indices[0])
                              for name in self.variations_group_in_names]
            target = root / group
            choices = [target] if target.is_dir() else candidates
            for candidate in choices:
                manifests = []
                for variation in self._variation_dirs(str(candidate)):
                    directory = candidate / f"variation-{variation}"
                    manifest = self._manifest(str(directory))
                    if manifest is not None:
                        manifests.append((directory, manifest))
                if not manifests:
                    continue
                first = manifests[0][1]
                items = first["items"]
                if len(items) != len(rows):
                    continue
                # New-format groups already have stable row identities.
                if all("moss_source_key" in v for item in items for v in item["variants"]):
                    continue
                old_concept = first.get("cache_layout", {}).get("concept") or {}
                old_fingerprint = old_concept.get("_moss_layout_fingerprint")
                if not old_fingerprint:
                    continue
                legacy_values = [
                    old_fingerprint if name == "concept._moss_layout_fingerprint" else value
                    for name, value in zip(self.variations_group_in_names, current_values, strict=True)
                ]
                # Includes source file signatures, image transforms, seed,
                # resolution and DPO configuration, just like the normal cache.
                if self._hash(legacy_values) != candidate.name:
                    continue
                upgraded = []
                for directory, manifest in manifests:
                    if len(manifest["items"]) != len(rows):
                        break
                    digest = hashlib.sha256(b"moss-layout-v1")
                    valid = True
                    for item, row in zip(manifest["items"], rows, strict=True):
                        variants = item["variants"]
                        if not variants or any(v.get("image_path") != row["path"] for v in variants):
                            valid = False
                            break
                        old_index = variants[0].get("moss_source_index")
                        if not isinstance(old_index, int):
                            valid = False
                            break
                        digest.update(json.dumps([old_index, row["path"], row["source"], row["raw"]]).encode())
                    if not valid or digest.hexdigest() != old_fingerprint:
                        break
                    manifest = copy.deepcopy(manifest)
                    for item, row in zip(manifest["items"], rows, strict=True):
                        for variant in item["variants"]:
                            variant["moss_source_key"] = row["source_key"]
                            # Retain the saved RNG index and resolution selection
                            # key: migration must not change existing prompts.
                    manifest["cache_layout"]["concept"] = rows[0]["concept"]
                    upgraded.append((directory, manifest))
                else:
                    for directory, manifest in upgraded:
                        self._save(manifest, str(directory / "aggregate.pt"), False)
                    if candidate != target:
                        os.replace(candidate, target)
                    print(f"[OT-MOSS-CACHE] Reused existing latents: {len(rows)} rows; "
                          "updated source mapping only.")
                    break

    def _refresh(self, out_variation):
        if not self._upgraded:
            if not self.variations_initialized:
                self._init_variations()
            self._upgrade_existing_groups()
            self._upgraded = True
        super()._refresh(out_variation)
