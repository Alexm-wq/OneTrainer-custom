from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import random
import shutil
from typing import Any, Callable

import torch
from tqdm import tqdm

from mgds.PipelineModule import PipelineModule
from mgds.crypto import CACHE_PURPOSE, is_encrypted_file, secure_torch_load, secure_torch_save
from mgds.pipelineModuleTypes.SingleVariationRandomAccessPipelineModule import SingleVariationRandomAccessPipelineModule
from mgds.pipelineModules.MultiResolutionVariation import encode_multi_resolution_variation


class MultiResolutionDiskCache(PipelineModule, SingleVariationRandomAccessPipelineModule):
    """Cache every configured resolution per image and choose one per epoch.

    Resolution variants for a DPO pair live in the same payload, so chosen and
    rejected examples cannot drift onto different resolutions.
    """

    FORMAT_VERSION = 2
    CACHE_KIND = "onetrainer_multi_resolution"
    LAYOUT_VERSION = 2
    LAYOUT_FILENAME = "cache_layout.pt"

    def __init__(
            self,
            cache_dir: str,
            split_names: list[str] | None = None,
            aggregate_names: list[str] | None = None,
            resolution_variants_in_name: str = "resolution_variants",
            selection_key_in_names: list[str] | None = None,
            variations_in_name: str | None = None,
            balancing_in_name: str | None = None,
            balancing_strategy_in_name: str | None = None,
            variations_group_in_name: str | list[str] | None = None,
            group_enabled_in_name: str | None = None,
            before_cache_fun: Callable[[], None] | None = None,
            strict_cache_validation: bool | None = None,
            encrypted: bool = False,
            encryption_context: str = "image",
            encrypt_all: bool = True,
            encryption_source_path_in_name: str | list[str] | None = None,
            cache_only: bool = False,
            cache_only_concepts: list[dict] | None = None,
            cache_only_layout: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.cache_dir = cache_dir
        self.split_names = list(split_names or [])
        self.aggregate_names = list(aggregate_names or [])
        self.resolution_variants_in_name = resolution_variants_in_name
        self.selection_key_in_names = list(selection_key_in_names or ["dpo_pair_key", "image_path"])
        self.variations_in_name = variations_in_name
        self.balancing_in_name = balancing_in_name
        self.balancing_strategy_in_name = balancing_strategy_in_name
        self.variations_group_in_names = (
            [variations_group_in_name] if isinstance(variations_group_in_name, str)
            else list(variations_group_in_name or [])
        )
        self.group_enabled_in_name = group_enabled_in_name
        self.before_cache_fun = before_cache_fun or (lambda: None)
        self.strict_cache_validation = bool(strict_cache_validation) if strict_cache_validation is not None else False
        self.encrypted = bool(encrypted)
        self.encrypt_all = bool(encrypt_all)
        self.encryption_source_path_in_names = (
            [encryption_source_path_in_name] if isinstance(encryption_source_path_in_name, str)
            else list(encryption_source_path_in_name or [])
        )
        self.encryption_purpose = CACHE_PURPOSE + b"/" + str(encryption_context or "image").encode()
        self.cache_only = bool(cache_only)
        self.cache_only_concepts = list(cache_only_concepts or [])
        self.cache_only_layout = cache_only_layout
        self.group_variations: dict[str, int] = {}
        self.group_indices: dict[str, list[int]] = {}
        self.group_output_samples: dict[str, int] = {}
        self.aggregate_cache: dict[str, list[Any]] = {}
        self._cache_only_records: dict[str, dict[str, Any]] = {}
        self.variations_initialized = False

    @staticmethod
    def _hash(value: Any) -> str:
        raw = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get_inputs(self) -> list[str]:
        if self.cache_only:
            return []
        names = self.split_names + self.aggregate_names + [self.resolution_variants_in_name]
        names += [n for n in self.selection_key_in_names if n not in names]
        if self.variations_in_name:
            names += [self.variations_in_name, self.balancing_in_name,
                      self.balancing_strategy_in_name, *self.variations_group_in_names]
            if self.group_enabled_in_name:
                names.append(self.group_enabled_in_name)
        names += self.encryption_source_path_in_names
        return list(dict.fromkeys(n for n in names if n is not None))

    def get_outputs(self) -> list[str]:
        return self.split_names + self.aggregate_names

    def length(self) -> int:
        if not self.variations_initialized:
            if self.cache_only:
                self._init_cache_only()
            elif not self._load_layout():
                names = self.split_names or self.aggregate_names
                if not names:
                    raise RuntimeError("MultiResolutionDiskCache requires cached names")
                return self._get_previous_length(names[0])
        return sum(self.group_output_samples.values())

    def _save(self, value: Any, path: str, encrypted: bool) -> None:
        secure_torch_save(value, os.path.realpath(path), encrypted=encrypted,
                          purpose=self.encryption_purpose)

    def _load(self, path: str) -> Any:
        return secure_torch_load(os.path.realpath(path), purpose=self.encryption_purpose,
                                 map_location="cpu")

    def _source_encrypted(self, variation: int, index: int) -> bool:
        if not self.encrypted:
            return False
        if self.encrypt_all:
            return True
        for name in self.encryption_source_path_in_names:
            try:
                value = self._get_previous_item(variation, name, index)
            except Exception:
                continue
            for path in value if isinstance(value, (list, tuple, set)) else [value]:
                if isinstance(path, (str, os.PathLike)) and path and is_encrypted_file(path):
                    return True
        return False

    @staticmethod
    def _clone(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().clone().cpu()
        if isinstance(value, dict):
            return {k: MultiResolutionDiskCache._clone(v) for k, v in value.items()}
        if isinstance(value, list):
            return [MultiResolutionDiskCache._clone(v) for v in value]
        if isinstance(value, tuple):
            return tuple(MultiResolutionDiskCache._clone(v) for v in value)
        return value

    def _to_device(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(self.pipeline.device, non_blocking=True)
        if isinstance(value, dict):
            return {k: self._to_device(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_device(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._to_device(v) for v in value)
        return value

    @staticmethod
    def _normalise_variants(value: Any) -> list[Any]:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list):
            value = [value]
        result, seen = [], set()
        for item in value:
            if isinstance(item, torch.Tensor):
                item = item.detach().cpu().tolist()
            if isinstance(item, tuple):
                item = list(item)
            key = json.dumps(item, sort_keys=True, separators=(",", ":"), default=str)
            if key not in seen:
                seen.add(key)
                result.append(item)
        if not result:
            raise RuntimeError("Resolution variant list is empty")
        return result

    def _concept(self, variation: int, index: int) -> dict | None:
        try:
            value = self._get_previous_item(variation, "concept", index)
        except Exception:
            return None
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        return dict(value) if isinstance(value, dict) else None

    @staticmethod
    def _logical_id(concept: dict | None, group: str) -> str:
        if concept:
            if concept.get("seed") not in (None, ""):
                return f"seed:{concept['seed']}"
            if concept.get("path"):
                return "path:" + hashlib.sha256(str(concept["path"]).encode()).hexdigest()
        return f"group:{group}"

    def _sample_identity(self, variation: int, index: int, aggregate_variants: list[dict]) -> str:
        first = aggregate_variants[0] if aggregate_variants else {}
        values = []
        for name in self.selection_key_in_names:
            value = first.get(name)
            if value in (None, "", False):
                try:
                    value = self._get_previous_item(variation, name, index)
                except Exception:
                    value = None
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().tolist()
            elif isinstance(value, os.PathLike):
                value = os.fspath(value)
            values.append([name, value])
        if not any(v[1] not in (None, "", False) for v in values):
            values = [["__index__", index]]
        return self._hash(values)

    def _selection_key(self, group: str, variation: int, index: int, aggregate_variants: list[dict]) -> str:
        first = aggregate_variants[0] if aggregate_variants else {}
        parts = [group]
        for name in self.selection_key_in_names:
            value = first.get(name)
            if value in (None, "", False):
                try:
                    value = self._get_previous_item(variation, name, index)
                except Exception:
                    value = None
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().tolist()
            if value not in (None, "", False):
                parts.append(f"{name}={value}")
        if len(parts) == 1:
            parts.append(f"index={index}")
        return "|".join(map(str, parts))

    def _layout_config(self) -> dict[str, Any]:
        return {
            "split_names": self.split_names,
            "aggregate_names": self.aggregate_names,
            "resolution_variants_in_name": self.resolution_variants_in_name,
            "selection_key_in_names": self.selection_key_in_names,
            "variations_in_name": self.variations_in_name,
            "balancing_in_name": self.balancing_in_name,
            "balancing_strategy_in_name": self.balancing_strategy_in_name,
            "variations_group_in_names": self.variations_group_in_names,
            "group_enabled_in_name": self.group_enabled_in_name,
        }

    def _layout_path(self) -> str:
        return os.path.join(self.cache_dir, self.LAYOUT_FILENAME)

    def _load_layout(self) -> bool:
        path = self._layout_path()
        if not os.path.isfile(path):
            return False
        try:
            layout = self._load(path)
        except Exception:
            return False
        if not isinstance(layout, dict) or layout.get("layout_version") != self.LAYOUT_VERSION:
            return False
        if layout.get("cache_kind") != self.CACHE_KIND or layout.get("config") != self._layout_config():
            return False
        gv, gi, go = layout.get("group_variations"), layout.get("group_indices"), layout.get("group_output_samples")
        if not all(isinstance(x, dict) for x in (gv, gi, go)):
            return False
        self.group_variations = {str(k): int(v) for k, v in gv.items()}
        self.group_indices = {str(k): list(v) for k, v in gi.items()}
        self.group_output_samples = {str(k): int(v) for k, v in go.items()}
        self.variations_initialized = True
        print("[OT-MULTIRES-CACHE] Loaded persisted cache layout; skipping upstream dataset traversal.")
        return True

    def _write_layout(self) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        self._save({
            "layout_version": self.LAYOUT_VERSION,
            "cache_kind": self.CACHE_KIND,
            "config": self._layout_config(),
            "group_variations": self.group_variations,
            "group_indices": self.group_indices,
            "group_output_samples": self.group_output_samples,
        }, self._layout_path(), self.encrypted)

    def _init_variations(self) -> None:
        if self.cache_only:
            self._init_cache_only()
            return
        if self._load_layout():
            return
        if self.variations_in_name:
            gv, gi, gb, gs = {}, {}, {}, {}
            for i in range(self._get_previous_length(self.variations_in_name)):
                if self.group_enabled_in_name and not self._get_previous_item(0, self.group_enabled_in_name, i):
                    continue
                key = self._hash([self._get_previous_item(0, n, i) for n in self.variations_group_in_names])
                gv.setdefault(key, int(self._get_previous_item(0, self.variations_in_name, i)))
                gi.setdefault(key, []).append(i)
                gb.setdefault(key, float(self._get_previous_item(0, self.balancing_in_name, i)))
                gs.setdefault(key, self._get_previous_item(0, self.balancing_strategy_in_name, i))
            go = {}
            for key, balancing in gb.items():
                strategy = getattr(gs[key], "value", gs[key])
                if strategy == "REPEATS":
                    go[key] = int(math.floor(len(gi[key]) * balancing))
                elif strategy == "SAMPLES":
                    go[key] = int(balancing)
                else:
                    raise RuntimeError(f"Unknown balancing strategy: {strategy}")
        else:
            name = (self.split_names or self.aggregate_names)[0]
            gv = {"": 1}
            gi = {"": list(range(self._get_previous_length(name)))}
            go = {"": len(gi[""])}
        self.group_variations, self.group_indices, self.group_output_samples = gv, gi, go
        self.variations_initialized = True
        self._write_layout()

    def _cache_dir(self, group: str, variation: int) -> str:
        return os.path.join(self.cache_dir, group,
                            f"variation-{variation % self.group_variations[group]}")

    def _manifest(self, directory: str) -> dict | None:
        path = os.path.join(directory, "aggregate.pt")
        if not os.path.isfile(path):
            return None
        try:
            value = self._load(path)
        except Exception:
            return None
        if not isinstance(value, dict) or value.get("format_version") != self.FORMAT_VERSION:
            return None
        if value.get("cache_kind") != self.CACHE_KIND or not isinstance(value.get("items"), list):
            return None
        return value

    def _build(self, group: str, variation: int) -> None:
        directory = self._cache_dir(group, variation)
        if os.path.isdir(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)
        size = len(self.group_indices[group])
        aggregate, item_encryption = [None] * size, [False] * size
        with tqdm(total=size, smoothing=0.1, desc="caching multi-resolution") as bar:
            def fn(pos: int, index: int, current_device: int | None):
                if torch.cuda.is_available() and current_device is not None:
                    torch.cuda.set_device(current_device)
                descriptors = self._normalise_variants(
                    self._get_previous_item(variation, self.resolution_variants_in_name, index)
                )
                split_variants, aggregate_variants = [], []
                with torch.no_grad():
                    for r in range(len(descriptors)):
                        encoded = encode_multi_resolution_variation(variation, r)
                        split = {n: self._clone(self._get_previous_item(encoded, n, index)) for n in self.split_names}
                        agg = {n: self._clone(self._get_previous_item(encoded, n, index)) for n in self.aggregate_names}
                        chosen, rejected = split.get("latent_image"), split.get("latent_image_rejected")
                        if isinstance(chosen, torch.Tensor) and isinstance(rejected, torch.Tensor) and chosen.shape != rejected.shape:
                            raise RuntimeError(
                                f"DPO chosen/rejected cache shape mismatch: chosen={tuple(chosen.shape)}, rejected={tuple(rejected.shape)}"
                            )
                        split_variants.append(split)
                        aggregate_variants.append(agg)
                enc = self._source_encrypted(variation, index)
                self._save({"format_version": self.FORMAT_VERSION, "cache_kind": self.CACHE_KIND,
                            "resolution_variants": descriptors, "variants": split_variants},
                           os.path.join(directory, f"{pos}.pt"), enc)
                aggregate[pos] = {
                    "resolution_variants": descriptors,
                    "selection_key": self._selection_key(group, variation, index, aggregate_variants),
                    "sample_identity": self._sample_identity(variation, index, aggregate_variants),
                    "variants": aggregate_variants,
                }
                item_encryption[pos] = enc
            device = torch.cuda.current_device() if torch.cuda.is_available() else None
            futures = [self._state.executor.submit(fn, p, i, device)
                       for p, i in enumerate(self.group_indices[group])]
            for n, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    future.result()
                except Exception:
                    self._state.executor.shutdown(wait=True, cancel_futures=True)
                    raise
                if n % 250 == 0:
                    self._torch_gc()
                bar.update(1)
        first = self.group_indices[group][0] if size else 0
        concept = self._concept(variation, first) if size else None
        manifest = {
            "format_version": self.FORMAT_VERSION,
            "cache_kind": self.CACHE_KIND,
            "size": size,
            "items": aggregate,
            "cache_layout": {
                "logical_id": self._logical_id(concept, group),
                "output_samples": int(self.group_output_samples[group]),
                "sample_identities": [str(x["sample_identity"]) for x in aggregate],
                "concept": concept,
            },
        }
        self._save(manifest, os.path.join(directory, "aggregate.pt"),
                   bool(self.encrypted and (self.encrypt_all or any(item_encryption))))

    def _refresh(self, out_variation: int) -> None:
        if not self.variations_initialized:
            self._init_variations()
        if self.cache_only:
            self._load_cache_only()
            return
        self.aggregate_cache = {g: [None] * v for g, v in self.group_variations.items()}
        prepared = False
        for group, variations in self.group_variations.items():
            count = self.group_output_samples[group]
            if count <= 0:
                continue
            size = len(self.group_indices[group])
            first = (count * out_variation) // size
            last = (count * (out_variation + 1) - 1) // size
            for variation in dict.fromkeys(x % variations for x in range(first, last + 1)):
                directory = self._cache_dir(group, variation)
                manifest = self._manifest(directory)
                valid = manifest is not None and len(manifest["items"]) == size
                if valid and self.strict_cache_validation:
                    valid = all(os.path.isfile(os.path.join(directory, f"{i}.pt")) for i in range(size))
                if not valid:
                    if not prepared:
                        prepared = True
                        self.before_cache_fun()
                    self._build(group, variation)
                    manifest = self._manifest(directory)
                if manifest is None:
                    raise RuntimeError(f"Invalid multi-resolution cache manifest: {directory}")
                self.aggregate_cache[group][variation] = manifest["items"]

    @staticmethod
    def _variation_dirs(group_dir: str) -> list[int]:
        if not os.path.isdir(group_dir):
            return []
        result = []
        for entry in os.scandir(group_dir):
            if entry.is_dir() and entry.name.startswith("variation-"):
                try:
                    result.append(int(entry.name[10:]))
                except ValueError:
                    pass
        return sorted(set(result))

    def _init_cache_only(self) -> None:
        if self.variations_initialized:
            return
        if not os.path.isdir(self.cache_dir):
            raise RuntimeError(f"Use Cache Only: missing image cache {self.cache_dir}")
        records = []
        groups = [""] + sorted(e.name for e in os.scandir(self.cache_dir)
                               if e.is_dir() and not e.name.startswith("variation-"))
        for group in groups:
            group_dir = os.path.join(self.cache_dir, group) if group else self.cache_dir
            physical = self._variation_dirs(group_dir)
            if not physical:
                continue
            manifest = self._manifest(os.path.join(group_dir, f"variation-{physical[0]}"))
            if manifest is None or not isinstance(manifest.get("cache_layout"), dict):
                raise RuntimeError("Use Cache Only requires current strict multi-resolution manifests; rebuild cache.")
            layout = manifest["cache_layout"]
            identities = layout.get("sample_identities")
            if not isinstance(identities, list) or len(identities) != len(manifest["items"]):
                raise RuntimeError("Use Cache Only: invalid image sample identities")
            records.append({"group_key": group, "variations": physical,
                            "size": len(identities), "sample_identities": identities,
                            "logical_id": str(layout.get("logical_id", "")),
                            "output_samples": int(layout.get("output_samples", len(identities))),
                            "concept": layout.get("concept")})
        if not records:
            raise RuntimeError("Use Cache Only: no complete image cache groups found")
        ids = [r["logical_id"] for r in records]
        if any(not x for x in ids) or len(ids) != len(set(ids)):
            raise RuntimeError("Use Cache Only: image cache logical IDs are missing or duplicated")
        self._cache_only_records = {r["group_key"]: r for r in records}
        self.group_variations = {r["group_key"]: len(r["variations"]) for r in records}
        self.group_indices = {r["group_key"]: list(range(r["size"])) for r in records}
        self.group_output_samples = {r["group_key"]: r["output_samples"] for r in records}
        self.variations_initialized = True
        if self.cache_only_layout is not None:
            self.cache_only_layout["image_slots"] = records
            self.cache_only_layout["image_cache"] = self

    def _load_cache_only(self) -> None:
        self.aggregate_cache = {g: [None] * v for g, v in self.group_variations.items()}
        for group, count in self.group_variations.items():
            record = self._cache_only_records[group]
            group_dir = os.path.join(self.cache_dir, group) if group else self.cache_dir
            for logical in range(count):
                physical = record["variations"][logical]
                manifest = self._manifest(os.path.join(group_dir, f"variation-{physical}"))
                if manifest is None:
                    raise RuntimeError("Use Cache Only: invalid image cache manifest")
                self.aggregate_cache[group][logical] = manifest["items"]

    def _index(self, out_variation: int, out_index: int) -> tuple[str, int, int]:
        offset = 0
        for group, count in self.group_output_samples.items():
            if out_index >= offset + count:
                offset += count
                continue
            size = len(self.group_indices[group])
            local = out_index - offset + out_variation * count
            return group, (local // size) % self.group_variations[group], local % size
        raise IndexError(out_index)

    @staticmethod
    def _raw_permutation(key: str, cycle: int, count: int) -> list[int]:
        digest = hashlib.sha256(f"{key}|resolution-cycle={cycle}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:16], "big"))
        order = list(range(count))
        rng.shuffle(order)
        return order

    @classmethod
    def _selected_resolution_index(cls, key: str, epoch: int, count: int) -> int:
        if count <= 1:
            return 0
        if count == 2:
            first = hashlib.sha256(f"{key}|two-resolution-start".encode()).digest()[0] & 1
            return first ^ (epoch & 1)
        cycle, pos = divmod(epoch, count)
        order = cls._raw_permutation(key, cycle, count)
        if cycle:
            previous = cls._raw_permutation(key, cycle - 1, count)[-1]
            if order[0] == previous:
                order[0], order[1] = order[1], order[0]
        return order[pos]

    def start(self, out_variation: int):
        self._refresh(out_variation)

    def get_item(self, index: int, requested_name: str = None) -> dict:
        group, variation, pos = self._index(self.current_variation, index)
        entry = self.aggregate_cache[group][variation][pos]
        descriptors = entry["resolution_variants"]
        selected_index = self._selected_resolution_index(
            entry["selection_key"], self.current_variation, len(descriptors)
        )
        if requested_name in self.aggregate_names:
            selected = entry["variants"][selected_index]
            result = {}
            for name in self.aggregate_names:
                if name in selected:
                    result[name] = self._to_device(selected[name])
                elif self.cache_only and name == "prompt":
                    result[name] = ""
                elif self.cache_only and name == "concept":
                    result[name] = self._cache_only_records[group].get("concept")
            return result
        if requested_name in self.split_names:
            if self.cache_only:
                record = self._cache_only_records[group]
                group_dir = os.path.join(self.cache_dir, group) if group else self.cache_dir
                directory = os.path.join(group_dir, f"variation-{record['variations'][variation]}")
            else:
                directory = self._cache_dir(group, variation)
            payload = self._load(os.path.join(directory, f"{pos}.pt"))
            if not isinstance(payload, dict) or payload.get("format_version") != self.FORMAT_VERSION:
                raise RuntimeError("Invalid multi-resolution cache item")
            if payload.get("resolution_variants") != descriptors:
                raise RuntimeError("Multi-resolution aggregate/split mismatch")
            selected = payload["variants"][selected_index]
            return {name: self._to_device(selected[name]) for name in self.split_names if name in selected}
        return {}
