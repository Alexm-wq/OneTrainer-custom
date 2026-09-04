from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import shutil
from typing import Any, Callable

import torch
from tqdm import tqdm

from mgds.PipelineModule import PipelineModule
from mgds.crypto import CACHE_PURPOSE, is_encrypted_file, secure_torch_load, secure_torch_save
from mgds.pipelineModuleTypes.SingleVariationRandomAccessPipelineModule import SingleVariationRandomAccessPipelineModule


class DiskCache(PipelineModule, SingleVariationRandomAccessPipelineModule):
    """Extended OneTrainer MGDS cache with authenticated cache encryption."""

    FORMAT_VERSION = 3
    CACHE_KIND = "onetrainer_disk_cache"

    def __init__(
            self,
            cache_dir: str,
            split_names: list[str] | None = None,
            aggregate_names: list[str] | None = None,
            variations_in_name: str | None = None,
            balancing_in_name: str | None = None,
            balancing_strategy_in_name: str | None = None,
            variations_group_in_name: str | list[str] | None = None,
            group_enabled_in_name: str | None = None,
            before_cache_fun: Callable[[], None] | None = None,
            encrypted: bool = False,
            encryption_context: str = "generic",
            encrypt_all: bool = True,
            encryption_source_path_in_name: str | list[str] | None = None,
            identity_in_names: list[str] | None = None,
            cache_only: bool = False,
            cache_only_concepts: list[dict] | None = None,
            cache_only_layout: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.cache_dir = cache_dir
        self.split_names = list(split_names or [])
        self.aggregate_names = list(aggregate_names or [])
        self.variations_in_name = variations_in_name
        self.balancing_in_name = balancing_in_name
        self.balancing_strategy_in_name = balancing_strategy_in_name
        self.variations_group_in_names = (
            [variations_group_in_name] if isinstance(variations_group_in_name, str)
            else list(variations_group_in_name or [])
        )
        self.group_enabled_in_name = group_enabled_in_name
        self.before_cache_fun = before_cache_fun or (lambda: None)
        self.encrypted = bool(encrypted)
        self.encrypt_all = bool(encrypt_all)
        self.encryption_source_path_in_names = (
            [encryption_source_path_in_name] if isinstance(encryption_source_path_in_name, str)
            else list(encryption_source_path_in_name or [])
        )
        self.identity_in_names = list(identity_in_names or [])
        self.encryption_purpose = CACHE_PURPOSE + b"/" + str(encryption_context or "generic").encode()
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

    def length(self) -> int:
        if not self.variations_initialized:
            if self.cache_only:
                self._init_cache_only()
            else:
                names = self.split_names or self.aggregate_names
                if not names:
                    raise RuntimeError("DiskCache requires at least one cached name")
                return self._get_previous_length(names[0])
        return sum(self.group_output_samples.values())

    def get_inputs(self) -> list[str]:
        if self.cache_only:
            return []
        names = self.split_names + self.aggregate_names
        if self.variations_in_name:
            names += [self.variations_in_name, self.balancing_in_name,
                      self.balancing_strategy_in_name, *self.variations_group_in_names]
            if self.group_enabled_in_name:
                names.append(self.group_enabled_in_name)
        names += self.encryption_source_path_in_names + self.identity_in_names
        return list(dict.fromkeys(x for x in names if x is not None))

    def get_outputs(self) -> list[str]:
        return self.split_names + self.aggregate_names

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

    def _identity(self, variation: int, index: int) -> str:
        values = []
        for name in self.identity_in_names:
            try:
                value = self._get_previous_item(variation, name, index)
            except Exception:
                value = None
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().tolist()
            elif isinstance(value, os.PathLike):
                value = os.fspath(value)
            values.append([name, value])
        return self._hash(values if any(v[1] not in (None, "", False) for v in values)
                          else [["__index__", index]])

    def _concept(self, variation: int, index: int) -> dict | None:
        try:
            value = self._get_previous_item(variation, "concept", index)
        except Exception:
            return None
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        return dict(value) if isinstance(value, dict) else None

    @staticmethod
    def _logical_id(concept: dict | None, group_key: str) -> str:
        if concept:
            if concept.get("seed") not in (None, ""):
                return f"seed:{concept['seed']}"
            if concept.get("path"):
                return "path:" + hashlib.sha256(str(concept["path"]).encode()).hexdigest()
        return f"group:{group_key}"

    def _init_variations(self) -> None:
        if self.cache_only:
            self._init_cache_only()
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

    @staticmethod
    def _clone(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().clone().cpu()
        if isinstance(value, dict):
            return {k: DiskCache._clone(v) for k, v in value.items()}
        if isinstance(value, list):
            return [DiskCache._clone(v) for v in value]
        if isinstance(value, tuple):
            return tuple(DiskCache._clone(v) for v in value)
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

    def _build(self, group: str, variation: int) -> None:
        directory = self._cache_dir(group, variation)
        if os.path.isdir(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)
        size = len(self.group_indices[group])
        aggregate, identities, encrypted_items = [None] * size, [None] * size, [False] * size
        with tqdm(total=size, smoothing=0.1, desc="caching") as bar:
            def fn(pos: int, index: int, current_device: int | None):
                if torch.cuda.is_available() and current_device is not None:
                    torch.cuda.set_device(current_device)
                with torch.no_grad():
                    split = {n: self._clone(self._get_previous_item(variation, n, index)) for n in self.split_names}
                    agg = {n: self._clone(self._get_previous_item(variation, n, index)) for n in self.aggregate_names}
                enc = self._source_encrypted(variation, index)
                self._save(split, os.path.join(directory, f"{pos}.pt"), enc)
                aggregate[pos], identities[pos], encrypted_items[pos] = agg, self._identity(variation, index), enc
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
            "items": aggregate,
            "cache_layout": {
                "logical_id": self._logical_id(concept, group),
                "output_samples": int(self.group_output_samples[group]),
                "sample_identities": [str(x) for x in identities],
                "concept": concept,
            },
        }
        self._save(manifest, os.path.join(directory, "aggregate.pt"),
                   bool(self.encrypted and (self.encrypt_all or any(encrypted_items))))

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
                if manifest is None or len(manifest["items"]) != size:
                    if not prepared:
                        prepared = True
                        self.before_cache_fun()
                    self._build(group, variation)
                    manifest = self._manifest(directory)
                if manifest is None:
                    raise RuntimeError(f"Invalid cache manifest: {directory}")
                self.aggregate_cache[group][variation] = manifest["items"]

    @staticmethod
    def _variation_dirs(group_dir: str) -> list[int]:
        if not os.path.isdir(group_dir):
            return []
        out = []
        for entry in os.scandir(group_dir):
            if entry.is_dir() and entry.name.startswith("variation-"):
                try:
                    out.append(int(entry.name[10:]))
                except ValueError:
                    pass
        return sorted(set(out))

    def _init_cache_only(self) -> None:
        if self.variations_initialized:
            return
        records = []
        if not os.path.isdir(self.cache_dir):
            raise RuntimeError(f"Use Cache Only: missing cache directory {self.cache_dir}")
        groups = [""] + sorted(e.name for e in os.scandir(self.cache_dir)
                               if e.is_dir() and not e.name.startswith("variation-"))
        for group in groups:
            group_dir = os.path.join(self.cache_dir, group) if group else self.cache_dir
            physical = self._variation_dirs(group_dir)
            if not physical:
                continue
            manifest = self._manifest(os.path.join(group_dir, f"variation-{physical[0]}"))
            if manifest is None or not isinstance(manifest.get("cache_layout"), dict):
                raise RuntimeError("Use Cache Only requires current strict cache manifests; rebuild from source data.")
            layout = manifest["cache_layout"]
            ids = layout.get("sample_identities")
            if not isinstance(ids, list) or len(ids) != len(manifest["items"]):
                raise RuntimeError("Use Cache Only: invalid sample identities")
            records.append({"group_key": group, "variations": physical,
                            "size": len(ids), "sample_identities": ids,
                            "logical_id": str(layout.get("logical_id", "")),
                            "output_samples": int(layout.get("output_samples", len(ids))),
                            "concept": layout.get("concept")})
        if not records:
            raise RuntimeError("Use Cache Only: no complete cache groups found")
        image_slots = self.cache_only_layout.get("image_slots") if self.cache_only_layout else None
        if isinstance(image_slots, list):
            ordered = []
            for slot in image_slots:
                matches = [r for r in records if r["logical_id"] == str(slot.get("logical_id", ""))]
                if len(matches) != 1 or matches[0]["sample_identities"] != slot.get("sample_identities"):
                    raise RuntimeError("Use Cache Only: image/text cache identity mismatch")
                matches[0]["output_samples"] = int(slot["output_samples"])
                ordered.append(matches[0])
            records = ordered
        self._cache_only_records = {r["group_key"]: r for r in records}
        self.group_variations = {r["group_key"]: len(r["variations"]) for r in records}
        self.group_indices = {r["group_key"]: list(range(r["size"])) for r in records}
        self.group_output_samples = {r["group_key"]: r["output_samples"] for r in records}
        self.variations_initialized = True
        if self.cache_only_layout is not None and image_slots is None:
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
                    raise RuntimeError("Use Cache Only: invalid cache manifest")
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

    def start(self, out_variation: int):
        self._refresh(out_variation)

    def get_item(self, index: int, requested_name: str = None) -> dict:
        group, variation, pos = self._index(self.current_variation, index)
        if requested_name in self.aggregate_names:
            item = self.aggregate_cache[group][variation][pos]
            return {n: self._to_device(item[n]) for n in self.aggregate_names if n in item}
        if requested_name in self.split_names:
            if self.cache_only:
                record = self._cache_only_records[group]
                group_dir = os.path.join(self.cache_dir, group) if group else self.cache_dir
                directory = os.path.join(group_dir, f"variation-{record['variations'][variation]}")
            else:
                directory = self._cache_dir(group, variation)
            item = self._load(os.path.join(directory, f"{pos}.pt"))
            return {n: self._to_device(item[n]) for n in self.split_names if n in item}
        return {}
