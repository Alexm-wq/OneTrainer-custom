"""Content-addressed conditioning *after* image variant selection.

An epoch's layouts are derived from the exact cached image geometries first.
Only missing prompt/context hashes run Qwen; repeated captions share one payload.
"""
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.SingleVariationRandomAccessPipelineModule import SingleVariationRandomAccessPipelineModule

import torch


class MOSSConditioningCache(PipelineModule, SingleVariationRandomAccessPipelineModule):
    NAMES = ["prompt", "tokens", "tokens_mask", "text_encoder_hidden_state"]

    def __init__(self, model, max_length, cache_dir, identity, before_cache, enabled=True, prepare_empty=False):
        super().__init__()
        self.model = model
        self.max_length = max_length or 4096
        self.cache_dir = Path(cache_dir)
        self.identity = identity
        self.before_cache = before_cache
        self.enabled = enabled
        self.prepare_empty = prepare_empty
        self.initialized = False
        self.entries = []

    def get_inputs(self):
        return ["prompt"]

    def get_outputs(self):
        return self.NAMES

    def length(self):
        return len(self.entries) if self.enabled and self.initialized else self._get_previous_length("prompt")

    def _key(self, prompt):
        value = json.dumps([self.identity, self.max_length, prompt], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(value.encode()).hexdigest()

    def _load(self, path):
        try:
            value = torch.load(path, map_location="cpu", weights_only=True)
            ids, mask, hidden = (value[name] for name in self.NAMES[1:])
            if ids.shape != (self.max_length,) or mask.shape != ids.shape or hidden.shape[0] != self.max_length:
                return None
            if hidden.ndim != 2 or not mask.bool().any():
                return None
            return value
        except (OSError, ValueError, KeyError, RuntimeError, EOFError, TypeError, AttributeError, pickle.UnpicklingError):
            return None

    def _encode(self, prompt):
        encoded = self.model.tokenize(prompt, self.max_length)
        tokens, mask = encoded.input_ids, encoded.attention_mask
        with torch.no_grad():
            hidden = self.model.encode_tokens(tokens, mask)[0]
        return {"tokens": tokens[0].detach().cpu(), "tokens_mask": mask[0].detach().cpu(),
                "text_encoder_hidden_state": hidden.detach().cpu()}

    def _save(self, path, value):
        # Unique temporary names permit concurrent DDP ranks to populate a key.
        fd, temporary = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        os.close(fd)
        try:
            torch.save(value, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def start(self, variation):
        self.entries = []
        if not self.enabled:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        prepared, seen, misses = False, set(), 0
        for index in range(self._get_previous_length("prompt")):
            prompt = self._get_previous_item(variation, "prompt", index)
            key = self._key(prompt)
            path = self.cache_dir / (key + ".pt")
            self.entries.append((prompt, path))
            if key in seen:
                continue
            seen.add(key)
            if self._load(path) is None:
                if not prepared:
                    self.before_cache()
                    prepared = True
                self._save(path, self._encode(prompt))
                misses += 1
        # Empty conditioning is process-local and must exist even on a cache hit.
        if (not prepared and self.entries and self.prepare_empty
                and self.max_length not in self.model.empty_text):
            self.before_cache()
        self.initialized = True
        print(f"[OT-MOSS-TEXT-CACHE] rows={len(self.entries)}, unique={len(seen)}, "
              f"encoded={misses}, reused={len(seen) - misses}")

    def get_item(self, index, requested_name=None):
        if self.enabled:
            prompt, path = self.entries[index]
            if requested_name == "prompt":
                return {"prompt": prompt}
            value = self._load(path)
            if value is None:
                raise RuntimeError(f"MOSS conditioning cache changed during the epoch: {path}. Restart to rebuild it.")
        else:
            prompt = self._get_previous_item(self.current_variation, "prompt", index)
            if requested_name == "prompt":
                return {"prompt": prompt}
            value = self._encode(prompt)
        return {"prompt": prompt, **{name: tensor.to(self.pipeline.device) for name, tensor in value.items()}}
