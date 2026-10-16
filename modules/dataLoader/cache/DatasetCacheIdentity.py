"""Fingerprint source inventories before expensive image/text processing."""
import hashlib
import json
import os

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class DatasetCacheIdentity(PipelineModule, RandomAccessPipelineModule):
    """Keep indexed image and text caches tied to the ordered source rows."""

    def __init__(self, image_path_names=None):
        super().__init__()
        self.image_path_names = list(image_path_names or ["image_path"])
        self.text_path_names = ["sample_prompt_path", "concept.text.prompt_path"]
        self.rows = []
        self.concepts = {}

    def get_inputs(self):
        return list(dict.fromkeys(["concept", "image_path", *self.image_path_names, *self.text_path_names]))

    def get_outputs(self):
        return ["concept"]

    def length(self):
        return len(self.rows)

    @staticmethod
    def _encoded(value):
        return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()

    @staticmethod
    def _file_signature(path, signatures):
        if not path:
            return None
        path = os.path.normcase(os.path.realpath(os.fspath(path)))
        if path not in signatures:
            try:
                stat = os.stat(path)
                signatures[path] = [path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
            except FileNotFoundError:
                # Missing optional captions/masks must also invalidate the cache
                # if they are created later. Required files fail in their loader.
                signatures[path] = [path, None]
        return signatures[path]

    def start(self, variation):
        self.rows = []
        self.concepts = {}
        images, texts, signatures = {}, {}, {}
        for index in range(self._get_previous_length("image_path")):
            concept = self._get_previous_item(variation, "concept", index)
            key = self._encoded(concept)
            if key not in self.concepts:
                self.concepts[key] = dict(concept)
                images[key] = hashlib.sha256(b"dataset-image-v1\n")
                texts[key] = hashlib.sha256(b"dataset-text-v1\n")
            self.rows.append(key)

            path = self._get_previous_item(variation, "image_path", index)
            image_files = [
                self._file_signature(self._get_previous_item(variation, name, index), signatures)
                for name in self.image_path_names
            ]
            text_files = [
                self._file_signature(self._get_previous_item(variation, name, index), signatures)
                for name in self.text_path_names
            ]
            # Include ordered image paths in both streams: cached text tensors
            # are addressed by row number even when their captions are unchanged.
            images[key].update(self._encoded([path, image_files]) + b"\n")
            texts[key].update(self._encoded([path, text_files]) + b"\n")

        for key, concept in self.concepts.items():
            concept["_image_cache_fingerprint"] = images[key].hexdigest()
            concept["_text_cache_fingerprint"] = texts[key].hexdigest()
        print(f"[OT-DATASET-CACHE] Checked {len(self.rows)} source rows in {len(self.concepts)} concepts.")

    def get_item(self, variation, index, requested_name=None):
        return {"concept": self.concepts[self.rows[index]]}
