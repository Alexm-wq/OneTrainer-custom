"""Layout-aware stages around the pinned MGDS image transforms."""
import hashlib
import json
import threading
from pathlib import Path

from modules.util.moss_layout import layout_prompt, parse_layout, read_layout_source, transform_layout

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModules.RandomFlip import RandomFlip
from mgds.pipelineModules.RandomMaskRotateCrop import RandomMaskRotateCrop
from mgds.pipelineModules.RandomRotate import RandomRotate
from mgds.pipelineModules.ScaleCropImage import ScaleCropImage
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class MOSSLayoutInventory(PipelineModule, RandomAccessPipelineModule):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.by_source = {}

    def get_inputs(self):
        return ["image_path", "concept"]

    def get_outputs(self):
        return ["moss_layout_source", "moss_source_index", "moss_source_key", "concept"]

    def length(self):
        return len(self.rows)

    def start(self, variation):
        self.rows = []
        self.by_source = {}
        fingerprints, concepts, counts = {}, {}, {}
        for index in range(self._get_previous_length("image_path")):
            concept = self._get_previous_item(variation, "concept", index)
            key = json.dumps(concept, sort_keys=True)
            path = self._get_previous_item(variation, "image_path", index)
            raw, source = read_layout_source(
                path, concept["text"].get("moss_layout_mode", "auto"),
                concept["text"]["prompt_source"],
            )
            concepts.setdefault(key, dict(concept))
            fingerprints.setdefault(key, hashlib.sha256(b"moss-layout-v1"))
            # Ordering matters within a concept, never across unrelated concepts.
            local_index = counts.get(key, 0)
            counts[key] = local_index + 1
            fingerprints[key].update(json.dumps([local_index, path, source, raw]).encode())
            source_key = json.dumps([path, *[
                concept.get(name) for name in (
                    "path", "seed", "include_subdirectories", "dpo_chosen_pattern",
                    "dpo_rejected_pattern", "_dpo_split", "_dpo_validation_percentage", "_dpo_validation_seed",
                )
            ]], sort_keys=True)
            self.rows.append({"key": key, "raw": raw, "source": source, "path": path,
                              "source_key": source_key, "source_index": local_index})
        for key, concept in concepts.items():
            concept["_moss_layout_fingerprint"] = fingerprints[key].hexdigest()
        for row in self.rows:
            row["concept"] = concepts[row.pop("key")]
            self.by_source[row["source_key"]] = row
        print(f"[OT-MOSS-LAYOUT] layout_rows={sum(bool(r['raw']) for r in self.rows)}, "
              f"caption_rows={sum(not r['raw'] for r in self.rows)}")

    def get_item(self, variation, index, requested_name=None):
        row = self.rows[index]
        return {"concept": row["concept"], "moss_layout_source": row["raw"],
                "moss_source_index": row["source_index"], "moss_source_key": row["source_key"]}


class LoadMOSSLayout(PipelineModule, RandomAccessPipelineModule):
    def get_inputs(self):
        return ["image", "moss_layout_source", "image_path"]

    def get_outputs(self):
        return ["moss_layout"]

    def length(self):
        return self._get_previous_length("image")

    def get_item(self, variation, index, requested_name=None):
        raw = self._get_previous_item(variation, "moss_layout_source", index)
        if not raw:
            return {"moss_layout": ""}
        image = self._get_previous_item(variation, "image", index)
        height, width = image.shape[-2:]
        try:
            layout = parse_layout(raw, width, height, match_image=True)
        except ValueError as error:
            path = self._get_previous_item(variation, "image_path", index)
            raise ValueError(f"{path}: {error}") from error
        return {"moss_layout": layout}


class MOSSScaleCrop(ScaleCropImage):
    def get_inputs(self):
        return super().get_inputs() + ["moss_layout"]

    def get_outputs(self):
        return super().get_outputs() + ["moss_layout"]

    def get_item(self, variation, index, requested_name=None):
        item = super().get_item(variation, index, requested_name)
        layout = self._get_previous_item(variation, "moss_layout", index)
        if layout:
            scale = self._get_previous_item(variation, self.scale_resolution_in_name, index)
            height, width = item[self.names[0]].shape[-2:]
            layout = transform_layout(
                layout, width, height,
                scale=(scale[-2] / layout["canvas_height"], scale[-1] / layout["canvas_width"]),
                offset=item[self.crop_offset_out_name],
            )
        item["moss_layout"] = layout
        return item


class MOSSRandomFlip(RandomFlip):
    def get_inputs(self):
        return super().get_inputs() + ["moss_layout"]

    def get_outputs(self):
        return super().get_outputs() + ["moss_layout"]

    def get_item(self, variation, index, requested_name=None):
        item = super().get_item(variation, index, requested_name)
        layout = self._get_previous_item(variation, "moss_layout", index)
        enabled = self._get_previous_item(variation, self.enabled_in_name, index)
        fixed = self._get_previous_item(variation, self.fixed_enabled_in_name, index)
        flip = self._get_rand(variation, index).random() < 0.5 if enabled else bool(fixed)
        if layout and flip:
            layout = transform_layout(layout, layout["canvas_width"], layout["canvas_height"], flip=True)
        item["moss_layout"] = layout
        return item


class MOSSRandomRotate(RandomRotate):
    def get_inputs(self):
        return super().get_inputs() + ["moss_layout"]

    def get_outputs(self):
        return super().get_outputs() + ["moss_layout"]

    def get_item(self, variation, index, requested_name=None):
        item = super().get_item(variation, index, requested_name)
        layout = self._get_previous_item(variation, "moss_layout", index)
        enabled = self._get_previous_item(variation, self.enabled_in_name, index)
        fixed = self._get_previous_item(variation, self.fixed_enabled_in_name, index)
        maximum = self._get_previous_item(variation, self.max_angle_in_name, index)
        angle = self._get_rand(variation, index).uniform(-maximum, maximum) if enabled else maximum if fixed else 0
        if layout and angle:
            layout = transform_layout(layout, layout["canvas_width"], layout["canvas_height"], angle=angle)
        item["moss_layout"] = layout
        return item


class MOSSMaskRotateCrop(RandomMaskRotateCrop):
    """Observe the exact affine/crop chosen by the pinned MGDS implementation.

    Its private transform helpers are the single source of truth. Thread-local
    recording avoids copying its mask-region/padding algorithm or racing workers.
    Tests exercise these hooks against the installed MGDS revision.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._geometry = threading.local()

    def get_inputs(self):
        return super().get_inputs() + ["moss_layout"]

    def get_outputs(self):
        return super().get_outputs() + ["moss_layout"]

    def _RandomMaskRotateCrop__rotate(self, tensor, center, angle):
        self._geometry.rotation = (center, angle)
        return RandomMaskRotateCrop._RandomMaskRotateCrop__rotate(tensor, center, angle)

    def _RandomMaskRotateCrop__crop(self, tensor, y_min, y_max, x_min, x_max):
        self._geometry.crop = (y_min, y_max, x_min, x_max)
        return RandomMaskRotateCrop._RandomMaskRotateCrop__crop(tensor, y_min, y_max, x_min, x_max)

    def get_item(self, variation, index, requested_name=None):
        self._geometry.rotation = self._geometry.crop = None
        item = super().get_item(variation, index, requested_name)
        layout = self._get_previous_item(variation, "moss_layout", index)
        if layout and self._geometry.crop:
            center, angle = self._geometry.rotation
            y0, y1, x0, x1 = self._geometry.crop
            # Clip at the original canvas before cropping, as torchvision does.
            layout = transform_layout(layout, layout["canvas_width"], layout["canvas_height"],
                                      angle=angle, center=center)
            layout = transform_layout(layout, x1 - x0, y1 - y0, offset=(y0, x0))
        item["moss_layout"] = layout
        return item


class MOSSRestoreConcept(PipelineModule, RandomAccessPipelineModule):
    """Resolve the current config by stable source identity, never a global row.

    This keeps balancing/resume/DPO ordering aligned without caching stale loss
    weights or recaching VAE latents when ordinary caption settings change.
    """
    def __init__(self, inventory):
        super().__init__()
        self.inventory = inventory

    def get_inputs(self):
        return ["moss_source_key"]

    def get_outputs(self):
        return ["concept"]

    def length(self):
        return self._get_previous_length("moss_source_key")

    def get_item(self, variation, index, requested_name=None):
        source = self._get_previous_item(variation, "moss_source_key", index)
        return {"concept": self.inventory.by_source[source]["concept"]}


class MOSSNormalPrompt(PipelineModule, RandomAccessPipelineModule):
    def __init__(self, text_caching):
        super().__init__()
        self.text_caching = text_caching

    def get_inputs(self):
        return ["moss_layout", "image_path", "moss_source_index", "concept"]

    def get_outputs(self):
        return ["prompt"]

    def length(self):
        return self._get_previous_length("image_path")

    def get_item(self, variation, index, requested_name=None):
        if self._get_previous_item(variation, "moss_layout", index):
            # Text augmentation operates only on this disposable empty caption.
            return {"prompt": ""}
        concept = self._get_previous_item(variation, "concept", index)
        path = Path(self._get_previous_item(variation, "image_path", index))
        source = concept["text"]["prompt_source"]
        if source == "filename":
            return {"prompt": path.stem}
        caption_path = Path(concept["text"]["prompt_path"]) if source == "concept" else path.with_suffix(".txt")
        if not caption_path.is_file():
            return {"prompt": ""}
        lines = [line.strip() for line in caption_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        source_index = self._get_previous_item(variation, "moss_source_index", index)
        text_variation = variation % max(1, concept["text_variations"]) if self.text_caching else variation
        return {"prompt": self._get_rand(text_variation, source_index).choice(lines) if lines else ""}


class MOSSLayoutPrompt(PipelineModule, RandomAccessPipelineModule):
    def get_inputs(self):
        return ["prompt", "moss_layout"]

    def get_outputs(self):
        return ["prompt"]

    def length(self):
        return self._get_previous_length("prompt")

    def get_item(self, variation, index, requested_name=None):
        layout = self._get_previous_item(variation, "moss_layout", index)
        prompt = layout_prompt(layout) if layout else self._get_previous_item(variation, "prompt", index)
        return {"prompt": prompt}


class MOSSTextAugmentation(PipelineModule, RandomAccessPipelineModule):
    """Preserve MGDS tag behavior, capped by text_variations when caching."""
    def __init__(self, module, text_caching):
        super().__init__()
        self.module = module
        self.text_caching = text_caching
        self._local = threading.local()
        self.module._get_rand = self._augmentation_rand

    def _augmentation_rand(self, variation, index=-1):
        count = self._local.count
        return self._get_rand(variation % count if self.text_caching else variation, self._local.source_index)

    def init(self, pipeline, base_seed, module_index, state):
        super().init(pipeline, base_seed, module_index, state)
        self.module.init(pipeline, base_seed, module_index, state)

    def get_inputs(self):
        return self.module.get_inputs() + ["moss_layout", "concept", "moss_source_index"]

    def get_outputs(self):
        return self.module.get_outputs()

    def length(self):
        return self._get_previous_length("prompt")

    def get_item(self, variation, index, requested_name=None):
        if self._get_previous_item(variation, "moss_layout", index):
            return {"prompt": ""}
        # Keep upstream resolution/row selection at the current epoch. Only the
        # augmentation RNG cycles over text_variations, never the image stream.
        concept = self._get_previous_item(variation, "concept", index)
        self._local.count = max(1, concept["text_variations"])
        self._local.source_index = self._get_previous_item(variation, "moss_source_index", index)
        return self.module.get_item(variation, index, requested_name)
