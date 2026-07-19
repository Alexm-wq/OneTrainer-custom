from __future__ import annotations

import math
import os
from fractions import Fraction

import numpy as np
from PIL import Image

from modules.util.path_util import supported_image_extensions


ALL_POSSIBLE_INPUT_ASPECTS = [
    (1.0, 1.0), (1.0, 1.05), (1.0, 1.10), (1.0, 1.15),
    (1.0, 1.20), (1.0, 1.25), (1.0, 1.3333333333), (1.0, 1.4),
    (1.0, 1.5), (1.0, 1.6), (1.0, 1.6666666667), (1.0, 1.75),
    (1.0, 1.7777777778), (1.0, 1.85), (1.0, 2.0), (1.0, 2.2),
    (1.0, 2.4), (1.0, 2.5), (1.0, 2.75), (1.0, 3.0),
    (1.0, 3.5), (1.0, 4.0),
]

MODEL_QUANTIZATION = {
    "Z_IMAGE": 64, "CHROMA_1": 64, "QWEN": 64, "FLUX_DEV_1": 64,
    "FLUX_FILL_DEV_1": 64, "FLUX_2": 64, "KREA_2": 64,
    "HUNYUAN_VIDEO": 64, "STABLE_DIFFUSION_XL_10_BASE": 64,
    "STABLE_DIFFUSION_XL_10_BASE_INPAINTING": 64, "STABLE_DIFFUSION_15": 8,
    "STABLE_DIFFUSION_15_INPAINTING": 8, "STABLE_DIFFUSION_20": 8,
    "STABLE_DIFFUSION_20_BASE": 8, "STABLE_DIFFUSION_20_INPAINTING": 8,
    "STABLE_DIFFUSION_20_DEPTH": 8, "STABLE_DIFFUSION_21": 8,
    "STABLE_DIFFUSION_21_BASE": 8, "STABLE_DIFFUSION_3": 64,
    "STABLE_DIFFUSION_35": 64, "HI_DREAM_FULL": 64, "SANA": 32,
    "PIXART_ALPHA": 16, "PIXART_SIGMA": 16, "WUERSTCHEN_2": 128,
    "STABLE_CASCADE_1": 128,
}


def quantization_for_model(model_type) -> int:
    name = getattr(model_type, "name", model_type)
    return MODEL_QUANTIZATION.get(str(name), 64)


def build_buckets(target_resolution: int, quantization: int):
    raw = [
        (
            h / math.sqrt(h * w) * target_resolution,
            w / math.sqrt(h * w) * target_resolution,
        )
        for h, w in ALL_POSSIBLE_INPUT_ASPECTS
    ]
    raw += [(w, h) for h, w in raw]
    buckets = sorted({
        (
            max(quantization, round(h / quantization) * quantization),
            max(quantization, round(w / quantization) * quantization),
        )
        for h, w in raw
    })
    return buckets, np.asarray([h / w for h, w in buckets], dtype=np.float64)


def label_aspect(height: int, width: int) -> str:
    fraction = Fraction(height, width).limit_denominator(64)
    if fraction.numerator > fraction.denominator:
        orientation = "landscape"
    elif fraction.numerator < fraction.denominator:
        orientation = "portrait"
    else:
        orientation = "square"
    return f"~{fraction.numerator}:{fraction.denominator} {orientation}"


def iter_images(path: str):
    extensions = supported_image_extensions()
    for root, dirs, files in os.walk(path):
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        for filename in files:
            stem, extension = os.path.splitext(filename)
            if extension.lower() not in extensions:
                continue
            if stem.endswith("-masklabel") or stem.endswith("-condlabel"):
                continue
            yield os.path.join(root, filename)


def analyze_concept(
    concept_path: str,
    batch_size: int,
    target_resolutions: list[int],
    quantization: int,
    include_subdirectories: bool = True,
) -> dict:
    if not os.path.isdir(concept_path):
        raise ValueError(f"Not a directory: {concept_path}")
    if batch_size <= 0 or quantization <= 0 or not target_resolutions:
        raise ValueError("Invalid batch size, quantization, or target resolutions.")

    dimensions = []
    scanned = unreadable = 0
    for path in iter_images(concept_path):
        scanned += 1
        try:
            with Image.open(path) as image:
                width, height = image.size
            dimensions.append((height, width))
        except Exception:
            unreadable += 1

    targets = []
    for target in target_resolutions:
        buckets, aspects = build_buckets(int(target), int(quantization))
        counts = {bucket: 0 for bucket in buckets}
        for height, width in dimensions:
            index = int(np.argmin(np.abs(aspects - height / width)))
            counts[buckets[index]] += 1

        rows = []
        for (height, width), count in counts.items():
            if count == 0:
                continue
            drops = count % batch_size
            rows.append({
                "h": int(height),
                "w": int(width),
                "count": int(count),
                "drops": int(drops),
                "add": int((batch_size - drops) % batch_size),
                "remove": int(drops),
                "aspect_label": label_aspect(height, width),
            })
        rows.sort(key=lambda row: (-row["count"], row["h"], row["w"]))
        targets.append({
            "target": int(target),
            "total_pairs": sum(row["count"] for row in rows),
            "total_drops": sum(row["drops"] for row in rows),
            "total_add": sum(row["add"] for row in rows),
            "total_remove": sum(row["remove"] for row in rows),
            "buckets": rows,
        })

    return {
        "concept_path": concept_path,
        "batch_size": int(batch_size),
        "quantization": int(quantization),
        "scanned": scanned,
        "unreadable": unreadable,
        "targets": targets,
    }


def parse_target_resolutions(value: str) -> list[int]:
    result = []
    for token in str(value or "").split(","):
        token = token.strip()
        if not token or "x" in token.lower():
            continue
        try:
            number = int(token)
        except ValueError:
            continue
        if number > 0:
            result.append(number)
    return result
