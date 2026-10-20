"""MOSS native layout prompts: centered absolute pixel boxes in yxyx order.

Annotations use the EXIF-oriented source image. Only the serialized prompt is
sent to Qwen; canvas metadata and clipped polygons are OneTrainer bookkeeping.
"""
import copy
import json
import math
from pathlib import Path


def is_layout(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return False
    return isinstance(value, dict) and "items" in value and "summary" in value


def empty_layout(width, height, caption=""):
    return {
        "summary": {"meta": [], "canvas": "single", "caption": caption},
        "items": [], "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx",
        "canvas_width": width, "canvas_height": height,
    }


def parse_layout(value, width=None, height=None, *, match_image=False):
    try:
        layout = json.loads(value) if isinstance(value, str) else copy.deepcopy(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Invalid MOSS layout JSON: {error}") from error
    if not is_layout(layout):
        raise ValueError("MOSS layout needs summary and items.")
    if layout.get("bbox_coord_space") != "abs_centralized" or layout.get("bbox_format") != "yxyx":
        raise ValueError("MOSS requires bbox_coord_space=abs_centralized and bbox_format=yxyx.")
    if not isinstance(layout["summary"], dict) or not isinstance(layout["summary"].get("caption"), str):
        raise ValueError("MOSS summary.caption must be a string.")
    if not isinstance(layout["items"], list):
        raise ValueError("MOSS items must be a list.")
    w, h = layout.get("canvas_width", width), layout.get("canvas_height", height)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           or v <= 0 or int(v) != v for v in (w, h)):
        raise ValueError("MOSS layout needs positive integer canvas_width and canvas_height.")
    if match_image and (w != width or h != height):
        raise ValueError(
            f"MOSS annotation canvas {w}x{h} does not match the oriented source image {width}x{height}."
        )
    layout["canvas_width"], layout["canvas_height"] = int(w), int(h)
    for index, item in enumerate(layout["items"]):
        if not isinstance(item, dict) or not isinstance(item.get("label"), str) or not isinstance(item.get("caption"), str):
            raise ValueError(f"MOSS item {index} needs string label and caption.")
        if "text" in item and not isinstance(item["text"], str):
            raise ValueError(f"MOSS item {index}.text must be a string.")
        box = item.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box
        ):
            raise ValueError(f"MOSS item {index}.bbox must contain four finite numbers.")
        y0, x0, y1, x1 = box
        if y1 <= y0 or x1 <= x0:
            raise ValueError(f"MOSS item {index}.bbox must have positive width and height.")
        # Internal polygons preserve rotated edges across later crops.
        item["_polygon"] = [
            [x0 + w / 2, y0 + h / 2], [x1 + w / 2, y0 + h / 2],
            [x1 + w / 2, y1 + h / 2], [x0 + w / 2, y1 + h / 2],
        ]
    return layout


def _clip_polygon(points, width, height):
    for axis, bound, lower in ((0, 0, True), (0, width, False), (1, 0, True), (1, height, False)):
        output = []
        for a, b in zip(points, points[1:] + points[:1], strict=True):
            inside_a = a[axis] >= bound if lower else a[axis] <= bound
            inside_b = b[axis] >= bound if lower else b[axis] <= bound
            if inside_a:
                output.append(a)
            if inside_a != inside_b:
                t = (bound - a[axis]) / (b[axis] - a[axis])
                output.append([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])])
        points = output
        if not points:
            break
    return points


def transform_layout(layout, width, height, *, scale=(1.0, 1.0), offset=(0.0, 0.0),
                     flip=False, angle=0.0, center=None):
    """Apply resize, rotation, crop and then horizontal flip to the actual polygon.

    scale/offset are (y, x), dimensions/center are (x, y), like torchvision.
    Positive rotation is counter-clockwise in image coordinates.
    """
    if not layout:
        return ""
    result = copy.deepcopy(layout)
    sy, sx = scale
    oy, ox = offset
    cx, cy = center or (layout["canvas_width"] * sx / 2, layout["canvas_height"] * sy / 2)
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    kept = []
    for item in result["items"]:
        points = item["_polygon"]
        transformed = []
        for x, y in points:
            x, y = x * sx, y * sy
            x, y = cx + c * (x - cx) + s * (y - cy), cy - s * (x - cx) + c * (y - cy)
            x, y = x - ox, y - oy
            if flip:
                x = width - x
            transformed.append([x, y])
        polygon = _clip_polygon(transformed, width, height)
        if len(polygon) < 3:
            continue
        # A polygon touching only a crop edge has no visible area.
        area = abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in
                       zip(polygon, polygon[1:] + polygon[:1], strict=True))) / 2
        if area <= 1e-8:
            continue
        xs, ys = zip(*polygon, strict=True)
        item["_polygon"] = polygon
        item["bbox"] = [min(ys) - height / 2, min(xs) - width / 2,
                        max(ys) - height / 2, max(xs) - width / 2]
        kept.append(item)
    result.update(items=kept, canvas_width=int(width), canvas_height=int(height))
    return result


def layout_document(layout):
    """Serializable sidecar/editor document; do not leak polygon bookkeeping."""
    result = copy.deepcopy(layout)
    for item in result["items"]:
        item.pop("_polygon", None)
        item["bbox"] = [round(float(v), 4) for v in item["bbox"]]
    return result


def layout_prompt(layout, extra_caption=""):
    """Serialize like the official layout compiler: integer pixel edges and compact JSON."""
    width, height = layout["canvas_width"], layout["canvas_height"]
    result = layout_document(layout)
    result.pop("canvas_width", None)
    result.pop("canvas_height", None)
    items = []
    for source, item in zip(layout["items"], result["items"], strict=True):
        y0, x0, y1, x1 = (float(v) for v in source["bbox"])
        # Crops can leave slivers narrower than a pixel; they show no object.
        if y1 - y0 < 1 or x1 - x0 < 1:
            continue
        # Floor the minima and ceil the maxima in top-left pixels, then recenter.
        item["bbox"] = [
            math.floor(round(y0 + height / 2, 9)) - height // 2,
            math.floor(round(x0 + width / 2, 9)) - width // 2,
            math.ceil(round(y1 + height / 2, 9)) - height // 2,
            math.ceil(round(x1 + width / 2, 9)) - width // 2,
        ]
        items.append(item)
    result["items"] = items
    if extra_caption.strip():
        caption = result["summary"]["caption"]
        if extra_caption.strip() != caption.strip():
            result["summary"]["caption"] = (caption + "\n" + extra_caption).strip()
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def read_layout_source(image_path, mode="auto", prompt_source="sample"):
    """Prefer .layout.json, then a MOSS .json, then a complete JSON .txt caption."""
    if mode not in ("auto", "off", "required"):
        raise ValueError(f"Unknown MOSS layout mode: {mode}")
    if mode == "off":
        return "", ""
    stem = Path(image_path).with_suffix("")
    candidates = [Path(str(stem) + ".layout.json"), Path(str(stem) + ".json")]
    if prompt_source == "sample":
        candidates.append(Path(str(stem) + ".txt"))
    for path in candidates:
        if not path.is_file():
            continue
        raw = path.read_text(encoding="utf-8-sig")
        explicit = str(path).endswith(".layout.json")
        try:
            value = json.loads(raw)
        except ValueError as error:
            if explicit or ('"bbox"' in raw and '"items"' in raw):
                raise ValueError(f"{path}: invalid MOSS layout JSON: {error}") from error
            continue
        if explicit or is_layout(value):
            if not is_layout(value):
                raise ValueError(f"{path}: MOSS layout needs summary and items.")
            return raw, str(path)
    if mode == "required":
        raise ValueError(f"{image_path}: missing MOSS layout (.layout.json, .json or JSON .txt).")
    return "", ""


def sample_layout_prompt(config, width, height):
    """Scale the editor's reference canvas to the actual quantized sample size."""
    raw = getattr(config, "moss_layout_json", "") if getattr(config, "moss_layout_enabled", False) else ""
    if raw:
        layout = parse_layout(raw, config.width, config.height)
        extra = "" if is_layout(config.prompt) else config.prompt
    elif getattr(config, "moss_layout_enabled", False):
        raise ValueError("MOSS layout is enabled but empty. Open the layout editor or disable layout.")
    elif is_layout(config.prompt):
        layout = parse_layout(config.prompt, config.width, config.height)
        extra = ""
    else:
        return config.prompt
    layout = transform_layout(
        layout, width, height,
        scale=(height / layout["canvas_height"], width / layout["canvas_width"]),
    )
    return layout_prompt(layout, extra)
