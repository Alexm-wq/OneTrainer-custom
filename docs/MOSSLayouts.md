# MOSS layouts: training, sampling and annotation

MOSS accepts its native layout JSON as ordinary text conditioning. No separate
ControlNet, region encoder or box loss is added. Plain captions and layouts can
share a concept and batch. The text encoder remains frozen.

## Training

In a MOSS concept, **MOSS layouts** defaults to **Auto (mixed captions/layouts)**.
For each EXIF-oriented image, the loader prefers:

1. `photo.layout.json`;
2. `photo.json`, if it contains a MOSS layout;
3. a complete layout JSON in `photo.txt`, when the caption source is per sample.

Without a layout, the ordinary caption source and text augmentations apply.
**Required** reports missing annotations; **Off** uses ordinary caption loading.
An invalid explicit layout raises an error including its path. Multiline JSON is
read as one document, never as randomly selected caption lines.

The format is the [official MOSS layout schema](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md#layout-prompts-experimental):
`summary.caption`, `items[].label`, `items[].caption`, optional
`items[].text`, `items[].bbox`, `bbox_coord_space: "abs_centralized"` and
`bbox_format: "yxyx"`. Coordinates are **absolute pixels**, ordered
`[y_min, x_min, y_max, x_max]`, with the image center as origin.

OneTrainer sidecars additionally record `canvas_width` and `canvas_height`.
These dimensions must match the actual oriented image. Official JSON without
dimensions uses the image dimensions. Canvas metadata is stripped from the
prompt sent to the model. Normalized 0–1000 watermark boxes are a different
format and cannot be substituted.

### Geometry

Annotations describe the source image. The final prompt is generated after the
actual bucket, resize, crop, flip and rotation have been chosen. Boxes follow
those exact transforms, including mask-driven rotate/crop. Polygons are clipped
to the visible canvas; completely removed objects are removed from `items`.
The final axis-aligned box encloses the remaining polygon.
The prompt sent to the model uses the official layout compiler's serialization:
compact JSON with integer pixel edges (minima floored, maxima ceiled). Items
narrower or shorter than one pixel after cropping are dropped.

The authored scene description and item text are retained verbatim. Avoid
absolute direction words such as “on the left” if random horizontal flips are
enabled: geometry cannot rewrite the meaning of natural-language captions.
For rendered text, keep flips and rotation disabled unless they are intentional.
Tag shuffle, tag dropout and capitalization never operate on layout JSON.
Whole-caption dropout still works through the model's normal empty conditioning.

### Caching and resume

The transformed layout and source row are stored in the **same resolution and
image-variation payload** as the latent. After the epoch selects a variant, its
exact serialized prompt is looked up in a content-addressed text cache. Different
boxes/resolutions/crops produce different text keys; identical ordinary captions
share a text payload. New layouts are encoded before training resumes each epoch.

Layout content changes, additions and removals invalidate the affected concept's
geometry cache. Ordinary caption edits only invalidate their conditioning.
Dataset additions and row ordering are included in inventory identities. Current
concept settings are restored from the actual source row instead of cached loss
weights. Both cache toggles, balancing, multi-resolution, mixed DPO and resume use
the same association.

Restart after changing layout files during an active run. No manual cache deletion
is required. Existing older MOSS caches use a separate namespace and are not read.

A layout that exceeds **Text Encoder Sequence Length** raises an actionable error
instead of silently truncating JSON. Increase that setting or shorten the layout.
4096 remains the normal MOSS default; small sequence lengths can be insufficient.

## Sampling

Open sample settings, choose **Edit MOSS layout**, draw/edit boxes and press
**Apply layout**. The **MOSS layout** switch enables the stored document.
The normal prompt adds scene details to the layout description; the negative
prompt continues to work normally. Box coordinates scale from the editor's
reference canvas to the actual, 16-aligned sample dimensions.

You can also paste a complete native layout directly into the normal prompt.
Sample configurations preserve the layout across save/load.

## Standalone bucketing and phrase annotation tool

Run in your existing OneTrainer environment:

```bash
pixi run --locked -e cuda13 python scripts/moss_layout_ui.py
```

Or open a specific image and caption:

```bash
pixi run --locked -e cuda13 python scripts/moss_layout_ui.py \
  --image /workspace/data/photo.png --caption /workspace/data/photo.txt \
  --resolution 1024
```

Use `--ui ctk` for the CustomTkinter frontend instead of Qt. The tool runs on CPU,
loads no model weights and does not start training.

1. Open an image; its caption/layout sidecar is loaded automatically. A separate
   caption can be loaded with **Load caption file**.
2. Enter the MGDS target and click **Apply MGDS bucket (center crop)**.
   This calls the fork's actual `AspectBucketing` and MGDS `ScaleCropImage`
   implementation with MOSS's quantization of 16. Targets such as `1024`,
   `768,1024` (choose the zero-based variant), and `1024x768` are supported.
3. Select a word or phrase in the caption, then draw a box. The selection becomes
   the object's description. **Use selected phrase for next box** can retain it
   explicitly. The original full caption remains the scene description.
4. Use **Select / move** to move a box or drag a corner to resize it. Descriptions,
   labels, rendered text and numeric edges can also be edited directly.
5. **Export image + caption + layout** writes a PNG and matching `.txt` and
   `.layout.json` files together. The source image is preserved.
   **Save .layout.json** alone is available for uncropped source annotations;
   after bucketing it routes to the bundle export.

The editor displays ordinary top-left pixel coordinates. It converts them to the
centered MOSS format on export. The same editor is available from **Annotate MOSS
layouts** in the concept window.

For an exported bucket that should stay exactly as annotated, train at that same
resolution (or use the concept's explicit width x height resolution override).
Further training resizing/cropping is supported and transforms the annotations
again. Center crop is the annotation tool's deliberate default; training jitter
can choose a different crop unless you disable it.

## Validation scope

Focused CPU tests cover layout parsing, geometry, native bucket/crop equivalence,
the real MGDS data/cache pipeline, mixed captions and DPO, sampling prompt
construction, truncation protection, and both actual GUI frontends, including
phrase selection and round-trip export. No model weights are required.
Image quality and layout adherence still require a real MOSS GPU sampling run;
MOSS's layout capability is experimental.
