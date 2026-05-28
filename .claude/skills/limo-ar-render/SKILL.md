---
name: limo-ar-render
description: Render a LIMO AR post-processed video from a ROS 2 db3 bag plus DJI LRF/MP4 footage. Use when the user provides (or asks to render) an `lrf`/`mp4` + `db3` + `metadata.yaml` set, wants an `ar_only` or `wide_with_map` AR overlay video, or needs to (re)compute `configs/homography.yaml`.
---

# LIMO AR Render

The `limo_ar` Python package in this repo already implements the full pipeline. This skill orchestrates its CLI — **do not modify pipeline code to "fix" a render issue before checking config, sync offset, and homography**.

## First-time setup

If `python3 -c "import rosbags, cv2, tqdm"` fails, install deps:

```bash
pip install -r requirements.txt
```

## Inputs to collect from the user

Ask for any of these that are missing — do not guess paths.

1. `video_path` — `.lrf` or `.mp4`
2. `bag_path` — `.db3` (its sibling `metadata.yaml` must exist in the same directory)
3. Desired mode — `ar_only` or `wide_with_map` (default `wide_with_map` if unspecified)
4. Whether `configs/homography.yaml` matches this video. The current file is self-noted as "approximate" — re-calibrate if AR overlay looks misaligned.

## Procedure

### 1. Pick or create a config

Configs are in [configs/](configs/); [example_limo.yaml](configs/example_limo.yaml) is the reference. Existing scenario configs use `<scenario>_limo.yaml` naming (e.g. `file_1_limo.yaml`).

- If the user points at an existing config, use it.
- Otherwise copy `example_limo.yaml` and edit only the `input:` block and `output.path`.

### 2. Inspect

Run this on any new config before rendering — surfaces topic / timeline / path issues fast:

```bash
python3 -m limo_ar.render --config <config.yaml> --inspect
```

Wrong paths produce a clear `FileNotFoundError`. Report any missing topics, large video↔bag time gap, or robots with no pose messages.

### 3. Short preview

Recommended for any new config or after changing sync/homography. Default 10 s:

```bash
python3 -m limo_ar.render \
  --config <config.yaml> \
  --mode <ar_only|wide_with_map> \
  --output outputs/<name>_preview.mp4 \
  --preview-sec 10
```

Show the user the preview path before doing a full render.

### 4. Full render

```bash
python3 -m limo_ar.render \
  --config <config.yaml> \
  --mode <ar_only|wide_with_map> \
  --output outputs/<name>.mp4
```

Add `--webm-output outputs/<name>.webm` for a web copy.

### Sync tuning

If AR objects are temporally off, adjust `sync.time_offset_sec` (relation: `bag_t = video_t + time_offset_sec`) or pass `--time-offset-sec <float>`. Suggest ±0.1 s steps with a re-preview.

For all other flags, run `python3 -m limo_ar.render --help`.

## Homography

[configs/homography.yaml](configs/homography.yaml) accepts three forms — pick based on environment:

1. **Interactive (preferred when display available)**:
   ```bash
   python3 -m limo_ar.calibration \
     --video <video_path> \
     --world-points configs/<scenario>_world_points.yaml \
     --output configs/homography.yaml \
     --frame-sec 0.0 \
     --save-frame outputs/calibration_frame.jpg
   ```
   Needs `cv2.imshow` (X11/Wayland). Click image points in the **same order** as the world-points YAML.

2. **Headless / manual**: open `configs/homography.yaml` in a text editor and fill in `world_points:` and `image_points:` (4+ pairs in matching order). `matrix:` is auto-computed at load.

3. **Direct matrix**: if you already have a 3×3 homography from elsewhere, write it under `matrix:` and omit the point lists.

If `homography.yaml` is missing the renderer silently drops video-space AR overlay and (for `wide_with_map`) still draws the right-hand map. So a missing/wrong homography degrades quietly — verify the preview visually.

## Don't

- Don't render full-length before a preview — renders are slow.
- Don't modify `limo_ar/*.py` to fix render output before checking config, sync, and homography.
- Don't invent file paths — ask the user.
- Don't launch `limo_ar.calibration` in a headless context; suggest form 2 or 3 above instead.
