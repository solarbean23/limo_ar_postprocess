# LIMO AR Post-processing

Minimal offline renderer for ROS 2 db3 bags plus DJI LRF/MP4 video.

## Quick start

Inspect bag topics and parsed timeline:

```bash
python -m limo_ar.render --config configs/example_limo.yaml --inspect
```

Render a short preview first:

```bash
python -m limo_ar.render \
  --config configs/example_limo.yaml \
  --mode wide_with_map \
  --output outputs/preview_10s.mp4 \
  --preview-sec 10
```

Render MP4 and a WebM copy:

```bash
python -m limo_ar.render \
  --config configs/example_limo.yaml \
  --mode wide_with_map \
  --output outputs/preview_10s.mp4 \
  --webm-output outputs/preview_10s.webm \
  --preview-sec 10
```

Render AR-only output:

```bash
python -m limo_ar.render \
  --config configs/example_limo.yaml \
  --mode ar_only \
  --output outputs/ar_only.mp4
```

`sync.time_offset_sec` or `--time-offset-sec` controls video-to-bag synchronization:

```text
bag_t = video_t + time_offset_sec
```

If `homography.yaml` is missing, the renderer skips video-space AR overlay and still renders
the right-side map for `wide_with_map`.

## Homography

Create or update `configs/homography.yaml` by clicking image points in the same order as
`configs/file_1_world_points.yaml`:

```bash
python -m limo_ar.calibration \
  --video /path/to/DJI_20000710120811_0008_D.LRF.lrf \
  --world-points configs/file_1_world_points.yaml \
  --output configs/homography.yaml \
  --frame-sec 0.0 \
  --save-frame outputs/calibration_frame.jpg
```

If the four field corners are not visible, replace `file_1_world_points.yaml` with four or
more visible known world points such as field markers or object positions.
