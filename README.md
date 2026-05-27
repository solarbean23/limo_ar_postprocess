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
