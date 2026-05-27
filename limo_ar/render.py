from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .bag_reader import read_bag
from .calibration import load_homography
from .drawing import draw_connectors, draw_map, draw_video_overlay
from .timeline import LimoTimeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline LIMO AR post-processing renderer.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--mode", choices=("ar_only", "wide_with_map"))
    parser.add_argument("--output")
    parser.add_argument("--webm-output")
    parser.add_argument("--start-sec", type=float)
    parser.add_argument("--end-sec", type=float)
    parser.add_argument("--preview-sec", type=float)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--time-offset-sec", type=float)
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = load_config(config_path)
    apply_overrides(config, args)
    resolve_config_paths(config, config_path.parent)

    bag = read_bag(
        config["input"]["bag_path"],
        config["input"].get("metadata_path"),
        config,
    )
    timeline = LimoTimeline(bag, config)

    if args.inspect:
        print_inspection(bag, timeline)
        return 0

    render_video(config, timeline)
    return 0


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    config.setdefault("render", {})
    config.setdefault("output", {})
    config.setdefault("sync", {})

    if args.mode:
        config["render"]["mode"] = args.mode
    if args.output:
        config["output"]["path"] = args.output
    if args.webm_output:
        config["output"]["webm_path"] = args.webm_output
    if args.start_sec is not None:
        config["render"]["start_sec"] = args.start_sec
    if args.end_sec is not None:
        config["render"]["end_sec"] = args.end_sec
    if args.preview_sec is not None:
        start = float(config["render"].get("start_sec") or 0.0)
        config["render"]["end_sec"] = start + args.preview_sec
    if args.max_frames is not None:
        config["render"]["max_frames"] = args.max_frames
    if args.fps is not None:
        config["render"]["fps"] = args.fps
    if args.time_offset_sec is not None:
        config["sync"]["time_offset_sec"] = args.time_offset_sec


def resolve_config_paths(config: dict[str, Any], config_dir: Path) -> None:
    input_cfg = config.setdefault("input", {})
    for key in ("video_path", "bag_path", "metadata_path", "homography_path"):
        if input_cfg.get(key):
            input_cfg[key] = str(_resolve_existing_or_relative(input_cfg[key], config_dir))
    output_cfg = config.setdefault("output", {})
    if output_cfg.get("path"):
        output_cfg["path"] = str(Path(output_cfg["path"]))
    if output_cfg.get("webm_path"):
        output_cfg["webm_path"] = str(Path(output_cfg["webm_path"]))


def render_video(config: dict[str, Any], timeline: LimoTimeline) -> None:
    input_cfg = config.get("input", {})
    render_cfg = config.get("render", {})
    output_cfg = config.get("output", {})
    visual_cfg = config.get("visual", {})
    sync_cfg = config.get("sync", {})

    video_path = Path(input_cfg.get("video_path", ""))
    output_path = Path(output_cfg.get("path", "outputs/preview.mp4"))
    webm_path = output_cfg.get("webm_path")
    mode = render_cfg.get("mode", "wide_with_map")
    start_sec = float(render_cfg.get("start_sec") or 0.0)
    end_sec = render_cfg.get("end_sec")
    end_sec = float(end_sec) if end_sec is not None else None
    max_frames = render_cfg.get("max_frames")
    max_frames = int(max_frames) if max_frames is not None else None
    time_offset_sec = float(sync_cfg.get("time_offset_sec", 0.0))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")

    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise RuntimeError("could not read video dimensions")

    output_fps = float(render_cfg.get("fps") or input_fps)
    if end_sec is None and frame_count > 0:
        end_sec = frame_count / input_fps
    if end_sec is None:
        end_sec = timeline.duration_sec
    end_sec = max(start_sec, end_sec)

    homography = load_homography(input_cfg.get("homography_path"))
    draw_video = bool(visual_cfg.get("draw_video_overlay", True))
    if draw_video and homography is None:
        print(
            "warning: homography file is missing or invalid; video AR overlay will be skipped",
            file=sys.stderr,
        )
        if mode == "ar_only":
            print("warning: ar_only output will mostly match the input video", file=sys.stderr)

    if mode == "ar_only":
        output_size = (width, height)
    elif mode == "wide_with_map":
        output_size = (width + height, height)
    else:
        raise ValueError(f"unknown render mode: {mode}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    codec = str(render_cfg.get("codec", "mp4v"))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        output_fps,
        output_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {output_path}")

    total_frames = int(round(max(0.0, end_sec - start_sec) * output_fps))
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    progress = _progress(total_frames)
    try:
        for out_idx in range(total_frames):
            video_t = start_sec + out_idx / output_fps
            source_frame = int(round(video_t * input_fps))
            if frame_count and source_frame >= frame_count:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            ok, frame = cap.read()
            if not ok:
                break

            bag_t = video_t + time_offset_sec
            state = timeline.state_at(bag_t)
            video_points: dict[str, tuple[int, int]] = {}
            if draw_video and homography is not None:
                video_points = draw_video_overlay(frame, state, homography, config)

            if mode == "ar_only":
                canvas = frame
            else:
                if bool(visual_cfg.get("draw_map", True)):
                    map_image, map_points = draw_map(height, state, config)
                else:
                    map_image = np.zeros((height, height, 3), dtype=np.uint8)
                    map_points = {}
                canvas = np.zeros((height, width + height, 3), dtype=np.uint8)
                canvas[:, :width] = frame
                canvas[:, width:] = map_image
                map_points = {
                    robot_id: (point[0] + width, point[1])
                    for robot_id, point in map_points.items()
                }
                if bool(visual_cfg.get("draw_connectors", True)) and video_points:
                    draw_connectors(canvas, video_points, map_points, state.robot_roles, config)

            writer.write(canvas)
            progress.update(1)
    finally:
        progress.close()
        writer.release()
        cap.release()

    print(f"wrote {output_path}")
    if webm_path:
        convert_to_webm(output_path, Path(webm_path))
        print(f"wrote {webm_path}")


def print_inspection(bag: Any, timeline: LimoTimeline) -> None:
    print(f"Bag: {bag.db3_path}")
    if bag.metadata_path:
        print(f"Metadata: {bag.metadata_path}")
    print(f"Duration: {bag.duration_sec:.3f} sec")
    print(f"Messages: {bag.message_count} total, {bag.skipped_messages} skipped")
    print("Topics:")
    for name, msg_type in sorted(bag.topics.items()):
        print(f"  {name}: {msg_type}")
    print("Timeline:")
    for key, value in timeline.summary().items():
        print(f"  {key}: {value}")


def convert_to_webm(input_path: Path, output_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for --webm-output, but it was not found")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libvpx-vp9",
        "-crf",
        "34",
        "-b:v",
        "0",
        "-an",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def _resolve_existing_or_relative(value: str, config_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    config_path = config_dir / path
    if config_path.exists():
        return config_path
    return cwd_path


def _progress(total: int):
    try:
        from tqdm import tqdm

        return tqdm(total=total, unit="frame")
    except Exception:
        return _PlainProgress(total)


class _PlainProgress:
    def __init__(self, total: int):
        self.total = total
        self.count = 0

    def update(self, amount: int) -> None:
        self.count += amount

    def close(self) -> None:
        if self.total:
            print(f"frames: {self.count}/{self.total}")


if __name__ == "__main__":
    raise SystemExit(main())
