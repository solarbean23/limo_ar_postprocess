from __future__ import annotations

import argparse
from dataclasses import replace
import math
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
from .drawing import draw_connectors, draw_info_panel, draw_map, draw_video_overlay
from .timeline import FrameState, LimoTimeline, Pose2D
from .video_tracking import VideoRobotTracker, state_with_tracked_robot_positions


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
    parser.add_argument("--time-scale", type=float)
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
        print_inspection(bag, timeline, config)
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
    if args.time_scale is not None:
        config["sync"]["time_scale"] = args.time_scale


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
    time_scale = float(sync_cfg.get("time_scale", 1.0))
    sample_at_source_frame_time = bool(sync_cfg.get("sample_at_source_frame_time", False))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")

    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if raw_width <= 0 or raw_height <= 0:
        raise RuntimeError("could not read video dimensions")
    width, height = _render_frame_size(raw_width, raw_height, render_cfg)

    output_fps = float(render_cfg.get("fps") or input_fps)
    if end_sec is None and frame_count > 0:
        end_sec = frame_count / input_fps
    if end_sec is None:
        end_sec = timeline.duration_sec
    end_sec = max(start_sec, end_sec)

    homography_path = input_cfg.get("homography_path")
    homography = None
    homography_error = None
    if homography_path:
        try:
            homography = load_homography(homography_path)
        except Exception as exc:
            homography_error = str(exc)
    draw_video = bool(visual_cfg.get("draw_video_overlay", True))
    if draw_video and homography is None:
        if homography_error:
            print(f"[WARN] Homography invalid: {homography_path} ({homography_error})", file=sys.stderr)
        else:
            print(f"[WARN] Homography missing: {homography_path}", file=sys.stderr)
        print("[WARN] Video AR overlay disabled. Run calibration or provide homography.yaml.", file=sys.stderr)
        if mode == "ar_only":
            print(
                "[WARN] ar_only mode without homography will be almost identical to the input video.",
                file=sys.stderr,
            )

    side_panel_width = height
    side_map_size = height
    side_info_height = 0
    if mode == "wide_with_map" and bool(visual_cfg.get("draw_map", True)) and bool(visual_cfg.get("info_panel_enabled", True)):
        side_info_height = _info_panel_height(height, visual_cfg)
        side_map_size = height - side_info_height
        side_panel_width = side_map_size

    if mode == "ar_only":
        output_size = (width, height)
    elif mode == "wide_with_map":
        output_size = (width + side_panel_width, height)
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
    smoother = StateSmoother(config)
    resize_to = (width, height) if (width, height) != (raw_width, raw_height) else None
    frame_sampler = VideoFrameSampler(
        cap,
        input_fps,
        frame_count,
        render_cfg.get("video_sampling", "blend"),
        resize_to=resize_to,
    )
    robot_tracker = VideoRobotTracker(homography, config) if draw_video and homography is not None else None
    try:
        for out_idx in range(total_frames):
            video_t = start_sec + out_idx / output_fps
            source_frame = int(math.floor(video_t * input_fps))
            if frame_count and source_frame >= frame_count:
                break
            frame = frame_sampler.frame_at(video_t)
            if frame is None:
                break

            sample_video_t = source_frame / input_fps if sample_at_source_frame_time else video_t
            bag_t = sample_video_t * time_scale + time_offset_sec
            state = smoother.smooth(timeline.state_at(bag_t), bag_t)
            trails = build_robot_trails(timeline, bag_t, config)
            tracked_points: dict[str, tuple[int, int]] = {}
            map_state = state
            video_points: dict[str, tuple[int, int]] = {}
            if draw_video and homography is not None:
                if robot_tracker is not None:
                    tracked_points = robot_tracker.update(frame, state)
                    map_state = state_with_tracked_robot_positions(
                        state,
                        tracked_points,
                        homography,
                        config,
                    )
                video_points = draw_video_overlay(
                    frame,
                    state,
                    homography,
                    config,
                    trails,
                    tracked_points,
                )

            if mode == "ar_only":
                canvas = frame
            else:
                if bool(visual_cfg.get("draw_map", True)):
                    if bool(visual_cfg.get("info_panel_enabled", True)):
                        info_height = side_info_height
                        map_size = side_map_size
                        panel_width = side_panel_width
                        right_panel = np.full((height, panel_width, 3), (20, 22, 24), dtype=np.uint8)
                        map_x = 0
                        map_y = 0
                        map_image, map_points = draw_map(map_size, map_state, config, trails)
                        right_panel[map_y : map_y + map_size, map_x : map_x + map_size] = map_image
                        draw_info_panel(right_panel[map_size:height, :panel_width], map_state, config)
                    else:
                        right_panel = None
                        map_x = 0
                        map_y = 0
                        map_image, map_points = draw_map(height, map_state, config, trails)
                else:
                    right_panel = None
                    map_x = 0
                    map_y = 0
                    map_image = np.zeros((height, height, 3), dtype=np.uint8)
                    map_points = {}
                panel_width = right_panel.shape[1] if right_panel is not None else map_image.shape[1]
                canvas = np.zeros((height, width + panel_width, 3), dtype=np.uint8)
                canvas[:, :width] = frame
                canvas[:, width:] = right_panel if right_panel is not None else map_image
                map_points = {
                    robot_id: (point[0] + width + map_x, point[1] + map_y)
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


def print_inspection(bag: Any, timeline: LimoTimeline, config: dict[str, Any]) -> None:
    print(f"Bag: {bag.db3_path}")
    if bag.metadata_path:
        print(f"Metadata: {bag.metadata_path}")
    print(f"Duration: {bag.duration_sec:.3f} sec")
    print(f"Messages: {bag.message_count} total, {bag.skipped_messages} skipped")
    video_meta = read_video_metadata(config.get("input", {}).get("video_path"))
    if video_meta:
        print(
            "Video: "
            f"{video_meta['path']} "
            f"{video_meta['width']}x{video_meta['height']} "
            f"{video_meta['fps']:.3f}fps "
            f"{video_meta['frames']} frames"
        )
    homography_path = config.get("input", {}).get("homography_path")
    homography_present = bool(homography_path and Path(homography_path).exists())
    print("Config:")
    print(f"  mode={config.get('render', {}).get('mode', 'wide_with_map')}")
    resize_height = config.get("render", {}).get("resize_height")
    if resize_height:
        print(f"  resize_height={resize_height}")
    print(f"  map.display_transform={config.get('map', {}).get('display_transform', 'none')}")
    print(f"  time_offset_sec={config.get('sync', {}).get('time_offset_sec', 0.0)}")
    print(f"  time_scale={config.get('sync', {}).get('time_scale', 1.0)}")
    print(
        "  sample_at_source_frame_time="
        f"{config.get('sync', {}).get('sample_at_source_frame_time', False)}"
    )
    print(f"  video_sampling={config.get('render', {}).get('video_sampling', 'blend')}")
    print(f"  homography={'present' if homography_present else 'missing'}")
    print("Topics:")
    for name, msg_type in sorted(bag.topics.items()):
        print(f"  {name}: {msg_type}")
    print("Robots:")
    for robot_id, item in timeline.robot_pose_summary().items():
        if not item.get("samples"):
            print(f"  {robot_id}: 0 samples")
            continue
        first = item["first"]
        last = item["last"]
        print(
            f"  {robot_id}: {item['samples']} samples, "
            f"first=({first.x:.2f}, {first.y:.2f}), "
            f"last=({last.x:.2f}, {last.y:.2f})"
        )
    print("Targets:")
    for object_id, item in timeline.target_event_summary().items():
        position = _format_position(item.get("position"))
        completed = item.get("inactive")
        completed_text = f"{completed:.2f}s" if completed is not None else "none"
        first_seen = item.get("first_seen")
        first_seen_text = f"{first_seen:.2f}s" if first_seen is not None else "none"
        print(
            f"  {object_id}: first_seen={first_seen_text}, "
            f"completed={completed_text}, position={position}"
        )
    print("Fires:")
    for object_id, item in timeline.fire_event_summary().items():
        position = _format_position(item.get("position"))
        suppressed = item.get("inactive")
        suppressed_text = f"{suppressed:.2f}s" if suppressed is not None else "none"
        first_seen = item.get("first_seen")
        first_seen_text = f"{first_seen:.2f}s" if first_seen is not None else "none"
        print(
            f"  {object_id}: first_seen={first_seen_text}, "
            f"suppressed={suppressed_text}, position={position}"
        )
    print("State probes:")
    for t in (0.0, 1.0, 23.0, 36.0, 50.2, 56.7, 64.0):
        state = timeline.state_at(t)
        print(
            f"  t={t:.1f}: "
            f"fires={sorted(state.fires)}, "
            f"targets={sorted(state.targets)}"
        )


def build_robot_trails(
    timeline: LimoTimeline,
    bag_t: float,
    config: dict[str, Any],
) -> dict[str, list[Pose2D]]:
    visual = config.get("visual", {})
    if not bool(visual.get("draw_robot_trails", True)):
        return {}
    duration = float(visual.get("trail_duration_sec", 8.0))
    step = max(0.1, float(visual.get("trail_step_sec", 0.5)))
    start = max(0.0, bag_t - duration)
    times = np.arange(start, bag_t + 1e-6, step)
    trails: dict[str, list[Pose2D]] = {}
    for t in times:
        state = timeline.state_at(float(t))
        for robot_id, pose in state.robot_poses.items():
            trails.setdefault(robot_id, []).append(pose)
    return trails


def _render_frame_size(
    raw_width: int,
    raw_height: int,
    render_cfg: dict[str, Any],
) -> tuple[int, int]:
    resize_width = render_cfg.get("resize_width")
    resize_height = render_cfg.get("resize_height")
    if resize_width is None and resize_height is None:
        return raw_width, raw_height
    if resize_width is not None and resize_height is not None:
        return int(resize_width), int(resize_height)
    if resize_height is not None:
        height = int(resize_height)
        width = int(round(raw_width * height / max(raw_height, 1)))
        return width, height
    width = int(resize_width)
    height = int(round(raw_height * width / max(raw_width, 1)))
    return width, height


def _info_panel_height(height: int, visual_cfg: dict[str, Any]) -> int:
    ratio = float(visual_cfg.get("info_panel_height_ratio", 0.32))
    info_height = int(round(height * ratio))
    min_height = int(visual_cfg.get("info_panel_min_height_px", 210))
    max_height = max(min_height, height - int(visual_cfg.get("map_panel_min_size_px", 430)))
    return max(min_height, min(max_height, info_height))


class VideoFrameSampler:
    def __init__(
        self,
        cap: cv2.VideoCapture,
        fps: float,
        frame_count: int,
        mode: str,
        resize_to: tuple[int, int] | None = None,
    ):
        self.cap = cap
        self.fps = fps
        self.frame_count = frame_count
        self.mode = str(mode or "nearest")
        self.resize_to = resize_to
        self.cache: dict[int, np.ndarray] = {}
        self.last_read_index = -1

    def frame_at(self, t: float) -> np.ndarray | None:
        frame_pos = max(0.0, t * self.fps)
        if self.mode == "blend":
            left_idx = int(math.floor(frame_pos))
            right_idx = left_idx + 1
            left = self._frame(left_idx)
            if left is None:
                return None
            right = self._frame(right_idx)
            if right is None:
                return left.copy()
            alpha = frame_pos - left_idx
            if alpha <= 1e-6:
                return left.copy()
            if alpha >= 1.0 - 1e-6:
                return right.copy()
            return cv2.addWeighted(left, 1.0 - alpha, right, alpha, 0.0)

        idx = int(round(frame_pos))
        frame = self._frame(idx)
        return frame.copy() if frame is not None else None

    def _frame(self, idx: int) -> np.ndarray | None:
        if self.frame_count:
            idx = min(max(idx, 0), self.frame_count - 1)
        if idx in self.cache:
            return self.cache[idx]
        if idx != self.last_read_index + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.last_read_index = idx
        if self.resize_to is not None:
            frame = cv2.resize(frame, self.resize_to, interpolation=cv2.INTER_AREA)
        self.cache[idx] = frame
        if len(self.cache) > 4:
            for key in sorted(self.cache)[:-4]:
                self.cache.pop(key, None)
        return frame


class StateSmoother:
    def __init__(self, config: dict[str, Any]):
        visual = config.get("visual", {})
        self.tau = float(visual.get("pose_smoothing_tau_sec", 0.32))
        self.previous_t: float | None = None
        self.previous_poses: dict[str, Pose2D] = {}

    def smooth(self, state: FrameState, t: float) -> FrameState:
        if self.tau <= 0.0:
            return state
        if self.previous_t is None:
            self.previous_t = t
            self.previous_poses = dict(state.robot_poses)
            return state
        dt = max(0.0, t - self.previous_t)
        alpha = 1.0 if dt <= 0.0 else 1.0 - math.exp(-dt / self.tau)
        smoothed: dict[str, Pose2D] = {}
        for robot_id, pose in state.robot_poses.items():
            prev = self.previous_poses.get(robot_id)
            if prev is None:
                smoothed[robot_id] = pose
                continue
            yaw_delta = (pose.yaw - prev.yaw + math.pi) % (2.0 * math.pi) - math.pi
            yaw = prev.yaw + yaw_delta * alpha
            yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi
            smoothed[robot_id] = Pose2D(
                x=prev.x + (pose.x - prev.x) * alpha,
                y=prev.y + (pose.y - prev.y) * alpha,
                z=prev.z + (pose.z - prev.z) * alpha,
                yaw=yaw,
            )
        self.previous_t = t
        self.previous_poses = smoothed
        return replace(state, robot_poses=smoothed)


def read_video_metadata(video_path: str | None) -> dict[str, Any] | None:
    if not video_path:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return {
            "path": str(video_path),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": fps,
            "frames": frames,
        }
    finally:
        cap.release()


def _format_position(position: Any) -> str:
    if not position:
        return "unknown"
    x, y = position
    return f"({x:.2f}, {y:.2f})"


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
        "-row-mt",
        "1",
        "-cpu-used",
        "4",
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
