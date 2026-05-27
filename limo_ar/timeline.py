from __future__ import annotations

import bisect
import json
import math
import re
from dataclasses import dataclass
from typing import Any

import yaml

from .bag_reader import BagData, PoseSample, quaternion_to_yaw


DEFAULT_TARGETS = [
    {"id": "Target_1", "x": 1.0, "y": 1.0, "active": True},
    {"id": "Target_2", "x": 1.0, "y": -1.0, "active": True},
]

DEFAULT_FIRES = [
    {"id": "Fire_1", "x": 0.5, "y": 1.5, "active": True},
    {"id": "Fire_2", "x": 0.5, "y": -0.5, "active": True},
    {"id": "Fire_3", "x": 1.2, "y": 1.2, "active": False},
    {"id": "Fire_4", "x": -1.0, "y": -1.0, "active": False},
]


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float
    z: float = 0.0


@dataclass(frozen=True)
class VisualObject:
    id: str
    x: float
    y: float
    active: bool
    status: str = "active"
    radius_scale: float = 1.0
    radius_m: float | None = None


@dataclass(frozen=True)
class FrameState:
    t: float
    robot_poses: dict[str, Pose2D]
    robot_roles: dict[str, str]
    robot_labels: dict[str, str]
    base_pose: Pose2D | None
    fires: dict[str, VisualObject]
    targets: dict[str, VisualObject]
    rescued_target_count: int
    robots_in_base: int


@dataclass(frozen=True)
class _ObjectEvent:
    t: float
    active: bool
    x: float | None = None
    y: float | None = None
    status: str = ""
    radius_m: float | None = None


class LimoTimeline:
    def __init__(self, bag: BagData, config: dict[str, Any]):
        self.bag = bag
        self.config = config
        self.visual_cfg = config.get("visual", {})
        self.base_area = config.get(
            "base_area",
            {"x_min": -2.0, "x_max": 0.0, "y_min": 0.0, "y_max": 2.0},
        )
        self.robot_roles = {
            robot.get("id"): robot.get("role", "robot")
            for robot in config.get("robots", [])
            if robot.get("id")
        }
        self.robot_labels = {
            robot.get("id"): robot.get("short_label") or robot.get("label") or robot.get("id")
            for robot in config.get("robots", [])
            if robot.get("id")
        }
        self.robot_pose_samples = {
            robot_id: _prepare_pose_samples(samples, self.visual_cfg)
            for robot_id, samples in bag.robot_poses.items()
        }
        self._robot_pose_times = {
            robot_id: [sample.t for sample in samples]
            for robot_id, samples in self.robot_pose_samples.items()
        }
        self._base_pose_times = [sample.t for sample in bag.base_poses]

        self._fire_initial = _load_static_objects(config, "fires", DEFAULT_FIRES)
        self._target_initial = _load_static_objects(config, "targets", DEFAULT_TARGETS)
        self._fire_events = self._build_events("fire", self._fire_initial)
        self._target_events = self._build_events("target", self._target_initial)
        self._fire_suppress_windows = self._build_fire_suppress_windows()
        self._target_done_times = self._first_inactive_times(self._target_events)

    @property
    def duration_sec(self) -> float:
        return self.bag.duration_sec

    def state_at(self, t: float) -> FrameState:
        robot_poses = {
            robot_id: pose
            for robot_id, pose in (
                (robot_id, self._interpolate_pose(samples, self._robot_pose_times[robot_id], t))
                for robot_id, samples in self.robot_pose_samples.items()
            )
            if pose is not None
        }
        base_pose = self._interpolate_pose(self.bag.base_poses, self._base_pose_times, t)
        fires = self._objects_at("fire", self._fire_initial, self._fire_events, t)
        targets = self._objects_at("target", self._target_initial, self._target_events, t)
        rescued_target_count = sum(done_t <= t for done_t in self._target_done_times.values())
        robots_in_base = sum(
            _point_in_area(pose.x, pose.y, self.base_area) for pose in robot_poses.values()
        )
        return FrameState(
            t=t,
            robot_poses=robot_poses,
            robot_roles=self.robot_roles,
            robot_labels=self.robot_labels,
            base_pose=base_pose,
            fires=fires,
            targets=targets,
            rescued_target_count=rescued_target_count,
            robots_in_base=robots_in_base,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "duration_sec": self.duration_sec,
            "robots": {
            robot_id: len(samples) for robot_id, samples in self.bag.robot_poses.items()
            },
            "base_pose_samples": len(self.bag.base_poses),
            "fire_state_messages": len(self.bag.fire_states),
            "target_state_messages": len(self.bag.target_states),
            "fires": sorted(self._fire_initial),
            "targets": sorted(self._target_initial),
            "skipped_messages": self.bag.skipped_messages,
        }

    def _build_events(
        self,
        kind: str,
        initial: dict[str, dict[str, float]],
    ) -> dict[str, list[_ObjectEvent]]:
        events: dict[str, list[_ObjectEvent]] = {
            object_id: [
                _ObjectEvent(
                    t=0.0,
                    active=bool(pos.get("active", True)),
                    x=pos.get("x"),
                    y=pos.get("y"),
                    status="active" if pos.get("active", True) else "inactive",
                )
            ]
            for object_id, pos in initial.items()
        }
        samples = self.bag.fire_states if kind == "fire" else self.bag.target_states
        known_ids = set(initial)
        current_active = {
            object_id: bool(object_events[-1].active)
            for object_id, object_events in events.items()
        }
        for sample in samples:
            parsed = parse_state_message(sample.text, kind, known_ids)
            for object_id, info in parsed.items():
                known_ids.add(object_id)
                if object_id not in initial:
                    initial[object_id] = {
                        "x": info.get("x", 0.0),
                        "y": info.get("y", 0.0),
                        "active": bool(info.get("active", True)),
                    }
                    events.setdefault(
                        object_id,
                        [
                            _ObjectEvent(
                                t=0.0,
                                active=bool(info.get("active", True)),
                                x=info.get("x"),
                                y=info.get("y"),
                                radius_m=info.get("radius_m"),
                            )
                        ],
                    )
                active = info.get("active")
                if active is None:
                    active = events.get(object_id, [_ObjectEvent(0.0, True)])[-1].active
                status = str(info.get("status") or ("active" if active else "inactive"))
                events.setdefault(object_id, []).append(
                    _ObjectEvent(
                        t=sample.t,
                        active=bool(active),
                        x=info.get("x"),
                        y=info.get("y"),
                        status=status,
                        radius_m=info.get("radius_m"),
                    )
                )
                current_active[object_id] = bool(active)

            if kind == "fire" and _message_has_authoritative_fire_list(sample.text):
                parsed_ids = set(parsed)
                for object_id in list(known_ids):
                    if current_active.get(object_id, False) and object_id not in parsed_ids:
                        events.setdefault(object_id, []).append(
                            _ObjectEvent(
                                t=sample.t,
                                active=False,
                                x=None,
                                y=None,
                                status="suppressed",
                            )
                        )
                        current_active[object_id] = False

        for object_events in events.values():
            object_events.sort(key=lambda event: event.t)
        return events

    def _objects_at(
        self,
        kind: str,
        initial: dict[str, dict[str, float]],
        events: dict[str, list[_ObjectEvent]],
        t: float,
    ) -> dict[str, VisualObject]:
        objects: dict[str, VisualObject] = {}
        for object_id, object_events in events.items():
            times = [event.t for event in object_events]
            idx = bisect.bisect_right(times, t) - 1
            event = object_events[max(idx, 0)]
            x = event.x if event.x is not None else initial.get(object_id, {}).get("x", 0.0)
            y = event.y if event.y is not None else initial.get(object_id, {}).get("y", 0.0)
            radius_m = event.radius_m or initial.get(object_id, {}).get("radius_m")
            active = event.active
            radius_scale = 1.0

            if kind == "fire":
                suppress_window = self._fire_suppress_windows.get(object_id)
                if active and suppress_window:
                    start, end = suppress_window
                    if start <= t <= end:
                        span = max(end - start, 1e-9)
                        radius_scale = max(0.0, min(1.0, (end - t) / span))
                if not active:
                    continue

            if kind == "target" and not active:
                continue

            objects[object_id] = VisualObject(
                id=object_id,
                x=float(x),
                y=float(y),
                active=active,
                status=event.status,
                radius_scale=radius_scale,
                radius_m=radius_m,
            )
        return objects

    def _interpolate_pose(
        self,
        samples: list[PoseSample],
        times: list[float],
        t: float,
    ) -> Pose2D | None:
        if not samples:
            return None
        if t <= times[0]:
            return _pose2d(samples[0])
        if t >= times[-1]:
            return _pose2d(samples[-1])

        idx = bisect.bisect_right(times, t)
        left = samples[idx - 1]
        right = samples[idx]
        span = max(right.t - left.t, 1e-9)
        alpha = (t - left.t) / span
        left_yaw = quaternion_to_yaw(left.qx, left.qy, left.qz, left.qw)
        right_yaw = quaternion_to_yaw(right.qx, right.qy, right.qz, right.qw)
        yaw_delta = (right_yaw - left_yaw + math.pi) % (2.0 * math.pi) - math.pi
        return Pose2D(
            x=left.x + (right.x - left.x) * alpha,
            y=left.y + (right.y - left.y) * alpha,
            z=left.z + (right.z - left.z) * alpha,
            yaw=left_yaw + yaw_delta * alpha,
        )

    def _build_fire_suppress_windows(self) -> dict[str, tuple[float, float]]:
        windows: dict[str, tuple[float, float]] = {}
        fire_robot_ids = [
            robot_id
            for robot_id, role in self.robot_roles.items()
            if role == "fire" and robot_id in self.robot_pose_samples
        ]
        fallback_duration = float(self.visual_cfg.get("fire_suppress_duration_sec", 2.0))
        lookback = float(self.visual_cfg.get("fire_suppress_max_window_sec", 10.0))
        start_distance = float(self.visual_cfg.get("fire_suppress_start_distance_m", 0.55))
        min_duration = float(self.visual_cfg.get("fire_suppress_min_duration_sec", 1.0))

        for fire_id, events in self._fire_events.items():
            inactive = _first_inactive_after_active(events)
            if inactive is None:
                continue
            fire_x, fire_y = _object_position_at(
                events,
                self._fire_initial.get(fire_id, {}),
                inactive.t,
            )
            arrival_start = self._first_fire_robot_arrival_time(
                fire_robot_ids,
                fire_x,
                fire_y,
                max(0.0, inactive.t - lookback),
                inactive.t,
                start_distance,
            )
            if arrival_start is None:
                start = inactive.t - fallback_duration
                if min_duration > 0.0 and inactive.t - start < min_duration:
                    start = inactive.t - min_duration
            else:
                start = arrival_start
            start = max(0.0, min(start, inactive.t))
            if inactive.t > start:
                windows[fire_id] = (start, inactive.t)
        return windows

    def _first_fire_robot_arrival_time(
        self,
        robot_ids: list[str],
        fire_x: float,
        fire_y: float,
        start_t: float,
        end_t: float,
        distance_m: float,
    ) -> float | None:
        first: float | None = None
        for robot_id in robot_ids:
            for sample in self.robot_pose_samples.get(robot_id, []):
                if sample.t < start_t:
                    continue
                if sample.t > end_t:
                    break
                if math.hypot(sample.x - fire_x, sample.y - fire_y) <= distance_m:
                    first = sample.t if first is None else min(first, sample.t)
                    break
        return first

    @staticmethod
    def _first_inactive_times(events: dict[str, list[_ObjectEvent]]) -> dict[str, float]:
        done: dict[str, float] = {}
        for object_id, object_events in events.items():
            for event in object_events:
                if not event.active:
                    done[object_id] = event.t
                    break
        return done

    def robot_pose_summary(self) -> dict[str, dict[str, Any]]:
        summary: dict[str, dict[str, Any]] = {}
        for robot_id, samples in self.robot_pose_samples.items():
            if not samples:
                summary[robot_id] = {"samples": 0}
                continue
            summary[robot_id] = {
                "samples": len(samples),
                "first": _pose2d(samples[0]),
                "last": _pose2d(samples[-1]),
            }
        return summary

    def fire_event_summary(self) -> dict[str, dict[str, Any]]:
        return self._event_summary(self._fire_events, first_seen_active=True)

    def target_event_summary(self) -> dict[str, dict[str, Any]]:
        return self._event_summary(self._target_events, first_seen_active=True)

    @staticmethod
    def _event_summary(
        events: dict[str, list[_ObjectEvent]],
        first_seen_active: bool,
    ) -> dict[str, dict[str, Any]]:
        summary: dict[str, dict[str, Any]] = {}
        for object_id, object_events in sorted(events.items()):
            first_seen = None
            inactive = None
            x = None
            y = None
            for event in object_events:
                if first_seen is None and (event.active or not first_seen_active):
                    first_seen = event.t
                if event.x is not None and event.y is not None:
                    x = event.x
                    y = event.y
                if event.active and first_seen is None:
                    first_seen = event.t
                if inactive is None and not event.active and first_seen is not None:
                    inactive = event.t
            summary[object_id] = {
                "first_seen": first_seen,
                "inactive": inactive,
                "position": (x, y) if x is not None and y is not None else None,
            }
        return summary


def parse_state_message(
    text: str,
    kind: str,
    known_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    known_ids = known_ids or set()
    loaded = _load_structured_text(text)

    if isinstance(loaded, dict):
        parsed = _parse_project_state_dict(loaded, kind)
        if parsed:
            return parsed

    parsed: dict[str, dict[str, Any]] = {}
    if loaded is not None and not isinstance(loaded, str):
        _collect_structured(loaded, kind, parsed)
        if parsed:
            return parsed

    _collect_from_regex(text, kind, parsed)

    lowered = text.lower()
    if kind == "target" and "alltargetcompleted" in lowered:
        for object_id in known_ids:
            parsed.setdefault(object_id, {})["active"] = False
            parsed[object_id]["status"] = "completed"
    if kind == "fire" and "allfiresuppressed" in lowered:
        for object_id in known_ids:
            parsed.setdefault(object_id, {})["active"] = False
            parsed[object_id]["status"] = "suppressed"
    return parsed


def _parse_project_state_dict(obj: dict[str, Any], kind: str) -> dict[str, dict[str, Any]]:
    root_key = "fires" if kind == "fire" else "targets"
    items = obj.get(root_key)
    if not isinstance(items, list):
        return {}
    parsed: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        object_id = _normalize_id(item.get("id") or item.get("name"), kind)
        if not object_id:
            continue
        info = _info_from_mapping(item)
        if "status_name" in item:
            status = str(item["status_name"])
            info["status"] = status
            active_from_status = _active_from_status(status)
            if active_from_status is not None:
                info["active"] = active_from_status
        if "active" in item:
            info["active"] = bool(item["active"])
        parsed[object_id] = info
    return parsed


def _message_has_authoritative_fire_list(text: str) -> bool:
    try:
        obj = json.loads(text)
    except Exception:
        return False
    return isinstance(obj, dict) and isinstance(obj.get("fires"), list)


def _load_static_objects(
    config: dict[str, Any],
    name: str,
    defaults: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    object_cfg = config.get("objects", {}).get(name)
    objects = object_cfg if isinstance(object_cfg, list) and object_cfg else defaults
    output: dict[str, dict[str, float]] = {}
    for obj in objects:
        object_id = str(obj.get("id") or obj.get("name") or "")
        if not object_id:
            continue
        output[object_id] = {
            "x": float(obj.get("x", 0.0)),
            "y": float(obj.get("y", 0.0)),
            "active": bool(obj.get("active", True)),
            "radius_m": float(obj["radius_m"]) if "radius_m" in obj else None,
        }
    return output


def _prepare_pose_samples(samples: list[PoseSample], visual_cfg: dict[str, Any]) -> list[PoseSample]:
    if not samples or not bool(visual_cfg.get("pose_preprocess_enabled", True)):
        return samples
    ordered = sorted(samples, key=lambda sample: sample.t)
    clustered = _collapse_pose_time_clusters(
        ordered,
        float(visual_cfg.get("pose_cluster_dt_sec", 0.002)),
    )
    filtered = _reject_pose_spikes(
        clustered,
        float(visual_cfg.get("pose_filter_max_speed_mps", 2.4)),
        float(visual_cfg.get("pose_filter_step_margin_m", 0.04)),
    )
    return _smooth_pose_samples(
        filtered,
        float(visual_cfg.get("pose_preprocess_window_sec", 0.18)),
    )


def _collapse_pose_time_clusters(samples: list[PoseSample], cluster_dt: float) -> list[PoseSample]:
    if cluster_dt <= 0.0 or len(samples) < 2:
        return samples
    output: list[PoseSample] = []
    group: list[PoseSample] = []

    def flush() -> None:
        if not group:
            return
        if len(group) == 1:
            output.append(group[0])
            return
        if output:
            prev = output[-1]
            chosen = min(group, key=lambda sample: _pose_distance(prev, sample))
        else:
            chosen = group[len(group) // 2]
        output.append(chosen)

    for sample in samples:
        if not group:
            group = [sample]
            continue
        if sample.t - group[-1].t <= cluster_dt:
            group.append(sample)
            continue
        flush()
        group = [sample]
    flush()
    return output


def _reject_pose_spikes(
    samples: list[PoseSample],
    max_speed_mps: float,
    step_margin_m: float,
) -> list[PoseSample]:
    if len(samples) < 3 or max_speed_mps <= 0.0:
        return samples
    output = [samples[0]]
    for sample in samples[1:]:
        prev = output[-1]
        dt = sample.t - prev.t
        if dt <= 0.0:
            continue
        distance = _pose_distance(prev, sample)
        allowed = step_margin_m + max_speed_mps * dt
        if distance > allowed and dt < 0.18:
            continue
        output.append(sample)
    return output


def _smooth_pose_samples(samples: list[PoseSample], window_sec: float) -> list[PoseSample]:
    if len(samples) < 3 or window_sec <= 0.0:
        return samples
    times = [sample.t for sample in samples]
    half_window = window_sec * 0.5
    sigma = max(window_sec / 3.0, 1e-6)
    output: list[PoseSample] = []
    left = 0
    right = 0
    for idx, sample in enumerate(samples):
        t = sample.t
        while left < len(samples) and times[left] < t - half_window:
            left += 1
        while right < len(samples) and times[right] <= t + half_window:
            right += 1
        weight_sum = 0.0
        x = y = z = qx = qy = qz = qw = 0.0
        for item in samples[left:right]:
            dt = item.t - t
            weight = math.exp(-0.5 * (dt / sigma) ** 2)
            weight_sum += weight
            x += item.x * weight
            y += item.y * weight
            z += item.z * weight
            qx += item.qx * weight
            qy += item.qy * weight
            qz += item.qz * weight
            qw += item.qw * weight
        if weight_sum <= 0.0:
            output.append(sample)
            continue
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm <= 1e-9:
            qx, qy, qz, qw = sample.qx, sample.qy, sample.qz, sample.qw
        else:
            qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
        output.append(
            PoseSample(
                t=sample.t,
                x=x / weight_sum,
                y=y / weight_sum,
                z=z / weight_sum,
                qx=qx,
                qy=qy,
                qz=qz,
                qw=qw,
                frame_id=sample.frame_id,
            )
        )
    return output


def _pose_distance(left: PoseSample, right: PoseSample) -> float:
    return math.hypot(left.x - right.x, left.y - right.y)


def _pose2d(sample: PoseSample) -> Pose2D:
    return Pose2D(
        x=sample.x,
        y=sample.y,
        z=sample.z,
        yaw=quaternion_to_yaw(sample.qx, sample.qy, sample.qz, sample.qw),
    )


def _point_in_area(x: float, y: float, area: dict[str, float]) -> bool:
    return (
        float(area.get("x_min", -math.inf)) <= x <= float(area.get("x_max", math.inf))
        and float(area.get("y_min", -math.inf)) <= y <= float(area.get("y_max", math.inf))
    )


def _first_inactive_after_active(events: list[_ObjectEvent]) -> _ObjectEvent | None:
    seen_active = False
    for event in events:
        if event.active:
            seen_active = True
        elif seen_active:
            return event
    return None


def _object_position_at(
    events: list[_ObjectEvent],
    initial: dict[str, Any],
    t: float,
) -> tuple[float, float]:
    x = initial.get("x", 0.0)
    y = initial.get("y", 0.0)
    for event in events:
        if event.t > t:
            break
        if event.x is not None:
            x = event.x
        if event.y is not None:
            y = event.y
    return float(x), float(y)


def _load_structured_text(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except Exception:
        pass
    if any(token in stripped for token in (":", "{", "[", "- ")):
        try:
            return yaml.safe_load(stripped)
        except Exception:
            return None
    return None


def _collect_structured(obj: Any, kind: str, out: dict[str, dict[str, Any]]) -> None:
    if isinstance(obj, list):
        for item in obj:
            _collect_structured(item, kind, out)
        return
    if not isinstance(obj, dict):
        return

    object_id = _normalize_id(obj.get("id") or obj.get("name"), kind)
    if object_id:
        out.setdefault(object_id, {}).update(_info_from_mapping(obj))

    for key, value in obj.items():
        key_text = str(key)
        normalized = _normalize_id(key_text, kind)
        status_active = _active_from_status(key_text)
        if normalized:
            out.setdefault(normalized, {}).update(_info_from_value(value))
        elif key_text.lower() in {"fires", "fire", "targets", "target", "objects"}:
            _collect_structured(value, kind, out)
        elif status_active is not None:
            for item_id in _ids_from_value(value, kind):
                out.setdefault(item_id, {})["active"] = status_active
                out[item_id]["status"] = key_text
        elif isinstance(value, (dict, list)):
            _collect_structured(value, kind, out)


def _collect_from_regex(text: str, kind: str, out: dict[str, dict[str, Any]]) -> None:
    for match in re.finditer(r"\b(?:fire|target)[_-]?\d+\b", text, flags=re.IGNORECASE):
        object_id = _normalize_id(match.group(0), kind)
        if not object_id:
            continue
        window = _segment_around_match(text, match.start(), match.end())
        info = out.setdefault(object_id, {})
        active = _active_from_status(window)
        if active is not None:
            info["active"] = active
            info["status"] = _status_word(window) or ("active" if active else "inactive")
        x, y = _xy_from_text(window)
        if x is not None and y is not None:
            info["x"] = x
            info["y"] = y


def _normalize_id(raw: Any, kind: str) -> str | None:
    if raw is None:
        return None
    text = str(raw)
    if "limo" in text.lower():
        return None
    pattern = r"fire[_-]?(\d+)" if kind == "fire" else r"target[_-]?(\d+)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return ("Fire" if kind == "fire" else "Target") + f"_{int(match.group(1))}"


def _segment_around_match(text: str, start: int, end: int) -> str:
    left = 0
    for separator in (";", "\n", "|"):
        pos = text.rfind(separator, 0, start)
        if pos >= 0:
            left = max(left, pos + 1)
    right = len(text)
    for separator in (";", "\n", "|"):
        pos = text.find(separator, end)
        if pos >= 0:
            right = min(right, pos)
    return text[left:right]


def _info_from_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    info: dict[str, Any] = {}
    for key in ("x", "y"):
        if key in mapping:
            info[key] = float(mapping[key])
    if "position" in mapping and isinstance(mapping["position"], dict):
        position = mapping["position"]
        if "x" in position and "y" in position:
            info["x"] = float(position["x"])
            info["y"] = float(position["y"])
    elif "position" in mapping and isinstance(mapping["position"], (list, tuple)):
        position = mapping["position"]
        if len(position) >= 2:
            info["x"] = float(position[0])
            info["y"] = float(position[1])
    if "radius" in mapping and mapping["radius"] is not None:
        info["radius_m"] = float(mapping["radius"])
    if "radius_m" in mapping and mapping["radius_m"] is not None:
        info["radius_m"] = float(mapping["radius_m"])
    for key in ("active", "is_active", "enabled"):
        if key in mapping:
            info["active"] = bool(mapping[key])
    for key in ("state", "status", "phase", "status_name"):
        if key in mapping:
            status = str(mapping[key])
            info["status"] = status
            active = _active_from_status(status)
            if active is not None:
                info["active"] = active
    return info


def _info_from_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return _info_from_mapping(value)
    if isinstance(value, bool):
        return {"active": value, "status": "active" if value else "inactive"}
    if isinstance(value, str):
        active = _active_from_status(value)
        return {"active": active, "status": value} if active is not None else {"status": value}
    return {}


def _ids_from_value(value: Any, kind: str) -> list[str]:
    if isinstance(value, list):
        ids: list[str] = []
        for item in value:
            ids.extend(_ids_from_value(item, kind))
        return ids
    normalized = _normalize_id(value, kind)
    return [normalized] if normalized else []


def _active_from_status(text: str) -> bool | None:
    lowered = text.lower()
    false_words = (
        "inactive",
        "suppressed",
        "rescued",
        "completed",
        "complete",
        "done",
        "false",
        "off",
        "cleared",
    )
    true_words = ("active", "burning", "pending", "waiting", "true", "on", "alive")
    if any(word in lowered for word in false_words):
        return False
    if any(word in lowered for word in true_words):
        return True
    return None


def _status_word(text: str) -> str | None:
    match = re.search(
        r"\b(inactive|suppressed|rescued|completed|complete|active|burning|pending|waiting)\b",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else None


def _xy_from_text(text: str) -> tuple[float | None, float | None]:
    x_match = re.search(r"\bx\s*[:=]\s*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    y_match = re.search(r"\by\s*[:=]\s*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if x_match and y_match:
        return float(x_match.group(1)), float(y_match.group(1))
    pair = re.search(r"\((-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\)", text)
    if pair:
        return float(pair.group(1)), float(pair.group(2))
    return None, None
