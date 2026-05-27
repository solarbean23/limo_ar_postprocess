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
        self._robot_pose_times = {
            robot_id: [sample.t for sample in samples]
            for robot_id, samples in bag.robot_poses.items()
        }
        self._base_pose_times = [sample.t for sample in bag.base_poses]

        self._fire_initial = _load_static_objects(config, "fires", DEFAULT_FIRES)
        self._target_initial = _load_static_objects(config, "targets", DEFAULT_TARGETS)
        self._fire_events = self._build_events("fire", self._fire_initial)
        self._target_events = self._build_events("target", self._target_initial)
        self._target_done_times = self._first_inactive_times(self._target_events)

    @property
    def duration_sec(self) -> float:
        return self.bag.duration_sec

    def state_at(self, t: float) -> FrameState:
        robot_poses = {
            robot_id: pose
            for robot_id, pose in (
                (robot_id, self._interpolate_pose(samples, self._robot_pose_times[robot_id], t))
                for robot_id, samples in self.bag.robot_poses.items()
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
                    )
                )

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
            active = event.active
            radius_scale = 1.0

            if kind == "fire":
                next_inactive = _next_inactive_event(object_events, t)
                duration = float(self.visual_cfg.get("fire_suppress_duration_sec", 2.0))
                if active and next_inactive and duration > 0:
                    start = next_inactive.t - duration
                    if start <= t <= next_inactive.t:
                        radius_scale = max(0.0, min(1.0, (next_inactive.t - t) / duration))
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

    @staticmethod
    def _first_inactive_times(events: dict[str, list[_ObjectEvent]]) -> dict[str, float]:
        done: dict[str, float] = {}
        for object_id, object_events in events.items():
            for event in object_events:
                if not event.active:
                    done[object_id] = event.t
                    break
        return done


def parse_state_message(
    text: str,
    kind: str,
    known_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    known_ids = known_ids or set()
    parsed: dict[str, dict[str, Any]] = {}
    loaded = _load_structured_text(text)
    if loaded is not None and not isinstance(loaded, str):
        _collect_structured(loaded, kind, parsed)
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
        }
    return output


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


def _next_inactive_event(events: list[_ObjectEvent], t: float) -> _ObjectEvent | None:
    for event in events:
        if event.t >= t and not event.active:
            return event
    return None


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
    for key in ("active", "is_active", "enabled"):
        if key in mapping:
            info["active"] = bool(mapping[key])
    for key in ("state", "status", "phase"):
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
