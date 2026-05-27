from __future__ import annotations

import math
import sqlite3
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PoseSample:
    t: float
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    frame_id: str = ""


@dataclass(frozen=True)
class StateSample:
    t: float
    text: str


@dataclass
class BagData:
    db3_path: Path
    metadata_path: Path | None
    metadata: dict[str, Any] = field(default_factory=dict)
    topics: dict[str, str] = field(default_factory=dict)
    robot_poses: dict[str, list[PoseSample]] = field(default_factory=dict)
    base_poses: list[PoseSample] = field(default_factory=list)
    fire_states: list[StateSample] = field(default_factory=list)
    target_states: list[StateSample] = field(default_factory=list)
    start_time_ns: int = 0
    end_time_ns: int = 0
    message_count: int = 0
    skipped_messages: int = 0

    @property
    def duration_sec(self) -> float:
        if self.end_time_ns <= self.start_time_ns:
            return 0.0
        return (self.end_time_ns - self.start_time_ns) / 1e9


def load_metadata(metadata_path: str | Path | None) -> dict[str, Any]:
    if not metadata_path:
        return {}
    path = Path(metadata_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def read_bag_from_config(config: dict[str, Any]) -> BagData:
    input_cfg = config.get("input", {})
    db3_path = input_cfg.get("bag_path")
    if not db3_path:
        raise ValueError("config input.bag_path is required")
    metadata_path = input_cfg.get("metadata_path")
    if not metadata_path:
        candidate = Path(db3_path).with_name("metadata.yaml")
        metadata_path = candidate if candidate.exists() else None
    return read_bag(db3_path, metadata_path, config)


def read_bag(
    db3_path: str | Path,
    metadata_path: str | Path | None,
    config: dict[str, Any],
) -> BagData:
    db3 = Path(db3_path)
    if not db3.exists():
        raise FileNotFoundError(f"bag db3 not found: {db3}")

    metadata = load_metadata(metadata_path)
    topics_cfg = config.get("topics", {})
    robots_cfg = config.get("robots", [])

    robot_topic_to_id = {
        robot["pose_topic"]: robot["id"]
        for robot in robots_cfg
        if robot.get("pose_topic") and robot.get("id")
    }
    base_pose_topic = topics_cfg.get("base_pose", "/world/base/pose")
    fire_state_topic = topics_cfg.get("fire_state", "/world/fire/state")
    target_state_topic = topics_cfg.get("target_state", "/world/target/state")

    selected_topics = set(robot_topic_to_id)
    selected_topics.update([base_pose_topic, fire_state_topic, target_state_topic])

    with sqlite3.connect(str(db3)) as conn:
        topics_by_id = _read_topics(conn)
        topic_name_to_id = {row["name"]: topic_id for topic_id, row in topics_by_id.items()}
        all_start, all_end, all_count = _read_bag_bounds(conn)

        selected_ids = {
            topic_name_to_id[name]: name
            for name in selected_topics
            if name in topic_name_to_id
        }

        data = BagData(
            db3_path=db3,
            metadata_path=Path(metadata_path) if metadata_path else None,
            metadata=metadata,
            topics={row["name"]: row["type"] for row in topics_by_id.values()},
            robot_poses={robot["id"]: [] for robot in robots_cfg if robot.get("id")},
            start_time_ns=all_start,
            end_time_ns=all_end,
            message_count=all_count,
        )

        if not selected_ids:
            return data

        placeholders = ",".join("?" for _ in selected_ids)
        rows = conn.execute(
            f"""
            SELECT topic_id, timestamp, data
            FROM messages
            WHERE topic_id IN ({placeholders})
            ORDER BY timestamp ASC
            """,
            tuple(selected_ids.keys()),
        )

        first_selected_ts: int | None = None
        last_selected_ts: int | None = None
        for topic_id, timestamp, blob in rows:
            topic_name = selected_ids[topic_id]
            if first_selected_ts is None:
                first_selected_ts = int(timestamp)
            last_selected_ts = int(timestamp)
            t = (int(timestamp) - all_start) / 1e9 if all_start else 0.0

            try:
                if topic_name in robot_topic_to_id:
                    pose = parse_pose_stamped_cdr(blob)
                    data.robot_poses.setdefault(robot_topic_to_id[topic_name], []).append(
                        _pose_with_time(pose, t)
                    )
                elif topic_name == base_pose_topic:
                    pose = parse_pose_stamped_cdr(blob)
                    data.base_poses.append(_pose_with_time(pose, t))
                elif topic_name == fire_state_topic:
                    data.fire_states.append(StateSample(t=t, text=parse_string_cdr(blob)))
                elif topic_name == target_state_topic:
                    data.target_states.append(StateSample(t=t, text=parse_string_cdr(blob)))
            except Exception:
                data.skipped_messages += 1

        if first_selected_ts is not None:
            data.start_time_ns = all_start
            data.end_time_ns = last_selected_ts or all_end

    return data


def inspect_bag(db3_path: str | Path, metadata_path: str | Path | None = None) -> dict[str, Any]:
    db3 = Path(db3_path)
    metadata = load_metadata(metadata_path)
    with sqlite3.connect(str(db3)) as conn:
        topics = _read_topics(conn)
        start_ns, end_ns, count = _read_bag_bounds(conn)
    return {
        "db3_path": str(db3),
        "metadata_path": str(metadata_path) if metadata_path else None,
        "metadata": metadata,
        "topics": {row["name"]: row["type"] for row in topics.values()},
        "message_count": count,
        "duration_sec": (end_ns - start_ns) / 1e9 if end_ns > start_ns else 0.0,
    }


def parse_string_cdr(blob: bytes) -> str:
    last_error: Exception | None = None
    for endian in _endian_candidates(blob):
        for base_offset in (4, 0):
            try:
                reader = _CdrReader(blob, endian=endian, base_offset=base_offset)
                reader.offset = 4
                return reader.read_string()
            except Exception as exc:
                last_error = exc
    raise ValueError(f"could not parse std_msgs/String: {last_error}")


def parse_pose_stamped_cdr(blob: bytes) -> PoseSample:
    last_error: Exception | None = None
    for endian in _endian_candidates(blob):
        for base_offset in (4, 0):
            try:
                pose = _parse_pose_with(blob, endian, base_offset)
                if _looks_like_pose(pose):
                    return pose
            except Exception as exc:
                last_error = exc
    raise ValueError(f"could not parse geometry_msgs/PoseStamped: {last_error}")


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _pose_with_time(pose: PoseSample, t: float) -> PoseSample:
    return PoseSample(
        t=t,
        x=pose.x,
        y=pose.y,
        z=pose.z,
        qx=pose.qx,
        qy=pose.qy,
        qz=pose.qz,
        qw=pose.qw,
        frame_id=pose.frame_id,
    )


def _read_topics(conn: sqlite3.Connection) -> dict[int, dict[str, str]]:
    rows = conn.execute("SELECT id, name, type FROM topics ORDER BY id ASC")
    return {
        int(topic_id): {"name": str(name), "type": str(msg_type)}
        for topic_id, name, msg_type in rows
    }


def _read_bag_bounds(conn: sqlite3.Connection) -> tuple[int, int, int]:
    row = conn.execute("SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM messages").fetchone()
    if not row or row[0] is None:
        return 0, 0, 0
    return int(row[0]), int(row[1]), int(row[2])


def _parse_pose_with(blob: bytes, endian: str, base_offset: int) -> PoseSample:
    reader = _CdrReader(blob, endian=endian, base_offset=base_offset)
    reader.offset = 4
    reader.read_int32()
    reader.read_uint32()
    frame_id = reader.read_string(max_length=4096)
    x = reader.read_float64()
    y = reader.read_float64()
    z = reader.read_float64()
    qx = reader.read_float64()
    qy = reader.read_float64()
    qz = reader.read_float64()
    qw = reader.read_float64()
    return PoseSample(0.0, x, y, z, qx, qy, qz, qw, frame_id)


def _looks_like_pose(pose: PoseSample) -> bool:
    values = [pose.x, pose.y, pose.z, pose.qx, pose.qy, pose.qz, pose.qw]
    if not all(math.isfinite(value) for value in values):
        return False
    if max(abs(pose.x), abs(pose.y), abs(pose.z)) > 1e6:
        return False
    q_norm = math.sqrt(pose.qx**2 + pose.qy**2 + pose.qz**2 + pose.qw**2)
    return 0.1 <= q_norm <= 2.0


def _endian_candidates(blob: bytes) -> list[str]:
    if len(blob) >= 2 and blob[1] in (1, 3):
        return ["<", ">"]
    if len(blob) >= 2 and blob[1] in (0, 2):
        return [">", "<"]
    return ["<", ">"]


class _CdrReader:
    def __init__(self, data: bytes, endian: str, base_offset: int = 4):
        self.data = data
        self.endian = endian
        self.base_offset = base_offset
        self.offset = 0

    def align(self, size: int) -> None:
        relative = self.offset - self.base_offset
        padding = (-relative) % size
        self.offset += padding

    def read_int32(self) -> int:
        self.align(4)
        return self._unpack("i", 4)

    def read_uint32(self) -> int:
        self.align(4)
        return self._unpack("I", 4)

    def read_float64(self) -> float:
        self.align(8)
        return self._unpack("d", 8)

    def read_string(self, max_length: int = 1_000_000) -> str:
        length = self.read_uint32()
        if length > max_length:
            raise ValueError(f"CDR string too long: {length}")
        raw = self._read_bytes(length)
        if raw.endswith(b"\x00"):
            raw = raw[:-1]
        return raw.decode("utf-8", errors="replace")

    def _unpack(self, fmt: str, size: int) -> Any:
        raw = self._read_bytes(size)
        return struct.unpack(self.endian + fmt, raw)[0]

    def _read_bytes(self, size: int) -> bytes:
        end = self.offset + size
        if end > len(self.data):
            raise ValueError("CDR buffer ended early")
        raw = self.data[self.offset:end]
        self.offset = end
        return raw
