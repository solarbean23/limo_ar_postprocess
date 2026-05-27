from __future__ import annotations

from dataclasses import dataclass, replace
import itertools
import math
from typing import Any

import cv2
import numpy as np

from .projection import video_to_world, world_to_video
from .timeline import FrameState, Pose2D


@dataclass(frozen=True)
class _Candidate:
    center: tuple[float, float]
    area: float
    bbox: tuple[int, int, int, int]


class VideoRobotTracker:
    """Small color tracker used only to anchor AR labels to the visible robots."""

    def __init__(self, homography: np.ndarray, config: dict[str, Any]):
        self.homography = homography
        self.visual = config.get("visual", {})
        self.previous: dict[str, tuple[float, float]] = {}
        self.previous_t: float | None = None

    def update(self, frame: np.ndarray, state: FrameState) -> dict[str, tuple[int, int]]:
        predicted = _projected_robot_points(state, self.homography)
        if not bool(self.visual.get("video_robot_tracking_enabled", True)):
            return {robot_id: _round_point(point) for robot_id, point in predicted.items()}

        red_candidates = _detect_candidates(frame, "fire", self.visual)
        white_candidates = _detect_candidates(frame, "rescue", self.visual)
        tracked: dict[str, tuple[float, float]] = {}

        fire_ids = [
            robot_id
            for robot_id, role in state.robot_roles.items()
            if role == "fire" and robot_id in predicted
        ]
        fire_assignment = _assign_candidates(
            fire_ids,
            red_candidates,
            predicted,
            self.previous,
            self.visual,
            "fire",
        )
        tracked.update(fire_assignment)

        for robot_id, role in state.robot_roles.items():
            if robot_id not in predicted or role == "fire":
                continue
            candidates = white_candidates if role == "rescue" else red_candidates + white_candidates
            center = _best_candidate(
                candidates,
                predicted[robot_id],
                self.previous.get(robot_id),
                self.visual,
                role,
            )
            if center is not None:
                tracked[robot_id] = center.center

        output: dict[str, tuple[int, int]] = {}
        for robot_id, point in predicted.items():
            detected = tracked.get(robot_id)
            center = detected if detected is not None else point
            role = state.robot_roles.get(robot_id, "robot")
            center = self._smooth(robot_id, center, state.t, role)
            self.previous[robot_id] = center
            output[robot_id] = _round_point(center)
        self.previous_t = state.t
        return output

    def _smooth(
        self,
        robot_id: str,
        center: tuple[float, float],
        t: float,
        role: str,
    ) -> tuple[float, float]:
        previous = self.previous.get(robot_id)
        tau = float(self.visual.get("video_robot_tracking_smoothing_tau_sec", 0.018))
        if previous is None or tau <= 0.0 or self.previous_t is None:
            return center
        dt = max(0.0, t - self.previous_t)
        if dt <= 0.0:
            return center
        max_speed = _role_float(self.visual, "video_robot_tracking_max_step_px_per_sec", role, 480.0)
        step_margin = float(self.visual.get("video_robot_tracking_max_step_margin_px", 2.0))
        max_step = max_speed * dt + step_margin
        distance = _distance(center, previous)
        if distance > max_step > 0.0:
            ratio = max_step / max(distance, 1e-9)
            center = (
                previous[0] + (center[0] - previous[0]) * ratio,
                previous[1] + (center[1] - previous[1]) * ratio,
            )
        alpha = 1.0 - math.exp(-dt / tau)
        alpha = max(0.0, min(1.0, alpha))
        return (
            previous[0] + (center[0] - previous[0]) * alpha,
            previous[1] + (center[1] - previous[1]) * alpha,
        )


def state_with_tracked_robot_positions(
    state: FrameState,
    tracked_points: dict[str, tuple[int, int]],
    homography: np.ndarray,
    config: dict[str, Any],
) -> FrameState:
    visual = config.get("visual", {})
    if not tracked_points or not bool(visual.get("video_tracking_updates_map", True)):
        return state

    blend = float(visual.get("video_tracking_map_blend", 1.0))
    blend = max(0.0, min(1.0, blend))
    if blend <= 0.0:
        return state

    poses: dict[str, Pose2D] = {}
    for robot_id, pose in state.robot_poses.items():
        point = tracked_points.get(robot_id)
        world = video_to_world(point[0], point[1], homography) if point else None
        if world is None:
            poses[robot_id] = pose
            continue
        poses[robot_id] = Pose2D(
            x=pose.x + (world[0] - pose.x) * blend,
            y=pose.y + (world[1] - pose.y) * blend,
            z=pose.z,
            yaw=pose.yaw,
        )
    return replace(state, robot_poses=poses)


def _projected_robot_points(
    state: FrameState,
    homography: np.ndarray,
) -> dict[str, tuple[float, float]]:
    points: dict[str, tuple[float, float]] = {}
    for robot_id, pose in state.robot_poses.items():
        point = world_to_video(pose.x, pose.y, homography)
        if point is not None:
            points[robot_id] = point
    return points


def _assign_candidates(
    robot_ids: list[str],
    candidates: list[_Candidate],
    predicted: dict[str, tuple[float, float]],
    previous: dict[str, tuple[float, float]],
    visual: dict[str, Any],
    role: str,
) -> dict[str, tuple[float, float]]:
    if not robot_ids or not candidates:
        return {}
    candidate_indexes = list(range(len(candidates)))
    best_cost: float | None = None
    best_assignment: dict[str, tuple[float, float]] = {}
    max_distance = _role_float(visual, "video_robot_tracking_max_distance_px", role, 130.0)
    missing_penalty = _role_float(visual, "video_robot_tracking_missing_penalty", role, 180.0)
    area_bonus = _role_float(visual, "video_robot_tracking_area_bonus", role, 0.006)
    max_matches = min(len(robot_ids), len(candidates))

    for match_count in range(1, max_matches + 1):
        for robot_subset in itertools.combinations(robot_ids, match_count):
            for indexes in itertools.permutations(candidate_indexes, match_count):
                assignment: dict[str, tuple[float, float]] = {}
                cost = (len(robot_ids) - match_count) * missing_penalty
                valid = True
                for robot_id, candidate_idx in zip(robot_subset, indexes):
                    candidate = candidates[candidate_idx]
                    score = _candidate_score(
                        candidate,
                        predicted[robot_id],
                        previous.get(robot_id),
                        max_distance,
                        area_bonus,
                    )
                    if score is None:
                        valid = False
                        break
                    assignment[robot_id] = candidate.center
                    cost += score
                if not valid:
                    continue
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_assignment = assignment
    return best_assignment


def _best_candidate(
    candidates: list[_Candidate],
    predicted: tuple[float, float],
    previous: tuple[float, float] | None,
    visual: dict[str, Any],
    role: str,
) -> _Candidate | None:
    max_distance = _role_float(visual, "video_robot_tracking_max_distance_px", role, 130.0)
    area_bonus = _role_float(visual, "video_robot_tracking_area_bonus", role, 0.006)
    best: _Candidate | None = None
    best_score: float | None = None
    for candidate in candidates:
        score = _candidate_score(candidate, predicted, previous, max_distance, area_bonus)
        if score is None:
            continue
        if best_score is None or score < best_score:
            best = candidate
            best_score = score
    return best


def _candidate_score(
    candidate: _Candidate,
    predicted: tuple[float, float],
    previous: tuple[float, float] | None,
    max_distance: float,
    area_bonus: float,
) -> float | None:
    pred_dist = _distance(candidate.center, predicted)
    prev_dist = _distance(candidate.center, previous) if previous is not None else pred_dist
    use_previous = previous is not None and _distance(predicted, previous) <= max_distance * 1.5
    gate_dist = min(pred_dist, prev_dist) if use_previous else pred_dist
    if gate_dist > max_distance:
        return None
    motion_cost = pred_dist * 0.62 + prev_dist * 0.38 if use_previous else pred_dist
    return motion_cost - min(candidate.area, 4000.0) * area_bonus


def _detect_candidates(
    frame: np.ndarray,
    role: str,
    visual: dict[str, Any],
) -> list[_Candidate]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    if role == "fire":
        mask = ((hue < 12) | (hue > 170)) & (sat > 65) & (val > 55)
    else:
        mask = (sat < 62) & (val > 135)

    mask_u8 = mask.astype(np.uint8) * 255
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))

    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8)
    height, width = frame.shape[:2]
    min_area = _role_float(visual, "video_robot_tracking_min_area_px", role, 90.0)
    max_area = _role_float(visual, "video_robot_tracking_max_area_px", role, 9000.0)
    max_w = _role_float(visual, "video_robot_tracking_max_bbox_width_px", role, 150.0)
    max_h = _role_float(visual, "video_robot_tracking_max_bbox_height_px", role, 130.0)

    candidates: list[_Candidate] = []
    for idx in range(1, count):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if w > max_w or h > max_h:
            continue
        if role == "rescue" and _touches_frame_edge(x, y, w, h, width, height):
            continue
        center = (float(centroids[idx][0]), float(centroids[idx][1]))
        candidates.append(_Candidate(center=center, area=area, bbox=(x, y, w, h)))
    candidates.sort(key=lambda candidate: candidate.area, reverse=True)
    limit = int(visual.get("video_robot_tracking_candidate_limit", 12))
    return candidates[: max(1, limit)]


def _touches_frame_edge(
    x: int,
    y: int,
    w: int,
    h: int,
    image_w: int,
    image_h: int,
) -> bool:
    margin = 3
    return x <= margin or y <= margin or x + w >= image_w - margin or y + h >= image_h - margin


def _role_float(visual: dict[str, Any], key: str, role: str, default: float) -> float:
    role_values = visual.get(f"{key}_by_role")
    if isinstance(role_values, dict) and role in role_values:
        return float(role_values[role])
    return float(visual.get(key, default))


def _distance(
    left: tuple[float, float],
    right: tuple[float, float] | None,
) -> float:
    if right is None:
        return 0.0
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _round_point(point: tuple[float, float]) -> tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))
