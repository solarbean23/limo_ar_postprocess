from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .projection import (
    WorldBounds,
    world_polygon_to_video,
    world_rect_to_map_polygon,
    world_to_map,
    world_to_video,
)
from .timeline import FrameState, Pose2D


WHITE = (245, 245, 245)
BLACK = (10, 10, 10)
RED = (70, 90, 245)
FIRE_RED = (70, 95, 255)
YELLOW = (40, 220, 245)
GRAY = (150, 150, 150)


def draw_video_overlay(
    frame: np.ndarray,
    state: FrameState,
    homography: np.ndarray,
    config: dict[str, Any],
) -> dict[str, tuple[int, int]]:
    visual = config.get("visual", {})
    base_area = config.get("base_area", {})
    video_points: dict[str, tuple[int, int]] = {}

    if base_area:
        corners = [
            (float(base_area["x_min"]), float(base_area["y_min"])),
            (float(base_area["x_max"]), float(base_area["y_min"])),
            (float(base_area["x_max"]), float(base_area["y_max"])),
            (float(base_area["x_min"]), float(base_area["y_max"])),
        ]
        polygon = world_polygon_to_video(corners, homography)
        if polygon is not None:
            alpha = min(0.55, float(visual.get("alpha_base", 0.28)) + 0.07 * state.robots_in_base)
            _fill_poly_alpha(frame, polygon, GRAY, alpha)

    for target in state.targets.values():
        point = world_to_video(target.x, target.y, homography)
        if point is None:
            continue
        center = _int_point(point)
        size = int(visual.get("video_target_size_px", 18))
        _draw_square(frame, center, size, YELLOW, fill=True)
        _draw_text(frame, target.id.replace("_", " "), (center[0] + 10, center[1] - 10), YELLOW, 0.45)

    for fire in state.fires.values():
        point = world_to_video(fire.x, fire.y, homography)
        if point is None:
            continue
        center = _int_point(point)
        base_radius = int(visual.get("video_fire_radius_px", 24))
        radius = max(1, int(round(base_radius * fire.radius_scale)))
        alpha = float(visual.get("alpha_fire", 0.55))
        _circle_alpha(frame, center, radius, FIRE_RED, alpha)
        cv2.circle(frame, center, radius, FIRE_RED, 2, cv2.LINE_AA)

    for robot_id, pose in state.robot_poses.items():
        point = world_to_video(pose.x, pose.y, homography)
        if point is None:
            continue
        center = _int_point(point)
        video_points[robot_id] = center
        role = state.robot_roles.get(robot_id, "robot")
        color = WHITE if role == "rescue" else RED
        radius = int(visual.get("video_robot_ring_radius_px", 26))
        ratio = float(visual.get("video_robot_ellipse_ratio", 0.55))
        thickness = int(visual.get("video_robot_ring_thickness_px", 3))
        axes = (radius, max(4, int(radius * ratio)))
        angle = -math.degrees(pose.yaw)
        cv2.ellipse(frame, center, axes, angle, 0, 360, color, thickness + 2, cv2.LINE_AA)
        cv2.ellipse(frame, center, axes, angle, 0, 360, color, thickness, cv2.LINE_AA)
        label = state.robot_labels.get(robot_id, robot_id)
        _draw_text(frame, label, (center[0] + 12, center[1] - 14), color, 0.52)
        if role == "rescue":
            _draw_rescue_count(frame, center, state.rescued_target_count)

    return video_points


def draw_map(
    size: int,
    state: FrameState,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    visual = config.get("visual", {})
    bounds = WorldBounds.from_config(config)
    image = np.full((size, size, 3), (38, 42, 44), dtype=np.uint8)
    map_points: dict[str, tuple[int, int]] = {}

    _draw_grid(image, bounds)

    base_area = config.get("base_area", {})
    if base_area:
        polygon = world_rect_to_map_polygon(base_area, bounds, size, size)
        alpha = min(0.65, float(visual.get("alpha_base", 0.28)) + 0.08 * state.robots_in_base)
        _fill_poly_alpha(image, polygon, (150, 150, 150), alpha)
        cv2.polylines(image, [polygon], True, (185, 185, 185), 1, cv2.LINE_AA)

    for target in state.targets.values():
        center = world_to_map(target.x, target.y, bounds, size, size)
        _draw_square(image, center, int(visual.get("map_target_size_px", 18)), YELLOW, fill=True)

    for fire in state.fires.values():
        center = world_to_map(fire.x, fire.y, bounds, size, size)
        radius = max(1, int(round(float(visual.get("map_fire_radius_px", 18)) * fire.radius_scale)))
        _circle_alpha(image, center, radius, FIRE_RED, float(visual.get("alpha_fire", 0.55)))
        cv2.circle(image, center, radius, FIRE_RED, 2, cv2.LINE_AA)

    for robot_id, pose in state.robot_poses.items():
        center = world_to_map(pose.x, pose.y, bounds, size, size)
        map_points[robot_id] = center
        role = state.robot_roles.get(robot_id, "robot")
        color = WHITE if role == "rescue" else RED
        label = state.robot_labels.get(robot_id, robot_id)
        _draw_top_car(
            image,
            center,
            pose,
            int(visual.get("map_robot_length_px", 34)),
            int(visual.get("map_robot_width_px", 22)),
            color,
            label,
        )

    cv2.rectangle(image, (0, 0), (size - 1, size - 1), (120, 125, 128), 1, cv2.LINE_AA)
    return image, map_points


def draw_connectors(
    canvas: np.ndarray,
    video_points: dict[str, tuple[int, int]],
    map_points: dict[str, tuple[int, int]],
    robot_roles: dict[str, str],
    config: dict[str, Any],
) -> None:
    alpha = float(config.get("visual", {}).get("alpha_connector", 0.35))
    overlay = canvas.copy()
    for robot_id, left in video_points.items():
        right = map_points.get(robot_id)
        if not right:
            continue
        role = robot_roles.get(robot_id, "robot")
        color = WHITE if role == "rescue" else RED
        cv2.line(overlay, left, right, color, 1, cv2.LINE_AA)
        cv2.circle(overlay, left, 3, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, right, 3, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0.0, dst=canvas)


def _draw_grid(image: np.ndarray, bounds: WorldBounds) -> None:
    h, w = image.shape[:2]
    for x in np.linspace(bounds.x_min, bounds.x_max, 5):
        px, _ = world_to_map(float(x), bounds.y_min, bounds, w, h)
        cv2.line(image, (px, 0), (px, h), (60, 65, 67), 1, cv2.LINE_AA)
    for y in np.linspace(bounds.y_min, bounds.y_max, 5):
        _, py = world_to_map(bounds.x_min, float(y), bounds, w, h)
        cv2.line(image, (0, py), (w, py), (60, 65, 67), 1, cv2.LINE_AA)
    origin = world_to_map(0.0, 0.0, bounds, w, h)
    cv2.line(image, (origin[0], 0), (origin[0], h), (82, 88, 91), 1, cv2.LINE_AA)
    cv2.line(image, (0, origin[1]), (w, origin[1]), (82, 88, 91), 1, cv2.LINE_AA)


def _draw_top_car(
    image: np.ndarray,
    center: tuple[int, int],
    pose: Pose2D,
    length: int,
    width: int,
    color: tuple[int, int, int],
    label: str,
) -> None:
    angle = -math.degrees(pose.yaw)
    rect = (center, (float(length), float(width)), angle)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(image, box, color, cv2.LINE_AA)
    cv2.polylines(image, [box], True, BLACK, 1, cv2.LINE_AA)

    for local_x in (-0.32 * length, 0.32 * length):
        for local_y in (-0.58 * width, 0.58 * width):
            wheel = _local_to_screen(center, local_x, local_y, pose.yaw)
            cv2.ellipse(image, wheel, (4, 2), angle, 0, 360, BLACK, -1, cv2.LINE_AA)

    nose = _local_to_screen(center, 0.52 * length, 0.0, pose.yaw)
    cv2.line(image, center, nose, BLACK, 2, cv2.LINE_AA)
    _draw_text(image, label, (center[0] - 11, center[1] + 5), BLACK, 0.43, outline=None)


def _local_to_screen(
    center: tuple[int, int],
    local_x: float,
    local_y: float,
    yaw: float,
) -> tuple[int, int]:
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    x = center[0] + local_x * c - local_y * s
    y = center[1] + local_x * s + local_y * c
    return int(round(x)), int(round(y))


def _draw_square(
    image: np.ndarray,
    center: tuple[int, int],
    size: int,
    color: tuple[int, int, int],
    fill: bool,
) -> None:
    half = max(2, size // 2)
    p1 = (center[0] - half, center[1] - half)
    p2 = (center[0] + half, center[1] + half)
    thickness = -1 if fill else 2
    cv2.rectangle(image, p1, p2, color, thickness, cv2.LINE_AA)
    cv2.rectangle(image, p1, p2, BLACK, 1, cv2.LINE_AA)


def _draw_rescue_count(image: np.ndarray, center: tuple[int, int], count: int) -> None:
    size = 8
    start_x = center[0] - (count * (size + 3)) // 2
    y = center[1] + 24
    for idx in range(count):
        x = start_x + idx * (size + 3)
        cv2.rectangle(image, (x, y), (x + size, y + size), YELLOW, -1, cv2.LINE_AA)
        cv2.rectangle(image, (x, y), (x + size, y + size), BLACK, 1, cv2.LINE_AA)


def _fill_poly_alpha(
    image: np.ndarray,
    polygon: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    overlay = image.copy()
    cv2.fillPoly(overlay, [polygon.astype(np.int32)], color, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)


def _circle_alpha(
    image: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    overlay = image.copy()
    cv2.circle(overlay, center, radius, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)


def _draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float,
    outline: tuple[int, int, int] | None = BLACK,
) -> None:
    if outline is not None:
        cv2.putText(
            image,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            outline,
            3,
            cv2.LINE_AA,
        )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def _int_point(point: tuple[float, float]) -> tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))
