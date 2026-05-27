from __future__ import annotations

import math
import itertools
from typing import Any

import cv2
import numpy as np

from .projection import (
    WorldBounds,
    apply_yaw_transform,
    map_transform_from_config,
    project_ground_circle,
    project_ground_ellipse,
    project_ground_square,
    world_polygon_to_video,
    transformed_world_rect_to_map_polygon,
    transformed_world_to_map,
    world_to_map,
    world_to_video,
)
from .timeline import FrameState, Pose2D


WHITE = (232, 235, 232)
BLACK = (12, 14, 15)
RED = (74, 86, 212)
FIRE_RED = (70, 92, 218)
YELLOW = (58, 176, 214)
GRAY = (132, 138, 136)
SAFE_BLUE = (212, 166, 88)
UNSAFE_ORANGE = (38, 128, 224)
MAP_BG = (26, 28, 28)
MAP_GRID = (43, 47, 47)
MAP_GRID_STRONG = (61, 65, 65)
MAP_TEXT = (184, 190, 188)
MAP_WHITE = (210, 215, 212)
MAP_RED = (58, 72, 170)
MAP_FIRE_RED = (52, 76, 184)
MAP_YELLOW = (46, 152, 174)
MAP_SAFE_BLUE = (168, 126, 76)


def draw_video_overlay(
    frame: np.ndarray,
    state: FrameState,
    homography: np.ndarray,
    config: dict[str, Any],
    trails: dict[str, list[Pose2D]] | None = None,
    tracked_points: dict[str, tuple[int, int]] | None = None,
) -> dict[str, tuple[int, int]]:
    visual = config.get("visual", {})
    base_area = config.get("base_area", {})
    video_points: dict[str, tuple[int, int]] = {}
    raw_frame = frame.copy()
    assigned_target_id = _assigned_target_id(state)
    full_robot_labels = _robot_full_labels(config)

    if base_area:
        corners = [
            (float(base_area["x_min"]), float(base_area["y_min"])),
            (float(base_area["x_max"]), float(base_area["y_min"])),
            (float(base_area["x_max"]), float(base_area["y_max"])),
            (float(base_area["x_min"]), float(base_area["y_max"])),
        ]
        polygon = world_polygon_to_video(corners, homography)
        if polygon is not None:
            polygon = _skew_video_ground_polygon(polygon, visual)
            alpha = min(
                float(visual.get("alpha_base_max", 0.34)),
                float(visual.get("alpha_base", 0.16))
                + float(visual.get("alpha_base_robot_bonus", 0.03)) * state.robots_in_base,
            )
            _fill_poly_alpha(frame, polygon, GRAY, alpha)
            cv2.polylines(frame, [polygon], True, (190, 196, 194), 2, cv2.LINE_AA)
            centroid = tuple(np.mean(polygon, axis=0).astype(int))
            _draw_text(frame, "BASE", (centroid[0] - 20, centroid[1]), (210, 214, 212), 0.5)

    if trails and bool(visual.get("draw_robot_trails", True)):
        _draw_video_trails(frame, trails, state.robot_roles, homography)

    for target in state.targets.values():
        point = world_to_video(target.x, target.y, homography)
        if point is None:
            continue
        center = _int_point(point)
        half_size_m = float(visual.get("video_target_half_size_m", 0.08))
        polygon = project_ground_square(homography, target.x, target.y, half_size_m)
        if polygon is not None:
            _fill_poly_alpha(frame, polygon, YELLOW, 0.60)
            cv2.polylines(frame, [polygon], True, BLACK, 1, cv2.LINE_AA)
        else:
            size = int(visual.get("video_target_size_px", 18))
            _draw_square(frame, center, size, YELLOW, fill=True)
        _draw_text(frame, _object_label(target.id), (center[0] + 10, center[1] - 10), YELLOW, 0.58)
        if target.id == assigned_target_id:
            _draw_bobbing_arrow(frame, center, YELLOW, state.t, scale=1.0)

    for fire in state.fires.values():
        point = world_to_video(fire.x, fire.y, homography)
        if point is None:
            continue
        center = _int_point(point)
        default_radius_m = float(visual.get("video_fire_radius_m_default", 0.2))
        radius_m = (
            (fire.radius_m or default_radius_m)
            * float(visual.get("video_fire_radius_m_scale", 1.0))
            * fire.radius_scale
        )
        alpha = float(visual.get("alpha_fire", 0.55))
        polygon = project_ground_circle(homography, fire.x, fire.y, max(radius_m, 0.01))
        if polygon is not None:
            _fill_poly_alpha(frame, polygon, FIRE_RED, alpha)
            cv2.polylines(frame, [polygon], True, FIRE_RED, 2, cv2.LINE_AA)
        else:
            base_radius = int(visual.get("video_fire_radius_px", 24))
            radius = max(1, int(round(base_radius * fire.radius_scale)))
            _circle_alpha(frame, center, radius, FIRE_RED, alpha)
            cv2.circle(frame, center, radius, FIRE_RED, 2, cv2.LINE_AA)
        _draw_text(frame, _object_label(fire.id), (center[0] + 9, center[1] - 9), FIRE_RED, 0.58)

    for robot_id, pose in state.robot_poses.items():
        point = world_to_video(pose.x, pose.y, homography)
        if point is not None:
            center = _int_point(point)
            role = state.robot_roles.get(robot_id, "robot")
            tracked_center = tracked_points.get(robot_id) if tracked_points else None
            video_points[robot_id] = tracked_center or _refine_robot_center(raw_frame, center, role, visual)

    _draw_rescue_radius_video(frame, state, homography, config, video_points)

    if bool(visual.get("draw_robot_links", True)):
        _draw_robot_links(frame, video_points, state.robot_roles, config)

    for robot_id, pose in state.robot_poses.items():
        if robot_id not in video_points:
            continue
        center = video_points[robot_id]
        role = state.robot_roles.get(robot_id, "robot")
        color = WHITE if role == "rescue" else RED
        if bool(visual.get("draw_video_robot_rings", False)):
            thickness = int(visual.get("video_robot_ring_thickness_px", 3))
            length_m = float(visual.get("video_robot_ring_length_m", 0.42))
            width_m = float(visual.get("video_robot_ring_width_m", 0.28))
            polygon = project_ground_ellipse(
                homography,
                pose.x,
                pose.y,
                max(length_m * 0.5, 0.01),
                max(width_m * 0.5, 0.01),
                pose.yaw,
            )
            if polygon is not None:
                projected_center = world_to_video(pose.x, pose.y, homography)
                if projected_center is not None:
                    delta = np.asarray(center, dtype=np.int32) - np.asarray(_int_point(projected_center), dtype=np.int32)
                    polygon = polygon + delta
                cv2.polylines(frame, [polygon], True, BLACK, thickness + 3, cv2.LINE_AA)
                cv2.polylines(frame, [polygon], True, color, thickness, cv2.LINE_AA)
            else:
                radius = int(visual.get("video_robot_ring_radius_px", 26))
                ratio = float(visual.get("video_robot_ellipse_ratio", 0.55))
                axes = (radius, max(4, int(radius * ratio)))
                angle = -math.degrees(pose.yaw)
                cv2.ellipse(frame, center, axes, angle, 0, 360, BLACK, thickness + 3, cv2.LINE_AA)
                cv2.ellipse(frame, center, axes, angle, 0, 360, color, thickness, cv2.LINE_AA)
        label = full_robot_labels.get(robot_id, state.robot_labels.get(robot_id, robot_id))
        label_anchor = _robot_label_anchor(robot_id, center, video_points, visual)
        if bool(visual.get("draw_video_label_leaders", True)):
            _draw_label_leader(frame, center, label_anchor, color, visual)
        _draw_centered_text(frame, label, label_anchor, color, 0.48)
        if role == "rescue":
            _draw_rescue_count(frame, center, state.rescued_target_count)

    return video_points


def draw_map(
    size: int,
    state: FrameState,
    config: dict[str, Any],
    trails: dict[str, list[Pose2D]] | None = None,
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    visual = config.get("visual", {})
    bounds = WorldBounds.from_config(config)
    transform_mode = map_transform_from_config(config)
    image = np.full((size, size, 3), MAP_BG, dtype=np.uint8)
    map_points: dict[str, tuple[int, int]] = {}
    assigned_target_id = _assigned_target_id(state)

    _draw_grid(image, bounds)
    _draw_text(image, "MAP VIEW", (14, 28), MAP_TEXT, 0.55, outline=None)

    base_area = config.get("base_area", {})
    if base_area:
        polygon = transformed_world_rect_to_map_polygon(base_area, bounds, size, size, transform_mode)
        alpha = min(
            float(visual.get("alpha_base_max", 0.34)),
            float(visual.get("alpha_base", 0.16))
            + float(visual.get("alpha_base_robot_bonus", 0.03)) * state.robots_in_base,
        )
        _fill_poly_alpha(image, polygon, (104, 110, 108), alpha * 0.85)
        cv2.polylines(image, [polygon], True, (128, 134, 132), 1, cv2.LINE_AA)
        centroid = tuple(np.mean(polygon, axis=0).astype(int))
        _draw_text(image, "BASE", (centroid[0] - 22, centroid[1] + 5), MAP_TEXT, 0.55)

    if trails and bool(visual.get("draw_robot_trails", True)):
        _draw_map_trails(image, trails, state.robot_roles, bounds, transform_mode, config)

    for target in state.targets.values():
        center = transformed_world_to_map(target.x, target.y, bounds, size, size, transform_mode)
        _draw_rotated_square(
            image,
            center,
            int(visual.get("map_target_size_px", 18)),
            MAP_YELLOW,
            float(visual.get("map_target_rotation_deg", -30.0)),
        )
        _draw_text(image, _object_label(target.id), (center[0] + 9, center[1] - 9), MAP_YELLOW, 0.58)
        if target.id == assigned_target_id:
            _draw_bobbing_arrow(image, center, MAP_YELLOW, state.t, scale=0.85)

    for fire in state.fires.values():
        center = transformed_world_to_map(fire.x, fire.y, bounds, size, size, transform_mode)
        radius = _map_fire_radius_px(fire.radius_m, fire.radius_scale, size, bounds, visual)
        _circle_alpha(image, center, radius, MAP_FIRE_RED, float(visual.get("map_alpha_fire", 0.34)))
        cv2.circle(image, center, radius, MAP_FIRE_RED, 2, cv2.LINE_AA)
        _draw_text(image, _object_label(fire.id), (center[0] + radius + 4, center[1] + 4), MAP_FIRE_RED, 0.58)

    for robot_id, pose in state.robot_poses.items():
        center = transformed_world_to_map(pose.x, pose.y, bounds, size, size, transform_mode)
        map_points[robot_id] = center

    _draw_rescue_radius_map(image, state, bounds, transform_mode, config, map_points)

    if bool(visual.get("draw_robot_links", True)):
        _draw_robot_links(image, map_points, state.robot_roles, config)

    for robot_id, pose in state.robot_poses.items():
        center = map_points[robot_id]
        role = state.robot_roles.get(robot_id, "robot")
        color = MAP_WHITE if role == "rescue" else MAP_RED
        label = state.robot_labels.get(robot_id, robot_id)
        transformed_pose = Pose2D(
            x=pose.x,
            y=pose.y,
            z=pose.z,
            yaw=apply_yaw_transform(pose.yaw, transform_mode),
        )
        map_robot_shape = str(visual.get("map_robot_shape", "car")).lower()
        if map_robot_shape == "triangle":
            _draw_map_triangle_robot(
                image,
                center,
                transformed_pose,
                int(visual.get("map_robot_triangle_length_px", 42)),
                int(visual.get("map_robot_triangle_width_px", 24)),
                color,
                label,
            )
        else:
            _draw_top_car(
                image,
                center,
                transformed_pose,
                int(visual.get("map_robot_length_px", 34)),
                int(visual.get("map_robot_width_px", 22)),
                color,
                label,
            )

    cv2.rectangle(image, (0, 0), (size - 1, size - 1), (86, 92, 94), 1, cv2.LINE_AA)
    return image, map_points


def draw_info_panel(
    image: np.ndarray,
    state: FrameState,
    config: dict[str, Any],
) -> None:
    image[:] = (22, 24, 25)
    h, w = image.shape[:2]
    cv2.rectangle(image, (0, 0), (w - 1, h - 1), (78, 84, 86), 1, cv2.LINE_AA)
    _draw_text(image, "INFO", (14, 23), MAP_TEXT, 0.56, outline=None)
    cv2.line(image, (12, 33), (w - 12, 33), (55, 60, 62), 1, cv2.LINE_AA)
    y = 51
    row_h = 17
    for idx, (line, color) in enumerate(_info_lines(state, config)):
        if idx % 2 == 0:
            cv2.rectangle(image, (8, y - 13), (w - 9, y + 3), (26, 29, 30), -1, cv2.LINE_AA)
        scale = 0.31 if len(line) > 49 else 0.34
        _draw_text(image, line, (14, y), color, scale, outline=None)
        y += row_h
        if idx in (2, 5):
            cv2.line(image, (12, y - 7), (w - 12, y - 7), (44, 49, 50), 1, cv2.LINE_AA)
        if y > h - 6:
            break


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
        cv2.circle(overlay, left, 2, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, right, 2, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0.0, dst=canvas)


def _draw_robot_links(
    image: np.ndarray,
    points: dict[str, tuple[int, int]],
    robot_roles: dict[str, str],
    config: dict[str, Any],
) -> None:
    visible_points = [
        (robot_id, point)
        for robot_id, point in sorted(points.items())
        if point is not None
    ]
    if len(visible_points) < 2:
        return
    overlay = image.copy()
    color = tuple(config.get("visual", {}).get("robot_link_color_bgr", [245, 245, 245]))
    alpha = float(config.get("visual", {}).get("alpha_robot_link", 0.38))
    for (_, left), (_, right) in itertools.combinations(visible_points, 2):
        cv2.line(overlay, left, right, color, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)


def _mission_phase(state: FrameState) -> tuple[str, str]:
    if state.targets:
        return "PHASE 1 TARGET", "AllTargetCompleted=FALSE"
    if state.fires:
        return "PHASE 2 FIRE", "AllFireSuppressed=FALSE"
    if state.robot_poses and state.robots_in_base < len(state.robot_poses):
        return "PHASE 3 BASE", "AllInBase=FALSE"
    return "COMPLETE", "AllInBase=TRUE"


def _info_lines(
    state: FrameState,
    config: dict[str, Any],
) -> list[tuple[str, tuple[int, int, int]]]:
    phase, condition = _mission_phase(state)
    safe_radius_m = float(config.get("visual", {}).get("rescue_safe_radius_m", 0.42))
    safe = _rescue_is_safe(state, safe_radius_m)
    robot_total = max(len(state.robot_poses), 1)
    lines: list[tuple[str, tuple[int, int, int]]] = [
        (f"TIME: {state.t:05.1f}s", (218, 220, 220)),
        (f"BT: mission_bt / ReactiveSequence", MAP_TEXT),
        (f"PHASE: {phase} / {condition}", (218, 220, 220)),
        (
            f"SAFETY: {'SAFE' if safe else 'UNSAFE'} / radius={safe_radius_m:.1f}m",
            (218, 220, 220) if safe else UNSAFE_ORANGE,
        ),
        (f"BASE: robots_in_base={state.robots_in_base}/{robot_total}", MAP_TEXT),
    ]
    unsafe_color = UNSAFE_ORANGE if not safe else (218, 220, 220)
    for label, task in _robot_assignments(state, config):
        color = unsafe_color if label.upper().startswith("RESCUE") else (218, 220, 220)
        lines.append((f"{label.upper()}: {task}", color))
    target_total = len(config.get("objects", {}).get("targets", [])) or len(state.targets)
    fire_total = len(config.get("objects", {}).get("fires", [])) or len(state.fires)
    lines.extend(
        [
            (f"TARGETS: active={len(state.targets)} rescued={state.rescued_target_count}/{target_total}", (218, 220, 220)),
            (f"FIRES: active={len(state.fires)} total={fire_total}", (218, 220, 220)),
            (f"ACTIVE FIRES: {_active_object_list(state.fires)}", MAP_TEXT),
            (f"ACTIVE TARGETS: {_active_object_list(state.targets)}", MAP_TEXT),
        ]
    )
    return lines


def _active_object_list(objects: dict[str, Any]) -> str:
    if not objects:
        return "none"
    return ", ".join(_object_label(object_id) for object_id in sorted(objects))


def _robot_assignments(
    state: FrameState,
    config: dict[str, Any],
) -> list[tuple[str, str]]:
    labels = {
        robot.get("id"): robot.get("label") or robot.get("id")
        for robot in config.get("robots", [])
        if robot.get("id")
    }
    output: list[tuple[str, str]] = []
    assigned_fires = _assigned_fires_by_robot(state)
    for robot_id in labels:
        pose = state.robot_poses.get(robot_id)
        if pose is None:
            continue
        role = state.robot_roles.get(robot_id, "robot")
        key = str(labels.get(robot_id, robot_id)).replace("_", " ")
        if role == "rescue":
            if state.targets:
                assignment = _nearest_object_id(pose, state.targets)
                task = f"CompleteTargetAfterSafe ({_object_label(assignment)})"
            else:
                task = "IsInBase (BASE)"
        elif role == "fire":
            if state.targets:
                task = "EscortLeader (Rescue Limo 1)"
            elif state.fires:
                assignment = assigned_fires.get(robot_id) or _nearest_object_id(pose, state.fires)
                task = f"SuppressFireAfterAssign ({_object_label(assignment)})"
            else:
                task = "IsInBase (BASE)"
        else:
            task = "Standby"
        output.append((key, task))
    return output


def _assigned_fires_by_robot(state: FrameState) -> dict[str, str]:
    fire_robot_ids = [
        robot_id
        for robot_id, role in state.robot_roles.items()
        if role == "fire" and robot_id in state.robot_poses
    ]
    fire_ids = list(state.fires)
    if not fire_robot_ids or not fire_ids:
        return {}
    if len(fire_ids) < len(fire_robot_ids):
        return {
            robot_id: _nearest_object_id(state.robot_poses[robot_id], state.fires)
            for robot_id in fire_robot_ids
        }

    best_cost: float | None = None
    best_assignment: dict[str, str] = {}
    for selected_fires in itertools.permutations(fire_ids, len(fire_robot_ids)):
        cost = 0.0
        for robot_id, fire_id in zip(fire_robot_ids, selected_fires):
            pose = state.robot_poses[robot_id]
            fire = state.fires[fire_id]
            cost += math.hypot(pose.x - fire.x, pose.y - fire.y)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_assignment = {
                robot_id: fire_id
                for robot_id, fire_id in zip(fire_robot_ids, selected_fires)
            }
    return best_assignment


def _nearest_object_id(pose: Pose2D, objects: dict[str, Any]) -> str:
    if not objects:
        return "BASE"
    return min(
        objects.values(),
        key=lambda obj: math.hypot(pose.x - obj.x, pose.y - obj.y),
    ).id


def _draw_video_trails(
    image: np.ndarray,
    trails: dict[str, list[Pose2D]],
    robot_roles: dict[str, str],
    homography: np.ndarray,
) -> None:
    for robot_id, poses in trails.items():
        points = []
        for pose in poses:
            point = world_to_video(pose.x, pose.y, homography)
            if point is not None:
                points.append(_int_point(point))
        if len(points) < 2:
            continue
        color = WHITE if robot_roles.get(robot_id) == "rescue" else RED
        overlay = image.copy()
        cv2.polylines(overlay, [np.asarray(points, dtype=np.int32)], False, color, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.28, image, 0.72, 0.0, dst=image)


def _draw_map_trails(
    image: np.ndarray,
    trails: dict[str, list[Pose2D]],
    robot_roles: dict[str, str],
    bounds: WorldBounds,
    transform_mode: str,
    config: dict[str, Any],
) -> None:
    h, w = image.shape[:2]
    for robot_id, poses in trails.items():
        points = [
            transformed_world_to_map(pose.x, pose.y, bounds, w, h, transform_mode)
            for pose in poses
        ]
        if len(points) < 2:
            continue
        color = WHITE if robot_roles.get(robot_id) == "rescue" else RED
        overlay = image.copy()
        cv2.polylines(overlay, [np.asarray(points, dtype=np.int32)], False, color, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.25, image, 0.75, 0.0, dst=image)


def _draw_grid(image: np.ndarray, bounds: WorldBounds) -> None:
    h, w = image.shape[:2]
    for x in np.linspace(bounds.x_min, bounds.x_max, 5):
        px, _ = world_to_map(float(x), bounds.y_min, bounds, w, h)
        cv2.line(image, (px, 0), (px, h), MAP_GRID, 1, cv2.LINE_AA)
    for y in np.linspace(bounds.y_min, bounds.y_max, 5):
        _, py = world_to_map(bounds.x_min, float(y), bounds, w, h)
        cv2.line(image, (0, py), (w, py), MAP_GRID, 1, cv2.LINE_AA)
    origin = world_to_map(0.0, 0.0, bounds, w, h)
    cv2.line(image, (origin[0], 0), (origin[0], h), MAP_GRID_STRONG, 1, cv2.LINE_AA)
    cv2.line(image, (0, origin[1]), (w, origin[1]), MAP_GRID_STRONG, 1, cv2.LINE_AA)


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
    text_color = BLACK if color in (WHITE, MAP_WHITE) else WHITE
    _draw_text(image, label, (center[0] - 11, center[1] + 5), text_color, 0.43, outline=BLACK if text_color == WHITE else None)


def _draw_map_triangle_robot(
    image: np.ndarray,
    center: tuple[int, int],
    pose: Pose2D,
    length: int,
    width: int,
    color: tuple[int, int, int],
    label: str,
) -> None:
    tip = _local_to_screen(center, 0.58 * length, 0.0, pose.yaw)
    rear_left = _local_to_screen(center, -0.42 * length, -0.50 * width, pose.yaw)
    rear_right = _local_to_screen(center, -0.42 * length, 0.50 * width, pose.yaw)
    triangle = np.asarray([tip, rear_left, rear_right], dtype=np.int32)

    cv2.fillConvexPoly(image, triangle, color, cv2.LINE_AA)
    cv2.polylines(image, [triangle], True, BLACK, 2, cv2.LINE_AA)
    nose_mid = _local_to_screen(center, 0.20 * length, 0.0, pose.yaw)
    cv2.line(image, center, nose_mid, BLACK, 1, cv2.LINE_AA)

    text_color = BLACK if color == MAP_WHITE else (224, 226, 224)
    text_scale = 0.39 if len(label) <= 2 else 0.36
    _draw_centered_text(image, label, center, text_color, text_scale)


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


def _draw_rotated_square(
    image: np.ndarray,
    center: tuple[int, int],
    size: int,
    color: tuple[int, int, int],
    angle_deg: float,
) -> None:
    rect = (center, (float(size), float(size)), angle_deg)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(image, box, color, cv2.LINE_AA)
    cv2.polylines(image, [box], True, BLACK, 1, cv2.LINE_AA)


def _draw_rescue_count(image: np.ndarray, center: tuple[int, int], count: int) -> None:
    size = 8
    start_x = center[0] - (count * (size + 3)) // 2
    y = center[1] + 24
    for idx in range(count):
        x = start_x + idx * (size + 3)
        cv2.rectangle(image, (x, y), (x + size, y + size), YELLOW, -1, cv2.LINE_AA)
        cv2.rectangle(image, (x, y), (x + size, y + size), BLACK, 1, cv2.LINE_AA)


def _robot_label_anchor(
    robot_id: str,
    center: tuple[int, int],
    video_points: dict[str, tuple[int, int]],
    visual: dict[str, Any],
) -> tuple[int, int]:
    base_offset = _robot_label_offset(robot_id, visual)
    if base_offset is None:
        base_offset = visual.get("video_robot_label_offset_px", [0, -22])
    if not isinstance(base_offset, (list, tuple)) or len(base_offset) < 2:
        base_offset = [0, -22]
    anchor = np.asarray(
        [center[0] + float(base_offset[0]), center[1] + float(base_offset[1])],
        dtype=np.float64,
    )
    if bool(visual.get("video_robot_label_lock_to_center", False)):
        return int(round(anchor[0])), int(round(anchor[1]))
    others = [
        np.asarray(point, dtype=np.float64)
        for other_id, point in video_points.items()
        if other_id != robot_id
    ]
    if not others:
        return int(round(anchor[0])), int(round(anchor[1]))

    center_vec = np.asarray(center, dtype=np.float64)
    distances = [float(np.linalg.norm(center_vec - other)) for other in others]
    close_distance = float(visual.get("video_robot_label_close_distance_px", 105.0))
    nearest = min(distances)
    if nearest >= close_distance:
        return int(round(anchor[0])), int(round(anchor[1]))

    group = [center_vec] + [other for other, distance in zip(others, distances) if distance < close_distance * 1.35]
    centroid = np.mean(group, axis=0)
    direction = center_vec - centroid
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = _stable_label_direction(robot_id)
    else:
        direction = direction / norm
    push = float(visual.get("video_robot_label_separation_px", 44.0))
    strength = 1.0 - min(max(nearest / close_distance, 0.0), 1.0)
    anchor += direction * push * (0.45 + 0.55 * strength)
    return int(round(anchor[0])), int(round(anchor[1]))


def _robot_label_offset(robot_id: str, visual: dict[str, Any]) -> list[float] | None:
    offsets = visual.get("video_robot_label_offset_px_by_id")
    if not isinstance(offsets, dict):
        return None
    offset = offsets.get(robot_id)
    if isinstance(offset, (list, tuple)) and len(offset) >= 2:
        return [float(offset[0]), float(offset[1])]
    return None


def _draw_label_leader(
    image: np.ndarray,
    center: tuple[int, int],
    anchor: tuple[int, int],
    color: tuple[int, int, int],
    visual: dict[str, Any],
) -> None:
    distance = math.hypot(anchor[0] - center[0], anchor[1] - center[1])
    if distance < float(visual.get("video_label_leader_min_distance_px", 34.0)):
        return
    overlay = image.copy()
    cv2.line(overlay, center, anchor, BLACK, 3, cv2.LINE_AA)
    cv2.line(overlay, center, anchor, color, 1, cv2.LINE_AA)
    alpha = float(visual.get("video_label_leader_alpha", 0.38))
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)


def _stable_label_direction(robot_id: str) -> np.ndarray:
    if "Fire_Limo_1" in robot_id:
        return np.asarray([-0.85, -0.52], dtype=np.float64)
    if "Fire_Limo_2" in robot_id:
        return np.asarray([0.85, -0.52], dtype=np.float64)
    return np.asarray([0.0, -1.0], dtype=np.float64)


def _skew_video_ground_polygon(polygon: np.ndarray, visual: dict[str, Any]) -> np.ndarray:
    shear = float(visual.get("video_ground_shear_x_per_y", 0.0))
    if abs(shear) < 1e-9:
        return polygon
    center_y = float(np.mean(polygon[:, 1]))
    out = polygon.astype(np.float64).copy()
    out[:, 0] += shear * (out[:, 1] - center_y)
    return np.rint(out).astype(np.int32)


def _draw_rescue_radius_video(
    image: np.ndarray,
    state: FrameState,
    homography: np.ndarray,
    config: dict[str, Any],
    video_points: dict[str, tuple[int, int]],
) -> None:
    rescue_id = _rescue_id(state)
    if not rescue_id:
        return
    pose = state.robot_poses.get(rescue_id)
    if not pose:
        return
    radius_m = float(config.get("visual", {}).get("rescue_safe_radius_m", 0.42))
    visual = config.get("visual", {})
    polygon = project_ground_ellipse(
        homography,
        pose.x,
        pose.y,
        radius_m * float(visual.get("video_safe_radius_x_scale", 1.0)),
        radius_m * float(visual.get("video_safe_radius_y_scale", 1.0)),
        math.radians(float(visual.get("video_safe_radius_yaw_deg", 0.0))),
        n=96,
    )
    if polygon is None:
        return
    polygon = _skew_video_ground_polygon(polygon, visual)
    projected_center = world_to_video(pose.x, pose.y, homography)
    center = video_points.get(rescue_id)
    if projected_center is not None and center is not None:
        delta = np.asarray(center, dtype=np.int32) - np.asarray(_int_point(projected_center), dtype=np.int32)
        polygon = polygon + delta
    safe = _rescue_is_safe(state, radius_m)
    _fill_poly_alpha(image, polygon, SAFE_BLUE, 0.13)
    cv2.polylines(image, [polygon], True, SAFE_BLUE, 2, cv2.LINE_AA)
    if not safe:
        cv2.polylines(image, [polygon], True, UNSAFE_ORANGE, 1, cv2.LINE_AA)


def _draw_rescue_radius_map(
    image: np.ndarray,
    state: FrameState,
    bounds: WorldBounds,
    transform_mode: str,
    config: dict[str, Any],
    map_points: dict[str, tuple[int, int]],
) -> None:
    rescue_id = _rescue_id(state)
    if not rescue_id or rescue_id not in state.robot_poses:
        return
    center = map_points.get(rescue_id)
    if center is None:
        return
    radius_m = float(config.get("visual", {}).get("rescue_safe_radius_m", 0.42))
    meters_per_px = max(bounds.x_max - bounds.x_min, bounds.y_max - bounds.y_min) / max(image.shape[0], 1)
    radius_px = max(2, int(round(radius_m / max(meters_per_px, 1e-9))))
    safe = _rescue_is_safe(state, radius_m)
    _circle_alpha(image, center, radius_px, MAP_SAFE_BLUE, 0.11)
    cv2.circle(image, center, radius_px, MAP_SAFE_BLUE, 2, cv2.LINE_AA)
    if not safe:
        cv2.circle(image, center, radius_px, UNSAFE_ORANGE, 1, cv2.LINE_AA)


def _rescue_id(state: FrameState) -> str | None:
    for robot_id, role in state.robot_roles.items():
        if role == "rescue":
            return robot_id
    return None


def _rescue_pose(state: FrameState) -> Pose2D | None:
    rescue_id = _rescue_id(state)
    return state.robot_poses.get(rescue_id) if rescue_id else None


def _rescue_is_safe(state: FrameState, radius_m: float) -> bool:
    pose = _rescue_pose(state)
    if pose is None:
        return True
    for fire in state.fires.values():
        fire_radius = (fire.radius_m or 0.2) * max(fire.radius_scale, 0.0)
        if math.hypot(pose.x - fire.x, pose.y - fire.y) <= radius_m + fire_radius:
            return False
    return True


def _assigned_target_id(state: FrameState) -> str | None:
    pose = _rescue_pose(state)
    if pose is None or not state.targets:
        return None
    return min(
        state.targets.values(),
        key=lambda target: math.hypot(pose.x - target.x, pose.y - target.y),
    ).id


def _draw_bobbing_arrow(
    image: np.ndarray,
    center: tuple[int, int],
    color: tuple[int, int, int],
    t: float,
    scale: float,
) -> None:
    bob = int(round(math.sin(t * 4.0) * 4.0 * scale))
    tip = (center[0], center[1] - int(20 * scale) + bob)
    tail = (center[0], center[1] - int(42 * scale) + bob)
    cv2.arrowedLine(image, tail, tip, color, max(1, int(round(2 * scale))), cv2.LINE_AA, tipLength=0.45)


def _refine_robot_center(
    image: np.ndarray,
    center: tuple[int, int],
    role: str,
    visual: dict[str, Any],
) -> tuple[int, int]:
    if not bool(visual.get("video_robot_refine_enabled", True)):
        return center
    refine_roles = visual.get("video_robot_refine_roles")
    if isinstance(refine_roles, str):
        refine_role_set = {refine_roles}
    elif refine_roles is not None:
        refine_role_set = {str(item) for item in refine_roles}
    else:
        refine_role_set = None
    if refine_role_set is not None and role not in refine_role_set:
        return center
    search = int(_visual_by_role(visual, "video_robot_refine_search_px", role, 90))
    max_shift = float(_visual_by_role(visual, "video_robot_refine_max_shift_px", role, 70))
    min_area = int(_visual_by_role(visual, "video_robot_refine_min_area_px", role, 40))
    h, w = image.shape[:2]
    x1 = max(0, center[0] - search)
    x2 = min(w, center[0] + search + 1)
    y1 = max(0, center[1] - search)
    y2 = min(h, center[1] + search + 1)
    if x2 <= x1 or y2 <= y1:
        return center
    roi = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    red_mask = ((hue < 10) | (hue > 170)) & (sat > 70) & (val > 60)
    white_mask = (sat < 80) & (val > 115)
    dark_mask = val < 65
    if role == "rescue":
        mask = white_mask
    elif role == "fire":
        mask = red_mask
    else:
        mask = red_mask | white_mask | dark_mask
    mask = mask.astype(np.uint8) * 255
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return center
    roi_center = np.array([center[0] - x1, center[1] - y1], dtype=np.float64)
    best_idx = None
    best_score = None
    for idx in range(1, count):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        centroid = centroids[idx]
        distance = float(np.linalg.norm(centroid - roi_center))
        score = distance - min(area, 3000) * 0.01
        if best_score is None or score < best_score:
            best_score = score
            best_idx = idx
    if best_idx is None:
        return center
    refined = (int(round(centroids[best_idx][0] + x1)), int(round(centroids[best_idx][1] + y1)))
    if math.hypot(refined[0] - center[0], refined[1] - center[1]) > max_shift:
        return center
    return refined


def _map_fire_radius_px(
    radius_m: float | None,
    radius_scale: float,
    size: int,
    bounds: WorldBounds,
    visual: dict[str, Any],
) -> int:
    if radius_m is None:
        radius = float(visual.get("map_fire_radius_px", 18))
    else:
        meters_per_px = max(bounds.x_max - bounds.x_min, bounds.y_max - bounds.y_min) / max(size, 1)
        radius = radius_m / max(meters_per_px, 1e-9)
        radius *= float(visual.get("map_fire_radius_m_scale", 1.0))
    return max(2, int(round(radius * radius_scale)))


def _visual_by_role(
    visual: dict[str, Any],
    key: str,
    role: str,
    default: float,
) -> float:
    role_values = visual.get(f"{key}_by_role")
    if isinstance(role_values, dict) and role in role_values:
        return float(role_values[role])
    return float(visual.get(key, default))


def _robot_full_labels(config: dict[str, Any]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for robot in config.get("robots", []):
        robot_id = robot.get("id")
        if robot_id:
            labels[str(robot_id)] = str(robot.get("label") or robot_id)
    return labels


def _object_label(object_id: str) -> str:
    if "_" in object_id:
        prefix, number = object_id.split("_", 1)
        if prefix.lower() == "fire":
            return f"Fire {number}"
        if prefix.lower() == "target":
            return f"T{number}"
        return f"{prefix[0].upper()}{number}"
    return object_id


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


def _draw_centered_text(
    image: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    color: tuple[int, int, int],
    scale: float,
    outline: tuple[int, int, int] | None = BLACK,
) -> None:
    (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = int(round(anchor[0] - width / 2))
    y = int(round(anchor[1] + height / 2))
    x = max(4, min(image.shape[1] - width - 4, x))
    y = max(height + 4, min(image.shape[0] - 4, y))
    _draw_text(image, text, (x, y), color, scale, outline=outline)


def _int_point(point: tuple[float, float]) -> tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))
