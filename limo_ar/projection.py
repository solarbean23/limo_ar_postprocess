from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WorldBounds:
    x_min: float = -2.0
    x_max: float = 2.0
    y_min: float = -2.0
    y_max: float = 2.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "WorldBounds":
        world = config.get("world", {})
        return cls(
            x_min=float(world.get("x_min", -2.0)),
            x_max=float(world.get("x_max", 2.0)),
            y_min=float(world.get("y_min", -2.0)),
            y_max=float(world.get("y_max", 2.0)),
        )


def world_to_map(
    x: float,
    y: float,
    bounds: WorldBounds,
    width: int,
    height: int,
) -> tuple[int, int]:
    sx = (x - bounds.x_min) / max(bounds.x_max - bounds.x_min, 1e-9)
    sy = (bounds.y_max - y) / max(bounds.y_max - bounds.y_min, 1e-9)
    px = int(round(sx * (width - 1)))
    py = int(round(sy * (height - 1)))
    return px, py


def apply_display_transform(x: float, y: float, mode: str) -> tuple[float, float]:
    if mode == "none":
        return x, y
    if mode == "rotate_ccw_90":
        return -y, x
    if mode == "rotate_cw_90":
        return y, -x
    if mode == "rotate_180":
        return -x, -y
    raise ValueError(f"unknown map.display_transform: {mode}")


def apply_yaw_transform(yaw: float, mode: str) -> float:
    if mode == "none":
        out = yaw
    elif mode == "rotate_ccw_90":
        out = yaw + math.pi / 2.0
    elif mode == "rotate_cw_90":
        out = yaw - math.pi / 2.0
    elif mode == "rotate_180":
        out = yaw + math.pi
    else:
        raise ValueError(f"unknown map.display_transform: {mode}")
    return (out + math.pi) % (2.0 * math.pi) - math.pi


def map_transform_from_config(config: dict[str, Any]) -> str:
    return str(config.get("map", {}).get("display_transform", "none"))


def transformed_world_to_map(
    x: float,
    y: float,
    bounds: WorldBounds,
    width: int,
    height: int,
    mode: str,
) -> tuple[int, int]:
    tx, ty = apply_display_transform(x, y, mode)
    return world_to_map(tx, ty, bounds, width, height)


def world_rect_to_map_polygon(
    area: dict[str, float],
    bounds: WorldBounds,
    width: int,
    height: int,
) -> np.ndarray:
    corners = [
        (float(area["x_min"]), float(area["y_min"])),
        (float(area["x_max"]), float(area["y_min"])),
        (float(area["x_max"]), float(area["y_max"])),
        (float(area["x_min"]), float(area["y_max"])),
    ]
    return np.array(
        [world_to_map(x, y, bounds, width, height) for x, y in corners],
        dtype=np.int32,
    )


def transformed_world_rect_to_map_polygon(
    area: dict[str, float],
    bounds: WorldBounds,
    width: int,
    height: int,
    mode: str,
) -> np.ndarray:
    corners = [
        (float(area["x_min"]), float(area["y_min"])),
        (float(area["x_max"]), float(area["y_min"])),
        (float(area["x_max"]), float(area["y_max"])),
        (float(area["x_min"]), float(area["y_max"])),
    ]
    return np.array(
        [transformed_world_to_map(x, y, bounds, width, height, mode) for x, y in corners],
        dtype=np.int32,
    )


def world_to_video(
    x: float,
    y: float,
    homography: np.ndarray,
) -> tuple[float, float] | None:
    point = homography @ np.array([x, y, 1.0], dtype=np.float64)
    if abs(point[2]) < 1e-9:
        return None
    return float(point[0] / point[2]), float(point[1] / point[2])


def video_to_world(
    px: float,
    py: float,
    homography: np.ndarray,
) -> tuple[float, float] | None:
    inverse = np.linalg.inv(homography)
    point = inverse @ np.array([px, py, 1.0], dtype=np.float64)
    if abs(point[2]) < 1e-9:
        return None
    return float(point[0] / point[2]), float(point[1] / point[2])


def world_polygon_to_video(
    points: list[tuple[float, float]],
    homography: np.ndarray,
) -> np.ndarray | None:
    projected = []
    for x, y in points:
        pt = world_to_video(x, y, homography)
        if pt is None:
            return None
        projected.append((int(round(pt[0])), int(round(pt[1]))))
    return np.array(projected, dtype=np.int32)


def project_ground_circle(
    homography: np.ndarray,
    x: float,
    y: float,
    radius_m: float,
    n: int = 48,
) -> np.ndarray | None:
    points = []
    for i in range(n):
        angle = 2.0 * math.pi * i / n
        point = world_to_video(
            x + radius_m * math.cos(angle),
            y + radius_m * math.sin(angle),
            homography,
        )
        if point is None:
            return None
        points.append((int(round(point[0])), int(round(point[1]))))
    return np.asarray(points, dtype=np.int32)


def project_ground_ellipse(
    homography: np.ndarray,
    x: float,
    y: float,
    radius_x_m: float,
    radius_y_m: float,
    yaw: float,
    n: int = 48,
) -> np.ndarray | None:
    points = []
    c = math.cos(yaw)
    s = math.sin(yaw)
    for i in range(n):
        angle = 2.0 * math.pi * i / n
        local_x = radius_x_m * math.cos(angle)
        local_y = radius_y_m * math.sin(angle)
        point = world_to_video(
            x + local_x * c - local_y * s,
            y + local_x * s + local_y * c,
            homography,
        )
        if point is None:
            return None
        points.append((int(round(point[0])), int(round(point[1]))))
    return np.asarray(points, dtype=np.int32)


def project_ground_square(
    homography: np.ndarray,
    x: float,
    y: float,
    half_size_m: float,
) -> np.ndarray | None:
    points = [
        (x - half_size_m, y - half_size_m),
        (x + half_size_m, y - half_size_m),
        (x + half_size_m, y + half_size_m),
        (x - half_size_m, y + half_size_m),
    ]
    return world_polygon_to_video(points, homography)
