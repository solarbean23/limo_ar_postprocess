from __future__ import annotations

from dataclasses import dataclass
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


def world_to_video(
    x: float,
    y: float,
    homography: np.ndarray,
) -> tuple[float, float] | None:
    point = homography @ np.array([x, y, 1.0], dtype=np.float64)
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
