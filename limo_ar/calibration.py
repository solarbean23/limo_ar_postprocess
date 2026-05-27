from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def load_homography(path: str | Path | None) -> np.ndarray | None:
    if not path:
        return None
    homography_path = Path(path)
    if not homography_path.exists():
        return None
    with homography_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "homography" in data:
        data = data["homography"]

    matrix = data.get("matrix") or data.get("H")
    if matrix is not None:
        h = np.asarray(matrix, dtype=np.float64)
        if h.shape != (3, 3):
            raise ValueError(f"homography matrix must be 3x3: {homography_path}")
        return h

    world_points = np.asarray(data.get("world_points", []), dtype=np.float64)
    image_points = np.asarray(data.get("image_points", []), dtype=np.float64)
    if world_points.shape[0] < 4 or image_points.shape[0] < 4:
        raise ValueError(
            "homography.yaml needs matrix/H or at least four world_points and image_points"
        )
    h, _ = cv2.findHomography(world_points[:, :2], image_points[:, :2])
    if h is None:
        raise ValueError(f"could not compute homography from {homography_path}")
    return h


def save_homography(
    path: str | Path,
    world_points: list[list[float]],
    image_points: list[list[float]],
) -> None:
    h, _ = cv2.findHomography(
        np.asarray(world_points, dtype=np.float64),
        np.asarray(image_points, dtype=np.float64),
    )
    if h is None:
        raise ValueError("could not compute homography")
    output = {
        "homography": {
            "world_points": world_points,
            "image_points": image_points,
            "matrix": h.tolist(),
        }
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(output, f, sort_keys=False)


def load_world_points(path: str | Path) -> list[list[float]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data: Any = yaml.safe_load(f) or {}
    if isinstance(data, dict):
        data = data.get("world_points") or data.get("points")
    if not isinstance(data, list) or len(data) < 4:
        raise ValueError("world point file needs a list of at least four [x, y] points")
    return [[float(point[0]), float(point[1])] for point in data]


def main() -> None:
    parser = argparse.ArgumentParser(description="Click image points for a homography.yaml file.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--world-points", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frame-sec", type=float, default=0.0)
    args = parser.parse_args()

    world_points = load_world_points(args.world_points)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(args.frame_sec * fps)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("could not read calibration frame")

    image_points: list[list[float]] = []
    window = "limo_ar_calibration"

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or len(image_points) >= len(world_points):
            return
        image_points.append([float(x), float(y)])
        cv2.circle(frame, (x, y), 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            str(len(image_points)),
            (x + 7, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    print(f"Click {len(world_points)} image points in the same order as world points.")
    while len(image_points) < len(world_points):
        cv2.imshow(window, frame)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            raise SystemExit("calibration cancelled")
    cv2.destroyWindow(window)
    save_homography(args.output, world_points, image_points)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
