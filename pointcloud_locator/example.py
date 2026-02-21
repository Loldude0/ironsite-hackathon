#!/usr/bin/env python3
"""Example: locate YOLOv8-detected objects in a SLAM point cloud.

This script synthesises a small point cloud and a handful of fake
detections so you can run it stand-alone without real data.

    python -m pointcloud_locator.example
"""

import numpy as np

from pointcloud_locator import BoundingBox, CameraIntrinsics, ObjectLocator


def main() -> None:
    # ------------------------------------------------------------------
    # 1.  Synthesise a tiny point cloud (a wall at z ≈ 5 m)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    n_points = 50_000
    # Flat wall spanning x ∈ [-3, 3], y ∈ [-2, 2], z ∈ [4.9, 5.1]
    cloud = np.column_stack([
        rng.uniform(-3.0, 3.0, n_points),
        rng.uniform(-2.0, 2.0, n_points),
        rng.uniform(4.9, 5.1, n_points),
    ])

    # ------------------------------------------------------------------
    # 2.  Define the camera
    # ------------------------------------------------------------------
    intrinsics = CameraIntrinsics(
        fx=525.0, fy=525.0,   # typical for a 640×480 RGB-D sensor
        cx=320.0, cy=240.0,
        width=640, height=480,
    )

    camera_position = np.array([0.0, 0.0, 0.0])
    # Camera looking straight down the +Z axis (identity rotation)
    camera_rotation = np.eye(3)

    # ------------------------------------------------------------------
    # 3.  Simulate YOLOv8 detections
    # ------------------------------------------------------------------
    detections = [
        BoundingBox(x_center=320, y_center=240, width=100, height=80,
                    class_id=0, class_name="person", confidence=0.92),
        BoundingBox(x_center=160, y_center=120, width=60, height=50,
                    class_id=56, class_name="chair", confidence=0.81),
        BoundingBox(x_center=500, y_center=400, width=80, height=70,
                    class_id=62, class_name="tv", confidence=0.75),
    ]

    # ------------------------------------------------------------------
    # 4.  Locate!
    # ------------------------------------------------------------------
    locator = ObjectLocator(cloud, intrinsics)
    print(f"Point cloud loaded: {locator.num_points:,} points\n")

    results = locator.locate(
        camera_position=camera_position,
        camera_rotation=camera_rotation,
        bounding_boxes=detections,
        radius=0.15,        # hit tolerance in metres
        max_range=20.0,     # search up to 20 m
    )

    # ------------------------------------------------------------------
    # 5.  Print results
    # ------------------------------------------------------------------
    for hit in results:
        tag = hit.bbox.class_name
        if hit.hit:
            x, y, z = hit.position
            print(f"[HIT]  {tag:>10s}  →  ({x:+.3f}, {y:+.3f}, {z:+.3f})  "
                  f"dist={hit.distance:.2f} m   (point #{hit.point_index})")
        else:
            print(f"[MISS] {tag:>10s}  →  no intersection")


if __name__ == "__main__":
    main()
