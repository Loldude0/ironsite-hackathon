"""
pointcloud_locator – Locate 3-D positions of objects detected by YOLOv8
by casting rays from a camera into a SLAM point cloud.

Usage
-----
>>> from pointcloud_locator import ObjectLocator
>>> locator = ObjectLocator("map.ply", camera_intrinsics)
>>> hits = locator.locate(camera_pos, camera_rot, bounding_boxes)
"""

from .types import CameraIntrinsics, BoundingBox, HitResult
from .locator import ObjectLocator

__all__ = [
    "ObjectLocator",
    "CameraIntrinsics",
    "BoundingBox",
    "HitResult",
]
