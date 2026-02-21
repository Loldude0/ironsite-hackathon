"""
pointcloud_locator – Locate 3-D positions of objects detected by YOLOv8
by casting rays from a camera into a SLAM point cloud.

Usage
-----
>>> from pointcloud_locator import ObjectLocator
>>> locator = ObjectLocator("map.ply", camera_intrinsics)
>>> hits = locator.locate(camera_pos, camera_rot, bounding_boxes)
"""

from .types import CameraIntrinsics, BoundingBox, BoundingBox2D, HitResult
from .locator import ObjectLocator
from .viewer import view_point_cloud, yolo_output_to_viewer_bboxes

__all__ = [
    "ObjectLocator",
    "view_point_cloud",
    "CameraIntrinsics",
    "BoundingBox",
    "BoundingBox2D",
    "HitResult",
    "yolo_output_to_viewer_bboxes",
]
