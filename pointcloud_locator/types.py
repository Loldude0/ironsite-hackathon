"""Shared data-classes used throughout the package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class CameraIntrinsics:
    """Pin-hole camera intrinsic parameters.

    Parameters
    ----------
    fx, fy : float
        Focal length in pixels (horizontal / vertical).
    cx, cy : float
        Principal point (optical centre) in pixels.
    width, height : int
        Image resolution used by YOLOv8 for detection.
    """
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def matrix(self) -> np.ndarray:
        """Return the 3×3 intrinsic matrix **K**."""
        return np.array([
            [self.fx, 0.0,     self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,     0.0,     1.0],
        ], dtype=np.float64)


@dataclass
class BoundingBox:
    """A single YOLOv8 detection bounding box.

    Coordinates are in pixels and follow the YOLO (x_center, y_center, w, h)
    convention, but absolute-pixel values (not normalised).

    Parameters
    ----------
    x_center, y_center : float
        Centre of the bounding box in pixel coordinates.
    width, height : float
        Width / height of the bounding box in pixels.
    class_id : int
        Detected class index.
    class_name : str
        Human-readable class label (e.g. ``"person"``).
    confidence : float
        Detection confidence ∈ [0, 1].
    """
    x_center: float
    y_center: float
    width: float
    height: float
    class_id: int = 0
    class_name: str = ""
    confidence: float = 1.0

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_xyxy(
        cls,
        x1: float, y1: float, x2: float, y2: float,
        class_id: int = 0, class_name: str = "", confidence: float = 1.0,
    ) -> "BoundingBox":
        """Create from top-left / bottom-right corners."""
        return cls(
            x_center=(x1 + x2) / 2.0,
            y_center=(y1 + y2) / 2.0,
            width=abs(x2 - x1),
            height=abs(y2 - y1),
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
        )

    @classmethod
    def from_xywh(
        cls,
        x: float, y: float, w: float, h: float,
        class_id: int = 0, class_name: str = "", confidence: float = 1.0,
    ) -> "BoundingBox":
        """Create from top-left corner + width/height."""
        return cls(
            x_center=x + w / 2.0,
            y_center=y + h / 2.0,
            width=w,
            height=h,
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
        )


@dataclass
class HitResult:
    """Result of a single ray-cast against the point cloud.

    Attributes
    ----------
    position : np.ndarray | None
        World-space (x, y, z) of the first hit, or *None* on miss.
    distance : float
        Distance from the camera origin to the hit point.  ``inf`` on miss.
    bbox : BoundingBox
        The originating detection.
    ray_direction : np.ndarray
        Unit direction vector of the cast ray (world frame).
    hit : bool
        Whether the ray intersected the point cloud.
    point_index : int | None
        Index of the hit point in the original point cloud array.
    """
    bbox: BoundingBox
    ray_direction: np.ndarray
    position: Optional[np.ndarray] = None
    distance: float = float("inf")
    hit: bool = False
    point_index: Optional[int] = None
