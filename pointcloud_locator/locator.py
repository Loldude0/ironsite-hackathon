"""High-level API – ties together point-cloud loading, camera model, and
ray casting into a single easy-to-use class.

Example
-------
>>> from pointcloud_locator import ObjectLocator, CameraIntrinsics, BoundingBox
>>> intrinsics = CameraIntrinsics(fx=525, fy=525, cx=320, cy=240, width=640, height=480)
>>> locator = ObjectLocator("map.ply", intrinsics)
>>> cam_pos = [1.0, 2.0, 0.5]
>>> cam_rot = [0.0, 0.0, 0.0]          # Euler [roll, pitch, yaw]
>>> boxes = [BoundingBox(320, 240, 100, 80, class_name="chair")]
>>> hits = locator.locate(cam_pos, cam_rot, boxes, radius=0.05)
>>> for h in hits:
...     print(h.bbox.class_name, h.position, h.distance)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from .camera import pixel_to_ray
from .point_cloud import load_point_cloud
from .ray_caster import RayCaster
from .types import BoundingBox, CameraIntrinsics, HitResult


class ObjectLocator:
    """Locate detected objects in 3-D by casting rays into a SLAM point cloud.

    Parameters
    ----------
    point_cloud : str, Path, or np.ndarray
        Either a file path to a supported point-cloud format **or** a
        pre-loaded ``(N, 3)`` NumPy array.
    intrinsics : CameraIntrinsics
        Pinhole camera intrinsic parameters that match the images fed to
        YOLOv8.
    kdtree_leafsize : int
        Leaf size for the internal KD-tree (default 32).
    """

    def __init__(
        self,
        point_cloud: Union[str, Path, np.ndarray],
        intrinsics: CameraIntrinsics,
        kdtree_leafsize: int = 32,
    ) -> None:
        if isinstance(point_cloud, np.ndarray):
            self.points = np.ascontiguousarray(point_cloud[:, :3], dtype=np.float64)
        else:
            self.points = load_point_cloud(point_cloud)

        self.intrinsics = intrinsics
        self._caster = RayCaster(self.points, leafsize=kdtree_leafsize)

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def locate(
        self,
        camera_position: Union[Sequence[float], np.ndarray],
        camera_rotation: Union[Sequence[float], np.ndarray],
        bounding_boxes: List[BoundingBox],
        radius: float = 0.05,
        max_range: float = 100.0,
        step: Optional[float] = None,
    ) -> List[HitResult]:
        """Cast a ray per bounding-box and return the first hit positions.

        Parameters
        ----------
        camera_position : array-like, shape (3,)
            Camera position in world coordinates ``(x, y, z)``.
        camera_rotation : array-like
            Rotation from **camera → world**.  Accepted shapes:

            * ``(3, 3)`` – rotation matrix
            * ``(4,)``   – quaternion ``[w, x, y, z]``
            * ``(3,)``   – Euler angles ``[roll, pitch, yaw]`` in radians

        bounding_boxes : list of BoundingBox
            YOLOv8 detections (pixel coordinates).
        radius : float
            Hit tolerance – a point is considered intersected if its
            perpendicular distance to the ray is ≤ *radius*.
        max_range : float
            Maximum search distance along the ray (world units).
        step : float or None
            Ray-march step size.  Defaults to *radius*.

        Returns
        -------
        list of HitResult
            One result per bounding box, in the same order.
        """
        cam_pos = np.asarray(camera_position, dtype=np.float64).ravel()
        cam_rot = np.asarray(camera_rotation, dtype=np.float64)

        results: List[HitResult] = []

        for bbox in bounding_boxes:
            # Build ray from pixel centre of bounding box
            origin, direction = pixel_to_ray(
                u=bbox.x_center,
                v=bbox.y_center,
                intrinsics=self.intrinsics,
                camera_position=cam_pos,
                camera_rotation=cam_rot,
            )

            pos, dist, idx = self._caster.cast_ray(
                origin=origin,
                direction=direction,
                radius=radius,
                max_range=max_range,
                step=step,
            )

            results.append(HitResult(
                bbox=bbox,
                ray_direction=direction,
                position=pos,
                distance=dist,
                hit=pos is not None,
                point_index=idx,
            ))

        return results

    # ------------------------------------------------------------------ #
    #  Convenience helpers                                                 #
    # ------------------------------------------------------------------ #

    def locate_from_yolo_results(
        self,
        camera_position: Union[Sequence[float], np.ndarray],
        camera_rotation: Union[Sequence[float], np.ndarray],
        yolo_results,  # ultralytics Results object
        radius: float = 0.05,
        max_range: float = 100.0,
    ) -> List[HitResult]:
        """Like :meth:`locate`, but accepts a raw ``ultralytics`` Results
        object directly so you don't have to convert bounding boxes
        manually.

        Parameters
        ----------
        yolo_results : ultralytics.engine.results.Results
            A single ``Results`` object returned by ``model(image)``.
        """
        boxes: List[BoundingBox] = []
        for det in yolo_results.boxes:
            xyxy = det.xyxy[0].cpu().numpy()
            cls_id = int(det.cls[0])
            conf = float(det.conf[0])
            name = yolo_results.names.get(cls_id, str(cls_id))
            boxes.append(BoundingBox.from_xyxy(
                x1=float(xyxy[0]), y1=float(xyxy[1]),
                x2=float(xyxy[2]), y2=float(xyxy[3]),
                class_id=cls_id,
                class_name=name,
                confidence=conf,
            ))

        return self.locate(
            camera_position=camera_position,
            camera_rotation=camera_rotation,
            bounding_boxes=boxes,
            radius=radius,
            max_range=max_range,
        )

    @property
    def num_points(self) -> int:
        """Number of points in the loaded cloud."""
        return len(self.points)
