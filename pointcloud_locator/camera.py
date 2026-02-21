"""Camera model – converts 2-D pixel coordinates to 3-D world-space rays."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .types import CameraIntrinsics


def pixel_to_ray(
    u: float,
    v: float,
    intrinsics: CameraIntrinsics,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a pixel coordinate to a world-space ray.

    Parameters
    ----------
    u, v : float
        Pixel coordinates (column, row) – e.g. the centre of a bounding box.
    intrinsics : CameraIntrinsics
        Camera intrinsic parameters.
    camera_position : np.ndarray
        Camera location in world coordinates, shape ``(3,)``.
    camera_rotation : np.ndarray
        Rotation from **camera frame → world frame**.
        Accepted shapes:

        * ``(3, 3)`` – rotation matrix
        * ``(4,)``   – quaternion ``[w, x, y, z]`` (scalar-first)
        * ``(3,)``   – Euler angles ``[roll, pitch, yaw]`` in **radians**
          applied in the order *X → Y → Z* (extrinsic / fixed-axes).

    Returns
    -------
    origin : np.ndarray
        Ray origin in world space (= *camera_position*), shape ``(3,)``.
    direction : np.ndarray
        Unit direction vector of the ray in world space, shape ``(3,)``.
    """
    camera_position = np.asarray(camera_position, dtype=np.float64).ravel()
    camera_rotation = np.asarray(camera_rotation, dtype=np.float64)

    R = _to_rotation_matrix(camera_rotation)

    # Back-project pixel into camera-frame direction (z-forward convention)
    x_cam = (u - intrinsics.cx) / intrinsics.fx
    y_cam = (v - intrinsics.cy) / intrinsics.fy
    dir_cam = np.array([x_cam, y_cam, 1.0], dtype=np.float64)

    # Transform to world frame
    dir_world = R @ dir_cam
    dir_world /= np.linalg.norm(dir_world)

    return camera_position.copy(), dir_world


# --------------------------------------------------------------------------- #
#  Internal helpers                                                            #
# --------------------------------------------------------------------------- #

def _to_rotation_matrix(rot: np.ndarray) -> np.ndarray:
    """Convert various rotation representations to a 3×3 matrix.

    Supports:
    * (3, 3) – returned as-is.
    * (4,)   – quaternion [w, x, y, z].
    * (3,)   – Euler angles [roll, pitch, yaw] in radians (XYZ extrinsic).
    """
    if rot.shape == (3, 3):
        return rot
    elif rot.shape == (4,):
        return _quat_to_matrix(rot)
    elif rot.shape == (3,):
        return _euler_to_matrix(rot)
    else:
        raise ValueError(
            f"Unsupported rotation shape {rot.shape}. "
            "Expected (3,3), (4,) [quat wxyz], or (3,) [euler rpy]."
        )


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Quaternion [w, x, y, z] → 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _euler_to_matrix(angles: np.ndarray) -> np.ndarray:
    """Euler angles [roll, pitch, yaw] (radians, XYZ extrinsic) → 3×3 matrix.

    The composition is R = Rz(yaw) · Ry(pitch) · Rx(roll).
    """
    roll, pitch, yaw = angles
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)

    return Rz @ Ry @ Rx
