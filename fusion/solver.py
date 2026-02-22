from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .schemas import SlamPose, UwbSample


@dataclass(slots=True)
class SolverConfig:
    max_skew_ms: int = 100
    min_pairs: int = 25
    min_motion_spread_m: float = 0.4
    min_second_singular: float = 0.02
    outlier_mad_scale: float = 2.5
    outlier_min_m: float = 0.12
    min_inliers: int = 15
    min_inlier_ratio: float = 0.6


def solver_config_from_dict(payload: dict[str, Any] | None) -> SolverConfig:
    if payload is None:
        return SolverConfig()
    return SolverConfig(
        max_skew_ms=int(payload.get("max_skew_ms", 100)),
        min_pairs=int(payload.get("min_pairs", 25)),
        min_motion_spread_m=float(payload.get("min_motion_spread_m", 0.4)),
        min_second_singular=float(payload.get("min_second_singular", 0.02)),
        outlier_mad_scale=float(payload.get("outlier_mad_scale", 2.5)),
        outlier_min_m=float(payload.get("outlier_min_m", 0.12)),
        min_inliers=int(payload.get("min_inliers", 15)),
        min_inlier_ratio=float(payload.get("min_inlier_ratio", 0.6)),
    )


def match_samples_by_timestamp(
    uwb_samples: list[UwbSample],
    slam_poses: list[SlamPose],
    max_skew_ms: int,
) -> list[dict[str, Any]]:
    if not uwb_samples or not slam_poses:
        return []

    uwb_sorted = sorted(uwb_samples, key=lambda s: s.ts_ms)
    slam_sorted = sorted(slam_poses, key=lambda s: s.ts_ms)
    slam_times = np.array([s.ts_ms for s in slam_sorted], dtype=np.int64)

    candidates: list[tuple[int, int, int]] = []
    for u_idx, uwb in enumerate(uwb_sorted):
        insert_idx = int(np.searchsorted(slam_times, uwb.ts_ms))
        for s_idx in (insert_idx - 1, insert_idx):
            if s_idx < 0 or s_idx >= len(slam_sorted):
                continue
            skew = abs(int(uwb.ts_ms - slam_sorted[s_idx].ts_ms))
            if skew <= max_skew_ms:
                candidates.append((skew, u_idx, s_idx))

    candidates.sort(key=lambda x: x[0])
    used_u: set[int] = set()
    used_s: set[int] = set()
    pairs: list[dict[str, Any]] = []

    for skew, u_idx, s_idx in candidates:
        if u_idx in used_u or s_idx in used_s:
            continue
        used_u.add(u_idx)
        used_s.add(s_idx)
        uwb = uwb_sorted[u_idx]
        slam = slam_sorted[s_idx]
        pairs.append({"uwb": uwb, "slam": slam, "skew_ms": int(skew)})

    pairs.sort(key=lambda p: p["uwb"].ts_ms)
    return pairs


def _estimate_rigid_umeyama(
    local_points: np.ndarray,  # Nx3
    global_points: np.ndarray,  # Nx3
) -> tuple[np.ndarray, np.ndarray]:
    if local_points.shape != global_points.shape or local_points.shape[1] != 3:
        raise ValueError("Points must be Nx3 and shape-matched")
    n = local_points.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points")

    mu_local = local_points.mean(axis=0)
    mu_global = global_points.mean(axis=0)
    x = local_points - mu_local
    y = global_points - mu_global

    covariance = (y.T @ x) / float(n)
    u, _s, vt = np.linalg.svd(covariance)
    d = np.eye(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        d[2, 2] = -1.0
    r = u @ d @ vt
    t = mu_global - (r @ mu_local)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = r
    T[:3, 3] = t
    return r, T


def _residuals_m(r: np.ndarray, t: np.ndarray, local_points: np.ndarray, global_points: np.ndarray) -> np.ndarray:
    pred = (r @ local_points.T).T + t
    return np.linalg.norm(pred - global_points, axis=1)


def run_calibration(
    uwb_samples: list[UwbSample],
    slam_poses: list[SlamPose],
    config: SolverConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pairs = match_samples_by_timestamp(uwb_samples, slam_poses, max_skew_ms=config.max_skew_ms)
    if len(pairs) < config.min_pairs:
        return (
            {
                "status": "failed",
                "reason": f"insufficient_pairs:{len(pairs)}",
                "sample_pairs": len(pairs),
            },
            pairs,
        )

    p_local = np.array([pair["slam"].t_xyz_m for pair in pairs], dtype=np.float64)
    p_global = np.array([pair["uwb"].relative_xyz_m for pair in pairs], dtype=np.float64)

    spread = float(np.linalg.norm(p_local.max(axis=0) - p_local.min(axis=0)))
    if spread < config.min_motion_spread_m:
        return (
            {
                "status": "failed",
                "reason": f"insufficient_motion_spread:{spread:.4f}",
                "sample_pairs": len(pairs),
            },
            pairs,
        )

    centered = p_local - p_local.mean(axis=0)
    svals = np.linalg.svd(centered, compute_uv=False)
    if svals.shape[0] < 2 or float(svals[1]) < config.min_second_singular:
        return (
            {
                "status": "failed",
                "reason": f"degenerate_geometry:s2={float(svals[1]) if svals.shape[0] > 1 else 0.0:.6f}",
                "sample_pairs": len(pairs),
            },
            pairs,
        )

    r0, T0 = _estimate_rigid_umeyama(p_local, p_global)
    t0 = T0[:3, 3]
    residuals0 = _residuals_m(r0, t0, p_local, p_global)

    med = float(np.median(residuals0))
    mad = float(np.median(np.abs(residuals0 - med)))
    robust_sigma = 1.4826 * mad
    threshold = max(config.outlier_min_m, med + config.outlier_mad_scale * robust_sigma)
    inlier_mask = residuals0 <= threshold
    inlier_count = int(np.count_nonzero(inlier_mask))
    inlier_ratio = float(inlier_count) / float(len(residuals0))

    if inlier_count < config.min_inliers or inlier_ratio < config.min_inlier_ratio:
        return (
            {
                "status": "failed",
                "reason": (
                    f"insufficient_inliers:{inlier_count}/{len(residuals0)} "
                    f"ratio={inlier_ratio:.3f}"
                ),
                "sample_pairs": len(pairs),
                "inliers": inlier_count,
                "inlier_ratio": inlier_ratio,
                "rms_error_m": float(np.sqrt(np.mean(residuals0 ** 2))),
            },
            pairs,
        )

    p_local_in = p_local[inlier_mask]
    p_global_in = p_global[inlier_mask]
    r1, T1 = _estimate_rigid_umeyama(p_local_in, p_global_in)
    t1 = T1[:3, 3]
    residuals1 = _residuals_m(r1, t1, p_local_in, p_global_in)
    rms = float(np.sqrt(np.mean(residuals1 ** 2)))

    for idx, pair in enumerate(pairs):
        pair["residual_m"] = float(residuals0[idx])
        pair["inlier"] = bool(inlier_mask[idx])

    result = {
        "status": "calibrated",
        "sample_pairs": len(pairs),
        "inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "rms_error_m": rms,
        "outlier_threshold_m": float(threshold),
        "T_G_Lw": T1.tolist(),
    }
    return result, pairs


def _quat_xyzw_to_rotation(q_xyzw: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = q_xyzw
    n = float(np.sqrt(x * x + y * y + z * z + w * w))
    if n < 1e-8:
        return np.eye(3, dtype=np.float64)
    x /= n
    y /= n
    z /= n
    w /= n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_to_quat_xyzw(r: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(r[0, 0] + r[1, 1] + r[2, 2])
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return (float(x), float(y), float(z), float(w))


def pose_to_matrix(t_xyz_m: tuple[float, float, float], q_xyzw: tuple[float, float, float, float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_xyzw_to_rotation(q_xyzw)
    T[:3, 3] = np.array(t_xyz_m, dtype=np.float64)
    return T


def matrix_to_pose(T: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    t = (float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
    q = _rotation_to_quat_xyzw(T[:3, :3])
    return t, q


def apply_transform_to_pose(
    T_G_Lw: np.ndarray,
    t_Lw_Cw: tuple[float, float, float],
    q_Lw_Cw: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    T_Lw_Cw = pose_to_matrix(t_Lw_Cw, q_Lw_Cw)
    T_G_Cw = T_G_Lw @ T_Lw_Cw
    return matrix_to_pose(T_G_Cw)


def wrap_angle_pi(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


def infer_face_anchor_yaw(relative_xyz_m: tuple[float, float, float], yaw_offset_rad: float = 0.0) -> float:
    x, y, _z = relative_xyz_m
    if np.hypot(x, y) < 1e-6:
        raise ValueError("Cannot infer yaw: relative xy is near zero")
    # Worker camera is assumed to face from worker toward anchor, i.e. vector -relative_xyz.
    yaw = np.arctan2(-y, -x) + yaw_offset_rad
    return wrap_angle_pi(float(yaw))


def yaw_from_quaternion_xyzw(q_xyzw: tuple[float, float, float, float]) -> float:
    r = _quat_xyzw_to_rotation(q_xyzw)
    return wrap_angle_pi(float(np.arctan2(r[1, 0], r[0, 0])))


def _rotation_z(yaw_rad: float) -> np.ndarray:
    c = np.cos(yaw_rad)
    s = np.sin(yaw_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def solve_initial_xyzyaw_transform(
    global_position_xyz_m: tuple[float, float, float],
    global_yaw_rad: float,
    local_t_xyz_m: tuple[float, float, float],
    local_q_xyzw: tuple[float, float, float, float],
) -> dict[str, Any]:
    local_yaw_rad = yaw_from_quaternion_xyzw(local_q_xyzw)
    yaw_offset_rad = wrap_angle_pi(global_yaw_rad - local_yaw_rad)

    R = _rotation_z(yaw_offset_rad)
    p_global = np.array(global_position_xyz_m, dtype=np.float64)
    p_local = np.array(local_t_xyz_m, dtype=np.float64)
    t = p_global - (R @ p_local)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t

    return {
        "status": "calibrated",
        "calibration_mode": "initial_xyzyaw",
        "sample_pairs": 1,
        "inliers": 1,
        "inlier_ratio": 1.0,
        "rms_error_m": 0.0,
        "assumed_global_yaw_rad": float(wrap_angle_pi(global_yaw_rad)),
        "local_bootstrap_yaw_rad": float(local_yaw_rad),
        "yaw_offset_rad": float(yaw_offset_rad),
        "bootstrap_global_xyz_m": [float(p_global[0]), float(p_global[1]), float(p_global[2])],
        "bootstrap_local_xyz_m": [float(p_local[0]), float(p_local[1]), float(p_local[2])],
        "T_G_Lw": T.tolist(),
    }
