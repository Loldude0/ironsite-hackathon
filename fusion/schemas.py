from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing or invalid string field: {key}")
    return value


def _require_float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Missing or invalid float field: {key}") from exc


def _require_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"Missing or invalid int field: {key}") from exc


def _require_vec3(payload: dict[str, Any], key: str) -> tuple[float, float, float]:
    value = payload.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Missing or invalid vec3 field: {key}")
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except Exception as exc:
        raise ValueError(f"Missing or invalid vec3 field: {key}") from exc


def _require_quat_xyzw(payload: dict[str, Any], key: str) -> tuple[float, float, float, float]:
    value = payload.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"Missing or invalid quaternion field: {key}")
    try:
        return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    except Exception as exc:
        raise ValueError(f"Missing or invalid quaternion field: {key}") from exc


@dataclass(slots=True)
class UwbSample:
    session_id: str
    anchor_id: str
    worker_id: str
    ts_ms: int
    distance_m: float
    direction_xyz: tuple[float, float, float]
    relative_xyz_m: tuple[float, float, float]
    server_rx_ms: int


@dataclass(slots=True)
class SlamPose:
    session_id: str
    node_id: str
    ts_ms: int
    t_xyz_m: tuple[float, float, float]
    q_xyzw: tuple[float, float, float, float]
    tracking_state: int | None
    server_rx_ms: int


@dataclass(slots=True)
class CalibrationStart:
    session_id: str
    anchor_id: str
    worker_id: str
    duration_s: float
    ts_ms: int
    calibration_mode: str
    yaw_mode: str
    yaw_offset_rad: float


def parse_calibration_start(payload: dict[str, Any], *, server_rx_ms: int) -> CalibrationStart:
    calibration_mode = str(payload.get("calibration_mode", "paired_6dof"))
    if calibration_mode not in ("paired_6dof", "initial_xyzyaw"):
        raise ValueError("Invalid calibration_mode")

    yaw_mode = str(payload.get("yaw_mode", "face_anchor"))
    if yaw_mode not in ("face_anchor",):
        raise ValueError("Invalid yaw_mode")

    return CalibrationStart(
        session_id=_require_str(payload, "session_id"),
        anchor_id=_require_str(payload, "anchor_id"),
        worker_id=_require_str(payload, "worker_id"),
        duration_s=max(1.0, float(payload.get("duration_s", 5.0))),
        ts_ms=_require_int(payload, "ts_ms"),
        calibration_mode=calibration_mode,
        yaw_mode=yaw_mode,
        yaw_offset_rad=float(payload.get("yaw_offset_rad", 0.0)),
    )


def parse_uwb_sample(payload: dict[str, Any], *, server_rx_ms: int) -> UwbSample:
    message_type = payload.get("type")
    if message_type not in (None, "uwb_sample"):
        raise ValueError("Invalid message type for UWB sample")
    return UwbSample(
        session_id=_require_str(payload, "session_id"),
        anchor_id=_require_str(payload, "anchor_id"),
        worker_id=_require_str(payload, "worker_id"),
        ts_ms=_require_int(payload, "ts_ms"),
        distance_m=_require_float(payload, "distance_m"),
        direction_xyz=_require_vec3(payload, "direction_xyz"),
        relative_xyz_m=_require_vec3(payload, "relative_xyz_m"),
        server_rx_ms=server_rx_ms,
    )


def parse_slam_pose(payload: dict[str, Any], *, server_rx_ms: int) -> SlamPose:
    message_type = payload.get("type")
    if message_type not in (None, "slam_pose"):
        raise ValueError("Invalid message type for SLAM pose")

    tracking_state: int | None = None
    if "tracking_state" in payload:
        try:
            tracking_state = int(payload["tracking_state"])
        except Exception as exc:
            raise ValueError("Invalid tracking_state") from exc

    return SlamPose(
        session_id=_require_str(payload, "session_id"),
        node_id=_require_str(payload, "node_id"),
        ts_ms=_require_int(payload, "ts_ms"),
        t_xyz_m=_require_vec3(payload, "t_xyz_m"),
        q_xyzw=_require_quat_xyzw(payload, "q_xyzw"),
        tracking_state=tracking_state,
        server_rx_ms=server_rx_ms,
    )
