#!/usr/bin/env python3
"""ARVOS LiDAR recorder.

Receives iPhone Pro LiDAR stream via ARVOS websocket server and writes:
1) per-frame point cloud to .pcd
2) per-frame pose/orientation metadata to .json

Example
-------
python arvos_server.py --port 9090 --output-dir data/arvos_capture
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
from arvos import ArvosServer


@dataclass
class PoseRecord:
    timestamp_unix: float
    frame_index: int
    position: list[float] | None
    quaternion_wxyz: list[float] | None
    euler_deg_rpy: list[float] | None
    rotation_matrix_3x3: list[list[float]] | None
    raw_pose: dict[str, Any] | None


@dataclass
class ImageRecord:
    timestamp_unix: float
    frame_index: int
    image_file: str
    width: int
    height: int
    channels: int
    position: list[float] | None
    quaternion_wxyz: list[float] | None
    euler_deg_rpy: list[float] | None
    rotation_matrix_3x3: list[list[float]] | None
    raw_pose: dict[str, Any] | None


def _to_dict(obj: Any) -> dict[str, Any] | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        out: dict[str, Any] = {}
        for k, v in vars(obj).items():
            if k.startswith("_"):
                continue
            if isinstance(v, (str, int, float, bool, type(None))):
                out[k] = v
            elif isinstance(v, (list, tuple)):
                out[k] = list(v)
            elif isinstance(v, dict):
                out[k] = v
        return out
    return None


def _extract_points(data: Any) -> np.ndarray:
    """Try multiple likely ARVOS payload layouts and return Nx3 points."""
    candidates = ["points", "point_cloud", "pointcloud", "xyz", "vertices"]

    source = data
    if not isinstance(source, dict):
        source = _to_dict(data) or {}

    for key in candidates:
        if key not in source:
            continue
        arr = np.asarray(source[key], dtype=np.float64)
        if arr.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        if arr.ndim == 1 and arr.size % 3 == 0:
            arr = arr.reshape(-1, 3)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return arr[:, :3]

    return np.empty((0, 3), dtype=np.float64)


def _extract_pose_dict(data: Any) -> dict[str, Any] | None:
    source = data if isinstance(data, dict) else (_to_dict(data) or {})
    for key in ("pose", "camera_pose", "transform", "attitude"):
        if key in source and isinstance(source[key], dict):
            return source[key]
    return source if source else None


def _extract_position(pose: dict[str, Any] | None) -> list[float] | None:
    if not pose:
        return None
    if "position" in pose and isinstance(pose["position"], (list, tuple)) and len(pose["position"]) >= 3:
        return [float(pose["position"][0]), float(pose["position"][1]), float(pose["position"][2])]
    if all(k in pose for k in ("x", "y", "z")):
        return [float(pose["x"]), float(pose["y"]), float(pose["z"])]
    if all(k in pose for k in ("tx", "ty", "tz")):
        return [float(pose["tx"]), float(pose["ty"]), float(pose["tz"])]
    return None


def _extract_quaternion_wxyz(pose: dict[str, Any] | None) -> list[float] | None:
    if not pose:
        return None
    if "quaternion" in pose and isinstance(pose["quaternion"], (list, tuple)) and len(pose["quaternion"]) >= 4:
        q = pose["quaternion"]
        return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
    if all(k in pose for k in ("qw", "qx", "qy", "qz")):
        return [float(pose["qw"]), float(pose["qx"]), float(pose["qy"]), float(pose["qz"])]
    if all(k in pose for k in ("w", "x", "y", "z")):
        return [float(pose["w"]), float(pose["x"]), float(pose["y"]), float(pose["z"])]
    return None


def _extract_rotation_matrix(pose: dict[str, Any] | None) -> list[list[float]] | None:
    if not pose:
        return None
    mat = pose.get("rotation_matrix")
    if isinstance(mat, list):
        arr = np.asarray(mat, dtype=np.float64)
        if arr.shape == (3, 3):
            return arr.tolist()
    return None


def _extract_euler_deg(pose: dict[str, Any] | None) -> list[float] | None:
    if not pose:
        return None
    if "euler_deg" in pose and isinstance(pose["euler_deg"], (list, tuple)) and len(pose["euler_deg"]) >= 3:
        return [float(pose["euler_deg"][0]), float(pose["euler_deg"][1]), float(pose["euler_deg"][2])]
    if all(k in pose for k in ("roll", "pitch", "yaw")):
        return [float(pose["roll"]), float(pose["pitch"]), float(pose["yaw"])]
    return None


def _decode_image_bytes(blob: bytes) -> np.ndarray | None:
    arr = np.frombuffer(blob, dtype=np.uint8)
    if arr.size == 0:
        return None
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _extract_image(data: Any) -> np.ndarray | None:
    source = data if isinstance(data, dict) else (_to_dict(data) or {})

    def _from_any(value: Any) -> np.ndarray | None:
        if value is None:
            return None

        if isinstance(value, np.ndarray):
            if value.ndim == 3 and value.shape[2] in (1, 3, 4):
                img = value
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                if img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                return img
            return None

        if isinstance(value, (bytes, bytearray)):
            return _decode_image_bytes(bytes(value))

        if isinstance(value, str):
            try:
                return _decode_image_bytes(base64.b64decode(value, validate=False))
            except Exception:
                return None

        if isinstance(value, list):
            arr = np.asarray(value)
            if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                if arr.shape[2] == 4:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
                return arr
            return None

        if isinstance(value, dict):
            for key in ("data", "bytes", "buffer", "image", "frame", "jpeg", "jpg", "png"):
                if key in value:
                    img = _from_any(value[key])
                    if img is not None:
                        return img
        return None

    for key in ("image", "photo", "camera_image", "camera_frame", "frame", "rgb", "jpeg", "jpg", "png", "data"):
        if key in source:
            img = _from_any(source[key])
            if img is not None:
                return img

    return _from_any(data)


async def run_server(port: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pcd_dir = output_dir / "pcd"
    pose_dir = output_dir / "pose"
    image_dir = output_dir / "images"
    image_pose_dir = output_dir / "image_pose"
    pcd_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    image_pose_dir.mkdir(parents=True, exist_ok=True)

    server = ArvosServer(port=port)
    frame_index = 0
    image_index = 0
    latest_pose: dict[str, Any] | None = None

    def _save_frame(points_xyz: np.ndarray, pose_payload: dict[str, Any] | None, timestamp: float) -> None:
        nonlocal frame_index
        if points_xyz.size == 0:
            return

        valid = np.isfinite(points_xyz).all(axis=1)
        points_xyz = points_xyz[valid]
        if len(points_xyz) == 0:
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_xyz)

        frame_tag = f"frame_{frame_index:06d}_{int(timestamp * 1000)}"
        pcd_path = pcd_dir / f"{frame_tag}.pcd"
        pose_path = pose_dir / f"{frame_tag}.json"

        ok = o3d.io.write_point_cloud(str(pcd_path), pcd)
        if not ok:
            raise RuntimeError(f"Failed to write point cloud: {pcd_path}")

        pose_dict = pose_payload if pose_payload is not None else latest_pose
        pose_record = PoseRecord(
            timestamp_unix=timestamp,
            frame_index=frame_index,
            position=_extract_position(pose_dict),
            quaternion_wxyz=_extract_quaternion_wxyz(pose_dict),
            euler_deg_rpy=_extract_euler_deg(pose_dict),
            rotation_matrix_3x3=_extract_rotation_matrix(pose_dict),
            raw_pose=pose_dict,
        )
        pose_path.write_text(json.dumps(asdict(pose_record), indent=2) + "\n", encoding="utf-8")

        frame_index += 1
        print(f"[saved] #{frame_index:06d} points={len(points_xyz):,} -> {pcd_path.name}")

    async def _on_pose_like(data: Any) -> None:
        nonlocal latest_pose
        latest_pose = _extract_pose_dict(data)

    async def _on_pointcloud_like(data: Any) -> None:
        ts = time.time()
        payload = _to_dict(data) or {}
        points = _extract_points(payload)
        pose_payload = _extract_pose_dict(payload)
        _save_frame(points_xyz=points, pose_payload=pose_payload, timestamp=ts)

    def _save_image_frame(image_bgr: np.ndarray, pose_payload: dict[str, Any] | None, timestamp: float) -> None:
        nonlocal image_index
        if image_bgr.size == 0:
            return

        if image_bgr.ndim != 3 or image_bgr.shape[2] not in (1, 3):
            return

        h, w = int(image_bgr.shape[0]), int(image_bgr.shape[1])
        channels = int(image_bgr.shape[2])

        frame_tag = f"image_{image_index:06d}_{int(timestamp * 1000)}"
        image_path = image_dir / f"{frame_tag}.jpg"
        pose_path = image_pose_dir / f"{frame_tag}.json"

        ok = cv2.imwrite(str(image_path), image_bgr)
        if not ok:
            raise RuntimeError(f"Failed to write image: {image_path}")

        pose_dict = pose_payload if pose_payload is not None else latest_pose
        image_record = ImageRecord(
            timestamp_unix=timestamp,
            frame_index=image_index,
            image_file=image_path.name,
            width=w,
            height=h,
            channels=channels,
            position=_extract_position(pose_dict),
            quaternion_wxyz=_extract_quaternion_wxyz(pose_dict),
            euler_deg_rpy=_extract_euler_deg(pose_dict),
            rotation_matrix_3x3=_extract_rotation_matrix(pose_dict),
            raw_pose=pose_dict,
        )
        pose_path.write_text(json.dumps(asdict(image_record), indent=2) + "\n", encoding="utf-8")

        image_index += 1
        print(f"[saved-image] #{image_index:06d} {w}x{h} -> {image_path.name}")

    async def _on_image_like(data: Any) -> None:
        ts = time.time()
        payload = _to_dict(data) or {}
        img = _extract_image(payload if payload else data)
        if img is None:
            return
        pose_payload = _extract_pose_dict(payload)
        _save_image_frame(image_bgr=img, pose_payload=pose_payload, timestamp=ts)

    # Show QR code for quick phone connection.
    if hasattr(server, "print_qr_code"):
        server.print_qr_code()

    # Register pose/imu callbacks (best effort for different ARVOS versions).
    for pose_cb_name in ("on_pose", "on_camera_pose", "on_transform", "on_imu"):
        if hasattr(server, pose_cb_name):
            setattr(server, pose_cb_name, _on_pose_like)

    # Register point cloud callbacks (best effort for different ARVOS versions).
    registered_point_cb = False
    for cloud_cb_name in ("on_point_cloud", "on_pointcloud", "on_lidar", "on_depth"):
        if hasattr(server, cloud_cb_name):
            setattr(server, cloud_cb_name, _on_pointcloud_like)
            registered_point_cb = True

    if not registered_point_cb:
        raise RuntimeError(
            "Could not find a point-cloud callback on ArvosServer. "
            "Expected one of: on_point_cloud, on_pointcloud, on_lidar, on_depth"
        )

    # Register image/photo callbacks (best effort for different ARVOS versions).
    registered_image_cb = False
    for image_cb_name in ("on_image", "on_photo", "on_camera_image", "on_camera_frame", "on_rgb", "on_video_frame"):
        if hasattr(server, image_cb_name):
            setattr(server, image_cb_name, _on_image_like)
            registered_image_cb = True

    if not registered_image_cb:
        print("[warn] no camera image callback found on ArvosServer; image capture disabled")

    print(f"[arvos] listening on ws://0.0.0.0:{port}")
    print(f"[arvos] saving PCD to {pcd_dir}")
    print(f"[arvos] saving poses to {pose_dir}")
    print(f"[arvos] saving images to {image_dir}")
    print(f"[arvos] saving image poses to {image_pose_dir}")
    await server.start()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record iPhone LiDAR from ARVOS websocket into PCD + pose JSON.")
    parser.add_argument("--port", type=int, default=9090, help="WebSocket port for ARVOS server")
    parser.add_argument("--output-dir", type=Path, default=Path("data/arvos_capture"), help="Output folder")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(run_server(port=args.port, output_dir=args.output_dir))


if __name__ == "__main__":
    main()