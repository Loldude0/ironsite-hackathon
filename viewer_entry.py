#!/usr/bin/env python3
"""Entry script: run YOLO on an image, then launch the point-cloud viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pointcloud_locator.viewer import view_point_cloud
from pointcloud_locator.point_cloud import load_point_cloud
from yolo import RealtimeYoloVideoBBoxStream


def _resolve_yolo_device(cli_device: str | None, config_device: str | None) -> str | None:
    """Resolve YOLO device with precedence: CLI > config > auto-detect CUDA > default."""
    if cli_device is not None:
        return cli_device
    if config_device is not None:
        return config_device

    try:
        import torch
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass

    return None


def _load_realtime_config(config_path: Path) -> tuple[float, float | None, float | None, str | None, bool, str, float, bool]:
    """Load realtime YOLO/viewer configuration.

    Returns
    -------
    update_interval_ms, conf, iou, device, show_window, window_name,
    point_cloud_update_interval_ms, point_cloud_loop
    """
    if not config_path.exists():
        return 200.0, None, None, None, True, "YOLO Realtime", 200.0, True

    payload: dict[str, Any] = json.loads(config_path.read_text())
    update_interval_ms = float(payload.get("update_interval_ms", 200.0))
    yolo_cfg = payload.get("yolo", {}) if isinstance(payload.get("yolo", {}), dict) else {}
    window_cfg = payload.get("window", {}) if isinstance(payload.get("window", {}), dict) else {}
    point_cloud_cfg = payload.get("point_cloud", {}) if isinstance(payload.get("point_cloud", {}), dict) else {}
    conf = yolo_cfg.get("conf")
    iou = yolo_cfg.get("iou")
    device = yolo_cfg.get("device")
    show_window = bool(window_cfg.get("show", True))
    window_name = str(window_cfg.get("name", "YOLO Realtime"))
    point_cloud_update_interval_ms = float(point_cloud_cfg.get("update_interval_ms", update_interval_ms))
    point_cloud_loop = bool(point_cloud_cfg.get("loop", True))

    conf_v = float(conf) if conf is not None else None
    iou_v = float(iou) if iou is not None else None
    device_v = str(device) if device is not None else None
    return (
        update_interval_ms,
        conf_v,
        iou_v,
        device_v,
        show_window,
        window_name,
        point_cloud_update_interval_ms,
        point_cloud_loop,
    )


def _build_point_cloud_folder_provider(pcd_files: list[Path], loop: bool):
    next_idx = 1  # index 0 is loaded as initial cloud

    def _provider():
        nonlocal next_idx
        if not pcd_files:
            return None
        if next_idx >= len(pcd_files):
            if not loop:
                return None
            next_idx = 0

        points = load_point_cloud(pcd_files[next_idx])
        next_idx += 1
        return points

    return _provider


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLO on an MP4 stream and open viewer with realtime bounding-box updates.",
    )
    parser.add_argument("point_cloud", type=Path, help="Path to point cloud (.pcd/.ply/.xyz/.txt/.csv/.npy)")
    parser.add_argument("--video", type=Path, required=True, help="Path to input MP4 for realtime YOLO")
    parser.add_argument("--camera-config", type=Path, default=Path("assets/sample_camera_config.json"), help="Path to camera config JSON")
    parser.add_argument("--model", type=Path, default=Path("yolo26n.pt"), help="YOLO model weights path")
    parser.add_argument("--realtime-config", type=Path, default=Path("config/realtime_yolo_config.json"), help="Realtime update config JSON")
    parser.add_argument("--device", type=str, default=None, help="YOLO device override, e.g. 0, 1, cpu")
    parser.add_argument("--point-size", type=float, default=2.0, help="Point size in viewer")
    parser.add_argument("--max-points", type=int, default=750_000, help="Max points after random downsampling")
    parser.add_argument("--ray-radius", type=float, default=0.08, help="Ray hit radius")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    (
        update_interval_ms,
        conf,
        iou,
        device,
        show_window,
        window_name,
        point_cloud_update_interval_ms,
        point_cloud_loop,
    ) = _load_realtime_config(args.realtime_config)

    point_cloud_path = args.point_cloud
    live_point_cloud_provider = None

    if point_cloud_path.is_dir():
        pcd_files = sorted(point_cloud_path.glob("*.pcd"))
        if not pcd_files:
            raise ValueError(f"No .pcd files found in folder: {point_cloud_path}")
        initial_point_cloud_path = pcd_files[0]
        live_point_cloud_provider = _build_point_cloud_folder_provider(
            pcd_files=pcd_files,
            loop=point_cloud_loop,
        )
        print(f"[live-pcd] source folder: {point_cloud_path} ({len(pcd_files)} files)")
        print(f"[live-pcd] frame update interval: {point_cloud_update_interval_ms:.0f} ms")
    else:
        initial_point_cloud_path = point_cloud_path

    resolved_device = _resolve_yolo_device(args.device, device)
    stream = RealtimeYoloVideoBBoxStream(
        video_path=args.video,
        model_path=args.model,
        conf=conf,
        iou=iou,
        device=resolved_device,
        show_window=show_window,
        window_name=window_name,
    )
    stream.start()

    last_seq = -1

    def _live_bbox_provider():
        nonlocal last_seq
        boxes, source_size, seq = stream.get_latest()
        if seq < 0 or seq == last_seq:
            err = stream.get_last_error()
            if err is not None:
                raise RuntimeError(f"YOLO realtime stream failed: {err}")
            return None
        last_seq = seq
        return boxes, source_size

    print(f"[live] update interval: {update_interval_ms:.0f} ms ({args.realtime_config})")
    print(f"[live] video source: {args.video}")
    print(f"[live] yolo device: {resolved_device if resolved_device is not None else 'default(auto)'}")

    try:
        view_point_cloud(
            path=initial_point_cloud_path,
            point_size=args.point_size,
            max_points=args.max_points,
            camera_config=args.camera_config,
            ray_radius=args.ray_radius,
            bounding_boxes=[],
            bbox_image_size=None,
            live_bbox_provider=_live_bbox_provider,
            live_update_interval_ms=update_interval_ms,
            live_point_cloud_provider=live_point_cloud_provider,
            point_cloud_update_interval_ms=point_cloud_update_interval_ms,
        )
    finally:
        stream.stop()


if __name__ == "__main__":
    main()
