#!/usr/bin/env python3
"""Entry script: run YOLO on an image, then launch the point-cloud viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pointcloud_locator.viewer import view_point_cloud
from yolo import RealtimeYoloVideoBBoxStream


def _load_realtime_config(config_path: Path) -> tuple[float, float | None, float | None, str | None]:
    """Load realtime YOLO/viewer configuration.

    Returns
    -------
    update_interval_ms, conf, iou, device
    """
    if not config_path.exists():
        return 200.0, None, None, None

    payload: dict[str, Any] = json.loads(config_path.read_text())
    update_interval_ms = float(payload.get("update_interval_ms", 200.0))
    yolo_cfg = payload.get("yolo", {}) if isinstance(payload.get("yolo", {}), dict) else {}
    conf = yolo_cfg.get("conf")
    iou = yolo_cfg.get("iou")
    device = yolo_cfg.get("device")

    conf_v = float(conf) if conf is not None else None
    iou_v = float(iou) if iou is not None else None
    device_v = str(device) if device is not None else None
    return update_interval_ms, conf_v, iou_v, device_v


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLO on an MP4 stream and open viewer with realtime bounding-box updates.",
    )
    parser.add_argument("point_cloud", type=Path, help="Path to point cloud (.pcd/.ply/.xyz/.txt/.csv/.npy)")
    parser.add_argument("--video", type=Path, required=True, help="Path to input MP4 for realtime YOLO")
    parser.add_argument("--camera-config", type=Path, default=Path("assets/sample_camera_config.json"), help="Path to camera config JSON")
    parser.add_argument("--model", type=Path, default=Path("yolo26n.pt"), help="YOLO model weights path")
    parser.add_argument("--realtime-config", type=Path, default=Path("assets/realtime_yolo_config.json"), help="Realtime update config JSON")
    parser.add_argument("--point-size", type=float, default=2.0, help="Point size in viewer")
    parser.add_argument("--max-points", type=int, default=750_000, help="Max points after random downsampling")
    parser.add_argument("--ray-radius", type=float, default=0.08, help="Ray hit radius")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    update_interval_ms, conf, iou, device = _load_realtime_config(args.realtime_config)

    stream = RealtimeYoloVideoBBoxStream(
        video_path=args.video,
        model_path=args.model,
        conf=conf,
        iou=iou,
        device=device,
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

    try:
        view_point_cloud(
            path=args.point_cloud,
            point_size=args.point_size,
            max_points=args.max_points,
            camera_config=args.camera_config,
            ray_radius=args.ray_radius,
            bounding_boxes=[],
            bbox_image_size=None,
            live_bbox_provider=_live_bbox_provider,
            live_update_interval_ms=update_interval_ms,
        )
    finally:
        stream.stop()


if __name__ == "__main__":
    main()
