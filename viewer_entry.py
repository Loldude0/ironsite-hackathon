#!/usr/bin/env python3
"""Entry script: run YOLO on an image, then launch the point-cloud viewer."""

from __future__ import annotations

import argparse
from pathlib import Path

from pointcloud_locator.viewer import view_point_cloud
from yolo import predict_bboxes_for_viewer_with_image_size


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLO on an image and open the point cloud viewer with detected boxes.",
    )
    parser.add_argument("point_cloud", type=Path, help="Path to point cloud (.pcd/.ply/.xyz/.txt/.csv/.npy)")
    parser.add_argument("--image", type=Path, required=True, help="Path to input image for YOLO detection")
    parser.add_argument("--camera-config", type=Path, default=Path("assets/sample_camera_config.json"), help="Path to camera config JSON")
    parser.add_argument("--model", type=Path, default=Path("yolo26n.pt"), help="YOLO model weights path")
    parser.add_argument("--point-size", type=float, default=2.0, help="Point size in viewer")
    parser.add_argument("--max-points", type=int, default=750_000, help="Max points after random downsampling")
    parser.add_argument("--ray-radius", type=float, default=0.08, help="Ray hit radius")
    parser.add_argument("--conf", type=float, default=None, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=None, help="YOLO IoU threshold")
    parser.add_argument("--device", type=str, default=None, help="YOLO device (e.g. cpu or 0)")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    bboxes, source_size = predict_bboxes_for_viewer_with_image_size(
        image_path=args.image,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )
    if source_size is not None:
        print(f"[yolo] detections converted for viewer: {len(bboxes)} (source image size={source_size[0]}x{source_size[1]})")
    else:
        print(f"[yolo] detections converted for viewer: {len(bboxes)}")

    view_point_cloud(
        path=args.point_cloud,
        point_size=args.point_size,
        max_points=args.max_points,
        camera_config=args.camera_config,
        ray_radius=args.ray_radius,
        bounding_boxes=bboxes,
        bbox_image_size=source_size,
    )


if __name__ == "__main__":
    main()
