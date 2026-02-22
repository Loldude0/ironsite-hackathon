#!/usr/bin/env python3
"""Entry script: run YOLO on an image, then launch the point-cloud viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from pointcloud_locator.viewer import view_point_cloud
from pointcloud_locator import yolo_output_to_viewer_bboxes
from pointcloud_locator.point_cloud import load_point_cloud
from yolo import RealtimeYoloVideoBBoxStream, RealtimeYoloImageFolderBBoxStream


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


def _list_image_files(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def _build_synced_folder_providers(
    pcd_files: list[Path],
    image_files: list[Path],
    *,
    model_path: Path,
    conf: float | None,
    iou: float | None,
    device: str | None,
    loop: bool,
    show_window: bool,
    window_name: str,
):
    model = YOLO(str(model_path))

    cv2 = None
    if show_window:
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except Exception as exc:
            raise RuntimeError(f"OpenCV is required for YOLO display window: {exc}") from exc

    pair_count = min(len(pcd_files), len(image_files))
    if pair_count <= 0:
        raise ValueError("Need at least one point cloud and one image file for synchronized mode")

    state: dict[str, Any] = {
        "next_idx": 1,  # index 0 is used for initial frame
        "latest_payload": None,  # tuple[list[BoundingBox], tuple[int, int] | None]
        "latest_seq": -1,
        "last_bbox_seq": -1,
        "stopped": False,
    }

    def _infer_image(image_path: Path):
        predict_kwargs: dict[str, Any] = {
            "source": str(image_path),
            "verbose": False,
        }
        if conf is not None:
            predict_kwargs["conf"] = conf
        if iou is not None:
            predict_kwargs["iou"] = iou
        if device is not None:
            predict_kwargs["device"] = device

        results = model.predict(**predict_kwargs)
        if not results:
                return [], None, None

        result = results[0]
        boxes = yolo_output_to_viewer_bboxes(result)
        source_size = None
        if hasattr(result, "orig_shape"):
            h, w = result.orig_shape[:2]
            source_size = (int(w), int(h))
            annotated = result.plot() if cv2 is not None else None
            return boxes, source_size, annotated

    def _advance_index(idx: int) -> int | None:
        if idx >= pair_count:
            if not loop:
                return None
            return 0
        return idx

    def _pc_provider():
        if bool(state["stopped"]):
            return None

        idx = _advance_index(int(state["next_idx"]))
        if idx is None:
            if cv2 is not None:
                try:
                    cv2.destroyWindow(window_name)
                except Exception:
                    pass
            return None

        points = load_point_cloud(pcd_files[idx])
        boxes, source_size, annotated = _infer_image(image_files[idx])

        if cv2 is not None:
            if annotated is not None:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    state["stopped"] = True
                    try:
                        cv2.destroyWindow(window_name)
                    except Exception:
                        pass
                    return None

        state["latest_payload"] = (boxes, source_size)
        state["latest_seq"] = int(state["latest_seq"]) + 1
        state["next_idx"] = idx + 1
        return points

    def _bbox_provider():
        latest_seq = int(state["latest_seq"])
        if latest_seq < 0 or latest_seq == int(state["last_bbox_seq"]):
            return None
        state["last_bbox_seq"] = latest_seq
        return state["latest_payload"]

    return _pc_provider, _bbox_provider, pair_count


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
    video_path = args.video
    live_point_cloud_provider = None
    live_bbox_provider = None
    stream = None
    bbox_update_interval_ms = update_interval_ms

    if point_cloud_path.is_dir():
        pcd_files = sorted(point_cloud_path.glob("*.pcd"))
        if not pcd_files:
            raise ValueError(f"No .pcd files found in folder: {point_cloud_path}")
        if video_path.is_dir():
            image_files = _list_image_files(video_path)
            if not image_files:
                raise ValueError(f"No image files found in folder: {video_path}")

            pair_count = min(len(pcd_files), len(image_files))
            if pair_count <= 0:
                raise ValueError("No usable synchronized point-cloud/image pairs found")

            initial_point_cloud_path = pcd_files[0]
            live_point_cloud_provider, live_bbox_provider, _pair_count = _build_synced_folder_providers(
                pcd_files=pcd_files,
                image_files=image_files,
                model_path=args.model,
                conf=conf,
                iou=iou,
                device=_resolve_yolo_device(args.device, device),
                loop=point_cloud_loop,
                show_window=show_window,
                window_name=window_name,
            )
            # Keep both updates on the same cadence in synchronized folder mode.
            bbox_update_interval_ms = point_cloud_update_interval_ms

            print(f"[live-sync] point-cloud folder: {point_cloud_path} ({len(pcd_files)} files)")
            print(f"[live-sync] image folder: {video_path} ({len(image_files)} files)")
            print(f"[live-sync] using {pair_count} synchronized pairs")
            print(f"[live-sync] synchronized frame interval: {bbox_update_interval_ms:.0f} ms")
        else:
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
    if live_bbox_provider is None:
        if args.video.is_dir():
            stream = RealtimeYoloImageFolderBBoxStream(
                image_dir=args.video,
                model_path=args.model,
                update_interval_ms=update_interval_ms,
                loop=True,
                conf=conf,
                iou=iou,
                device=resolved_device,
                show_window=show_window,
                window_name=window_name,
            )
            print(f"[live] image-folder source: {args.video}")
            print(f"[live] image frame interval: {update_interval_ms:.0f} ms")
        else:
            stream = RealtimeYoloVideoBBoxStream(
                video_path=args.video,
                model_path=args.model,
                conf=conf,
                iou=iou,
                device=resolved_device,
                show_window=show_window,
                window_name=window_name,
            )
            print(f"[live] video source: {args.video}")

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

        live_bbox_provider = _live_bbox_provider

    print(f"[live] update interval: {bbox_update_interval_ms:.0f} ms ({args.realtime_config})")
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
            live_bbox_provider=live_bbox_provider,
            live_update_interval_ms=bbox_update_interval_ms,
            live_point_cloud_provider=live_point_cloud_provider,
            point_cloud_update_interval_ms=point_cloud_update_interval_ms,
        )
    finally:
        if stream is not None:
            stream.stop()


if __name__ == "__main__":
    main()
