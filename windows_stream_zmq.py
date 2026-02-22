#!/usr/bin/env python3
"""Stream Record3D RGB-D frames from Windows to Linux over ZeroMQ."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import zmq
import zstandard as zstd

try:
    import cv2
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("opencv-python is required") from exc

from windows_capture_record3d import Record3DCapture


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish live Record3D RGB-D frames as ZMQ multipart messages.",
    )
    parser.add_argument(
        "--endpoint",
        required=True,
        help="ZMQ endpoint, e.g. tcp://100.101.102.103:5555",
    )
    parser.add_argument(
        "--bind",
        action="store_true",
        help="Bind PUB socket instead of connect (default: connect)",
    )
    parser.add_argument("--topic", default="rgbd", help="ZMQ topic")
    parser.add_argument("--device-index", type=int, default=0, help="Record3D device index")
    parser.add_argument(
        "--source-color-order",
        choices=("RGB", "BGR"),
        default="RGB",
        help="Color order returned by Record3D API",
    )
    parser.add_argument(
        "--depth-units",
        choices=("auto", "meters", "millimeters"),
        default="auto",
        help="Depth units from Record3D API",
    )
    parser.add_argument(
        "--depth-scale-mm",
        type=float,
        default=0.001,
        help="Meter scale for millimeter depth",
    )
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality [1..100]")
    parser.add_argument("--zstd-level", type=int, default=3, help="zstd compression level")
    parser.add_argument("--snd-hwm", type=int, default=2, help="ZMQ send high-water mark")
    parser.add_argument(
        "--max-fps",
        type=float,
        default=0.0,
        help="Optional send FPS cap (0 = no cap)",
    )
    parser.add_argument("--timeout-s", type=float, default=1.0, help="Capture frame timeout")
    parser.add_argument("--preview", action="store_true", help="Show outgoing RGB/depth preview")
    parser.add_argument(
        "--output-width",
        type=int,
        default=0,
        help="Optional output width for both RGB and depth (0 = keep source)",
    )
    parser.add_argument(
        "--output-height",
        type=int,
        default=0,
        help="Optional output height for both RGB and depth (0 = keep source)",
    )
    parser.add_argument(
        "--save-root",
        type=Path,
        default=None,
        help="Optional folder to save per-frame RGB/depth/meta/PCD while streaming",
    )
    parser.add_argument(
        "--save-every-n",
        type=int,
        default=1,
        help="Save every Nth frame (default: 1)",
    )
    parser.add_argument(
        "--save-rgb-jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for saved RGB frames [1..100]",
    )
    parser.add_argument(
        "--save-pcd-stride",
        type=int,
        default=2,
        help="Pixel stride for point cloud generation (1 = full res, default: 2)",
    )
    parser.add_argument(
        "--save-pcd-min-depth",
        type=float,
        default=0.05,
        help="Minimum valid depth (meters) for saved PCD",
    )
    parser.add_argument(
        "--save-pcd-max-depth",
        type=float,
        default=8.0,
        help="Maximum valid depth (meters) for saved PCD",
    )
    return parser


def _to_bgr(frame_rgb: np.ndarray, color_order: str) -> np.ndarray:
    if color_order == "BGR":
        return frame_rgb
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def _depth_preview(depth_m: np.ndarray) -> np.ndarray:
    clipped = np.clip(depth_m, 0.0, 5.0)
    u8 = (clipped / 5.0 * 255.0).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)


def _build_header(
    *,
    seq: int,
    ts: float,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    jpeg_quality: int,
    depth_scale: float,
    depth_uncompressed_bytes: int,
) -> dict[str, Any]:
    return {
        "seq": seq,
        "ts": ts,
        "w": width,
        "h": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "rgb_encoding": "BGR",
        "rgb_codec": "jpeg",
        "jpeg_quality": int(jpeg_quality),
        "depth_encoding": "f32_meters",
        "depth_codec": "zstd",
        "depth_scale": float(depth_scale),
        "depth_uncompressed_bytes": int(depth_uncompressed_bytes),
    }


def _resize_rgbd(
    bgr: np.ndarray,
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    out_w: int,
    out_h: int,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    src_h, src_w = bgr.shape[:2]
    if out_w <= 0 or out_h <= 0 or (out_w == src_w and out_h == src_h):
        return bgr, depth_m, fx, fy, cx, cy

    rgb_resized = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
    depth_resized = cv2.resize(depth_m, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    sx = float(out_w) / float(src_w)
    sy = float(out_h) / float(src_h)
    fx2 = float(fx) * sx
    fy2 = float(fy) * sy
    cx2 = float(cx) * sx
    cy2 = float(cy) * sy
    return (
        np.ascontiguousarray(rgb_resized),
        np.ascontiguousarray(depth_resized.astype(np.float32, copy=False)),
        fx2,
        fy2,
        cx2,
        cy2,
    )


def _depth_to_xyz(
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    stride: int,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    step = max(1, int(stride))
    sampled = depth_m[::step, ::step]

    y_idx, x_idx = np.indices(sampled.shape, dtype=np.float32)
    x_pix = x_idx * float(step)
    y_pix = y_idx * float(step)

    z = sampled
    valid = np.isfinite(z) & (z >= float(min_depth_m)) & (z <= float(max_depth_m))
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    z_v = z[valid].astype(np.float32, copy=False)
    x_v = ((x_pix[valid] - float(cx)) * z_v / float(fx)).astype(np.float32, copy=False)
    y_v = ((y_pix[valid] - float(cy)) * z_v / float(fy)).astype(np.float32, copy=False)
    return np.column_stack((x_v, y_v, z_v)).astype(np.float32, copy=False)


def _write_pcd_xyz_binary(path: Path, points_xyz: np.ndarray) -> None:
    n = int(points_xyz.shape[0])
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        if n > 0:
            np.ascontiguousarray(points_xyz, dtype=np.float32).tofile(f)


def _save_frame_artifacts(
    *,
    save_root: Path,
    seq: int,
    ts: float,
    bgr: np.ndarray,
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rgb_jpeg_quality: int,
    pcd_stride: int,
    pcd_min_depth: float,
    pcd_max_depth: float,
) -> None:
    rgb_dir = save_root / "rgb"
    depth_dir = save_root / "depth"
    pcd_dir = save_root / "pcd"
    meta_dir = save_root / "meta"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    pcd_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{seq:08d}"
    rgb_path = rgb_dir / f"{stem}.jpg"
    depth_path = depth_dir / f"{stem}.npy"
    pcd_path = pcd_dir / f"{stem}.pcd"
    meta_path = meta_dir / f"{stem}.json"

    ok = cv2.imwrite(str(rgb_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(rgb_jpeg_quality, 1, 100))])
    if not ok:
        raise RuntimeError(f"Failed to save RGB image to {rgb_path}")
    np.save(depth_path, depth_m.astype(np.float32, copy=False))

    points_xyz = _depth_to_xyz(
        depth_m=depth_m,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        stride=pcd_stride,
        min_depth_m=pcd_min_depth,
        max_depth_m=pcd_max_depth,
    )
    _write_pcd_xyz_binary(pcd_path, points_xyz)

    meta = {
        "seq": int(seq),
        "ts": float(ts),
        "w": int(bgr.shape[1]),
        "h": int(bgr.shape[0]),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "depth_encoding": "f32_meters",
        "pcd_points": int(points_xyz.shape[0]),
        "pcd_stride": int(max(1, pcd_stride)),
        "pcd_min_depth_m": float(pcd_min_depth),
        "pcd_max_depth_m": float(pcd_max_depth),
        "rgb_path": str(rgb_path.name),
        "depth_path": str(depth_path.name),
        "pcd_path": str(pcd_path.name),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.save_every_n <= 0:
        raise ValueError("--save-every-n must be >= 1")

    topic = args.topic.encode("ascii")
    compressor = zstd.ZstdCompressor(level=args.zstd_level)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(args.jpeg_quality, 1, 100))]

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, max(1, args.snd_hwm))
    sock.setsockopt(zmq.LINGER, 0)
    if args.bind:
        sock.bind(args.endpoint)
        print(f"[stream] PUB bind: {args.endpoint}")
    else:
        sock.connect(args.endpoint)
        print(f"[stream] PUB connect: {args.endpoint}")

    capture = Record3DCapture(
        device_index=args.device_index,
        source_color_order=args.source_color_order,
        depth_units=args.depth_units,
        depth_scale_mm=args.depth_scale_mm,
    )
    capture.start()

    frames = 0
    bytes_sent = 0
    t_log = time.monotonic()
    t_next = t_log
    printed_intrinsics = False
    printed_resize = False
    saved_frames = 0

    save_root: Path | None = None
    if args.save_root is not None:
        save_root = args.save_root.expanduser().resolve()
        save_root.mkdir(parents=True, exist_ok=True)
        print(f"[save] enabled root: {save_root}")
        print(
            f"[save] cadence=1/{args.save_every_n}, pcd_stride={max(1, args.save_pcd_stride)}, "
            f"pcd_depth=[{args.save_pcd_min_depth:.2f},{args.save_pcd_max_depth:.2f}] m"
        )

    try:
        while True:
            if args.max_fps > 0.0:
                now = time.monotonic()
                if now < t_next:
                    time.sleep(max(0.0, t_next - now))
                t_next = max(t_next + (1.0 / args.max_fps), time.monotonic())

            frame = capture.get_frame(timeout_s=args.timeout_s)
            if frame is None:
                continue

            bgr = _to_bgr(frame.rgb, frame.color_order)

            i = frame.intrinsics
            depth_m = np.ascontiguousarray(frame.depth_m.astype(np.float32, copy=False))
            bgr, depth_m, fx, fy, cx, cy = _resize_rgbd(
                bgr=bgr,
                depth_m=depth_m,
                fx=i.fx,
                fy=i.fy,
                cx=i.cx,
                cy=i.cy,
                out_w=args.output_width,
                out_h=args.output_height,
            )
            if not printed_resize and args.output_width > 0 and args.output_height > 0:
                print(
                    f"[stream] resizing enabled: {i.width}x{i.height} -> "
                    f"{args.output_width}x{args.output_height}"
                )
                printed_resize = True

            ok, encoded_rgb = cv2.imencode(".jpg", bgr, encode_params)
            if not ok:
                print("[stream] warning: RGB JPEG encode failed")
                continue

            depth_raw = depth_m.tobytes(order="C")
            depth_zstd = compressor.compress(depth_raw)

            if not printed_intrinsics:
                print(
                    "[stream] intrinsics "
                    f"w={bgr.shape[1]} h={bgr.shape[0]} fx={fx:.3f} fy={fy:.3f} "
                    f"cx={cx:.3f} cy={cy:.3f}"
                )
                printed_intrinsics = True

            header = _build_header(
                seq=frame.seq,
                ts=frame.timestamp,
                width=bgr.shape[1],
                height=bgr.shape[0],
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                jpeg_quality=int(np.clip(args.jpeg_quality, 1, 100)),
                depth_scale=1.0,
                depth_uncompressed_bytes=len(depth_raw),
            )
            header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
            rgb_bytes = encoded_rgb.tobytes()

            sock.send_multipart([topic, header_bytes, rgb_bytes, depth_zstd], copy=False)

            if save_root is not None and (frame.seq % args.save_every_n == 0):
                _save_frame_artifacts(
                    save_root=save_root,
                    seq=frame.seq,
                    ts=frame.timestamp,
                    bgr=bgr,
                    depth_m=depth_m,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    rgb_jpeg_quality=args.save_rgb_jpeg_quality,
                    pcd_stride=args.save_pcd_stride,
                    pcd_min_depth=args.save_pcd_min_depth,
                    pcd_max_depth=args.save_pcd_max_depth,
                )
                saved_frames += 1

            frames += 1
            bytes_sent += len(header_bytes) + len(rgb_bytes) + len(depth_zstd)

            now = time.monotonic()
            if now - t_log >= 1.0:
                elapsed = now - t_log
                mbps = (bytes_sent * 8.0) / (elapsed * 1_000_000.0)
                fps = frames / elapsed
                print(
                    f"[stream] seq={frame.seq} fps={fps:.1f} "
                    f"tx={mbps:.2f} Mbps rgb={len(rgb_bytes)}B depth={len(depth_zstd)}B "
                    f"saved={saved_frames}"
                )
                frames = 0
                bytes_sent = 0
                t_log = now

            if args.preview:
                cv2.imshow("ZMQ Stream RGB", bgr)
                cv2.imshow("ZMQ Stream Depth (m)", _depth_preview(depth_m))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        if args.preview:
            cv2.destroyAllWindows()
        sock.close(0)
        ctx.term()
        print("[stream] stopped")


if __name__ == "__main__":
    main()
