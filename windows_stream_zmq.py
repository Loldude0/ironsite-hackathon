#!/usr/bin/env python3
"""Stream Record3D RGB-D frames from Windows to Linux over ZeroMQ."""

from __future__ import annotations

import argparse
import json
import time
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


def main() -> None:
    args = _build_arg_parser().parse_args()

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
            ok, encoded_rgb = cv2.imencode(".jpg", bgr, encode_params)
            if not ok:
                print("[stream] warning: RGB JPEG encode failed")
                continue

            depth_m = np.ascontiguousarray(frame.depth_m.astype(np.float32, copy=False))
            depth_raw = depth_m.tobytes(order="C")
            depth_zstd = compressor.compress(depth_raw)

            i = frame.intrinsics
            if not printed_intrinsics:
                print(
                    "[stream] intrinsics "
                    f"w={i.width} h={i.height} fx={i.fx:.3f} fy={i.fy:.3f} "
                    f"cx={i.cx:.3f} cy={i.cy:.3f}"
                )
                printed_intrinsics = True

            header = _build_header(
                seq=frame.seq,
                ts=frame.timestamp,
                width=i.width,
                height=i.height,
                fx=i.fx,
                fy=i.fy,
                cx=i.cx,
                cy=i.cy,
                jpeg_quality=int(np.clip(args.jpeg_quality, 1, 100)),
                depth_scale=1.0,
                depth_uncompressed_bytes=len(depth_raw),
            )
            header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
            rgb_bytes = encoded_rgb.tobytes()

            sock.send_multipart([topic, header_bytes, rgb_bytes, depth_zstd], copy=False)

            frames += 1
            bytes_sent += len(header_bytes) + len(rgb_bytes) + len(depth_zstd)

            now = time.monotonic()
            if now - t_log >= 1.0:
                elapsed = now - t_log
                mbps = (bytes_sent * 8.0) / (elapsed * 1_000_000.0)
                fps = frames / elapsed
                print(
                    f"[stream] seq={frame.seq} fps={fps:.1f} "
                    f"tx={mbps:.2f} Mbps rgb={len(rgb_bytes)}B depth={len(depth_zstd)}B"
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
