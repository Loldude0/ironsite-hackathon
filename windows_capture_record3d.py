#!/usr/bin/env python3
"""Capture aligned RGB-D frames from Record3D on Windows.

This module is intentionally self-contained so it can be copied with the repo
to a Windows machine and used by `windows_stream_zmq.py`.
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional for headless capture
    cv2 = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class RGBDFrame:
    seq: int
    timestamp: float
    rgb: np.ndarray  # HxWx3 uint8
    depth_m: np.ndarray  # HxW float32 meters
    intrinsics: Intrinsics
    color_order: str  # "RGB" or "BGR"


def _call_first(obj: Any, names: Iterable[str]) -> Any:
    for name in names:
        attr = getattr(obj, name, None)
        if attr is None:
            continue
        if callable(attr):
            try:
                return attr()
            except TypeError:
                # Some pybind methods require a simple mode flag. Try common values.
                for args in ((0,), (False,), (1,), (True,)):
                    try:
                        return attr(*args)
                    except TypeError:
                        continue
                continue
        return attr
    return None


def _to_monotonic_seconds(value: float | int | None) -> float:
    if value is None:
        return time.monotonic()
    ts = float(value)
    # Heuristic normalization for epoch-like ns/us/ms timestamps.
    if ts > 1e17:
        return ts * 1e-9
    if ts > 1e14:
        return ts * 1e-6
    if ts > 1e11:
        return ts * 1e-3
    return ts


class Record3DCapture:
    """Live capture wrapper around the Record3D Python API."""

    DEVICE_TYPE_TRUEDEPTH = 0
    DEVICE_TYPE_LIDAR = 1

    def __init__(
        self,
        device_index: int = 0,
        source_color_order: str = "RGB",
        depth_units: str = "auto",
        depth_scale_mm: float = 0.001,
    ) -> None:
        self.device_index = int(device_index)
        self.source_color_order = source_color_order.upper()
        if self.source_color_order not in {"RGB", "BGR"}:
            raise ValueError("source_color_order must be RGB or BGR")
        self.depth_units = depth_units.lower()
        if self.depth_units not in {"auto", "meters", "millimeters"}:
            raise ValueError("depth_units must be auto|meters|millimeters")
        self.depth_scale_mm = float(depth_scale_mm)

        self._stream: Any | None = None
        self._new_frame_event = threading.Event()
        self._started = False
        self._seq = 0
        self._device_type: int | None = None
        self._use_poll_fallback = False
        self._fallback_notice_printed = False
        self._poll_interval_s = 1.0 / 60.0
        self._cached_intrinsics: Intrinsics | None = None
        self._intrinsics_warning_printed = False

    @staticmethod
    def _load_record3d() -> tuple[Any, Any, list[Any]]:
        try:
            import record3d as r3d  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on host
            raise RuntimeError(
                "Record3D Python module is not installed. "
                "Install it on Windows from the Record3D SDK instructions."
            ) from exc

        stream_cls = getattr(r3d, "Record3DStream", None)
        if stream_cls is None:
            raise RuntimeError(
                "record3d.Record3DStream was not found in the installed package. "
                "Check the package version against the Record3D demo-main.py API."
            )

        get_devices = getattr(stream_cls, "get_connected_devices", None)
        if callable(get_devices):
            devices = list(get_devices())
        else:
            module_get_devices = getattr(r3d, "get_connected_devices", None)
            if not callable(module_get_devices):
                raise RuntimeError(
                    "Record3D device enumeration API not found. "
                    "Expected Record3DStream.get_connected_devices() or record3d.get_connected_devices()."
                )
            devices = list(module_get_devices())
        return r3d, stream_cls, devices

    @staticmethod
    def list_devices() -> list[Any]:
        _, _, devices = Record3DCapture._load_record3d()
        return devices

    def _connect_stream(self, stream: Any, device: Any) -> None:
        connect_fn = getattr(stream, "connect", None)
        if not callable(connect_fn):
            raise RuntimeError("Record3D stream object has no connect() method")

        attempts = [device]
        for attr_name in ("product_id", "udid", "serial"):
            value = getattr(device, attr_name, None)
            if value is not None:
                attempts.append(value)

        last_error: Exception | None = None
        for target in attempts:
            try:
                connect_fn(target)
                return
            except Exception as exc:  # pragma: no cover - SDK-dependent
                last_error = exc
        raise RuntimeError(f"Unable to connect to Record3D device: {last_error}")

    def start(self) -> None:
        if self._started:
            return
        _, stream_cls, devices = self._load_record3d()
        if not devices:
            raise RuntimeError(
                "No Record3D devices detected over USB.\n"
                "Troubleshooting:\n"
                "1) Install iTunes on Windows (Record3D prerequisite).\n"
                "2) Ensure 'Apple Mobile Device Service' is running.\n"
                "3) Unlock iPhone, accept 'Trust This Computer'.\n"
                "4) Open Record3D app on iPhone and enable USB Streaming mode.\n"
                "5) Use a direct USB cable (avoid hubs/adapters during setup)."
            )
        if self.device_index < 0 or self.device_index >= len(devices):
            raise RuntimeError(
                f"device_index {self.device_index} out of range (found {len(devices)} devices)"
            )
        device = devices[self.device_index]
        stream = stream_cls()

        def _on_new_frame(*_args: Any, **_kwargs: Any) -> None:
            self._new_frame_event.set()

        def _on_stream_stopped(*_args: Any, **_kwargs: Any) -> None:
            print("[record3d] stream stopped")

        if hasattr(stream, "on_new_frame"):
            stream.on_new_frame = _on_new_frame
        if hasattr(stream, "on_stream_stopped"):
            stream.on_stream_stopped = _on_stream_stopped

        self._connect_stream(stream, device)
        start_fn = getattr(stream, "start", None)
        if callable(start_fn):
            start_fn()
        device_type = _call_first(stream, ("get_device_type", "device_type"))
        if device_type is not None:
            try:
                self._device_type = int(device_type)
            except Exception:
                self._device_type = None
        self._stream = stream
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        assert self._stream is not None
        stop_fn = getattr(self._stream, "stop", None)
        if callable(stop_fn):
            try:
                stop_fn()
            except Exception:
                pass
        self._stream = None
        self._started = False
        self._device_type = None
        self._use_poll_fallback = False
        self._fallback_notice_printed = False
        self._cached_intrinsics = None
        self._intrinsics_warning_printed = False
        self._new_frame_event.clear()

    @staticmethod
    def _as_float_maybe(value: Any) -> float | None:
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _read_numeric_field(obj: Any, *names: str) -> float | None:
        for name in names:
            attr = getattr(obj, name, None)
            if attr is None:
                continue
            if callable(attr):
                try:
                    value = attr()
                except TypeError:
                    continue
            else:
                value = attr
            parsed = Record3DCapture._as_float_maybe(value)
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _intrinsics_from_coeffs(coeffs: Any, width: int, height: int) -> Intrinsics | None:
        if coeffs is None:
            return None

        fx = fy = cx = cy = None
        if isinstance(coeffs, dict):
            fx = Record3DCapture._as_float_maybe(coeffs.get("fx"))
            fy = Record3DCapture._as_float_maybe(coeffs.get("fy"))
            cx = Record3DCapture._as_float_maybe(coeffs.get("tx", coeffs.get("cx")))
            cy = Record3DCapture._as_float_maybe(coeffs.get("ty", coeffs.get("cy")))
        elif isinstance(coeffs, (tuple, list)) and len(coeffs) >= 4:
            fx = Record3DCapture._as_float_maybe(coeffs[0])
            fy = Record3DCapture._as_float_maybe(coeffs[1])
            cx = Record3DCapture._as_float_maybe(coeffs[2])
            cy = Record3DCapture._as_float_maybe(coeffs[3])
        else:
            fx = Record3DCapture._read_numeric_field(coeffs, "fx")
            fy = Record3DCapture._read_numeric_field(coeffs, "fy")
            cx = Record3DCapture._read_numeric_field(coeffs, "tx", "cx")
            cy = Record3DCapture._read_numeric_field(coeffs, "ty", "cy")

        if fx is None or fy is None or cx is None or cy is None:
            return None
        if fx <= 0.0 or fy <= 0.0:
            return None
        return Intrinsics(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)

    @staticmethod
    def _intrinsics_from_matrix(matrix: Any, width: int, height: int) -> Intrinsics | None:
        if matrix is None:
            return None
        try:
            mat = np.asarray(matrix, dtype=np.float64)
        except Exception:
            return None
        if mat.shape == (9,):
            mat = mat.reshape(3, 3)
        if mat.shape != (3, 3):
            return None
        fx = float(mat[0, 0])
        fy = float(mat[1, 1])
        cx = float(mat[0, 2])
        cy = float(mat[1, 2])
        if fx <= 0.0 or fy <= 0.0:
            return None
        return Intrinsics(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)

    def _extract_intrinsics(self, frame: Any, width: int, height: int) -> Intrinsics | None:
        candidates: list[Any] = []
        candidate_sources: list[str] = []
        method_names = (
            "get_intrinsic_mat",
            "get_intrinsics_mat",
            "get_intrinsics",
            "get_intrinsic_coeffs",
            "get_camera_intrinsics",
        )
        if frame is not None:
            for name in method_names:
                value = _call_first(frame, (name,))
                if value is not None:
                    candidates.append(value)
                    candidate_sources.append(f"frame.{name}")
            for name in ("intrinsic_mat", "intrinsics"):
                value = getattr(frame, name, None)
                if value is not None:
                    candidates.append(value)
                    candidate_sources.append(f"frame.{name}")
        if self._stream is not None:
            for name in method_names:
                value = _call_first(self._stream, (name,))
                if value is not None:
                    candidates.append(value)
                    candidate_sources.append(f"stream.{name}")
            for name in ("intrinsic_mat", "intrinsics"):
                value = getattr(self._stream, name, None)
                if value is not None:
                    candidates.append(value)
                    candidate_sources.append(f"stream.{name}")

        for idx, candidate in enumerate(candidates):
            intr = self._intrinsics_from_matrix(candidate, width=width, height=height)
            if intr is not None:
                self._cached_intrinsics = intr
                return intr
            intr = self._intrinsics_from_coeffs(candidate, width=width, height=height)
            if intr is not None:
                self._cached_intrinsics = intr
                return intr

        if self._cached_intrinsics is not None:
            return self._cached_intrinsics

        if not self._intrinsics_warning_printed:
            desc: list[str] = []
            for idx, candidate in enumerate(candidates):
                source = candidate_sources[idx] if idx < len(candidate_sources) else f"candidate[{idx}]"
                desc.append(f"{source}:{type(candidate).__name__}")
            stream_type = type(self._stream).__name__ if self._stream is not None else "None"
            print(
                "[capture] intrinsics unavailable yet; waiting for metadata. "
                f"stream_type={stream_type} candidates={desc}"
            )
            self._intrinsics_warning_printed = True
        return None

    def _extract_rgb(self, frame: Any) -> np.ndarray:
        rgb_raw = None
        if frame is not None:
            rgb_raw = _call_first(frame, ("get_rgb_frame", "get_color_frame", "get_image"))
        if rgb_raw is None and self._stream is not None:
            rgb_raw = _call_first(self._stream, ("get_rgb_frame", "get_color_frame", "get_image"))
        if rgb_raw is None:
            raise RuntimeError("Record3D frame did not provide RGB data")
        rgb = np.asarray(rgb_raw)
        if rgb.ndim == 2:
            rgb = np.stack((rgb, rgb, rgb), axis=-1)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise RuntimeError(f"Unexpected RGB shape: {rgb.shape}")
        if rgb.shape[2] > 3:
            rgb = rgb[:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(rgb)

    def _extract_depth(self, frame: Any) -> np.ndarray:
        depth_raw = None
        if frame is not None:
            depth_raw = _call_first(frame, ("get_depth_frame", "get_depth_map", "get_depth"))
        if depth_raw is None and self._stream is not None:
            depth_raw = _call_first(self._stream, ("get_depth_frame", "get_depth_map", "get_depth"))
        if depth_raw is None:
            raise RuntimeError("Record3D frame did not provide depth data")
        depth = np.asarray(depth_raw)
        if depth.ndim == 3 and depth.shape[2] == 1:
            depth = depth[:, :, 0]
        if depth.ndim != 2:
            raise RuntimeError(f"Unexpected depth shape: {depth.shape}")

        if self.depth_units == "millimeters":
            depth_m = depth.astype(np.float32) * self.depth_scale_mm
        elif self.depth_units == "meters":
            depth_m = depth.astype(np.float32)
        else:
            if np.issubdtype(depth.dtype, np.integer):
                depth_m = depth.astype(np.float32) * self.depth_scale_mm
            else:
                depth_m = depth.astype(np.float32)
                finite = depth_m[np.isfinite(depth_m)]
                if finite.size > 0 and float(np.median(finite)) > 20.0:
                    # Depth values look like millimeters despite float dtype.
                    depth_m = depth_m * self.depth_scale_mm

        depth_m = np.nan_to_num(depth_m, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        depth_m[depth_m < 0.0] = 0.0
        return np.ascontiguousarray(depth_m)

    def _extract_timestamp(self, frame: Any) -> float:
        ts_raw = None
        if frame is not None:
            ts_raw = _call_first(
                frame,
                ("get_timestamp", "get_frame_timestamp", "timestamp", "time", "ts"),
            )
        if ts_raw is None and self._stream is not None:
            ts_raw = _call_first(
                self._stream,
                ("get_timestamp", "get_frame_timestamp", "timestamp", "time", "ts"),
            )
        return _to_monotonic_seconds(ts_raw)

    def _apply_device_specific_transforms(self, rgb: np.ndarray, depth_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Per Record3D demo-main.py, TrueDepth frames must be mirrored.
        if self._device_type == self.DEVICE_TYPE_TRUEDEPTH:
            rgb = np.ascontiguousarray(np.flip(rgb, axis=1))
            depth_m = np.ascontiguousarray(np.flip(depth_m, axis=1))
        return rgb, depth_m

    def get_frame(self, timeout_s: float = 1.0) -> RGBDFrame | None:
        if not self._started:
            raise RuntimeError("Capture stream is not started")
        if not self._use_poll_fallback:
            if not self._new_frame_event.wait(timeout=timeout_s):
                self._use_poll_fallback = True
                if not self._fallback_notice_printed:
                    print(
                        "[capture] on_new_frame callback timed out; "
                        "switching to direct polling fallback"
                    )
                    self._fallback_notice_printed = True
            else:
                self._new_frame_event.clear()
        else:
            # Poll gently to avoid spinning while waiting for the next frame.
            time.sleep(self._poll_interval_s)

        assert self._stream is not None
        frame = _call_first(self._stream, ("get_current_frame", "get_frame"))

        try:
            rgb = self._extract_rgb(frame)
            depth_m = self._extract_depth(frame)
        except RuntimeError:
            return None
        rgb, depth_m = self._apply_device_specific_transforms(rgb, depth_m)
        if depth_m.shape[:2] != rgb.shape[:2]:
            raise RuntimeError(
                f"Depth/RGB shape mismatch. RGB={rgb.shape[:2]} depth={depth_m.shape[:2]}"
            )
        intrinsics = self._extract_intrinsics(frame, width=rgb.shape[1], height=rgb.shape[0])
        if intrinsics is None:
            return None
        ts = self._extract_timestamp(frame)
        seq = self._seq
        self._seq += 1

        return RGBDFrame(
            seq=seq,
            timestamp=ts,
            rgb=rgb,
            depth_m=depth_m,
            intrinsics=intrinsics,
            color_order=self.source_color_order,
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate live Record3D USB RGB-D capture on Windows.",
    )
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
        help="Unit used by depth data in Record3D API",
    )
    parser.add_argument(
        "--depth-scale-mm",
        type=float,
        default=0.001,
        help="Meter scale for millimeter depth (usually 0.001)",
    )
    parser.add_argument("--timeout-s", type=float, default=1.0, help="Frame wait timeout")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = infinite)")
    parser.add_argument("--preview", action="store_true", help="Open RGB/depth preview windows")
    return parser


def _depth_preview(depth_m: np.ndarray) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for preview mode")
    clipped = np.clip(depth_m, 0.0, 5.0)
    u8 = (clipped / 5.0 * 255.0).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)


def main() -> None:
    args = _build_arg_parser().parse_args()
    capture = Record3DCapture(
        device_index=args.device_index,
        source_color_order=args.source_color_order,
        depth_units=args.depth_units,
        depth_scale_mm=args.depth_scale_mm,
    )
    capture.start()
    print("[capture] started")

    frames = 0
    last_log = time.monotonic()
    first_intrinsics_printed = False

    try:
        while True:
            frame = capture.get_frame(timeout_s=args.timeout_s)
            if frame is None:
                print("[capture] timeout waiting for frame")
                continue

            frames += 1
            now = time.monotonic()
            if not first_intrinsics_printed:
                i = frame.intrinsics
                print(
                    "[capture] intrinsics "
                    f"w={i.width} h={i.height} fx={i.fx:.3f} fy={i.fy:.3f} "
                    f"cx={i.cx:.3f} cy={i.cy:.3f}"
                )
                first_intrinsics_printed = True

            if now - last_log >= 1.0:
                depth_valid = float(np.count_nonzero(frame.depth_m > 0.0)) / float(frame.depth_m.size)
                print(
                    f"[capture] seq={frame.seq} fps~{frames/(now-last_log):.1f} "
                    f"depth_valid={depth_valid*100.0:.1f}%"
                )
                frames = 0
                last_log = now

            if args.preview:
                if cv2 is None:
                    raise RuntimeError("opencv-python is required for --preview")
                rgb = frame.rgb
                if frame.color_order == "RGB":
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow("Record3D RGB", rgb)
                cv2.imshow("Record3D Depth (m)", _depth_preview(frame.depth_m))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            if args.max_frames > 0 and frame.seq + 1 >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        if cv2 is not None:
            cv2.destroyAllWindows()
        print("[capture] stopped")


if __name__ == "__main__":
    main()
