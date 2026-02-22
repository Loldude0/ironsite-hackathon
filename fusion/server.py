#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from .schemas import CalibrationStart, SlamPose, UwbSample, parse_calibration_start, parse_slam_pose, parse_uwb_sample
from .solver import (
    SolverConfig,
    apply_transform_to_pose,
    infer_face_anchor_yaw,
    run_calibration,
    solve_initial_xyzyaw_transform,
    solver_config_from_dict,
)

try:
    import yaml
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("pyyaml is required for fusion/server.py") from exc


def now_ms() -> int:
    return int(time.time() * 1000.0)


@dataclass(slots=True)
class SessionState:
    session_id: str
    anchor_id: str
    worker_id: str
    calibration_status: str = "idle"  # idle|calibrating|calibrated|failed
    calibration_mode: str = "paired_6dof"
    yaw_mode: str = "face_anchor"
    yaw_offset_rad: float = 0.0
    calibration_start_rx_ms: int | None = None
    calibration_duration_ms: int = 5000
    calibration_result: dict[str, Any] | None = None
    T_G_Lw: np.ndarray | None = None
    bootstrap_global_position_xyz_m: tuple[float, float, float] | None = None
    bootstrap_global_yaw_rad: float | None = None
    bootstrap_uwb_ts_ms: int | None = None
    uwb_samples: list[UwbSample] = field(default_factory=list)
    slam_poses: list[SlamPose] = field(default_factory=list)
    output_dir: Path | None = None
    trajectory_global_path: Path | None = None

    def calibration_end_rx_ms(self) -> int | None:
        if self.calibration_start_rx_ms is None:
            return None
        return self.calibration_start_rx_ms + self.calibration_duration_ms


class FusionRuntime:
    def __init__(self, config: dict[str, Any]) -> None:
        self._lock = threading.Lock()
        self.sessions: dict[str, SessionState] = {}

        self.output_root = Path(str(config.get("output_root", "outputs/fusion"))).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.solver_config: SolverConfig = solver_config_from_dict(config.get("solver"))
        self.min_initial_xy_distance_m = float(config.get("min_initial_xy_distance_m", 0.15))
        self.global_pose_udp_endpoint = str(config.get("global_pose_udp_endpoint", "")).strip()
        self._global_pose_sock: socket.socket | None = None
        self._global_pose_addr: tuple[str, int] | None = None
        if self.global_pose_udp_endpoint:
            host, port = self._parse_host_port(self.global_pose_udp_endpoint)
            self._global_pose_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._global_pose_addr = (host, port)

    @staticmethod
    def _parse_host_port(value: str) -> tuple[str, int]:
        if ":" not in value:
            raise ValueError(f"Expected host:port, got {value}")
        host, port_str = value.rsplit(":", 1)
        return host.strip(), int(port_str)

    def _get_or_create_session(self, session_id: str, anchor_id: str, worker_id: str) -> SessionState:
        session = self.sessions.get(session_id)
        if session is None:
            output_dir = self.output_root / session_id
            output_dir.mkdir(parents=True, exist_ok=True)
            session = SessionState(
                session_id=session_id,
                anchor_id=anchor_id,
                worker_id=worker_id,
                output_dir=output_dir,
                trajectory_global_path=output_dir / "trajectory_global_tum.txt",
            )
            self.sessions[session_id] = session
        return session

    def start_calibration(self, request: CalibrationStart) -> dict[str, Any]:
        with self._lock:
            session = self._get_or_create_session(request.session_id, request.anchor_id, request.worker_id)
            session.anchor_id = request.anchor_id
            session.worker_id = request.worker_id
            session.calibration_mode = request.calibration_mode
            session.yaw_mode = request.yaw_mode
            session.yaw_offset_rad = request.yaw_offset_rad
            session.calibration_status = "calibrating"
            session.calibration_start_rx_ms = now_ms()
            session.calibration_duration_ms = int(request.duration_s * 1000.0)
            session.calibration_result = None
            session.T_G_Lw = None
            session.bootstrap_global_position_xyz_m = None
            session.bootstrap_global_yaw_rad = None
            session.bootstrap_uwb_ts_ms = None
            session.uwb_samples.clear()
            session.slam_poses.clear()
            if session.trajectory_global_path and session.trajectory_global_path.exists():
                session.trajectory_global_path.unlink()
            return {
                "status": "ok",
                "session_id": session.session_id,
                "calibration_status": session.calibration_status,
                "calibration_mode": session.calibration_mode,
                "window_seconds": request.duration_s,
            }

    def ingest_uwb_sample(self, sample: UwbSample) -> dict[str, Any]:
        with self._lock:
            session = self._get_or_create_session(sample.session_id, sample.anchor_id, sample.worker_id)
            session.uwb_samples.append(sample)
            self._maybe_finalize_calibration(session)
            return {"status": "ok"}

    def ingest_slam_pose(self, pose: SlamPose) -> None:
        with self._lock:
            if pose.tracking_state is not None and pose.tracking_state != 2:
                return
            session = self._get_or_create_session(pose.session_id, anchor_id="unknown_anchor", worker_id=pose.node_id)
            session.slam_poses.append(pose)
            self._maybe_finalize_calibration(session)
            if session.calibration_status == "waiting_for_first_slam_pose":
                self._finalize_initial_xyzyaw_with_first_pose(session, pose)
            if session.calibration_status == "calibrated" and session.T_G_Lw is not None:
                self._write_global_pose(session, pose)

    def _maybe_finalize_calibration(self, session: SessionState) -> None:
        if session.calibration_status != "calibrating":
            return
        end_rx_ms = session.calibration_end_rx_ms()
        if end_rx_ms is None or now_ms() < end_rx_ms:
            return

        start_rx_ms = session.calibration_start_rx_ms or (end_rx_ms - session.calibration_duration_ms)
        uwb_window = [s for s in session.uwb_samples if start_rx_ms <= s.server_rx_ms <= end_rx_ms]
        slam_window = [s for s in session.slam_poses if start_rx_ms <= s.server_rx_ms <= end_rx_ms]
        if session.calibration_mode == "initial_xyzyaw":
            self._finalize_initial_xyzyaw_bootstrap(session, uwb_window)
            return

        result, pairs = run_calibration(uwb_window, slam_window, config=self.solver_config)
        session.calibration_result = result
        session.calibration_status = str(result.get("status", "failed"))
        self._write_calibration_result(session)
        self._write_matched_pairs(session, pairs)

        if session.calibration_status == "calibrated":
            session.T_G_Lw = np.array(result["T_G_Lw"], dtype=np.float64)

    def _finalize_initial_xyzyaw_bootstrap(self, session: SessionState, uwb_window: list[UwbSample]) -> None:
        if not uwb_window:
            session.calibration_status = "failed"
            session.calibration_result = {
                "status": "failed",
                "calibration_mode": "initial_xyzyaw",
                "reason": "no_uwb_samples_in_window",
                "sample_pairs": 0,
            }
            self._write_calibration_result(session)
            self._write_initial_uwb_window(session, [])
            return

        p_samples = np.array([s.relative_xyz_m for s in uwb_window], dtype=np.float64)
        p_global = p_samples.mean(axis=0)
        if float(np.hypot(p_global[0], p_global[1])) < self.min_initial_xy_distance_m:
            session.calibration_status = "failed"
            session.calibration_result = {
                "status": "failed",
                "calibration_mode": "initial_xyzyaw",
                "reason": "uwb_xy_too_small_for_yaw_inference",
                "sample_pairs": len(uwb_window),
            }
            self._write_calibration_result(session)
            self._write_initial_uwb_window(session, uwb_window)
            return

        try:
            yaw_rad = infer_face_anchor_yaw(
                (float(p_global[0]), float(p_global[1]), float(p_global[2])),
                yaw_offset_rad=session.yaw_offset_rad,
            )
        except ValueError as exc:
            session.calibration_status = "failed"
            session.calibration_result = {
                "status": "failed",
                "calibration_mode": "initial_xyzyaw",
                "reason": str(exc),
                "sample_pairs": len(uwb_window),
            }
            self._write_calibration_result(session)
            self._write_initial_uwb_window(session, uwb_window)
            return

        session.bootstrap_global_position_xyz_m = (float(p_global[0]), float(p_global[1]), float(p_global[2]))
        session.bootstrap_global_yaw_rad = float(yaw_rad)
        session.bootstrap_uwb_ts_ms = int(uwb_window[-1].ts_ms)
        session.calibration_status = "waiting_for_first_slam_pose"
        session.calibration_result = {
            "status": "waiting_for_first_slam_pose",
            "calibration_mode": "initial_xyzyaw",
            "reason": "uwb_bootstrap_ready_waiting_for_orb_pose",
            "sample_pairs": len(uwb_window),
            "bootstrap_uwb_ts_ms": session.bootstrap_uwb_ts_ms,
            "bootstrap_global_xyz_m": list(session.bootstrap_global_position_xyz_m),
            "assumed_global_yaw_rad": session.bootstrap_global_yaw_rad,
        }
        self._write_calibration_result(session)
        self._write_initial_uwb_window(session, uwb_window)

    def _finalize_initial_xyzyaw_with_first_pose(self, session: SessionState, pose: SlamPose) -> None:
        if session.bootstrap_global_position_xyz_m is None or session.bootstrap_global_yaw_rad is None:
            session.calibration_status = "failed"
            session.calibration_result = {
                "status": "failed",
                "calibration_mode": "initial_xyzyaw",
                "reason": "missing_uwb_bootstrap_before_first_pose",
            }
            self._write_calibration_result(session)
            return

        result = solve_initial_xyzyaw_transform(
            global_position_xyz_m=session.bootstrap_global_position_xyz_m,
            global_yaw_rad=session.bootstrap_global_yaw_rad,
            local_t_xyz_m=pose.t_xyz_m,
            local_q_xyzw=pose.q_xyzw,
        )
        result["bootstrap_uwb_ts_ms"] = session.bootstrap_uwb_ts_ms
        result["first_slam_pose_ts_ms"] = int(pose.ts_ms)
        result["status"] = "calibrated"

        session.T_G_Lw = np.array(result["T_G_Lw"], dtype=np.float64)
        session.calibration_result = result
        session.calibration_status = "calibrated"
        self._write_calibration_result(session)

    def _write_calibration_result(self, session: SessionState) -> None:
        if session.output_dir is None:
            return
        result = session.calibration_result or {"status": "unknown"}
        (session.output_dir / "calibration_result.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

    def _write_matched_pairs(self, session: SessionState, pairs: list[dict[str, Any]]) -> None:
        if session.output_dir is None:
            return
        pairs_path = session.output_dir / "matched_samples.jsonl"
        with pairs_path.open("w", encoding="utf-8") as f:
            for pair in pairs:
                record = {
                    "uwb_server_rx_ms": pair["uwb"].server_rx_ms,
                    "slam_server_rx_ms": pair["slam"].server_rx_ms,
                    "uwb_ts_ms": pair["uwb"].ts_ms,
                    "slam_ts_ms": pair["slam"].ts_ms,
                    "skew_ms": pair["skew_ms"],
                    "uwb_relative_xyz_m": list(pair["uwb"].relative_xyz_m),
                    "slam_t_xyz_m": list(pair["slam"].t_xyz_m),
                    "residual_m": pair.get("residual_m"),
                    "inlier": pair.get("inlier"),
                }
                f.write(json.dumps(record) + "\n")

    def _write_initial_uwb_window(self, session: SessionState, uwb_samples: list[UwbSample]) -> None:
        if session.output_dir is None:
            return
        pairs_path = session.output_dir / "matched_samples.jsonl"
        with pairs_path.open("w", encoding="utf-8") as f:
            for sample in uwb_samples:
                record = {
                    "uwb_server_rx_ms": sample.server_rx_ms,
                    "slam_server_rx_ms": None,
                    "uwb_ts_ms": sample.ts_ms,
                    "slam_ts_ms": None,
                    "skew_ms": None,
                    "uwb_relative_xyz_m": list(sample.relative_xyz_m),
                    "slam_t_xyz_m": None,
                    "residual_m": None,
                    "inlier": None,
                }
                f.write(json.dumps(record) + "\n")

    def _write_global_pose(self, session: SessionState, pose: SlamPose) -> None:
        assert session.T_G_Lw is not None
        t_global, q_global = apply_transform_to_pose(session.T_G_Lw, pose.t_xyz_m, pose.q_xyzw)
        ts_sec = float(pose.ts_ms) / 1000.0
        line = (
            f"{ts_sec:.6f} "
            f"{t_global[0]:.6f} {t_global[1]:.6f} {t_global[2]:.6f} "
            f"{q_global[0]:.6f} {q_global[1]:.6f} {q_global[2]:.6f} {q_global[3]:.6f}\n"
        )

        if session.trajectory_global_path is not None:
            session.trajectory_global_path.parent.mkdir(parents=True, exist_ok=True)
            with session.trajectory_global_path.open("a", encoding="utf-8") as f:
                f.write(line)

        if self._global_pose_sock is not None and self._global_pose_addr is not None:
            payload = {
                "type": "global_pose",
                "session_id": session.session_id,
                "worker_id": session.worker_id,
                "ts_ms": pose.ts_ms,
                "t_xyz_m": [float(t_global[0]), float(t_global[1]), float(t_global[2])],
                "q_xyzw": [float(q_global[0]), float(q_global[1]), float(q_global[2]), float(q_global[3])],
            }
            try:
                self._global_pose_sock.sendto(json.dumps(payload).encode("utf-8"), self._global_pose_addr)
            except OSError:
                pass

    def status(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                return {
                    "session_id": session_id,
                    "calibration_status": "unknown",
                    "calibration_mode": "unknown",
                    "sample_pairs": 0,
                    "inliers": 0,
                    "rms_error_m": None,
                    "window_remaining_s": None,
                }

            remaining_s: float | None = None
            if session.calibration_status == "calibrating":
                end_ms = session.calibration_end_rx_ms()
                if end_ms is not None:
                    remaining_s = max(0.0, (end_ms - now_ms()) / 1000.0)

            result = session.calibration_result or {}
            return {
                "session_id": session.session_id,
                "anchor_id": session.anchor_id,
                "worker_id": session.worker_id,
                "calibration_status": session.calibration_status,
                "calibration_mode": session.calibration_mode,
                "sample_pairs": int(result.get("sample_pairs", 0)),
                "inliers": int(result.get("inliers", 0)),
                "rms_error_m": result.get("rms_error_m"),
                "inlier_ratio": result.get("inlier_ratio"),
                "window_remaining_s": remaining_s,
                "reason": result.get("reason"),
            }


class FusionRequestHandler(BaseHTTPRequestHandler):
    runtime: FusionRuntime

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _write_json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json_body()
            rx_ms = now_ms()

            if parsed.path == "/calibration/start":
                req = parse_calibration_start(payload, server_rx_ms=rx_ms)
                result = self.runtime.start_calibration(req)
                self._write_json(200, result)
                return
            if parsed.path == "/uwb":
                sample = parse_uwb_sample(payload, server_rx_ms=rx_ms)
                result = self.runtime.ingest_uwb_sample(sample)
                self._write_json(200, result)
                return

            self._write_json(404, {"error": "not_found"})
        except Exception as exc:
            self._write_json(400, {"error": str(exc)})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._write_json(200, {"ok": True, "ts_ms": now_ms()})
            return
        if parsed.path == "/status":
            params = parse_qs(parsed.query)
            session_id = params.get("session_id", [None])[0]
            if not session_id:
                self._write_json(400, {"error": "missing session_id"})
                return
            self._write_json(200, self.runtime.status(session_id))
            return
        self._write_json(404, {"error": "not_found"})


def udp_pose_listener(runtime: FusionRuntime, host: str, port: int, stop_event: threading.Event) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(0.5)
    print(f"[fusion] UDP pose listener on {host}:{port}")
    while not stop_event.is_set():
        try:
            data, _addr = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except OSError:
            break

        rx_ms = now_ms()
        try:
            payload = json.loads(data.decode("utf-8"))
            pose = parse_slam_pose(payload, server_rx_ms=rx_ms)
            runtime.ingest_slam_pose(pose)
        except Exception:
            continue
    sock.close()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config root must be a mapping")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Fusion service for UWB + SLAM global transform calibration.")
    parser.add_argument("--config", type=Path, default=Path("fusion/config.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    runtime = FusionRuntime(config)

    http_host = str(config.get("http_host", "0.0.0.0"))
    http_port = int(config.get("http_port", 8080))
    udp_host = str(config.get("slam_pose_udp_host", "0.0.0.0"))
    udp_port = int(config.get("slam_pose_udp_port", 8091))

    stop_event = threading.Event()
    udp_thread = threading.Thread(
        target=udp_pose_listener,
        args=(runtime, udp_host, udp_port, stop_event),
        daemon=True,
    )
    udp_thread.start()

    FusionRequestHandler.runtime = runtime
    server = ThreadingHTTPServer((http_host, http_port), FusionRequestHandler)
    print(f"[fusion] HTTP server on {http_host}:{http_port}")
    print(f"[fusion] output root: {runtime.output_root}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        server.shutdown()
        server.server_close()
        udp_thread.join(timeout=1.0)
        print("[fusion] stopped")


if __name__ == "__main__":
    main()
