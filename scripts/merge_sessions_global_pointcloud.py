#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Ensure local package imports work when invoked as:
# `python scripts/merge_sessions_global_pointcloud.py ...`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fusion.solver import apply_transform_to_pose, matrix_to_pose, pose_to_matrix

# Import point cloud loader module directly from file to avoid importing
# pointcloud_locator package-level dependencies (e.g. scipy in __init__).
POINT_CLOUD_MODULE_PATH = REPO_ROOT / "pointcloud_locator" / "point_cloud.py"
_point_cloud_spec = importlib.util.spec_from_file_location("point_cloud_loader", POINT_CLOUD_MODULE_PATH)
if _point_cloud_spec is None or _point_cloud_spec.loader is None:
    raise RuntimeError(f"Failed to load point cloud module: {POINT_CLOUD_MODULE_PATH}")
_point_cloud_module = importlib.util.module_from_spec(_point_cloud_spec)
_point_cloud_spec.loader.exec_module(_point_cloud_module)
load_point_cloud = _point_cloud_module.load_point_cloud


SESSION_COLOR_PALETTE_RGB = np.array(
    [
        [239, 83, 80],   # red
        [66, 165, 245],  # blue
        [102, 187, 106], # green
        [255, 167, 38],  # orange
        [171, 71, 188],  # violet
        [38, 198, 218],  # cyan
        [255, 238, 88],  # yellow
    ],
    dtype=np.uint8,
)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _as_matrix_4x4(payload: Any, source: str) -> np.ndarray:
    candidate = payload
    if isinstance(payload, dict):
        if "T_G_Lw" in payload:
            candidate = payload["T_G_Lw"]
        elif "transform" in payload:
            candidate = payload["transform"]

    arr = np.array(candidate, dtype=np.float64)
    if arr.shape == (4, 4):
        pass
    elif arr.size == 16:
        arr = arr.reshape((4, 4))
    else:
        raise ValueError(
            f"{source} does not contain a 4x4 matrix. "
            "Expected 4x4 array, 16-element array, or object with key 'T_G_Lw'."
        )
    if not np.isfinite(arr).all():
        raise ValueError(f"{source} contains NaN/Inf matrix entries")
    return arr


def _load_samples_index(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing samples index: {path}") from exc

    records: list[dict[str, Any]] = []
    for line_idx, line in enumerate(lines, start=1):
        raw = line.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_idx}: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"Expected object record in {path}:{line_idx}")
        records.append(obj)
    return records


def _parse_vec3(record: dict[str, Any], key: str) -> tuple[float, float, float]:
    value = record.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Missing or invalid vec3 key '{key}'")
    return (float(value[0]), float(value[1]), float(value[2]))


def _parse_quat_xyzw(record: dict[str, Any], key: str = "q_xyzw") -> tuple[float, float, float, float]:
    value = record.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"Missing or invalid quaternion key '{key}'")
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def _invert_pose(
    t_xyz_m: tuple[float, float, float],
    q_xyzw: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    T = pose_to_matrix(t_xyz_m, q_xyzw)
    T_inv = np.linalg.inv(T)
    return matrix_to_pose(T_inv)


def _resolve_relative_path(path_value: str, base_dir: Path) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    # Prefer path relative to repository root when the record stores "outputs/...".
    root_candidate = (REPO_ROOT / p).resolve()
    if root_candidate.exists():
        return root_candidate
    # Fallback: relative to samples_index directory.
    return (base_dir / p).resolve()


def _tracking_ok(record: dict[str, Any], tracking_only: bool) -> bool:
    if not tracking_only:
        return True
    if "tracking_state" not in record:
        return True
    try:
        return int(record["tracking_state"]) == 2
    except Exception:
        return False


def _transform_points(T_G_Cw: np.ndarray, points_c: np.ndarray) -> np.ndarray:
    R = T_G_Cw[:3, :3]
    t = T_G_Cw[:3, 3]
    return (points_c @ R.T) + t


def _write_ply(path: Path, points_xyz: np.ndarray, colors_rgb_u8: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"Expected points shape (N,3), got {points_xyz.shape}")
    n = points_xyz.shape[0]

    with path.open("wb") as f:
        if colors_rgb_u8 is None:
            header = (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {n}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "end_header\n"
            )
            f.write(header.encode("ascii"))
            data = np.asarray(points_xyz, dtype=np.float32)
            f.write(data.tobytes(order="C"))
            return

        if colors_rgb_u8.shape != (n, 3):
            raise ValueError(
                f"Color shape must match points (N,3). got points={points_xyz.shape}, colors={colors_rgb_u8.shape}"
            )

        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))

        structured = np.empty(
            n,
            dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
        )
        structured["x"] = points_xyz[:, 0].astype(np.float32, copy=False)
        structured["y"] = points_xyz[:, 1].astype(np.float32, copy=False)
        structured["z"] = points_xyz[:, 2].astype(np.float32, copy=False)
        structured["red"] = colors_rgb_u8[:, 0]
        structured["green"] = colors_rgb_u8[:, 1]
        structured["blue"] = colors_rgb_u8[:, 2]
        f.write(structured.tobytes(order="C"))


def _limit_points(
    points_chunks: list[np.ndarray],
    color_chunks: list[np.ndarray] | None,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[np.ndarray] | None, int]:
    if max_points <= 0:
        total = sum(chunk.shape[0] for chunk in points_chunks)
        return points_chunks, color_chunks, total

    total = sum(chunk.shape[0] for chunk in points_chunks)
    if total <= max_points:
        return points_chunks, color_chunks, total

    merged_points = np.concatenate(points_chunks, axis=0)
    pick = rng.choice(merged_points.shape[0], size=max_points, replace=False)
    merged_points = merged_points[pick]

    if color_chunks is not None:
        merged_colors = np.concatenate(color_chunks, axis=0)[pick]
        return [merged_points], [merged_colors], max_points

    return [merged_points], None, max_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge multiple ORB+UWB sessions into one global point cloud. "
            "Each session provides samples_index.jsonl + T_G_Lw."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path, help="Session manifest JSON path")
    parser.add_argument("--output-ply", required=True, type=Path, help="Merged point cloud output (.ply)")
    parser.add_argument(
        "--frame-point-stride",
        type=int,
        default=4,
        help="Keep every Nth point from each frame PCD (default: 4)",
    )
    parser.add_argument(
        "--max-frames-per-session",
        type=int,
        default=0,
        help="Optional frame cap per session (0 => all)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=4_000_000,
        help="Hard cap for merged points via random downsampling while building (0 => unlimited)",
    )
    parser.add_argument(
        "--tracking-only",
        action="store_true",
        help="Use only samples with tracking_state==2",
    )
    parser.add_argument(
        "--invert-local-pose-default",
        action="store_true",
        help="Invert local sample pose by default unless session overrides invert_local_pose",
    )
    parser.add_argument(
        "--color-by-session",
        action="store_true",
        help="Color points by session index in output PLY",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for capped downsampling")
    return parser.parse_args()


def _load_transform_for_session(session: dict[str, Any]) -> np.ndarray:
    calibration_result_path = session.get("calibration_result")
    transform_file_path = session.get("transform_file")
    if bool(calibration_result_path) == bool(transform_file_path):
        raise ValueError(
            "Each session must specify exactly one of 'calibration_result' or 'transform_file'"
        )

    if calibration_result_path:
        path = _resolve_relative_path(str(calibration_result_path), REPO_ROOT)
        payload = _load_json(path)
        return _as_matrix_4x4(payload, f"calibration_result:{path}")

    path = _resolve_relative_path(str(transform_file_path), REPO_ROOT)
    payload = _load_json(path)
    return _as_matrix_4x4(payload, f"transform_file:{path}")


def _validate_manifest(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("Manifest root must be an object")
    sessions = payload.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("Manifest must contain non-empty array key 'sessions'")
    out: list[dict[str, Any]] = []
    for idx, session in enumerate(sessions):
        if not isinstance(session, dict):
            raise ValueError(f"Manifest sessions[{idx}] must be an object")
        if "samples_index" not in session:
            raise ValueError(f"Manifest sessions[{idx}] missing 'samples_index'")
        out.append(session)
    return out


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    manifest_payload = _load_json(args.manifest)
    sessions = _validate_manifest(manifest_payload)

    if args.frame_point_stride <= 0:
        raise ValueError("--frame-point-stride must be >= 1")
    if args.max_frames_per_session < 0:
        raise ValueError("--max-frames-per-session must be >= 0")
    if args.max_points < 0:
        raise ValueError("--max-points must be >= 0")

    merged_points_chunks: list[np.ndarray] = []
    merged_color_chunks: list[np.ndarray] | None = [] if args.color_by_session else None

    total_sessions = len(sessions)
    total_frames_used = 0
    total_frames_skipped = 0
    total_pcd_missing = 0

    summary: dict[str, Any] = {"sessions": []}

    for session_idx, session in enumerate(sessions):
        session_name = str(session.get("name", f"session_{session_idx}"))
        samples_index_path = _resolve_relative_path(str(session["samples_index"]), REPO_ROOT)
        session_records = _load_samples_index(samples_index_path)
        T_G_Lw = _load_transform_for_session(session)
        invert_local_pose = bool(session.get("invert_local_pose", args.invert_local_pose_default))
        session_stride = int(session.get("frame_point_stride", args.frame_point_stride))
        if session_stride <= 0:
            raise ValueError(f"Session '{session_name}' has invalid frame_point_stride={session_stride}")

        session_frames_used = 0
        session_frames_skipped = 0
        session_points_added = 0
        session_pcd_missing = 0

        for record in session_records:
            if args.max_frames_per_session > 0 and session_frames_used >= args.max_frames_per_session:
                break

            if not _tracking_ok(record, tracking_only=args.tracking_only):
                session_frames_skipped += 1
                continue
            if not bool(record.get("pcd_saved", True)):
                session_frames_skipped += 1
                continue
            pcd_path_raw = record.get("pcd_path")
            if not isinstance(pcd_path_raw, str) or not pcd_path_raw:
                session_frames_skipped += 1
                session_pcd_missing += 1
                continue
            pcd_path = _resolve_relative_path(pcd_path_raw, samples_index_path.parent)
            if not pcd_path.exists():
                session_frames_skipped += 1
                session_pcd_missing += 1
                continue

            t_local = _parse_vec3(record, "t_xyz_m")
            q_local = _parse_quat_xyzw(record, "q_xyzw")
            if invert_local_pose:
                t_local, q_local = _invert_pose(t_local, q_local)
            T_G_Cw = pose_to_matrix(*apply_transform_to_pose(T_G_Lw, t_local, q_local))

            points_c = load_point_cloud(pcd_path)
            if session_stride > 1:
                points_c = points_c[::session_stride]
            if points_c.shape[0] == 0:
                session_frames_skipped += 1
                continue

            points_g = _transform_points(T_G_Cw, points_c)
            merged_points_chunks.append(points_g.astype(np.float32, copy=False))

            if merged_color_chunks is not None:
                color = SESSION_COLOR_PALETTE_RGB[session_idx % len(SESSION_COLOR_PALETTE_RGB)]
                colors = np.repeat(color[None, :], points_g.shape[0], axis=0)
                merged_color_chunks.append(colors)

            session_frames_used += 1
            session_points_added += int(points_g.shape[0])

            merged_points_chunks, merged_color_chunks, _ = _limit_points(
                merged_points_chunks, merged_color_chunks, max_points=args.max_points, rng=rng
            )

        total_frames_used += session_frames_used
        total_frames_skipped += session_frames_skipped
        total_pcd_missing += session_pcd_missing
        summary["sessions"].append(
            {
                "name": session_name,
                "samples_index": str(samples_index_path),
                "frames_total": len(session_records),
                "frames_used": session_frames_used,
                "frames_skipped": session_frames_skipped,
                "points_added_pre_cap": session_points_added,
                "invert_local_pose": invert_local_pose,
                "frame_point_stride": session_stride,
                "pcd_missing_or_invalid": session_pcd_missing,
            }
        )
        print(
            f"[merge] session={session_name} used_frames={session_frames_used} "
            f"skipped={session_frames_skipped} points_pre_cap={session_points_added} "
            f"invert_local_pose={1 if invert_local_pose else 0}"
        )

    if not merged_points_chunks:
        raise ValueError("No points were produced. Check manifest paths, tracking filters, and PCD files.")

    points_merged = np.concatenate(merged_points_chunks, axis=0)
    if merged_color_chunks is not None:
        colors_merged = np.concatenate(merged_color_chunks, axis=0)
    else:
        colors_merged = None

    _write_ply(args.output_ply, points_merged, colors_merged)

    summary["output_ply"] = str(args.output_ply)
    summary["sessions_count"] = total_sessions
    summary["frames_used_total"] = total_frames_used
    summary["frames_skipped_total"] = total_frames_skipped
    summary["pcd_missing_total"] = total_pcd_missing
    summary["points_written"] = int(points_merged.shape[0])
    summary["color_by_session"] = bool(args.color_by_session)
    summary["tracking_only"] = bool(args.tracking_only)
    summary["frame_point_stride_default"] = int(args.frame_point_stride)
    summary["max_points"] = int(args.max_points)

    summary_path = args.output_ply.with_suffix(args.output_ply.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[merge] done")
    print(f"[merge] output_ply={args.output_ply}")
    print(f"[merge] points_written={points_merged.shape[0]}")
    print(f"[merge] summary={summary_path}")


if __name__ == "__main__":
    main()
