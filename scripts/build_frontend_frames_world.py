#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _resolve_path(raw: str, *, base_dir: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    root_candidate = (REPO_ROOT / p).resolve()
    if root_candidate.exists():
        return root_candidate
    return (base_dir / p).resolve()


def _as_matrix_4x4(payload: Any, source: str) -> np.ndarray:
    arr = np.array(payload, dtype=np.float64)
    if arr.shape == (4, 4):
        pass
    elif arr.size == 16:
        arr = arr.reshape((4, 4))
    else:
        raise ValueError(f"{source} is not a 4x4 transform matrix")
    if not np.isfinite(arr).all():
        raise ValueError(f"{source} contains NaN/Inf")
    return arr


def _load_samples_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing samples index: {path}") from exc

    out: list[dict[str, Any]] = []
    for line_idx, line in enumerate(lines, start=1):
        raw = line.strip()
        if not raw:
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_idx}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Expected object at {path}:{line_idx}")
        out.append(record)
    return out


def _load_trajectory_tum(path: Path) -> tuple[list[float], list[dict[str, Any]]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing trajectory file: {path}") from exc

    points: list[dict[str, Any]] = []
    for line_idx, line in enumerate(lines, start=1):
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split()
        if len(parts) < 8:
            raise ValueError(f"Invalid TUM row at {path}:{line_idx}")
        ts = float(parts[0])
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
        points.append({"ts": ts, "xyz": (x, y, z), "q_xyzw": (qx, qy, qz, qw)})
    if not points:
        raise ValueError(f"No trajectory points found in {path}")
    points.sort(key=lambda p: p["ts"])
    ts_sorted = [float(p["ts"]) for p in points]
    return ts_sorted, points


def _nearest_trajectory_point(
    ts: float,
    ts_sorted: list[float],
    points_sorted: list[dict[str, Any]],
) -> tuple[dict[str, Any], float]:
    idx = bisect.bisect_left(ts_sorted, ts)
    candidates: list[tuple[float, dict[str, Any]]] = []
    if idx < len(ts_sorted):
        p = points_sorted[idx]
        candidates.append((abs(float(p["ts"]) - ts), p))
    if idx > 0:
        p = points_sorted[idx - 1]
        candidates.append((abs(float(p["ts"]) - ts), p))
    if not candidates:
        raise ValueError("No trajectory candidates available")
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1], float(candidates[0][0])


def _transform_point(T: np.ndarray, xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    v = np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64)
    out = T @ v
    return (float(out[0]), float(out[1]), float(out[2]))


def _resolve_session_matrix(
    session: dict[str, Any],
    blender_transforms: dict[str, np.ndarray] | None,
) -> np.ndarray:
    if "matrix_world" in session:
        return _as_matrix_4x4(session["matrix_world"], "session.matrix_world")
    if "transform_matrix" in session:
        return _as_matrix_4x4(session["transform_matrix"], "session.transform_matrix")
    if "blender_object" in session:
        if blender_transforms is None:
            raise ValueError(
                f"Session '{session.get('name', 'unknown')}' uses blender_object but --blender-transforms not provided"
            )
        key = str(session["blender_object"])
        if key not in blender_transforms:
            available = ", ".join(sorted(blender_transforms.keys()))
            raise ValueError(f"blender_object '{key}' not found. Available: {available}")
        return blender_transforms[key]
    raise ValueError(
        f"Session '{session.get('name', 'unknown')}' must define one of matrix_world, transform_matrix, or blender_object"
    )


def _load_blender_transforms(path: Path | None) -> dict[str, np.ndarray] | None:
    if path is None:
        return None
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError("Blender transforms file must be an object: {object_name: 4x4_matrix}")
    out: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        out[str(key)] = _as_matrix_4x4(value, f"blender_transforms[{key}]")
    return out


def _rgb_name_from_record(record: dict[str, Any]) -> str:
    rgb_path = record.get("rgb_path")
    if isinstance(rgb_path, str) and rgb_path.strip():
        return Path(rgb_path).name
    sample_id = int(record.get("sample_id", -1))
    seq = int(record.get("seq", -1))
    return f"sample_{sample_id:06d}_seq{seq}_rgb.jpg"


def _resolve_rgb_source_path(
    sample: dict[str, Any],
    *,
    samples_index_path: Path,
    session: dict[str, Any],
) -> Path:
    rgb_name = _rgb_name_from_record(sample)

    rgb_path_raw = sample.get("rgb_path")
    if isinstance(rgb_path_raw, str) and rgb_path_raw.strip():
        path = _resolve_path(rgb_path_raw, base_dir=samples_index_path.parent)
        if path.exists():
            return path

    # Optional override when samples_index rgb_path is stale.
    rgb_root = session.get("rgb_root")
    if isinstance(rgb_root, str) and rgb_root.strip():
        root_path = _resolve_path(rgb_root, base_dir=samples_index_path.parent)
        candidate = root_path / rgb_name
        if candidate.exists():
            return candidate

    # Last fallback: try next to samples_index.
    local_candidate = samples_index_path.parent / rgb_name
    if local_candidate.exists():
        return local_candidate

    return Path("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build frontend /frames/frames_world.jsonl from manual Blender-aligned sessions.\n"
            "Each session provides samples_index.jsonl + trajectory_tum.txt + transform matrix."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path, help="Session manifest JSON")
    parser.add_argument(
        "--blender-transforms",
        type=Path,
        default=None,
        help="Optional JSON mapping {object_name: 4x4 matrix} exported from Blender",
    )
    parser.add_argument("--output-jsonl", required=True, type=Path, help="Output frames_world.jsonl path")
    parser.add_argument(
        "--output-trajectory",
        type=Path,
        default=None,
        help="Optional merged trajectory_tum output in transformed coordinates",
    )
    parser.add_argument(
        "--max-match-dt-s",
        type=float,
        default=0.06,
        help="Max allowed timestamp mismatch for sample->trajectory match (seconds)",
    )
    parser.add_argument(
        "--copy-images-to",
        type=Path,
        default=None,
        help="Optional directory to copy RGB images into (unique filenames per session)",
    )
    parser.add_argument(
        "--image-url-prefix",
        type=str,
        default="/frames",
        help="URL prefix for copied images (default: /frames)",
    )
    parser.add_argument(
        "--tracking-only",
        action="store_true",
        help="Include only samples with tracking_state==2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_payload = _load_json(args.manifest)
    if not isinstance(manifest_payload, dict):
        raise ValueError("Manifest root must be an object")
    sessions = manifest_payload.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("Manifest must contain non-empty 'sessions' array")

    blender_transforms = _load_blender_transforms(args.blender_transforms)

    if args.copy_images_to is not None:
        args.copy_images_to.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.output_trajectory is not None:
        args.output_trajectory.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    total_written = 0
    total_skipped = 0
    total_copied_images = 0

    trajectory_lines: list[str] = []
    summary_sessions: list[dict[str, Any]] = []

    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for session_idx, session in enumerate(sessions):
            if not isinstance(session, dict):
                raise ValueError(f"sessions[{session_idx}] must be an object")
            if "samples_index" not in session or "trajectory_tum" not in session:
                raise ValueError(f"sessions[{session_idx}] must contain samples_index and trajectory_tum")

            session_name = str(session.get("name", f"session_{session_idx}"))
            samples_index_path = _resolve_path(str(session["samples_index"]), base_dir=args.manifest.parent)
            trajectory_path = _resolve_path(str(session["trajectory_tum"]), base_dir=args.manifest.parent)
            T_session = _resolve_session_matrix(session, blender_transforms)

            samples = _load_samples_jsonl(samples_index_path)
            ts_sorted, traj_points = _load_trajectory_tum(trajectory_path)

            session_written = 0
            session_skipped = 0
            session_match_dts: list[float] = []

            for sample in samples:
                total_records += 1
                if args.tracking_only:
                    try:
                        if int(sample.get("tracking_state", 2)) != 2:
                            total_skipped += 1
                            session_skipped += 1
                            continue
                    except Exception:
                        total_skipped += 1
                        session_skipped += 1
                        continue

                if "ts" not in sample:
                    total_skipped += 1
                    session_skipped += 1
                    continue
                ts = float(sample["ts"])
                traj_point, dt = _nearest_trajectory_point(ts, ts_sorted, traj_points)
                session_match_dts.append(dt)
                if dt > args.max_match_dt_s:
                    total_skipped += 1
                    session_skipped += 1
                    continue

                xyz_local = traj_point["xyz"]
                xyz_world = _transform_point(T_session, xyz_local)

                out_record = dict(sample)
                out_record["session_name"] = session_name
                out_record["worldPos"] = [xyz_world[0], xyz_world[1], xyz_world[2]]
                out_record["trajectory_match_dt_s"] = float(dt)
                out_record["trajectory_source"] = str(trajectory_path)

                rgb_name = _rgb_name_from_record(sample)
                if args.copy_images_to is not None:
                    src_rgb = _resolve_rgb_source_path(
                        sample,
                        samples_index_path=samples_index_path,
                        session=session,
                    )
                    dst_name = f"{session_name}_{rgb_name}"
                    dst_rgb = args.copy_images_to / dst_name
                    if src_rgb.exists():
                        shutil.copy2(src_rgb, dst_rgb)
                        total_copied_images += 1
                    out_record["image_url"] = f"{args.image_url_prefix.rstrip('/')}/{dst_name}"
                else:
                    if "image_url" not in out_record:
                        out_record["image_url"] = f"{args.image_url_prefix.rstrip('/')}/{rgb_name}"

                out.write(json.dumps(out_record) + "\n")
                total_written += 1
                session_written += 1

                if args.output_trajectory is not None:
                    qx, qy, qz, qw = traj_point["q_xyzw"]
                    trajectory_lines.append(
                        f"{ts:.6f} {xyz_world[0]:.6f} {xyz_world[1]:.6f} {xyz_world[2]:.6f} "
                        f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
                    )

            summary_sessions.append(
                {
                    "name": session_name,
                    "samples_index": str(samples_index_path),
                    "trajectory_tum": str(trajectory_path),
                    "written": session_written,
                    "skipped": session_skipped,
                    "max_match_dt_s": max(session_match_dts) if session_match_dts else None,
                    "avg_match_dt_s": (sum(session_match_dts) / len(session_match_dts)) if session_match_dts else None,
                }
            )
            print(
                f"[frames-world] session={session_name} written={session_written} "
                f"skipped={session_skipped} max_dt={max(session_match_dts) if session_match_dts else 'n/a'}"
            )

    if args.output_trajectory is not None:
        args.output_trajectory.write_text("".join(trajectory_lines), encoding="utf-8")

    summary = {
        "manifest": str(args.manifest),
        "output_jsonl": str(args.output_jsonl),
        "output_trajectory": str(args.output_trajectory) if args.output_trajectory is not None else None,
        "tracking_only": bool(args.tracking_only),
        "max_match_dt_s": float(args.max_match_dt_s),
        "records_total": total_records,
        "records_written": total_written,
        "records_skipped": total_skipped,
        "copied_images": total_copied_images,
        "sessions": summary_sessions,
    }
    summary_path = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[frames-world] done")
    print(f"[frames-world] output_jsonl={args.output_jsonl}")
    if args.output_trajectory is not None:
        print(f"[frames-world] output_trajectory={args.output_trajectory}")
    if args.copy_images_to is not None:
        print(f"[frames-world] copied_images_to={args.copy_images_to}")
    print(f"[frames-world] summary={summary_path}")


if __name__ == "__main__":
    main()
