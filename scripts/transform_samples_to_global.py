#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Ensure local package imports work when invoked as:
# `python scripts/transform_samples_to_global.py ...`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fusion.solver import (
    apply_transform_to_pose,
    matrix_to_pose,
    pose_to_matrix,
    solve_initial_xyzyaw_transform,
    yaw_from_quaternion_xyzw,
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
            "Expected a 4x4 array, 16-element array, or object with key 'T_G_Lw'."
        )

    if not np.isfinite(arr).all():
        raise ValueError(f"{source} contains NaN/Inf matrix entries")
    return arr


def _parse_vec3(payload: dict[str, Any], key: str) -> tuple[float, float, float]:
    value = payload.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Missing or invalid vec3 key '{key}' in sample record")
    return (float(value[0]), float(value[1]), float(value[2]))


def _parse_quat_xyzw(payload: dict[str, Any], key: str = "q_xyzw") -> tuple[float, float, float, float]:
    value = payload.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"Missing or invalid quaternion key '{key}' in sample record")
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def _sample_ts_sec(record: dict[str, Any]) -> float:
    if "ts" in record:
        return float(record["ts"])
    if "ts_ms" in record:
        return float(record["ts_ms"]) / 1000.0
    raise ValueError("Sample record has neither 'ts' nor 'ts_ms'")


def _load_samples_index(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing samples index: {path}") from exc

    samples: list[dict[str, Any]] = []
    for line_idx, line in enumerate(lines, start=1):
        raw = line.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_idx}: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_idx}")
        samples.append(obj)
    if not samples:
        raise ValueError(f"No samples found in {path}")
    return samples


def _tracking_ok(record: dict[str, Any], allow_untracked: bool) -> bool:
    if allow_untracked:
        return True
    if "tracking_state" not in record:
        return True
    try:
        return int(record["tracking_state"]) == 2
    except Exception:
        return False


def _choose_bootstrap_sample(
    samples: list[dict[str, Any]],
    sample_id: int | None,
    ts_sec: float | None,
    allow_untracked: bool,
) -> dict[str, Any]:
    candidates = [s for s in samples if _tracking_ok(s, allow_untracked)]
    if not candidates:
        raise ValueError("No bootstrap candidates available (tracking_state==2 required by default)")

    if sample_id is not None:
        for sample in candidates:
            if int(sample.get("sample_id", -1)) == sample_id:
                return sample
        raise ValueError(f"bootstrap sample_id={sample_id} not found in samples_index")

    if ts_sec is not None:
        return min(candidates, key=lambda s: abs(_sample_ts_sec(s) - ts_sec))

    return candidates[0]


def _resolve_global_yaw_rad(args: argparse.Namespace) -> float:
    provided = 0
    provided += int(args.global_yaw_rad is not None)
    provided += int(args.global_yaw_deg is not None)
    provided += int(args.global_yaw_q_xyzw is not None)
    provided += int(args.global_yaw_q_wxyz is not None)
    if provided != 1:
        raise ValueError(
            "Provide exactly one of: --global-yaw-rad, --global-yaw-deg, "
            "--global-yaw-q-xyzw, --global-yaw-q-wxyz"
        )

    if args.global_yaw_rad is not None:
        return float(args.global_yaw_rad)
    if args.global_yaw_deg is not None:
        return float(np.deg2rad(float(args.global_yaw_deg)))
    if args.global_yaw_q_xyzw is not None:
        q = tuple(float(v) for v in args.global_yaw_q_xyzw)
        return float(yaw_from_quaternion_xyzw(q))
    assert args.global_yaw_q_wxyz is not None
    w, x, y, z = (float(v) for v in args.global_yaw_q_wxyz)
    return float(yaw_from_quaternion_xyzw((x, y, z, w)))


def _invert_pose(
    t_xyz_m: tuple[float, float, float],
    q_xyzw: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    T = pose_to_matrix(t_xyz_m, q_xyzw)
    T_inv = np.linalg.inv(T)
    return matrix_to_pose(T_inv)


def _resolve_transform(args: argparse.Namespace, samples: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    if args.calibration_result is not None:
        payload = _load_json(args.calibration_result)
        T_G_Lw = _as_matrix_4x4(payload, f"calibration_result:{args.calibration_result}")
        return T_G_Lw, {"source": "calibration_result", "path": str(args.calibration_result)}

    if args.transform_file is not None:
        payload = _load_json(args.transform_file)
        T_G_Lw = _as_matrix_4x4(payload, f"transform_file:{args.transform_file}")
        return T_G_Lw, {"source": "transform_file", "path": str(args.transform_file)}

    assert args.bootstrap
    if args.global_xyz is None:
        raise ValueError("--bootstrap requires --global-xyz X Y Z")

    global_pos = (float(args.global_xyz[0]), float(args.global_xyz[1]), float(args.global_xyz[2]))
    global_yaw_rad = _resolve_global_yaw_rad(args)
    bootstrap_sample = _choose_bootstrap_sample(
        samples,
        sample_id=args.bootstrap_sample_id,
        ts_sec=args.bootstrap_ts,
        allow_untracked=args.bootstrap_allow_untracked,
    )
    local_t = _parse_vec3(bootstrap_sample, "t_xyz_m")
    local_q = _parse_quat_xyzw(bootstrap_sample, "q_xyzw")
    if args.invert_local_pose:
        local_t, local_q = _invert_pose(local_t, local_q)

    result = solve_initial_xyzyaw_transform(
        global_position_xyz_m=global_pos,
        global_yaw_rad=global_yaw_rad,
        local_t_xyz_m=local_t,
        local_q_xyzw=local_q,
    )
    T_G_Lw = _as_matrix_4x4(result["T_G_Lw"], "bootstrap_result")
    return T_G_Lw, {
        "source": "bootstrap",
        "bootstrap_sample_id": int(bootstrap_sample.get("sample_id", -1)),
        "bootstrap_ts_sec": _sample_ts_sec(bootstrap_sample),
        "global_xyz_m": [global_pos[0], global_pos[1], global_pos[2]],
        "global_yaw_rad": float(global_yaw_rad),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a fixed global transform T_G_Lw to ORB-SLAM3 samples_index.jsonl poses "
            "and export global poses."
        )
    )
    parser.add_argument("--samples-index", required=True, type=Path, help="Path to samples_index.jsonl")
    parser.add_argument("--output-jsonl", required=True, type=Path, help="Output path for transformed JSONL")
    parser.add_argument(
        "--output-tum",
        type=Path,
        default=None,
        help="Optional output path for trajectory in TUM format (ts x y z qx qy qz qw)",
    )
    parser.add_argument(
        "--write-transform-file",
        type=Path,
        default=None,
        help="Optional output path to write the resolved T_G_Lw matrix JSON",
    )
    parser.add_argument(
        "--invert-local-pose",
        action="store_true",
        help=(
            "Invert each local sample pose before applying T_G_Lw. "
            "Useful if your sample pose is actually T_Cw_Lw but stored as T_Lw_Cw."
        ),
    )
    parser.add_argument(
        "--keep-nontracking",
        action="store_true",
        help="Include records where tracking_state is not 2. Default behavior only exports tracking_state==2 records.",
    )

    transform_group = parser.add_mutually_exclusive_group(required=True)
    transform_group.add_argument(
        "--calibration-result",
        type=Path,
        default=None,
        help="Fusion calibration_result.json containing T_G_Lw",
    )
    transform_group.add_argument(
        "--transform-file",
        type=Path,
        default=None,
        help="JSON file containing 4x4 matrix or object with key T_G_Lw",
    )
    transform_group.add_argument(
        "--bootstrap",
        action="store_true",
        help="Build T_G_Lw from one bootstrap sample plus global xyz+yaw",
    )

    parser.add_argument("--global-xyz", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--global-yaw-rad", type=float, default=None)
    parser.add_argument("--global-yaw-deg", type=float, default=None)
    parser.add_argument("--global-yaw-q-xyzw", type=float, nargs=4, default=None, metavar=("QX", "QY", "QZ", "QW"))
    parser.add_argument("--global-yaw-q-wxyz", type=float, nargs=4, default=None, metavar=("QW", "QX", "QY", "QZ"))
    parser.add_argument("--bootstrap-sample-id", type=int, default=None)
    parser.add_argument("--bootstrap-ts", type=float, default=None, help="Desired bootstrap timestamp in seconds")
    parser.add_argument(
        "--bootstrap-allow-untracked",
        action="store_true",
        help="Allow non-tracking samples as bootstrap candidates",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = _load_samples_index(args.samples_index)
    T_G_Lw, transform_meta = _resolve_transform(args, samples)

    if args.write_transform_file is not None:
        args.write_transform_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "T_G_Lw": T_G_Lw.tolist(),
            "meta": transform_meta,
        }
        args.write_transform_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.output_tum is not None:
        args.output_tum.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    written_records = 0
    skipped_records = 0

    tum_handle = args.output_tum.open("w", encoding="utf-8") if args.output_tum is not None else None
    try:
        with args.output_jsonl.open("w", encoding="utf-8") as out:
            for record in samples:
                total_records += 1
                if not args.keep_nontracking and not _tracking_ok(record, allow_untracked=False):
                    skipped_records += 1
                    continue

                t_local = _parse_vec3(record, "t_xyz_m")
                q_local = _parse_quat_xyzw(record, "q_xyzw")
                if args.invert_local_pose:
                    t_local, q_local = _invert_pose(t_local, q_local)

                t_global, q_global = apply_transform_to_pose(T_G_Lw, t_local, q_local)
                ts_sec = _sample_ts_sec(record)

                out_record = dict(record)
                out_record["t_global_xyz_m"] = [float(t_global[0]), float(t_global[1]), float(t_global[2])]
                out_record["q_global_xyzw"] = [float(q_global[0]), float(q_global[1]), float(q_global[2]), float(q_global[3])]
                out_record["pose_frame_global"] = "T_G_Cw"
                out_record["local_pose_inverted"] = bool(args.invert_local_pose)
                out_record["transform_source"] = transform_meta

                out.write(json.dumps(out_record) + "\n")
                written_records += 1

                if tum_handle is not None:
                    tum_handle.write(
                        f"{ts_sec:.6f} "
                        f"{t_global[0]:.9f} {t_global[1]:.9f} {t_global[2]:.9f} "
                        f"{q_global[0]:.9f} {q_global[1]:.9f} {q_global[2]:.9f} {q_global[3]:.9f}\n"
                    )
    finally:
        if tum_handle is not None:
            tum_handle.close()

    print("[transform] done")
    print(f"[transform] samples_index={args.samples_index}")
    print(f"[transform] output_jsonl={args.output_jsonl}")
    if args.output_tum is not None:
        print(f"[transform] output_tum={args.output_tum}")
    print(
        "[transform] records_total="
        f"{total_records} written={written_records} skipped={skipped_records} "
        f"invert_local_pose={1 if args.invert_local_pose else 0}"
    )
    print(f"[transform] source={transform_meta.get('source')} meta={json.dumps(transform_meta)}")
    print("[transform] T_G_Lw=" + json.dumps(T_G_Lw.tolist()))


if __name__ == "__main__":
    main()
