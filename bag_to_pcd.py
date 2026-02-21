#!/usr/bin/env python3
"""Convert PointCloud2 data from a .bag file into per-frame .pcd files.

Example
-------
python bag_to_pcd.py --bag data/run.bag --output-dir data/pcd --topic /velodyne_points
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


# ROS PointField datatypes
_POINTFIELD_TO_DTYPE = {
    1: "i1",  # INT8
    2: "u1",  # UINT8
    3: "i2",  # INT16
    4: "u2",  # UINT16
    5: "i4",  # INT32
    6: "u4",  # UINT32
    7: "f4",  # FLOAT32
    8: "f8",  # FLOAT64
}


def _pointcloud2_to_xyz(msg) -> np.ndarray:
    """Decode ROS PointCloud2 message into Nx3 xyz float64 array."""
    if not msg.fields:
        return np.empty((0, 3), dtype=np.float64)

    endian = ">" if msg.is_bigendian else "<"

    names: list[str] = []
    formats: list[np.dtype] = []
    offsets: list[int] = []

    for field in msg.fields:
        base = _POINTFIELD_TO_DTYPE.get(int(field.datatype))
        if base is None:
            continue

        dtype = np.dtype(endian + base)
        if int(field.count) > 1:
            dtype = np.dtype((dtype, int(field.count)))

        names.append(str(field.name))
        formats.append(dtype)
        offsets.append(int(field.offset))

    if not names:
        return np.empty((0, 3), dtype=np.float64)

    structured_dtype = np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": int(msg.point_step),
        }
    )

    point_count = int(msg.width) * int(msg.height)
    arr = np.frombuffer(msg.data, dtype=structured_dtype, count=point_count)

    name_map = {n.lower(): n for n in arr.dtype.names or []}
    if "x" not in name_map or "y" not in name_map or "z" not in name_map:
        return np.empty((0, 3), dtype=np.float64)

    x = np.asarray(arr[name_map["x"]], dtype=np.float64).reshape(-1)
    y = np.asarray(arr[name_map["y"]], dtype=np.float64).reshape(-1)
    z = np.asarray(arr[name_map["z"]], dtype=np.float64).reshape(-1)

    xyz = np.column_stack((x, y, z))
    valid = np.isfinite(xyz).all(axis=1)
    return xyz[valid]


def _pick_pointcloud_topic(reader: AnyReader, requested_topic: str | None) -> str:
    candidates = [
        c for c in reader.connections
        if c.msgtype in ("sensor_msgs/msg/PointCloud2", "sensor_msgs/PointCloud2")
    ]

    if requested_topic:
        for c in candidates:
            if c.topic == requested_topic:
                return requested_topic
        available = sorted({c.topic for c in candidates})
        raise ValueError(f"Topic {requested_topic!r} not found. Available PointCloud2 topics: {available}")

    if not candidates:
        raise ValueError("No PointCloud2 topics found in bag.")

    return candidates[0].topic


def convert_bag_to_pcd(
    bag_path: Path,
    output_dir: Path,
    topic: str | None = None,
    every_n: int = 1,
    max_frames: int | None = None,
) -> int:
    """Convert PointCloud2 stream in bag to a folder of .pcd files."""
    if every_n < 1:
        raise ValueError("every_n must be >= 1")

    output_dir.mkdir(parents=True, exist_ok=True)

    typestore = get_typestore(Stores.ROS1_NOETIC)
    saved = 0
    seen = 0

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        selected_topic = _pick_pointcloud_topic(reader, topic)
        conns = [
            c for c in reader.connections
            if c.topic == selected_topic and c.msgtype in ("sensor_msgs/msg/PointCloud2", "sensor_msgs/PointCloud2")
        ]

        print(f"[bag] reading topic: {selected_topic}")

        for conn, timestamp, rawdata in reader.messages(connections=conns):
            msg = reader.deserialize(rawdata, conn.msgtype)
            seen += 1

            if (seen - 1) % every_n != 0:
                continue

            xyz = _pointcloud2_to_xyz(msg)
            if xyz.shape[0] == 0:
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)

            out_path = output_dir / f"frame_{saved:06d}_{int(timestamp)}.pcd"
            ok = o3d.io.write_point_cloud(str(out_path), pcd)
            if not ok:
                raise RuntimeError(f"Failed to write {out_path}")

            saved += 1
            if max_frames is not None and saved >= max_frames:
                break

    return saved


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert .bag PointCloud2 stream to multiple PCD files.")
    parser.add_argument("--bag", type=Path, required=True, help="Path to .bag file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save .pcd files")
    parser.add_argument("--topic", type=str, default=None, help="PointCloud2 topic (auto-select first if omitted)")
    parser.add_argument("--every-n", type=int, default=1, help="Save every Nth frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after saving this many frames")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    num_saved = convert_bag_to_pcd(
        bag_path=args.bag,
        output_dir=args.output_dir,
        topic=args.topic,
        every_n=args.every_n,
        max_frames=args.max_frames,
    )
    print(f"[done] saved {num_saved} PCD files to {args.output_dir}")


if __name__ == "__main__":
    main()
