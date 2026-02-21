#!/usr/bin/env python3
"""Interactive 3-D point cloud viewer for PCD/PLY/XYZ/NPY files.

Example
-------
python -m pointcloud_locator.viewer assets/sample.pcd
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .types import BoundingBox, CameraIntrinsics
from .point_cloud import load_point_cloud


HARD_CODED_SAMPLE_BBOX = BoundingBox(
    x_center=320.0,
    y_center=240.0,
    width=180.0,
    height=120.0,
    class_name="sample_object",
    confidence=0.95,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open and explore a 3D point cloud with mouse controls.",
    )
    parser.add_argument(
        "point_cloud",
        type=Path,
        help="Path to point cloud file (.pcd, .ply, .xyz, .txt, .csv, .npy)",
    )
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=Path("assets/sample_camera_config.json"),
        help="Path to camera config json (default: assets/sample_camera_config.json)",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="Point size in the viewer (default: 2.0)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=750_000,
        help="Randomly downsample if cloud is larger than this (default: 750000)",
    )
    return parser


def _maybe_downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(42)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def _color_by_height(points: np.ndarray) -> np.ndarray:
    z = points[:, 2]
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    span = max(z_max - z_min, 1e-9)
    t = (z - z_min) / span

    # Simple blue->cyan->yellow palette for depth/height perception.
    r = np.clip(1.5 * t - 0.5, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * t - 1.0) * 1.5, 0.0, 1.0)
    b = np.clip(1.0 - 1.5 * t, 0.0, 1.0)
    return np.column_stack((r, g, b))


def _create_xy_grid(o3d, half_extent: float, spacing: float):
    """Create an XY grid centered at origin as an Open3D LineSet."""
    n = int(np.floor(half_extent / spacing))

    grid_points = []
    grid_lines = []
    grid_colors = []

    line_idx = 0

    # Lines parallel to X (vary Y)
    for i in range(-n, n + 1):
        y = i * spacing
        grid_points.append([-half_extent, y, 0.0])
        grid_points.append([half_extent, y, 0.0])
        grid_lines.append([line_idx, line_idx + 1])
        grid_colors.append([0.35, 0.35, 0.35])
        line_idx += 2

    # Lines parallel to Y (vary X)
    for i in range(-n, n + 1):
        x = i * spacing
        grid_points.append([x, -half_extent, 0.0])
        grid_points.append([x, half_extent, 0.0])
        grid_lines.append([line_idx, line_idx + 1])
        grid_colors.append([0.35, 0.35, 0.35])
        line_idx += 2

    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(np.asarray(grid_points, dtype=np.float64))
    grid.lines = o3d.utility.Vector2iVector(np.asarray(grid_lines, dtype=np.int32))
    grid.colors = o3d.utility.Vector3dVector(np.asarray(grid_colors, dtype=np.float64))
    return grid


def _create_axis_indicator(o3d, size: float, origin: np.ndarray, rotation: np.ndarray | None = None):
    """Create a thin XYZ axis indicator as colored lines."""
    base_points = np.array([
        [0.0, 0.0, 0.0],
        [size, 0.0, 0.0],
        [0.0, size, 0.0],
        [0.0, 0.0, size],
    ], dtype=np.float64)

    if rotation is not None:
        base_points = (rotation @ base_points.T).T

    base_points = base_points + origin

    lines = np.array([
        [0, 1],
        [0, 2],
        [0, 3],
    ], dtype=np.int32)
    colors = np.array([
        [1.0, 0.1, 0.1],  # X
        [0.1, 1.0, 0.1],  # Y
        [0.1, 0.4, 1.0],  # Z
    ], dtype=np.float64)

    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(base_points)
    axis.lines = o3d.utility.Vector2iVector(lines)
    axis.colors = o3d.utility.Vector3dVector(colors)
    return axis


def _euler_deg_to_matrix(euler_deg: np.ndarray) -> np.ndarray:
    euler_rad = np.deg2rad(euler_deg)
    roll, pitch, yaw = euler_rad

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _load_camera_config(config_path: Path) -> tuple[CameraIntrinsics, np.ndarray, np.ndarray, float]:
    payload: dict[str, Any] = json.loads(config_path.read_text())

    intr = payload["intrinsics"]
    intrinsics = CameraIntrinsics(
        fx=float(intr["fx"]),
        fy=float(intr["fy"]),
        cx=float(intr["cx"]),
        cy=float(intr["cy"]),
        width=int(intr["width"]),
        height=int(intr["height"]),
    )

    cam = payload["camera_pose"]
    position = np.asarray(cam.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)

    euler_deg = np.asarray(cam.get("euler_deg", [0.0, 0.0, 0.0]), dtype=np.float64)
    rotation = _euler_deg_to_matrix(euler_deg)

    plane_distance = float(payload.get("visualization", {}).get("image_plane_distance", 1.5))
    return intrinsics, position, euler_deg, plane_distance


def _save_camera_config(
    config_path: Path,
    intrinsics: CameraIntrinsics,
    camera_position: np.ndarray,
    euler_deg: np.ndarray,
    plane_distance: float,
) -> None:
    payload = {
        "intrinsics": {
            "fx": float(intrinsics.fx),
            "fy": float(intrinsics.fy),
            "cx": float(intrinsics.cx),
            "cy": float(intrinsics.cy),
            "width": int(intrinsics.width),
            "height": int(intrinsics.height),
        },
        "camera_pose": {
            "position": [float(camera_position[0]), float(camera_position[1]), float(camera_position[2])],
            "euler_deg": [float(euler_deg[0]), float(euler_deg[1]), float(euler_deg[2])],
        },
        "visualization": {
            "image_plane_distance": float(plane_distance),
        },
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload, indent=2) + "\n")


def _image_plane_and_bbox(
    o3d,
    intrinsics: CameraIntrinsics,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    bbox: BoundingBox,
    plane_distance: float,
):
    def pixel_to_world(u: float, v: float) -> np.ndarray:
        x = (u - intrinsics.cx) * plane_distance / intrinsics.fx
        y = (v - intrinsics.cy) * plane_distance / intrinsics.fy
        cam_pt = np.array([x, y, plane_distance], dtype=np.float64)
        return camera_position + camera_rotation @ cam_pt

    plane_corners_px = [
        (0.0, 0.0),
        (float(intrinsics.width), 0.0),
        (float(intrinsics.width), float(intrinsics.height)),
        (0.0, float(intrinsics.height)),
    ]
    plane_corners_w = [pixel_to_world(u, v) for u, v in plane_corners_px]

    x1 = bbox.x_center - bbox.width / 2.0
    y1 = bbox.y_center - bbox.height / 2.0
    x2 = bbox.x_center + bbox.width / 2.0
    y2 = bbox.y_center + bbox.height / 2.0

    box_corners_px = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    box_corners_w = [pixel_to_world(u, v) for u, v in box_corners_px]
    box_center_w = pixel_to_world(bbox.x_center, bbox.y_center)

    points = np.asarray(plane_corners_w + box_corners_w + [camera_position, box_center_w], dtype=np.float64)
    plane_offset = 0
    box_offset = 4
    cam_idx = 8
    center_idx = 9

    lines = np.asarray([
        [plane_offset + 0, plane_offset + 1],
        [plane_offset + 1, plane_offset + 2],
        [plane_offset + 2, plane_offset + 3],
        [plane_offset + 3, plane_offset + 0],
        [box_offset + 0, box_offset + 1],
        [box_offset + 1, box_offset + 2],
        [box_offset + 2, box_offset + 3],
        [box_offset + 3, box_offset + 0],
        [cam_idx, center_idx],
    ], dtype=np.int32)

    colors = np.asarray([
        [0.1, 0.8, 1.0],
        [0.1, 0.8, 1.0],
        [0.1, 0.8, 1.0],
        [0.1, 0.8, 1.0],
        [1.0, 0.25, 0.25],
        [1.0, 0.25, 0.25],
        [1.0, 0.25, 0.25],
        [1.0, 0.25, 0.25],
        [1.0, 0.9, 0.1],
    ], dtype=np.float64)

    overlay = o3d.geometry.LineSet()
    overlay.points = o3d.utility.Vector3dVector(points)
    overlay.lines = o3d.utility.Vector2iVector(lines)
    overlay.colors = o3d.utility.Vector3dVector(colors)

    return overlay, box_center_w


def view_point_cloud(
    path: Path,
    point_size: float = 2.0,
    max_points: int = 750_000,
    camera_config: Path = Path("assets/sample_camera_config.json"),
) -> None:
    """Open a point cloud in an interactive Open3D viewer."""
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - runtime dependency guidance
        raise RuntimeError(
            "open3d is required for interactive viewing. "
            "Install it with: pip install open3d"
        ) from exc

    points = load_point_cloud(path)
    if len(points) == 0:
        raise ValueError(f"No points found in file: {path}")

    intrinsics, camera_position, camera_euler_deg, plane_distance = _load_camera_config(camera_config)
    camera_rotation = _euler_deg_to_matrix(camera_euler_deg)

    points = _maybe_downsample(points, max_points=max_points)
    colors = _color_by_height(points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    diag = float(np.linalg.norm(maxs - mins))
    axis_size = max(diag * 0.07, 0.5)
    grid_half_extent = max(diag * 0.75, 2.0)
    grid_spacing = max(grid_half_extent / 20.0, 0.25)

    axis = _create_axis_indicator(
        o3d=o3d,
        size=axis_size,
        origin=np.array([0.0, 0.0, 0.0], dtype=np.float64),
    )
    grid = _create_xy_grid(o3d, half_extent=grid_half_extent, spacing=grid_spacing)

    dynamic_geometries: dict[str, Any] = {}

    def rebuild_camera_geometries(vis_obj) -> np.ndarray:
        nonlocal camera_rotation

        camera_rotation = _euler_deg_to_matrix(camera_euler_deg)

        for key in ("camera_axis", "camera_marker", "image_overlay"):
            if key in dynamic_geometries:
                vis_obj.remove_geometry(dynamic_geometries[key], reset_bounding_box=False)

        camera_axis = _create_axis_indicator(
            o3d=o3d,
            size=max(axis_size * 0.6, 0.35),
            origin=camera_position,
            rotation=camera_rotation,
        )

        camera_marker = o3d.geometry.TriangleMesh.create_sphere(radius=max(axis_size * 0.05, 0.07))
        camera_marker.compute_vertex_normals()
        camera_marker.paint_uniform_color([1.0, 0.65, 0.1])
        camera_marker.translate(camera_position)

        image_overlay, bbox_center_local = _image_plane_and_bbox(
            o3d=o3d,
            intrinsics=intrinsics,
            camera_position=camera_position,
            camera_rotation=camera_rotation,
            bbox=HARD_CODED_SAMPLE_BBOX,
            plane_distance=plane_distance,
        )

        dynamic_geometries["camera_axis"] = camera_axis
        dynamic_geometries["camera_marker"] = camera_marker
        dynamic_geometries["image_overlay"] = image_overlay

        vis_obj.add_geometry(camera_axis, reset_bounding_box=False)
        vis_obj.add_geometry(camera_marker, reset_bounding_box=False)
        vis_obj.add_geometry(image_overlay, reset_bounding_box=False)
        vis_obj.update_renderer()
        return bbox_center_local

    title = f"Point Cloud Viewer - {path.name} ({len(points):,} points)"

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=title, width=1280, height=800)
    vis.add_geometry(pcd)
    vis.add_geometry(grid)
    vis.add_geometry(axis)
    bbox_center = rebuild_camera_geometries(vis)

    # --- camera param edit state ---------------------------------------
    step_pos = max(diag * 0.02, 0.05)
    step_ang_deg = 2.0
    step_focal = 10.0
    step_plane = 0.05

    def _print_camera_state(prefix: str) -> None:
        print(
            f"{prefix} pos=({camera_position[0]:+.3f}, {camera_position[1]:+.3f}, {camera_position[2]:+.3f}) "
            f"rpy_deg=({camera_euler_deg[0]:+.1f}, {camera_euler_deg[1]:+.1f}, {camera_euler_deg[2]:+.1f}) "
            f"fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f} plane_d={plane_distance:.2f}"
        )

    def _apply_and_refresh(prefix: str) -> None:
        nonlocal bbox_center
        bbox_center = rebuild_camera_geometries(vis)
        # _print_camera_state(prefix)

    def _move_local(dx: float, dy: float, dz: float) -> None:
        nonlocal camera_position
        rot = _euler_deg_to_matrix(camera_euler_deg)
        delta_world = rot @ np.array([dx, dy, dz], dtype=np.float64)
        camera_position = camera_position + delta_world

    # Open3D key callbacks use ASCII integer key codes.
    vis.register_key_callback(ord("W"), lambda _v: (_move_local(0.0, 0.0, +step_pos), _apply_and_refresh("[cam]"), False)[-1])
    vis.register_key_callback(ord("S"), lambda _v: (_move_local(0.0, 0.0, -step_pos), _apply_and_refresh("[cam]"), False)[-1])
    vis.register_key_callback(ord("A"), lambda _v: (_move_local(-step_pos, 0.0, 0.0), _apply_and_refresh("[cam]"), False)[-1])
    vis.register_key_callback(ord("D"), lambda _v: (_move_local(+step_pos, 0.0, 0.0), _apply_and_refresh("[cam]"), False)[-1])
    vis.register_key_callback(ord("T"), lambda _v: (_move_local(0.0, +step_pos, 0.0), _apply_and_refresh("[cam]"), False)[-1])
    vis.register_key_callback(ord("G"), lambda _v: (_move_local(0.0, -step_pos, 0.0), _apply_and_refresh("[cam]"), False)[-1])

    vis.register_key_callback(ord("I"), lambda _v: (camera_euler_deg.__setitem__(1, camera_euler_deg[1] + step_ang_deg), _apply_and_refresh("[cam]"), False)[-1])
    vis.register_key_callback(ord("K"), lambda _v: (camera_euler_deg.__setitem__(1, camera_euler_deg[1] - step_ang_deg), _apply_and_refresh("[cam]"), False)[-1])
    vis.register_key_callback(ord("J"), lambda _v: (camera_euler_deg.__setitem__(2, camera_euler_deg[2] + step_ang_deg), _apply_and_refresh("[cam]"), False)[-1])
    vis.register_key_callback(ord("L"), lambda _v: (camera_euler_deg.__setitem__(2, camera_euler_deg[2] - step_ang_deg), _apply_and_refresh("[cam]"), False)[-1])
    vis.register_key_callback(ord("U"), lambda _v: (camera_euler_deg.__setitem__(0, camera_euler_deg[0] + step_ang_deg), _apply_and_refresh("[cam]"), False)[-1])
    vis.register_key_callback(ord("O"), lambda _v: (camera_euler_deg.__setitem__(0, camera_euler_deg[0] - step_ang_deg), _apply_and_refresh("[cam]"), False)[-1])

    vis.register_key_callback(ord("Z"), lambda _v: (setattr(intrinsics, "fx", max(50.0, intrinsics.fx - step_focal)), setattr(intrinsics, "fy", max(50.0, intrinsics.fy - step_focal)), _apply_and_refresh("[intrinsics]"), False)[-1])
    vis.register_key_callback(ord("X"), lambda _v: (setattr(intrinsics, "fx", intrinsics.fx + step_focal), setattr(intrinsics, "fy", intrinsics.fy + step_focal), _apply_and_refresh("[intrinsics]"), False)[-1])
    vis.register_key_callback(ord("H"), lambda _v: (_print_camera_state("[camera-state]"), False)[-1])

    def _plane_minus(_v):
        nonlocal plane_distance
        plane_distance = max(0.05, plane_distance - step_plane)
        _apply_and_refresh("[plane]")
        return False

    def _plane_plus(_v):
        nonlocal plane_distance
        plane_distance = plane_distance + step_plane
        _apply_and_refresh("[plane]")
        return False

    vis.register_key_callback(ord("["), _plane_minus)
    vis.register_key_callback(ord("]"), _plane_plus)

    def _save_cfg(_v):
        _save_camera_config(
            config_path=camera_config,
            intrinsics=intrinsics,
            camera_position=camera_position,
            euler_deg=camera_euler_deg,
            plane_distance=plane_distance,
        )
        print(f"[saved] camera config -> {camera_config}")
        return False

    vis.register_key_callback(ord("P"), _save_cfg)

    render_opt = vis.get_render_option()
    render_opt.point_size = max(1.0, float(point_size))
    render_opt.background_color = np.array([0.05, 0.05, 0.05])

    view_ctl = vis.get_view_control()
    view_ctl.set_zoom(0.7)

    print("Controls:")
    print("  Left drag: rotate")
    print("  Right drag / Shift+Left drag: pan")
    print("  Mouse wheel: zoom")
    print("  R: reset view (viewer camera)")
    print("  Q or Esc: quit")
    print("  XYZ axis + XY grid are shown at origin")
    print(f"  Camera config: {camera_config}")
    print("  Edit camera params:")
    print("    W/S A/D T/G : move +Z/-Z, -X/+X, +Y/-Y in camera local frame")
    print("    I/K J/L U/O : pitch+/-, yaw+/-, roll+/- (deg)")
    print("    Z/X         : decrease/increase fx, fy")
    print("    [ / ]       : move image plane nearer/farther")
    print("    H           : print current camera parameters")
    print("    P           : save current camera params to config file")
    print(f"  Camera world position: ({camera_position[0]:+.3f}, {camera_position[1]:+.3f}, {camera_position[2]:+.3f})")
    print(
        "  Hard-coded bbox on image plane: "
        f"{HARD_CODED_SAMPLE_BBOX.class_name} "
        f"(xc={HARD_CODED_SAMPLE_BBOX.x_center:.1f}, yc={HARD_CODED_SAMPLE_BBOX.y_center:.1f}, "
        f"w={HARD_CODED_SAMPLE_BBOX.width:.1f}, h={HARD_CODED_SAMPLE_BBOX.height:.1f})"
    )
    print(f"  BBox center (world on floating plane): ({bbox_center[0]:+.3f}, {bbox_center[1]:+.3f}, {bbox_center[2]:+.3f})")

    vis.run()
    vis.destroy_window()


def main() -> None:
    args = _build_parser().parse_args()
    view_point_cloud(
        path=args.point_cloud,
        point_size=args.point_size,
        max_points=args.max_points,
        camera_config=args.camera_config,
    )


if __name__ == "__main__":
    main()
