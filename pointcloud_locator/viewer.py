#!/usr/bin/env python3
"""Interactive 3-D point cloud viewer for PCD/PLY/XYZ/NPY files.

Example
-------
python -m pointcloud_locator.viewer assets/sample.pcd
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .point_cloud import load_point_cloud


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


def view_point_cloud(path: Path, point_size: float = 2.0, max_points: int = 750_000) -> None:
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

    points = _maybe_downsample(points, max_points=max_points)
    colors = _color_by_height(points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    diag = float(np.linalg.norm(maxs - mins))
    axis_size = max(diag * 0.15, 1.0)
    grid_half_extent = max(diag * 0.75, 2.0)
    grid_spacing = max(grid_half_extent / 20.0, 0.25)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0.0, 0.0, 0.0])
    grid = _create_xy_grid(o3d, half_extent=grid_half_extent, spacing=grid_spacing)

    title = f"Point Cloud Viewer - {path.name} ({len(points):,} points)"

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=800)
    vis.add_geometry(pcd)
    vis.add_geometry(grid)
    vis.add_geometry(axis)

    render_opt = vis.get_render_option()
    render_opt.point_size = max(1.0, float(point_size))
    render_opt.background_color = np.array([0.05, 0.05, 0.05])

    view_ctl = vis.get_view_control()
    view_ctl.set_zoom(0.7)

    print("Controls:")
    print("  Left drag: rotate")
    print("  Right drag / Shift+Left drag: pan")
    print("  Mouse wheel: zoom")
    print("  R: reset view")
    print("  Q or Esc: quit")
    print("  XYZ axis + XY grid are shown at origin")

    vis.run()
    vis.destroy_window()


def main() -> None:
    args = _build_parser().parse_args()
    view_point_cloud(
        path=args.point_cloud,
        point_size=args.point_size,
        max_points=args.max_points,
    )


if __name__ == "__main__":
    main()
