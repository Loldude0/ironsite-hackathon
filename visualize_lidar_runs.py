from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import open3d as o3d


@dataclass
class RunCloud:
	name: str
	path: Path
	cloud: o3d.geometry.PointCloud


def _trim_cloud_for_bbox(cloud: o3d.geometry.PointCloud, trim_percent: float) -> o3d.geometry.PointCloud:
	"""Trim outliers by removing trim_percent on both +/- ends of x,y,z."""
	if trim_percent <= 0:
		return cloud

	pts = np.asarray(cloud.points)
	if pts.shape[0] < 32:
		return cloud

	q_low = trim_percent / 100.0
	q_high = 1.0 - q_low
	if q_low <= 0.0 or q_high >= 1.0 or q_low >= q_high:
		return cloud

	low = np.quantile(pts, q_low, axis=0)
	high = np.quantile(pts, q_high, axis=0)
	mask = np.all((pts >= low) & (pts <= high), axis=1)
	idx = np.where(mask)[0]

	# Keep robustness: if trimming is too aggressive, fall back to original cloud.
	if idx.size < 16:
		return cloud

	return cloud.select_by_index(idx.tolist())


def _bbox_align_transform(
	source: o3d.geometry.PointCloud,
	target: o3d.geometry.PointCloud,
	trim_percent: float = 0.0,
) -> np.ndarray:
	"""Return 4x4 transform that aligns source center position to target center position."""
	source_for_bbox = _trim_cloud_for_bbox(source, trim_percent=trim_percent)
	target_for_bbox = _trim_cloud_for_bbox(target, trim_percent=trim_percent)

	obb_s = source_for_bbox.get_oriented_bounding_box()
	obb_t = target_for_bbox.get_oriented_bounding_box()

	c_s = np.asarray(obb_s.center, dtype=np.float64)
	c_t = np.asarray(obb_t.center, dtype=np.float64)
	trans = c_t - c_s

	t = np.eye(4, dtype=np.float64)
	t[:3, 3] = trans
	return t


def _align_runs_by_bbox(runs: List[RunCloud], trim_percent: float = 0.0) -> List[RunCloud]:
	"""Align all runs to run[0] using center-position matching only."""
	if len(runs) <= 1:
		return runs

	ref = runs[0]
	aligned: List[RunCloud] = [ref]

	for run in runs[1:]:
		cloud = copy.deepcopy(run.cloud)
		tf = _bbox_align_transform(cloud, ref.cloud, trim_percent=trim_percent)
		cloud.transform(tf)
		aligned.append(RunCloud(name=run.name, path=run.path, cloud=cloud))

	return aligned


def _discover_cloud_paths(workspace_root: Path, user_glob: str | None) -> List[Path]:
	"""Find candidate map_points.ply files in the workspace."""
	if user_glob:
		paths = sorted(workspace_root.glob(user_glob))
	else:
		# Covers common folder shapes like:
		# outputs1chair/outputs1chair/map_points.ply
		# outputs2chairs/output2chairs/map_points.ply
		patterns = [
			"outputs*/**/map_points.ply",
			"output*/**/map_points.ply",
			"**/map_points.ply",
		]
		found = set()
		for pattern in patterns:
			for p in workspace_root.glob(pattern):
				found.add(p)
		paths = sorted(found)

	return [p for p in paths if p.is_file()]


def _load_cloud(path: Path, voxel_size: float) -> o3d.geometry.PointCloud:
	cloud = o3d.io.read_point_cloud(str(path))
	if cloud.is_empty():
		raise ValueError(f"Empty point cloud: {path}")

	if voxel_size > 0:
		cloud = cloud.voxel_down_sample(voxel_size=voxel_size)

	return cloud


def _colorize_cloud(cloud: o3d.geometry.PointCloud, color_rgb: np.ndarray) -> o3d.geometry.PointCloud:
	c = copy.deepcopy(cloud)
	n = np.asarray(c.points).shape[0]
	colors = np.tile(color_rgb.reshape(1, 3), (n, 1))
	c.colors = o3d.utility.Vector3dVector(colors)
	return c


def _compute_scene_center_and_diag(runs: List[RunCloud]) -> tuple[np.ndarray, float]:
	mins: List[np.ndarray] = []
	maxs: List[np.ndarray] = []
	for run in runs:
		pts = np.asarray(run.cloud.points)
		if pts.shape[0] == 0:
			continue
		mins.append(np.min(pts, axis=0))
		maxs.append(np.max(pts, axis=0))

	if not mins:
		return np.zeros(3, dtype=np.float64), 10.0

	bmin = np.min(np.vstack(mins), axis=0)
	bmax = np.max(np.vstack(maxs), axis=0)
	center = 0.5 * (bmin + bmax)
	diag = float(np.linalg.norm(bmax - bmin))
	if diag <= 1e-6:
		diag = 10.0
	return center, diag


def _create_grid_xy(center: np.ndarray, size: float, step: float, z: float = 0.0) -> o3d.geometry.LineSet:
	half = size / 2.0
	x_min = center[0] - half
	x_max = center[0] + half
	y_min = center[1] - half
	y_max = center[1] + half

	x_vals = np.arange(x_min, x_max + step * 0.5, step)
	y_vals = np.arange(y_min, y_max + step * 0.5, step)

	pts: List[List[float]] = []
	lines: List[List[int]] = []

	for x in x_vals:
		i0 = len(pts)
		pts.append([float(x), float(y_min), float(z)])
		pts.append([float(x), float(y_max), float(z)])
		lines.append([i0, i0 + 1])

	for y in y_vals:
		i0 = len(pts)
		pts.append([float(x_min), float(y), float(z)])
		pts.append([float(x_max), float(y), float(z)])
		lines.append([i0, i0 + 1])

	grid = o3d.geometry.LineSet(
		points=o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64)),
		lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32)),
	)
	grid.colors = o3d.utility.Vector3dVector(
		np.tile(np.array([[0.35, 0.35, 0.35]], dtype=np.float64), (len(lines), 1))
	)
	return grid


class LidarRunViewer:
	def __init__(self, runs: List[RunCloud], point_size: float = 2.0) -> None:
		self.runs = runs
		self.point_size = point_size
		self.index = 0
		self.scene_center, self.scene_diag = _compute_scene_center_and_diag(runs)

		self.vis = o3d.visualization.VisualizerWithKeyCallback()
		self.vis.create_window(window_name="LiDAR Runs Overlay", width=1440, height=900)
		render_opt = self.vis.get_render_option()
		render_opt.point_size = point_size
		render_opt.background_color = np.array([0.05, 0.05, 0.05])

		grid_size = max(5.0, self.scene_diag * 1.2)
		grid_step = max(grid_size / 20.0, 0.25)
		self.grid = _create_grid_xy(
			center=self.scene_center,
			size=grid_size,
			step=grid_step,
			z=0.0,
		)
		axis_size = max(0.5, self.scene_diag * 0.12)
		self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
			size=axis_size,
			origin=self.scene_center.tolist(),
		)

		self._prepare_colored_clouds()
		self._refresh_scene(reset_camera=True)
		self._register_keys()

	def _prepare_colored_clouds(self) -> None:
		self.active_clouds: List[o3d.geometry.PointCloud] = []
		for run in self.runs:
			self.active_clouds.append(_colorize_cloud(run.cloud, np.array([1.0, 1.0, 1.0])))

	def _register_keys(self) -> None:
		# Next / previous
		self.vis.register_key_callback(ord("N"), self._next_cloud)
		self.vis.register_key_callback(ord("B"), self._prev_cloud)
		self.vis.register_key_callback(ord("L"), self._next_cloud)
		self.vis.register_key_callback(ord("H"), self._prev_cloud)
		# Quick help output
		self.vis.register_key_callback(ord("?"), self._print_help)

	def _print_help(self, _vis) -> bool:
		print("\nControls:")
		print("  N / L : next cloud")
		print("  B / H : previous cloud")
		print("  ?     : print this help")
		print("  Q/ESC : quit")
		return False

	def _update_title(self) -> None:
		run = self.runs[self.index]
		print(f"[{self.index + 1}/{len(self.runs)}] {run.name} | mode=Single | {run.path}")

	def _refresh_scene(self, reset_camera: bool = False) -> None:
		self.vis.clear_geometries()
		self.vis.add_geometry(self.grid, reset_bounding_box=False)
		self.vis.add_geometry(self.axis, reset_bounding_box=False)
		self.vis.add_geometry(self.active_clouds[self.index], reset_bounding_box=False)

		if reset_camera:
			self.vis.reset_view_point(True)

		self._update_title()
		self.vis.poll_events()
		self.vis.update_renderer()

	def _next_cloud(self, _vis) -> bool:
		self.index = (self.index + 1) % len(self.runs)
		self._refresh_scene()
		return False

	def _prev_cloud(self, _vis) -> bool:
		self.index = (self.index - 1) % len(self.runs)
		self._refresh_scene()
		return False

	def run(self) -> None:
		self._print_help(None)
		self.vis.run()
		self.vis.destroy_window()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Visualize multiple LiDAR map point clouds and step through runs while "
			"overlaying them in one view."
		)
	)
	parser.add_argument(
		"--root",
		type=Path,
		default=Path(__file__).resolve().parent,
		help="Workspace root directory to search from.",
	)
	parser.add_argument(
		"--glob",
		type=str,
		default=None,
		help=(
			"Optional glob pattern relative to --root (e.g. "
			"'outputs*chair*/**/map_points.ply')."
		),
	)
	parser.add_argument(
		"--voxel-size",
		type=float,
		default=0.0,
		help="Optional voxel down-sample size in meters (0 disables down-sampling).",
	)
	parser.add_argument(
		"--point-size",
		type=float,
		default=2.0,
		help="Rendering point size.",
	)
	parser.add_argument(
		"--no-align-bbox",
		action="store_true",
		help="Disable center-based alignment.",
	)
	parser.add_argument(
		"--trim-percent",
		type=float,
		default=2.0,
		help=(
			"Before center alignment, remove this percent of points from each +/- axis tail "
			"(per axis quantiles). Example: 2.0 removes 2%% low and 2%% high on x/y/z."
		),
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	root = args.root.resolve()

	paths = _discover_cloud_paths(root, args.glob)
	if not paths:
		raise SystemExit(
			"No map_points.ply files found. "
			"Try setting --glob 'outputs*chair*/**/map_points.ply'."
		)

	runs: List[RunCloud] = []
	for p in paths:
		cloud = _load_cloud(p, voxel_size=args.voxel_size)
		name = p.parent.name
		runs.append(RunCloud(name=name, path=p, cloud=cloud))

	if not args.no_align_bbox:
		if args.trim_percent < 0 or args.trim_percent >= 50:
			raise SystemExit("--trim-percent must be in [0, 50).")
		runs = _align_runs_by_bbox(runs, trim_percent=args.trim_percent)

	viewer = LidarRunViewer(runs, point_size=args.point_size)
	viewer.run()


if __name__ == "__main__":
	main()
