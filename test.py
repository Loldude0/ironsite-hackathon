from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Open a PLY, edit a red wireframe box with keys, and save the resulting PLY."
	)
	parser.add_argument("ply", type=Path, help="Input .ply file")
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Output .obj file (default: <input>_boxed.obj)",
	)
	parser.add_argument(
		"--point-size",
		type=float,
		default=2.0,
		help="Render point size",
	)
	parser.add_argument(
		"--move-step",
		type=float,
		default=0.10,
		help="Movement step for box center",
	)
	parser.add_argument(
		"--resize-step",
		type=float,
		default=0.10,
		help="Resize step (meters) for box dimensions",
	)
	parser.add_argument(
		"--obj-max-point-cubes",
		type=int,
		default=50000,
		help="Max number of points exported as tiny cubes in OBJ (for visibility).",
	)
	parser.add_argument(
		"--obj-point-cube-size",
		type=float,
		default=0.0,
		help="Cube size for OBJ points (0 = auto from cloud scale).",
	)
	return parser.parse_args()


class BoxEditor:
	def __init__(
		self,
		cloud: o3d.geometry.PointCloud,
		output_path: Path,
		point_size: float,
		move_step: float,
		resize_step: float,
		obj_max_point_cubes: int,
		obj_point_cube_size: float,
	) -> None:
		self.cloud = cloud
		self.output_path = output_path
		self.move_step = move_step
		self.resize_step = resize_step
		self.obj_max_point_cubes = max(1, int(obj_max_point_cubes))
		self.obj_point_cube_size = float(obj_point_cube_size)

		aabb = self.cloud.get_axis_aligned_bounding_box()
		init_center = np.asarray(aabb.get_center(), dtype=np.float64)
		init_extent = np.asarray(aabb.get_extent(), dtype=np.float64)
		init_extent = np.maximum(init_extent * 0.25, np.array([0.25, 0.25, 0.25], dtype=np.float64))
		self.default_center = init_center
		self.default_extent = init_extent

		self.box_centers: list[np.ndarray] = []
		self.box_extents: list[np.ndarray] = []
		self.box_lines: list[o3d.geometry.LineSet] = []
		self.active_box_index = -1

		self.vis = o3d.visualization.VisualizerWithKeyCallback()
		self.vis.create_window("PLY Box Editor", width=1400, height=900)

		self._make_points_white()
		self.vis.add_geometry(self.cloud)
		self._refresh_boxes()

		render_opt = self.vis.get_render_option()
		render_opt.point_size = point_size
		render_opt.background_color = np.array([0.06, 0.06, 0.06])

		self._register_keys()

	def _make_points_white(self) -> None:
		n = np.asarray(self.cloud.points).shape[0]
		colors = np.ones((n, 3), dtype=np.float64)
		self.cloud.colors = o3d.utility.Vector3dVector(colors)

	def _build_wire_box(self, center: np.ndarray, extent: np.ndarray, active: bool) -> o3d.geometry.LineSet:
		pts, lines = self._box_points_and_lines(center, extent)

		ls = o3d.geometry.LineSet(
			points=o3d.utility.Vector3dVector(pts),
			lines=o3d.utility.Vector2iVector(lines),
		)
		if active:
			color = np.array([[1.0, 0.0, 0.0]])
		else:
			color = np.array([[0.6, 0.1, 0.1]])
		ls.colors = o3d.utility.Vector3dVector(np.tile(color, (lines.shape[0], 1)))
		return ls

	def _box_points_and_lines(self, center: np.ndarray, extent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		half = extent / 2.0
		cx, cy, cz = center
		hx, hy, hz = half

		pts = np.array(
			[
				[cx - hx, cy - hy, cz - hz],
				[cx + hx, cy - hy, cz - hz],
				[cx + hx, cy + hy, cz - hz],
				[cx - hx, cy + hy, cz - hz],
				[cx - hx, cy - hy, cz + hz],
				[cx + hx, cy - hy, cz + hz],
				[cx + hx, cy + hy, cz + hz],
				[cx - hx, cy + hy, cz + hz],
			],
			dtype=np.float64,
		)

		lines = np.array(
			[
				[0, 1], [1, 2], [2, 3], [3, 0],
				[4, 5], [5, 6], [6, 7], [7, 4],
				[0, 4], [1, 5], [2, 6], [3, 7],
			],
			dtype=np.int32,
		)

		return pts, lines

	def _refresh_boxes(self) -> None:
		for ls in self.box_lines:
			self.vis.remove_geometry(ls, reset_bounding_box=False)

		self.box_lines = []
		for i, (center, extent) in enumerate(zip(self.box_centers, self.box_extents)):
			ls = self._build_wire_box(center, extent, active=(i == self.active_box_index))
			self.box_lines.append(ls)
			self.vis.add_geometry(ls, reset_bounding_box=False)

		self.vis.poll_events()
		self.vis.update_renderer()
		self._print_active_box_status()

	def _print_active_box_status(self) -> None:
		if self.active_box_index < 0 or not self.box_centers:
			print("No active box. Press N to add a box.")
			return
		c = self.box_centers[self.active_box_index]
		e = self.box_extents[self.active_box_index]
		print(
			f"Active box {self.active_box_index + 1}/{len(self.box_centers)} "
			f"center={c.round(3).tolist()} extent={e.round(3).tolist()}"
		)

	def _add_box(self, _vis) -> bool:
		if self.active_box_index >= 0 and self.box_centers:
			new_center = np.array(self.box_centers[self.active_box_index], dtype=np.float64)
			new_center += np.array([self.move_step, self.move_step, 0.0], dtype=np.float64)
			new_extent = np.array(self.box_extents[self.active_box_index], dtype=np.float64)
		else:
			new_center = np.array(self.default_center, dtype=np.float64)
			new_extent = np.array(self.default_extent, dtype=np.float64)
		self.box_centers.append(new_center)
		self.box_extents.append(new_extent)
		self.active_box_index = len(self.box_centers) - 1
		self._refresh_boxes()
		return False

	def _cycle_active_box(self, _vis) -> bool:
		if not self.box_centers:
			print("No boxes to cycle. Press N to add one.")
			return False
		self.active_box_index = (self.active_box_index + 1) % len(self.box_centers)
		self._refresh_boxes()
		return False

	def _move(self, dx: float, dy: float, dz: float) -> bool:
		if self.active_box_index < 0 or not self.box_centers:
			return False
		self.box_centers[self.active_box_index] += np.array([dx, dy, dz], dtype=np.float64)
		self._refresh_boxes()
		return False

	def _resize(self, axis: int, delta: float) -> bool:
		if self.active_box_index < 0 or not self.box_extents:
			return False
		extent = self.box_extents[self.active_box_index]
		extent[axis] = max(0.01, extent[axis] + delta)
		self._refresh_boxes()
		return False

	def _save(self, _vis) -> bool:
		self.output_path.parent.mkdir(parents=True, exist_ok=True)
		ok = self._save_obj(self.output_path)
		if ok:
			print(f"Saved OBJ (points + {len(self.box_centers)} wireframe box(es)): {self.output_path}")
			for i, (center, extent) in enumerate(zip(self.box_centers, self.box_extents), start=1):
				print(f"Box {i}: center={center.tolist()} extent={extent.tolist()}")
		else:
			print(f"Failed to save OBJ: {self.output_path}")
		return False

	def _save_obj(self, obj_path: Path) -> bool:
		try:
			pts = np.asarray(self.cloud.points)
			if pts.size == 0:
				return False

			# Export cloud as tiny cubes so most OBJ viewers can display it.
			if pts.shape[0] > self.obj_max_point_cubes:
				idx = np.linspace(0, pts.shape[0] - 1, self.obj_max_point_cubes, dtype=np.int64)
				pts_for_obj = pts[idx]
			else:
				pts_for_obj = pts

			if self.obj_point_cube_size > 0:
				cube_size = self.obj_point_cube_size
			else:
				aabb = self.cloud.get_axis_aligned_bounding_box()
				diag = float(np.linalg.norm(np.asarray(aabb.get_extent(), dtype=np.float64)))
				cube_size = max(diag * 0.002, 1e-4)

			h = cube_size / 2.0
			cube_offsets = np.array(
				[
					[-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
					[-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
				],
				dtype=np.float64,
			)
			cube_faces = np.array(
				[
					[1, 2, 3], [1, 3, 4],
					[5, 6, 7], [5, 7, 8],
					[1, 2, 6], [1, 6, 5],
					[2, 3, 7], [2, 7, 6],
					[3, 4, 8], [3, 8, 7],
					[4, 1, 5], [4, 5, 8],
				],
				dtype=np.int32,
			)

			with obj_path.open("w", encoding="utf-8") as f:
				f.write("# Exported by test.py\n")
				f.write("# White point cloud (as tiny cubes) + red wireframe boxes\n")

				f.write("o point_cloud_cubes\n")
				vertex_count = 0
				for p in pts_for_obj:
					cube_vs = p.reshape(1, 3) + cube_offsets
					for v in cube_vs:
						f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} 1.0 1.0 1.0\n")
					base = vertex_count
					for tri in cube_faces:
						i, j, k = tri + base
						f.write(f"f {int(i)} {int(j)} {int(k)}\n")
					vertex_count += 8

				for box_i, (center, extent) in enumerate(zip(self.box_centers, self.box_extents), start=1):
					box_pts, box_lines = self._box_points_and_lines(center, extent)
					f.write(f"o wire_box_{box_i}\n")
					for p in box_pts:
						f.write(f"v {p[0]:.8f} {p[1]:.8f} {p[2]:.8f} 1.0 0.0 0.0\n")

					# OBJ indexing is 1-based.
					for i0, i1 in box_lines:
						f.write(f"l {vertex_count + int(i0) + 1} {vertex_count + int(i1) + 1}\n")
					vertex_count += box_pts.shape[0]

			return True
		except OSError:
			return False

	def _print_help(self, _vis) -> bool:
		print("\nControls")
		print(" Box selection:")
		print("  N         : add a new box and make it active")
		print("  TAB       : cycle active box")
		print(" Movement:")
		print("  W/A/S/D   : move ACTIVE box in XY plane")
		print("  SPACE     : move +Z")
		print("  SHIFT     : move -Z (also Q as fallback)")
		print(" Resize:")
		print("  I / K     : ACTIVE box X longer / shorter")
		print("  O / L     : ACTIVE box Y longer / shorter")
		print("  P / ;     : ACTIVE box Z longer / shorter")
		print(" Save:")
		print("  ENTER     : save resulting OBJ (white points as tiny cubes + wireframe box)")
		print("  ?         : help")
		print("  ESC / Q   : quit")
		return False

	def _register_keys(self) -> None:
		# Movement in XY
		self.vis.register_key_callback(ord("W"), lambda v: self._move(0.0, +self.move_step, 0.0))
		self.vis.register_key_callback(ord("S"), lambda v: self._move(0.0, -self.move_step, 0.0))
		self.vis.register_key_callback(ord("A"), lambda v: self._move(-self.move_step, 0.0, 0.0))
		self.vis.register_key_callback(ord("D"), lambda v: self._move(+self.move_step, 0.0, 0.0))

		# Z movement: space up, shift down (glfw key code 340), Q fallback down.
		self.vis.register_key_callback(32, lambda v: self._move(0.0, 0.0, +self.move_step))
		self.vis.register_key_callback(340, lambda v: self._move(0.0, 0.0, -self.move_step))
		self.vis.register_key_callback(ord("Q"), lambda v: self._move(0.0, 0.0, -self.move_step))

		# Resize axes
		self.vis.register_key_callback(ord("I"), lambda v: self._resize(0, +self.resize_step))
		self.vis.register_key_callback(ord("K"), lambda v: self._resize(0, -self.resize_step))
		self.vis.register_key_callback(ord("O"), lambda v: self._resize(1, +self.resize_step))
		self.vis.register_key_callback(ord("L"), lambda v: self._resize(1, -self.resize_step))
		self.vis.register_key_callback(ord("P"), lambda v: self._resize(2, +self.resize_step))
		self.vis.register_key_callback(59, lambda v: self._resize(2, -self.resize_step))  # ';'

		# Multi-box controls
		self.vis.register_key_callback(ord("N"), self._add_box)
		self.vis.register_key_callback(258, self._cycle_active_box)  # TAB

		# Save/help
		self.vis.register_key_callback(257, self._save)  # ENTER
		self.vis.register_key_callback(ord("?"), self._print_help)

	def run(self) -> None:
		self._print_help(None)
		self.vis.run()
		self.vis.destroy_window()


def main() -> None:
	args = parse_args()
	in_path = args.ply.resolve()
	if not in_path.exists() or in_path.suffix.lower() != ".ply":
		raise SystemExit(f"Input must be an existing .ply file: {in_path}")

	out_path = args.output.resolve() if args.output else in_path.with_name(f"{in_path.stem}_boxed.obj")
	if out_path.suffix.lower() != ".obj":
		out_path = out_path.with_suffix(".obj")

	cloud = o3d.io.read_point_cloud(str(in_path))
	if cloud.is_empty():
		raise SystemExit(f"Point cloud is empty: {in_path}")

	editor = BoxEditor(
		cloud=cloud,
		output_path=out_path,
		point_size=args.point_size,
		move_step=args.move_step,
		resize_step=args.resize_step,
		obj_max_point_cubes=args.obj_max_point_cubes,
		obj_point_cube_size=args.obj_point_cube_size,
	)
	editor.run()


if __name__ == "__main__":
	main()
