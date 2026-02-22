from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rotate a PLY model by 90 degrees with keyboard controls and save."
    )
    parser.add_argument("ply", type=Path, help="Input .ply file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .ply path (default: <input>_rotated.ply)",
    )
    parser.add_argument("--point-size", type=float, default=2.0, help="Render point size")
    return parser.parse_args()


class RotateModelApp:
    def __init__(self, cloud: o3d.geometry.PointCloud, output_path: Path, point_size: float) -> None:
        self.cloud = cloud
        self.output_path = output_path

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Rotate Model", width=1280, height=820)
        self.vis.add_geometry(self.cloud)

        render_opt = self.vis.get_render_option()
        render_opt.point_size = point_size
        render_opt.background_color = np.array([0.06, 0.06, 0.06])

        self._register_keys()

    def _rotate(self, axis: str, sign: int) -> bool:
        angle = sign * np.pi / 2.0
        if axis == "x":
            r = self.cloud.get_rotation_matrix_from_xyz((angle, 0.0, 0.0))
        elif axis == "y":
            r = self.cloud.get_rotation_matrix_from_xyz((0.0, angle, 0.0))
        else:
            r = self.cloud.get_rotation_matrix_from_xyz((0.0, 0.0, angle))

        self.cloud.rotate(r, center=self.cloud.get_center())
        self.vis.update_geometry(self.cloud)
        self.vis.poll_events()
        self.vis.update_renderer()
        print(f"Rotated {axis.upper()} by {90 * sign:+d}°")
        return False

    def _save(self, _vis) -> bool:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        ok = o3d.io.write_point_cloud(str(self.output_path), self.cloud)
        if ok:
            print(f"Saved: {self.output_path}")
        else:
            print(f"Failed to save: {self.output_path}")
        return False

    def _help(self, _vis) -> bool:
        print("\nControls:")
        print("  X / x : +90° / -90° around X")
        print("  Y / y : +90° / -90° around Y")
        print("  Z / z : +90° / -90° around Z")
        print("  S     : save PLY")
        print("  ?     : show help")
        print("  ESC/Q : quit")
        return False

    def _register_keys(self) -> None:
        self.vis.register_key_callback(ord("X"), lambda v: self._rotate("x", +1))
        self.vis.register_key_callback(ord("x"), lambda v: self._rotate("x", -1))
        self.vis.register_key_callback(ord("Y"), lambda v: self._rotate("y", +1))
        self.vis.register_key_callback(ord("y"), lambda v: self._rotate("y", -1))
        self.vis.register_key_callback(ord("Z"), lambda v: self._rotate("z", +1))
        self.vis.register_key_callback(ord("z"), lambda v: self._rotate("z", -1))
        self.vis.register_key_callback(ord("S"), self._save)
        self.vis.register_key_callback(ord("?"), self._help)

    def run(self) -> None:
        self._help(None)
        self.vis.run()
        self.vis.destroy_window()


def main() -> None:
    args = parse_args()
    input_path = args.ply.resolve()
    if not input_path.exists() or input_path.suffix.lower() != ".ply":
        raise SystemExit(f"Input must be an existing .ply file: {input_path}")

    output_path = args.output.resolve() if args.output else input_path.with_name(f"{input_path.stem}_rotated.ply")

    cloud = o3d.io.read_point_cloud(str(input_path))
    if cloud.is_empty():
        raise SystemExit(f"Input point cloud is empty: {input_path}")

    app = RotateModelApp(cloud=cloud, output_path=output_path, point_size=args.point_size)
    app.run()


if __name__ == "__main__":
    main()
