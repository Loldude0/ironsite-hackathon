"""Load point clouds from common file formats produced by SLAM pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def load_point_cloud(
    filepath: str | Path,
) -> np.ndarray:
    """Load a 3-D point cloud and return an (N, 3) float64 array of XYZ coords.

    Supported formats
    -----------------
    * ``.ply``  – Stanford PLY (ASCII and binary little-endian)
    * ``.pcd``  – Point Cloud Data (ASCII and binary)
    * ``.xyz`` / ``.txt`` / ``.csv`` – Whitespace- or comma-delimited text
        with at least three numeric columns (X Y Z …).
    * ``.npy`` – NumPy binary array of shape (N, ≥3).

    Parameters
    ----------
    filepath : str or Path
        Path to the point cloud file.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 3)`` with dtype ``float64``.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the file format is not recognised or the data cannot be parsed.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Point cloud file not found: {filepath}")

    suffix = filepath.suffix.lower()

    if suffix == ".ply":
        return _load_ply(filepath)
    elif suffix == ".pcd":
        return _load_pcd(filepath)
    elif suffix in (".xyz", ".txt", ".csv"):
        return _load_text(filepath)
    elif suffix == ".npy":
        return _load_npy(filepath)
    else:
        raise ValueError(
            f"Unsupported point cloud format '{suffix}'. "
            "Supported: .ply, .pcd, .xyz, .txt, .csv, .npy"
        )


# --------------------------------------------------------------------------- #
#  Format-specific loaders                                                     #
# --------------------------------------------------------------------------- #

def _load_ply(filepath: Path) -> np.ndarray:
    """Parse a Stanford PLY file (ASCII or binary_little_endian)."""
    with open(filepath, "rb") as f:
        # --- Read header ---------------------------------------------------
        header_lines: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected end of PLY file in header.")
            decoded = line.decode("ascii", errors="replace").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break

        n_vertices = 0
        fmt = "ascii"
        prop_names: list[str] = []
        in_vertex_element = False

        for hl in header_lines:
            parts = hl.split()
            if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                n_vertices = int(parts[2])
                in_vertex_element = True
            elif parts[0] == "element" and parts[1] != "vertex":
                in_vertex_element = False
            elif parts[0] == "property" and in_vertex_element:
                prop_names.append(parts[-1])
            elif parts[0] == "format":
                fmt = parts[1]

        if n_vertices == 0:
            raise ValueError("PLY file declares 0 vertices.")

        # Determine x, y, z column indices
        try:
            xi = prop_names.index("x")
            yi = prop_names.index("y")
            zi = prop_names.index("z")
        except ValueError:
            # Fallback: use first three columns
            xi, yi, zi = 0, 1, 2

        if fmt == "ascii":
            data_offset = f.tell()
            f.seek(data_offset)
            rows = []
            for _ in range(n_vertices):
                vals = f.readline().decode("ascii").split()
                rows.append((float(vals[xi]), float(vals[yi]), float(vals[zi])))
            return np.array(rows, dtype=np.float64)

        elif fmt == "binary_little_endian":
            # Build a numpy dtype from the property declarations
            _ply_type_map = {
                "float": np.float32, "float32": np.float32,
                "double": np.float64, "float64": np.float64,
                "uchar": np.uint8, "uint8": np.uint8,
                "char": np.int8, "int8": np.int8,
                "ushort": np.uint16, "uint16": np.uint16,
                "short": np.int16, "int16": np.int16,
                "uint": np.uint32, "uint32": np.uint32,
                "int": np.int32, "int32": np.int32,
            }
            dtypes = []
            in_vertex_element = False
            for hl in header_lines:
                parts = hl.split()
                if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                    in_vertex_element = True
                elif parts[0] == "element" and parts[1] != "vertex":
                    in_vertex_element = False
                elif parts[0] == "property" and in_vertex_element and parts[1] != "list":
                    dt = _ply_type_map.get(parts[1], np.float32)
                    dtypes.append((parts[-1], dt))

            vertex_dtype = np.dtype(dtypes)
            raw = np.frombuffer(f.read(n_vertices * vertex_dtype.itemsize), dtype=vertex_dtype)
            points = np.column_stack([
                raw[prop_names[xi]].astype(np.float64),
                raw[prop_names[yi]].astype(np.float64),
                raw[prop_names[zi]].astype(np.float64),
            ])
            return points

        else:
            raise ValueError(f"Unsupported PLY format: {fmt}")


def _load_pcd(filepath: Path) -> np.ndarray:
    """Parse a PCD (Point Cloud Data) file – ASCII and binary modes."""
    with open(filepath, "rb") as f:
        header: dict[str, str] = {}
        header_size = 0
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected end of PCD file in header.")
            decoded = line.decode("ascii", errors="replace").strip()
            if decoded.startswith("DATA"):
                header["DATA"] = decoded.split()[1]
                header_size = f.tell()
                break
            key = decoded.split()[0]
            header[key] = decoded[len(key):].strip()

        fields = header.get("FIELDS", "x y z").split()
        n_points = int(header.get("POINTS", header.get("WIDTH", "0")))
        data_mode = header.get("DATA", "ascii").lower()

        try:
            xi = fields.index("x")
            yi = fields.index("y")
            zi = fields.index("z")
        except ValueError:
            xi, yi, zi = 0, 1, 2

        if data_mode == "ascii":
            rows = []
            for _ in range(n_points):
                vals = f.readline().decode("ascii").split()
                rows.append((float(vals[xi]), float(vals[yi]), float(vals[zi])))
            return np.array(rows, dtype=np.float64)

        elif data_mode == "binary":
            sizes = list(map(int, header.get("SIZE", "4 4 4").split()))
            types = header.get("TYPE", "F F F").split()
            _pcd_type_map = {
                ("F", 4): np.float32, ("F", 8): np.float64,
                ("U", 1): np.uint8, ("U", 2): np.uint16, ("U", 4): np.uint32,
                ("I", 1): np.int8, ("I", 2): np.int16, ("I", 4): np.int32,
            }
            dtypes = []
            for i, field_name in enumerate(fields):
                dt = _pcd_type_map.get((types[i], sizes[i]), np.float32)
                dtypes.append((field_name, dt))
            rec_dtype = np.dtype(dtypes)
            raw = np.frombuffer(f.read(n_points * rec_dtype.itemsize), dtype=rec_dtype)
            points = np.column_stack([
                raw[fields[xi]].astype(np.float64),
                raw[fields[yi]].astype(np.float64),
                raw[fields[zi]].astype(np.float64),
            ])
            return points

        else:
            raise ValueError(f"Unsupported PCD data mode: {data_mode}")


def _load_text(filepath: Path) -> np.ndarray:
    """Load a whitespace- or comma-separated text file (X Y Z …)."""
    # Attempt comma-separated first, fall back to whitespace
    text = filepath.read_text()
    if "," in text.split("\n", 1)[0]:
        data = np.loadtxt(filepath, delimiter=",", dtype=np.float64, comments="#")
    else:
        data = np.loadtxt(filepath, dtype=np.float64, comments="#")

    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(
            f"Text file must have at least 3 columns (X Y Z), got {data.shape[1]}."
        )
    return data[:, :3].copy()


def _load_npy(filepath: Path) -> np.ndarray:
    """Load a NumPy ``.npy`` binary file of shape (N, ≥ 3)."""
    data = np.load(filepath).astype(np.float64)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(
            f"Expected (N, >=3) array, got shape {data.shape}."
        )
    return data[:, :3].copy()
