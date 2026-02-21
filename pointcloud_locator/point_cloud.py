"""Load point clouds from common file formats produced by SLAM pipelines."""

from __future__ import annotations

from pathlib import Path
import numpy as np


def load_point_cloud(
    filepath: str | Path,
) -> np.ndarray:
    """Load a 3-D point cloud and return an (N, 3) float64 array of XYZ coords.

    Supported formats
    -----------------
    * ``.ply``  – Stanford PLY (ASCII and binary little-endian)
    * ``.pcd``  – Point Cloud Data (ASCII, binary, and binary_compressed)
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
        points = _load_ply(filepath)
    elif suffix == ".pcd":
        points = _load_pcd(filepath)
    elif suffix in (".xyz", ".txt", ".csv"):
        points = _load_text(filepath)
    elif suffix == ".npy":
        points = _load_npy(filepath)
    else:
        raise ValueError(
            f"Unsupported point cloud format '{suffix}'. "
            "Supported: .ply, .pcd, .xyz, .txt, .csv, .npy"
        )

    return _drop_non_finite_points(points)


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
    """Parse a PCD (Point Cloud Data) file – ASCII, binary, and compressed."""
    with open(filepath, "rb") as f:
        header: dict[str, str] = {}
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected end of PCD file in header.")
            decoded = line.decode("ascii", errors="replace").strip()
            if decoded.startswith("DATA"):
                header["DATA"] = decoded.split()[1]
                break
            key = decoded.split()[0]
            header[key] = decoded[len(key):].strip()

        fields = header.get("FIELDS", "x y z").split()
        n_points = int(header.get("POINTS", header.get("WIDTH", "0")))
        data_mode = header.get("DATA", "ascii").lower()
        sizes = list(map(int, header.get("SIZE", "4 4 4").split()))
        types = header.get("TYPE", "F F F").split()
        counts = list(map(int, header.get("COUNT", " ".join(["1"] * len(fields))).split()))

        if not (len(fields) == len(sizes) == len(types) == len(counts)):
            raise ValueError(
                "Malformed PCD header: FIELDS/SIZE/TYPE/COUNT lengths differ."
            )

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
            _pcd_type_map = {
                ("F", 4): np.dtype("<f4"), ("F", 8): np.dtype("<f8"),
                ("U", 1): np.dtype("<u1"), ("U", 2): np.dtype("<u2"), ("U", 4): np.dtype("<u4"),
                ("I", 1): np.dtype("<i1"), ("I", 2): np.dtype("<i2"), ("I", 4): np.dtype("<i4"),
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

        elif data_mode == "binary_compressed":
            hdr = f.read(8)
            if len(hdr) != 8:
                raise ValueError("Malformed binary_compressed PCD payload header.")

            compressed_size = int(np.frombuffer(hdr[:4], dtype="<u4")[0])
            uncompressed_size = int(np.frombuffer(hdr[4:], dtype="<u4")[0])
            compressed_blob = f.read(compressed_size)
            if len(compressed_blob) != compressed_size:
                raise ValueError("Unexpected end of file in compressed PCD payload.")

            raw_blob = _lzf_decompress(compressed_blob, uncompressed_size)

            _pcd_type_map = {
                ("F", 4): np.dtype("<f4"), ("F", 8): np.dtype("<f8"),
                ("U", 1): np.dtype("<u1"), ("U", 2): np.dtype("<u2"), ("U", 4): np.dtype("<u4"),
                ("I", 1): np.dtype("<i1"), ("I", 2): np.dtype("<i2"), ("I", 4): np.dtype("<i4"),
            }

            field_offsets: list[int] = []
            offset = 0
            for i in range(len(fields)):
                field_offsets.append(offset)
                offset += n_points * sizes[i] * counts[i]

            if offset != len(raw_blob):
                raise ValueError(
                    "Decompressed binary_compressed payload size does not match header layout."
                )

            def _extract_xyz_component(field_idx: int) -> np.ndarray:
                dt = _pcd_type_map.get((types[field_idx], sizes[field_idx]))
                if dt is None:
                    raise ValueError(
                        f"Unsupported PCD field type/size for '{fields[field_idx]}': "
                        f"TYPE={types[field_idx]} SIZE={sizes[field_idx]}"
                    )

                c = counts[field_idx]
                start = field_offsets[field_idx]
                n_vals = n_points * c
                arr = np.frombuffer(raw_blob, dtype=dt, count=n_vals, offset=start)
                if c > 1:
                    arr = arr.reshape(n_points, c)[:, 0]
                return arr.astype(np.float64)

            points = np.column_stack([
                _extract_xyz_component(xi),
                _extract_xyz_component(yi),
                _extract_xyz_component(zi),
            ])
            return points

        else:
            raise ValueError(f"Unsupported PCD data mode: {data_mode}")


def _lzf_decompress(data: bytes, expected_size: int) -> bytes:
    """Decompress PCD ``binary_compressed`` payload using LZF.

    PCD stores a 32-bit compressed-size and uncompressed-size prefix,
    followed by LZF-compressed bytes.
    """
    out = bytearray(expected_size)
    ip = 0
    op = 0
    n = len(data)

    while ip < n:
        ctrl = data[ip]
        ip += 1

        if ctrl < 32:
            length = ctrl + 1
            if ip + length > n or op + length > expected_size:
                raise ValueError("Invalid LZF stream (literal run out of bounds).")
            out[op:op + length] = data[ip:ip + length]
            ip += length
            op += length
        else:
            length = ctrl >> 5
            ref = op - ((ctrl & 0x1F) << 8) - 1

            if length == 7:
                if ip >= n:
                    raise ValueError("Invalid LZF stream (missing length extension).")
                length += data[ip]
                ip += 1

            if ip >= n:
                raise ValueError("Invalid LZF stream (missing back-reference byte).")
            ref -= data[ip]
            ip += 1

            length += 2
            if ref < 0 or op + length > expected_size:
                raise ValueError("Invalid LZF stream (back-reference out of bounds).")

            for _ in range(length):
                out[op] = out[ref]
                op += 1
                ref += 1

    if op != expected_size:
        raise ValueError(
            f"Invalid LZF stream (decoded {op} bytes, expected {expected_size})."
        )

    return bytes(out)


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


def _drop_non_finite_points(points: np.ndarray) -> np.ndarray:
    """Remove rows that contain NaN/Inf values in XYZ coordinates."""
    points = np.ascontiguousarray(points[:, :3], dtype=np.float64)
    finite_mask = np.isfinite(points).all(axis=1)
    if np.all(finite_mask):
        return points

    filtered = points[finite_mask]
    if len(filtered) == 0:
        raise ValueError("Point cloud contains no finite XYZ points after filtering NaN/Inf.")
    return filtered
