"""Ray–point-cloud intersection engine.

The core algorithm:

1. Build a ``scipy.spatial.KDTree`` over the point cloud (done once).
2. For each ray, march along the ray at step intervals and query the
   KD-tree for points within radius *r* of each sample.
3. Among all candidate points, compute the exact perpendicular distance
   to the ray and keep only those within *r*.
4. Return the candidate closest to the camera (the *first hit*).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree


class RayCaster:
    """Spatial index for efficient ray–point-cloud queries.

    Parameters
    ----------
    points : np.ndarray
        Point cloud of shape ``(N, 3)``.
    leafsize : int
        KD-tree leaf size (tune for your cloud density; default 32).
    """

    def __init__(self, points: np.ndarray, leafsize: int = 32) -> None:
        self.points = np.ascontiguousarray(points, dtype=np.float64)
        self.tree = KDTree(self.points, leafsize=leafsize)

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def cast_ray(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        radius: float,
        max_range: float = 100.0,
        step: Optional[float] = None,
    ) -> Tuple[Optional[np.ndarray], float, Optional[int]]:
        """Cast a single ray and return the first hit.

        Parameters
        ----------
        origin : np.ndarray
            Ray origin, shape ``(3,)``.
        direction : np.ndarray
            Unit direction vector, shape ``(3,)``.
        radius : float
            Maximum perpendicular distance from the ray for a point to
            count as a hit.
        max_range : float
            How far along the ray to search (world-space units).
        step : float or None
            March step size.  Defaults to ``radius`` so that the query
            spheres overlap sufficiently.

        Returns
        -------
        position : np.ndarray or None
            (x, y, z) of the hit, or *None* on miss.
        distance : float
            Distance from the origin along the ray to the hit.  ``inf``
            on miss.
        index : int or None
            Index into the original point cloud array.  *None* on miss.
        """
        if step is None:
            step = radius

        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        direction = direction / np.linalg.norm(direction)

        # Collect candidate point indices from KD-tree along the ray
        n_steps = max(1, int(np.ceil(max_range / step)))
        candidate_indices: set[int] = set()

        for i in range(n_steps):
            t = i * step
            sample = origin + t * direction
            idxs = self.tree.query_ball_point(sample, r=radius)
            candidate_indices.update(idxs)

        if not candidate_indices:
            return None, float("inf"), None

        idx_arr = np.array(sorted(candidate_indices))
        pts = self.points[idx_arr]  # (M, 3)

        # ---- exact cylinder test ----------------------------------------
        # For each candidate point p, compute:
        #   v = p - origin
        #   t_proj = dot(v, direction)            (projection along ray)
        #   perp_dist = ||v - t_proj * direction|| (perpendicular distance)
        #
        # Keep only points with perp_dist ≤ radius and t_proj ≥ 0 (in front
        # of the camera).

        v = pts - origin  # (M, 3)
        t_proj = v @ direction  # (M,)
        proj = np.outer(t_proj, direction)  # (M, 3)
        perp = v - proj  # (M, 3)
        perp_dist = np.linalg.norm(perp, axis=1)  # (M,)

        mask = (perp_dist <= radius) & (t_proj >= 0)
        if not np.any(mask):
            return None, float("inf"), None

        # Among valid hits, pick the one with the smallest t (closest)
        valid_t = t_proj[mask]
        valid_global_idx = idx_arr[mask]

        best = np.argmin(valid_t)
        hit_idx = int(valid_global_idx[best])
        hit_pos = self.points[hit_idx]
        hit_dist = float(valid_t[best])

        return hit_pos.copy(), hit_dist, hit_idx

    def cast_rays(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        radius: float,
        max_range: float = 100.0,
        step: Optional[float] = None,
    ) -> List[Tuple[Optional[np.ndarray], float, Optional[int]]]:
        """Cast multiple rays (convenience batch wrapper).

        Parameters have the same meaning as :meth:`cast_ray`, but
        *origins* and *directions* are ``(K, 3)`` arrays.
        """
        results = []
        for o, d in zip(origins, directions):
            results.append(self.cast_ray(o, d, radius, max_range, step))
        return results
