from __future__ import annotations

import math
import unittest

import numpy as np

from fusion.schemas import SlamPose, UwbSample
from fusion.solver import (
    SolverConfig,
    apply_transform_to_pose,
    infer_face_anchor_yaw,
    match_samples_by_timestamp,
    run_calibration,
    solve_initial_xyzyaw_transform,
)


def _yaw_rotation(yaw_rad: float) -> np.ndarray:
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


class FusionSolverTests(unittest.TestCase):
    def _build_samples(self, with_outliers: bool = False) -> tuple[list[UwbSample], list[SlamPose]]:
        rng = np.random.default_rng(42)
        yaw = 0.6
        r = _yaw_rotation(yaw)
        t = np.array([1.2, -0.4, 0.8], dtype=np.float64)

        uwb_samples: list[UwbSample] = []
        slam_samples: list[SlamPose] = []
        for i in range(80):
            ts = 1_000_000 + i * 70
            local = np.array([0.03 * i, 0.5 * math.sin(i * 0.08), 0.02 * math.cos(i * 0.1)], dtype=np.float64)
            global_p = r @ local + t + rng.normal(0.0, 0.01, size=3)
            if with_outliers and i % 13 == 0:
                global_p += np.array([0.8, -0.6, 0.5], dtype=np.float64)

            uwb_samples.append(
                UwbSample(
                    session_id="s1",
                    anchor_id="anchor",
                    worker_id="worker",
                    ts_ms=ts,
                    distance_m=float(np.linalg.norm(global_p)),
                    direction_xyz=(0.0, 0.0, 1.0),
                    relative_xyz_m=(float(global_p[0]), float(global_p[1]), float(global_p[2])),
                    server_rx_ms=ts + int(rng.integers(-20, 20)),
                )
            )
            slam_samples.append(
                SlamPose(
                    session_id="s1",
                    node_id="worker",
                    ts_ms=ts + 10,
                    t_xyz_m=(float(local[0]), float(local[1]), float(local[2])),
                    q_xyzw=(0.0, 0.0, 0.0, 1.0),
                    tracking_state=2,
                    server_rx_ms=ts + int(rng.integers(-20, 20)),
                )
            )
        return uwb_samples, slam_samples

    def test_calibration_succeeds_with_noise(self) -> None:
        uwb, slam = self._build_samples(with_outliers=False)
        result, pairs = run_calibration(uwb, slam, config=SolverConfig())
        self.assertEqual(result["status"], "calibrated")
        self.assertGreaterEqual(result["sample_pairs"], 25)
        self.assertGreaterEqual(result["inliers"], 15)
        self.assertLess(result["rms_error_m"], 0.08)
        self.assertGreater(len(pairs), 25)

    def test_calibration_robust_to_outliers(self) -> None:
        uwb, slam = self._build_samples(with_outliers=True)
        result, _pairs = run_calibration(uwb, slam, config=SolverConfig())
        self.assertEqual(result["status"], "calibrated")
        self.assertLess(result["rms_error_m"], 0.25)

    def test_calibration_fails_for_degenerate_motion(self) -> None:
        uwb_samples: list[UwbSample] = []
        slam_samples: list[SlamPose] = []
        for i in range(40):
            ts = 2_000_000 + i * 80
            p = (1.0, 1.0, 1.0)
            uwb_samples.append(
                UwbSample(
                    session_id="s2",
                    anchor_id="anchor",
                    worker_id="worker",
                    ts_ms=ts,
                    distance_m=1.0,
                    direction_xyz=(0.0, 0.0, 1.0),
                    relative_xyz_m=p,
                    server_rx_ms=ts,
                )
            )
            slam_samples.append(
                SlamPose(
                    session_id="s2",
                    node_id="worker",
                    ts_ms=ts,
                    t_xyz_m=p,
                    q_xyzw=(0.0, 0.0, 0.0, 1.0),
                    tracking_state=2,
                    server_rx_ms=ts,
                )
            )

        result, _pairs = run_calibration(uwb_samples, slam_samples, config=SolverConfig())
        self.assertEqual(result["status"], "failed")
        self.assertIn("motion", result.get("reason", ""))

    def test_apply_transform_pose(self) -> None:
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        t_out, q_out = apply_transform_to_pose(T, (0.5, 0.0, -0.5), (0.0, 0.0, 0.0, 1.0))
        self.assertAlmostEqual(t_out[0], 1.5, places=6)
        self.assertAlmostEqual(t_out[1], 2.0, places=6)
        self.assertAlmostEqual(t_out[2], 2.5, places=6)
        self.assertAlmostEqual(q_out[3], 1.0, places=6)

    def test_timestamp_matcher_respects_skew_gate(self) -> None:
        uwb_samples = [
            UwbSample(
                session_id="s3",
                anchor_id="anchor",
                worker_id="worker",
                ts_ms=1_000,
                distance_m=1.0,
                direction_xyz=(0.0, 0.0, 1.0),
                relative_xyz_m=(0.1, 0.0, 0.0),
                server_rx_ms=1_010,
            ),
            UwbSample(
                session_id="s3",
                anchor_id="anchor",
                worker_id="worker",
                ts_ms=2_000,
                distance_m=1.0,
                direction_xyz=(0.0, 0.0, 1.0),
                relative_xyz_m=(0.2, 0.0, 0.0),
                server_rx_ms=2_200,
            ),
        ]
        slam_samples = [
            SlamPose(
                session_id="s3",
                node_id="worker",
                ts_ms=1_040,
                t_xyz_m=(0.1, 0.0, 0.0),
                q_xyzw=(0.0, 0.0, 0.0, 1.0),
                tracking_state=2,
                server_rx_ms=1_030,
            ),
            SlamPose(
                session_id="s3",
                node_id="worker",
                ts_ms=2_180,
                t_xyz_m=(0.2, 0.0, 0.0),
                q_xyzw=(0.0, 0.0, 0.0, 1.0),
                tracking_state=2,
                server_rx_ms=2_020,
            ),
        ]

        pairs = match_samples_by_timestamp(uwb_samples, slam_samples, max_skew_ms=100)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["uwb"].ts_ms, 1_000)

    def test_initial_xyzyaw_transform_bootstrap(self) -> None:
        # Local bootstrap pose: yaw=0.4rad, position=(1,2,0.3)
        c = math.cos(0.4 / 2.0)
        s = math.sin(0.4 / 2.0)
        result = solve_initial_xyzyaw_transform(
            global_position_xyz_m=(5.0, -2.0, 1.2),
            global_yaw_rad=1.1,
            local_t_xyz_m=(1.0, 2.0, 0.3),
            local_q_xyzw=(0.0, 0.0, s, c),
        )
        self.assertEqual(result["status"], "calibrated")
        T = np.array(result["T_G_Lw"], dtype=np.float64)
        t_out, _q_out = apply_transform_to_pose(T, (1.0, 2.0, 0.3), (0.0, 0.0, s, c))
        self.assertAlmostEqual(t_out[0], 5.0, places=5)
        self.assertAlmostEqual(t_out[1], -2.0, places=5)
        self.assertAlmostEqual(t_out[2], 1.2, places=5)

    def test_infer_face_anchor_yaw(self) -> None:
        yaw = infer_face_anchor_yaw((2.0, 0.0, 0.0))
        self.assertAlmostEqual(abs(yaw), math.pi, places=6)


if __name__ == "__main__":
    unittest.main()
