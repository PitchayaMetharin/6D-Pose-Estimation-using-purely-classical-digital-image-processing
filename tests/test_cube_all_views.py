"""Deterministic synthetic checks for the unmarked all-view cube tracker."""

import sys
import unittest
from pathlib import Path

import cv2 as cv
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pose6d.calibration import load_camera_calibration  # noqa: E402
from pose6d.cube_detection import detect_cube, detect_cube_partial  # noqa: E402
from pose6d.cube_geometry import cube_object_points  # noqa: E402
from pose6d.cube_pose import (  # noqa: E402
    estimate_cube_pose_from_detection,
    estimate_cube_pose_from_partial_detection,
)
from pose6d.tracking import CubeTracker  # noqa: E402


class CubeAllViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.camera_matrix, cls.dist_coeffs, _ = load_camera_calibration(1280, 720)

    def render(self, rvec, depth=0.35):
        tvec = np.asarray([[0.0], [0.0], [depth]], dtype=np.float64)
        projected, _ = cv.projectPoints(
            cube_object_points(), np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            tvec, self.camera_matrix, self.dist_coeffs,
        )
        hull = cv.convexHull(projected.astype(np.float32)).reshape(-1, 2)
        frame = np.full((720, 1280, 3), 220, dtype=np.uint8)
        cv.fillConvexPoly(frame, np.rint(hull).astype(np.int32), (20, 20, 20))
        return frame

    def test_face_transition_and_multi_face_views(self):
        for degrees in (0, 5, 10, 20, 45, 90):
            detection, _ = detect_cube(
                self.render([np.deg2rad(degrees), 0, 0])
            )
            self.assertIsNotNone(detection)
            self.assertTrue({len(item["points"]) for item in detection["point_hypotheses"]} & {4, 5, 6})
            pose, diagnostics = estimate_cube_pose_from_detection(
                detection, self.camera_matrix, self.dist_coeffs
            )
            self.assertIsNotNone(pose, diagnostics)
            self.assertGreater(float(pose["center_tvec"][2, 0]), 0.0)

    def test_tracker_keeps_face_on_orientation_ambiguous(self):
        tracker = CubeTracker()
        for index, degrees in enumerate((0, 2, 4, 6, 8, 10)):
            detection, _ = detect_cube(
                self.render([0, 0, np.deg2rad(degrees)]),
                tracker.reference_pose,
            )
            result = tracker.update(
                detection, self.camera_matrix, self.dist_coeffs,
                100.0 + index / 30.0,
            )
            self.assertTrue(result["accepted"], result)
            self.assertTrue(result["pose"]["orientation_ambiguous"])

    def test_timeout_resets_global_reference(self):
        tracker = CubeTracker(hold_seconds=0.75)
        detection, _ = detect_cube(self.render([0, 0, 0]))
        first = tracker.update(detection, self.camera_matrix, self.dist_coeffs, 10.0)
        self.assertTrue(first["accepted"])
        held = tracker.update(None, self.camera_matrix, self.dist_coeffs, 10.2)
        self.assertTrue(held["held"])
        expired = tracker.update(None, self.camera_matrix, self.dist_coeffs, 11.0)
        self.assertIsNone(expired["pose"])
        self.assertIsNone(tracker.raw_pose)

    def test_partial_cube_refines_only_from_trusted_pose(self):
        detection, _ = detect_cube(self.render([0.2, 0.1, 0.3]))
        pose, diagnostics = estimate_cube_pose_from_detection(
            detection, self.camera_matrix, self.dist_coeffs,
        )
        self.assertIsNotNone(pose, diagnostics)
        partial = detect_cube_partial(self.render([0.2, 0.1, 0.3]), pose)
        self.assertIsNotNone(partial)
        self.assertGreaterEqual(partial["point_count"], 3)
        self.assertGreaterEqual(len(partial["matched_edge_ids"]), 3)
        self.assertGreaterEqual(partial["edge_coverage"], 0.35)
        self.assertTrue(partial["line_samples"])
        refined, partial_diagnostics = estimate_cube_pose_from_partial_detection(
            partial, self.camera_matrix, self.dist_coeffs, pose,
        )
        self.assertIsNotNone(refined, partial_diagnostics)
        self.assertTrue(refined["partial"])
        self.assertLessEqual(
            partial_diagnostics["independent_edge_residual_pixels"], 3.0,
        )
        self.assertIsNone(
            detect_cube_partial(self.render([0.2, 0.1, 0.3]), None)
        )

    def test_partial_clutter_requires_physical_edges(self):
        rng = np.random.default_rng(7)
        frame = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        accepts = 0
        for index in range(45):
            rvec = np.asarray([
                -0.45 + index * 0.01,
                -0.25 + index * 0.006,
                -0.35 + index * 0.012,
            ])
            tvec = np.asarray([
                [-0.18 + (index % 9) * 0.04],
                [-0.12 + (index % 5) * 0.05],
                [0.28 + (index % 7) * 0.045],
            ])
            projected, _ = cv.projectPoints(
                cube_object_points(), rvec, tvec,
                self.camera_matrix, self.dist_coeffs,
            )
            previous = {
                "projected_points": projected.reshape(8, 2),
                "rvec": rvec.reshape(3, 1),
                "center_tvec": tvec,
            }
            accepts += detect_cube_partial(frame, previous) is not None
        self.assertEqual(accepts, 0)

    def test_ball_exclusion_removes_dark_round_distractor(self):
        frame = np.full((720, 1280, 3), 220, dtype=np.uint8)
        cv.circle(frame, (640, 360), 100, (20, 20, 20), -1)
        exclusion = np.zeros((720, 1280), dtype=np.uint8)
        cv.circle(exclusion, (640, 360), 100, 255, -1)
        detection, debug = detect_cube(frame, exclusion_mask=exclusion)
        self.assertIsNone(detection)
        self.assertEqual(int(np.max(debug)), 0)

    def test_rejected_partial_does_not_poison_tracker_state(self):
        detection, _ = detect_cube(self.render([0.2, 0.1, 0.3]))
        tracker = CubeTracker()
        first = tracker.update(detection, self.camera_matrix, self.dist_coeffs, 1.0)
        self.assertTrue(first["accepted"])
        trusted = tracker.raw_pose["center_tvec"].copy()
        last_pose_time = tracker.last_pose_time
        rejected = {
            "partial": True,
            "rejection_reason": "PARTIAL EDGE SUPPORT",
        }
        result = tracker.update(
            rejected, self.camera_matrix, self.dist_coeffs, 1.1,
        )
        self.assertTrue(result["held"])
        np.testing.assert_array_equal(tracker.raw_pose["center_tvec"], trusted)
        self.assertEqual(tracker.last_pose_time, last_pose_time)
        self.assertIsNone(tracker.lost_since)
        expired = tracker.update(
            rejected, self.camera_matrix, self.dist_coeffs, 1.9,
        )
        self.assertEqual(expired["status"], "LOST")
        self.assertIsNone(tracker.raw_pose)


if __name__ == "__main__":
    unittest.main()
