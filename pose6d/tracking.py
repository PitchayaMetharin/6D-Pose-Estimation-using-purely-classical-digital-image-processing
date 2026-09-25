"""Accepted-pose smoothing, motion gates, and transactional cube tracking."""

import math
import time

import cv2 as cv
import numpy as np

from .cube_geometry import rotation_distance_radians
from .cube_pose import estimate_cube_pose_from_detection, refresh_cube_pose_geometry
from .settings import (
    CAMERA_FPS, CUBE_MAX_CENTER_SPEED_MPS, CUBE_MAX_HOLD_SECONDS,
    CUBE_MAX_ROTATION_SPEED_DEG_PER_S, CUBE_MAX_Z_SPEED_MPS,
    CUBE_POSE_TIME_CONSTANT_S,
)


def smooth_cube_pose(previous_pose, current_pose, camera_matrix, dist_coeffs, delta_time_s=1.0 / CAMERA_FPS):
    """Smooth translation and interpolate rotation on SO(3)."""

    smoothed_pose = {key: value for key, value in current_pose.items()}
    delta_time_s = max(1e-3, min(float(delta_time_s), 1.0))
    alpha = 1.0 - math.exp(-delta_time_s / CUBE_POSE_TIME_CONSTANT_S)
    previous_rotation, _ = cv.Rodrigues(previous_pose["rvec"])
    current_rotation, _ = cv.Rodrigues(current_pose["rvec"])
    relative_rotation = previous_rotation.T @ current_rotation
    relative_rvec, _ = cv.Rodrigues(relative_rotation)
    smoothed_rotation = previous_rotation @ cv.Rodrigues(alpha * relative_rvec)[0]
    smoothed_pose["rvec"] = cv.Rodrigues(smoothed_rotation)[0]
    smoothed_pose["tvec"] = (
        (1.0 - alpha) * previous_pose["center_tvec"]
        + alpha * current_pose["center_tvec"]
    )
    return refresh_cube_pose_geometry(smoothed_pose, camera_matrix, dist_coeffs)


def cube_pose_is_continuous(previous_pose, current_pose, delta_time_s=1.0 / CAMERA_FPS):
    """Reject impossible translation, depth, and rotation jumps."""

    current_center = np.asarray(current_pose["center_tvec"], dtype=np.float64).reshape(3, 1)
    if not np.isfinite(current_center).all() or float(current_center[2, 0]) <= 0:
        return False
    if previous_pose is None:
        return True
    previous_center = np.asarray(previous_pose["center_tvec"], dtype=np.float64).reshape(3, 1)
    delta_time_s = max(1e-3, min(float(delta_time_s), 1.0))
    center_speed = float(np.linalg.norm(current_center - previous_center)) / delta_time_s
    z_speed = abs(float(current_center[2, 0] - previous_center[2, 0])) / delta_time_s
    rotation_speed = math.degrees(rotation_distance_radians(current_pose["rvec"], previous_pose["rvec"])) / delta_time_s
    return (
        center_speed <= CUBE_MAX_CENTER_SPEED_MPS
        and z_speed <= CUBE_MAX_Z_SPEED_MPS
        and rotation_speed <= CUBE_MAX_ROTATION_SPEED_DEG_PER_S
    )


class CubeTracker:
    """Transactional all-view cube tracker.

    Detection and solving happen before any accepted state is changed.  A
    failed local solve therefore cannot poison the next frame; after the
    configured hold timeout the reference is cleared and the following frame
    performs global reacquisition.
    """

    def __init__(self, hold_seconds=CUBE_MAX_HOLD_SECONDS):
        self.hold_seconds = float(hold_seconds)
        self.raw_pose = None
        self.filtered_pose = None
        self.last_pose_time = None
        self.lost_since = None
        self._clock_time = None

    @property
    def trusted_pose(self):
        if self.raw_pose is None:
            return None
        if self.lost_since is None:
            return self.raw_pose
        current_time = self._clock_time if self._clock_time is not None else self._now()
        return self.raw_pose if current_time - self.lost_since <= self.hold_seconds else None

    @property
    def reference_pose(self):
        """Pose used by silhouette association while held/stale."""

        return self.trusted_pose

    @property
    def track_points(self):
        if self.raw_pose is None:
            return None
        return np.asarray(self.raw_pose.get("image_points"), dtype=np.float32).copy()

    def _now(self):
        return time.monotonic()

    def reset(self):
        self.raw_pose = None
        self.filtered_pose = None
        self.last_pose_time = None
        self.lost_since = None

    def update(self, detection, camera_matrix, dist_coeffs, timestamp=None):
        now = self._now() if timestamp is None else float(timestamp)
        self._clock_time = now
        # A timeout is a global-reset boundary.  Do this before solving so a
        # newly visible contour is never constrained by a stale orientation.
        if self.lost_since is not None and now - self.lost_since > self.hold_seconds:
            self.reset()

        previous_pose = self.raw_pose
        solve_mode = "local" if previous_pose is not None else "global"
        pose, diagnostics = estimate_cube_pose_from_detection(
            detection,
            camera_matrix,
            dist_coeffs,
            previous_pose,
            solve_mode=solve_mode,
        ) if detection is not None else (None, {
            "rejection_reason": "NO CUBE DETECTION",
            "solve_mode": solve_mode,
        })

        rejection_reason = diagnostics.get("rejection_reason") if diagnostics else None
        accepted = False
        if pose is not None:
            delta_time_s = (
                now - self.last_pose_time
                if self.last_pose_time is not None else 1.0 / CAMERA_FPS
            )
            if cube_pose_is_continuous(previous_pose, pose, delta_time_s):
                was_lost = self.lost_since is not None
                if previous_pose is not None and self.filtered_pose is not None and not was_lost:
                    next_filtered = smooth_cube_pose(
                        self.filtered_pose,
                        pose,
                        camera_matrix,
                        dist_coeffs,
                        delta_time_s,
                    )
                else:
                    next_filtered = pose
                # Commit only after all geometry and motion gates pass.
                self.raw_pose = pose
                self.filtered_pose = next_filtered
                self.last_pose_time = now
                self.lost_since = None
                accepted = True
                return {
                    "pose": self.filtered_pose,
                    "raw_pose": self.raw_pose,
                    "accepted": True,
                    "held": False,
                    "stale": False,
                    "rejection_reason": None,
                    "diagnostics": diagnostics,
                }
            rejection_reason = "MOTION"

        if self.lost_since is None and self.raw_pose is not None:
            self.lost_since = now
        if self.raw_pose is not None and self.filtered_pose is not None and self.lost_since is not None:
            elapsed = now - self.lost_since
            if elapsed <= self.hold_seconds:
                return {
                    "pose": self.filtered_pose,
                    "raw_pose": self.raw_pose,
                    "accepted": False,
                    "held": True,
                    "stale": True,
                    "held_seconds": elapsed,
                    "rejection_reason": rejection_reason,
                    "diagnostics": diagnostics,
                }
        return {
            "pose": None,
            "raw_pose": self.raw_pose,
            "accepted": accepted,
            "held": False,
            "stale": False,
            "rejection_reason": rejection_reason,
            "diagnostics": diagnostics,
        }
