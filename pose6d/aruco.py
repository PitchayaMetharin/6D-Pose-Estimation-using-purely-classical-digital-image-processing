"""ArUco detection, pose estimation, and marker matching."""

import math

import cv2 as cv
import numpy as np

from .settings import MARKER_SIZE_M

def create_aruco_detector():
    dictionary = cv.aruco.getPredefinedDictionary(
        cv.aruco.DICT_4X4_50
    )

    if hasattr(cv.aruco, "ArucoDetector"):
        parameters = cv.aruco.DetectorParameters()

        return cv.aruco.ArucoDetector(
            dictionary,
            parameters
        )

    parameters = cv.aruco.DetectorParameters_create()

    return dictionary, parameters


def detect_aruco_markers(detector, gray):
    if isinstance(detector, tuple):
        dictionary, parameters = detector
        return cv.aruco.detectMarkers(
            gray,
            dictionary,
            parameters=parameters
        )

    return detector.detectMarkers(gray)

def estimate_marker_pose(
    marker_corners,
    camera_matrix,
    dist_coeffs
):
    half = MARKER_SIZE_M / 2.0

    # Marker coordinates in metres
    #
    # (-,+) -------- (+,+)
    #   |               |
    #   |               |
    # (-,-) -------- (+,-)

    object_points = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0]
    ], dtype=np.float32)

    image_points = marker_corners.reshape(
        4,
        2
    ).astype(np.float32)

    success, rvec, tvec = cv.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv.SOLVEPNP_IPPE_SQUARE
    )

    return success, rvec, tvec


# ============================================================
# ROTATION MATRIX -> EULER ANGLES
# ============================================================

def rotation_matrix_to_euler(R):
    sy = math.sqrt(
        R[0, 0] * R[0, 0] +
        R[1, 0] * R[1, 0]
    )

    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(
            R[2, 1],
            R[2, 2]
        )

        pitch = math.atan2(
            -R[2, 0],
            sy
        )

        yaw = math.atan2(
            R[1, 0],
            R[0, 0]
        )

    else:
        roll = math.atan2(
            -R[1, 2],
            R[1, 1]
        )

        pitch = math.atan2(
            -R[2, 0],
            sy
        )

        yaw = 0

    return np.degrees([
        roll,
        pitch,
        yaw
    ])

def find_object_for_marker(marker_center, objects):
    x, y = marker_center

    candidates = []

    for obj in objects:
        result = cv.pointPolygonTest(
            obj["contour"],
            (float(x), float(y)),
            False
        )

        if result >= 0:
            candidates.append(obj)

    if not candidates:
        return None

    # Choose smallest contour containing the marker
    return min(
        candidates,
        key=lambda obj: obj["area"]
    )
