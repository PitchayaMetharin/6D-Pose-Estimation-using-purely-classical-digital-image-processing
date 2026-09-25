"""Camera calibration loading and interactive capture."""

import math
import os
import time

import cv2 as cv
import numpy as np

from .camera import open_configured_camera, read_frame_with_recovery
from .settings import (
    CALIBRATION_BOARD_SIZE, CALIBRATION_DUPLICATE_ANGLE_DEG,
    CALIBRATION_DUPLICATE_CENTER_DISTANCE,
    CALIBRATION_DUPLICATE_LOG_AREA_DISTANCE, CALIBRATION_FILE,
    CALIBRATION_MAX_RMS_ERROR_PIXELS,
    CALIBRATION_MAX_VIEW_ERROR_PIXELS, CALIBRATION_REQUIRED_VIEWS,
    CALIBRATION_SQUARE_SIZE_M, CAMERA_INDEX, MILLIMETRES_PER_METRE,
)

def create_approximate_camera_calibration(frame_width, frame_height):
    focal_length = frame_width

    camera_matrix = np.array([
        [focal_length, 0, frame_width / 2],
        [0, focal_length, frame_height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    return camera_matrix, dist_coeffs


def get_calibration_path():
    if os.path.isabs(CALIBRATION_FILE):
        return CALIBRATION_FILE

    script_directory = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    return os.path.join(script_directory, CALIBRATION_FILE)


def load_camera_calibration(frame_width, frame_height):
    calibration_path = get_calibration_path()

    if not os.path.exists(calibration_path):
        print("\nWARNING:")
        print(f"{calibration_path} not found.")
        print("Using an approximate camera matrix.")
        print("6D pose will NOT be very accurate until calibration is completed.\n")

        camera_matrix, dist_coeffs = create_approximate_camera_calibration(
            frame_width,
            frame_height
        )
        return camera_matrix, dist_coeffs, False

    try:
        with np.load(calibration_path) as data:
            required_fields = {
                "camera_matrix",
                "dist_coeffs",
                "image_width",
                "image_height"
            }

            missing_fields = required_fields.difference(data.files)

            if missing_fields:
                missing = ", ".join(sorted(missing_fields))
                raise ValueError(
                    f"calibration file is missing: {missing}"
                )

            camera_matrix = np.asarray(
                data["camera_matrix"],
                dtype=np.float64
            )

            dist_coeffs = np.asarray(
                data["dist_coeffs"],
                dtype=np.float64
            )

            calibrated_width = int(
                np.asarray(data["image_width"]).item()
            )

            calibrated_height = int(
                np.asarray(data["image_height"]).item()
            )

            if (calibrated_width, calibrated_height) != (
                frame_width,
                frame_height
            ):
                raise ValueError(
                    "calibration resolution is "
                    f"{calibrated_width}x{calibrated_height}, "
                    f"but the camera frame is {frame_width}x{frame_height}"
                )

            if camera_matrix.shape != (3, 3):
                raise ValueError("camera_matrix must have shape (3, 3)")

            if dist_coeffs.size == 0:
                raise ValueError("dist_coeffs cannot be empty")

            if not np.isfinite(camera_matrix).all():
                raise ValueError("camera_matrix contains non-finite values")

            if not np.isfinite(dist_coeffs).all():
                raise ValueError("dist_coeffs contains non-finite values")

            focal_x = float(camera_matrix[0, 0])
            focal_y = float(camera_matrix[1, 1])
            principal_x = float(camera_matrix[0, 2])
            principal_y = float(camera_matrix[1, 2])

            if focal_x <= 0 or focal_y <= 0:
                raise ValueError("camera focal lengths must be positive")

            if not np.allclose(
                camera_matrix[2],
                np.array([0.0, 0.0, 1.0]),
                atol=1e-6
            ):
                raise ValueError("camera_matrix has an invalid final row")

            if not (
                -0.25 * frame_width <= principal_x <= 1.25 * frame_width
                and -0.25 * frame_height <= principal_y <= 1.25 * frame_height
            ):
                raise ValueError(
                    "camera principal point is outside a plausible range"
                )

            if "rms_error" in data.files:
                rms_error = float(np.asarray(data["rms_error"]).item())

                if not np.isfinite(rms_error):
                    raise ValueError("rms_error is not finite")

                if rms_error > 2.0:
                    raise ValueError(
                        f"rms_error is too high: {rms_error:.3f} pixels"
                    )

    except (OSError, KeyError, TypeError, ValueError) as error:
        print("\nWARNING:")
        print(f"Calibration file is not usable: {error}")
        print("Using an approximate camera matrix.")
        print("6D pose will NOT be very accurate until calibration is completed.\n")

        camera_matrix, dist_coeffs = create_approximate_camera_calibration(
            frame_width,
            frame_height
        )
        return camera_matrix, dist_coeffs, False

    print("Loaded camera calibration:")
    print(camera_matrix)
    print(dist_coeffs)

    return camera_matrix, dist_coeffs, True


# ============================================================
# CAMERA CALIBRATION CAPTURE
# ============================================================

def find_calibration_corners(gray):
    flags = (
        cv.CALIB_CB_ADAPTIVE_THRESH
        | cv.CALIB_CB_NORMALIZE_IMAGE
    )

    found, corners = cv.findChessboardCorners(
        gray,
        CALIBRATION_BOARD_SIZE,
        flags
    )

    if not found:
        return False, None

    criteria = (
        cv.TERM_CRITERIA_EPS
        + cv.TERM_CRITERIA_MAX_ITER,
        30,
        0.001
    )

    corners = cv.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        criteria
    )

    return True, corners


def calibration_view_signature(corners, image_size):
    points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    image_width, image_height = image_size
    board_columns, _ = CALIBRATION_BOARD_SIZE

    center = np.mean(points, axis=0)
    center = np.array([
        center[0] / image_width,
        center[1] / image_height
    ])

    hull = cv.convexHull(points.astype(np.float32))
    area_fraction = abs(cv.contourArea(hull)) / (
        float(image_width) * float(image_height)
    )

    first_row_end = points[board_columns - 1]
    row_vector = first_row_end - points[0]
    angle = math.degrees(math.atan2(row_vector[1], row_vector[0]))

    return {
        "center": center,
        "log_area": math.log(max(area_fraction, 1e-12)),
        "angle_deg": angle
    }


def calibration_view_is_duplicate(signature, previous_signatures):
    for previous in previous_signatures:
        center_distance = float(np.linalg.norm(
            signature["center"] - previous["center"]
        ))

        log_area_distance = abs(
            signature["log_area"] - previous["log_area"]
        )

        angle_distance = abs(
            signature["angle_deg"] - previous["angle_deg"]
        )
        angle_distance %= 180.0
        angle_distance = min(angle_distance, 180.0 - angle_distance)

        if (
            center_distance < CALIBRATION_DUPLICATE_CENTER_DISTANCE
            and log_area_distance < CALIBRATION_DUPLICATE_LOG_AREA_DISTANCE
            and angle_distance < CALIBRATION_DUPLICATE_ANGLE_DEG
        ):
            return True

    return False


def calibration_views_have_coverage(signatures):
    if len(signatures) < 10:
        return False

    centers = np.array([
        signature["center"]
        for signature in signatures
    ])

    log_areas = np.array([
        signature["log_area"]
        for signature in signatures
    ])

    return (
        float(np.ptp(centers[:, 0])) >= 0.20
        and float(np.ptp(centers[:, 1])) >= 0.20
        and float(np.ptp(log_areas)) >= math.log(1.5)
    )


def save_camera_calibration(
    object_points,
    image_points,
    image_size,
    signatures=None
):
    try:
        calibrate_extended = getattr(
            cv,
            "calibrateCameraExtended",
            None
        )

        if calibrate_extended is not None:
            calibration_result = calibrate_extended(
                object_points,
                image_points,
                image_size,
                None,
                None
            )
        else:
            calibration_result = cv.calibrateCamera(
                object_points,
                image_points,
                image_size,
                None,
                None
            )

        rms_error = calibration_result[0]
        camera_matrix = calibration_result[1]
        dist_coeffs = calibration_result[2]

        if len(calibration_result) >= 8:
            per_view_errors = np.asarray(
                calibration_result[7],
                dtype=np.float64
            ).reshape(-1)
        else:
            per_view_errors = np.array([], dtype=np.float64)
    except cv.error as error:
        print(f"Calibration failed: {error}")
        return False

    if not np.isfinite(rms_error):
        print("Calibration rejected: RMS error is not finite.")
        return False

    if rms_error > CALIBRATION_MAX_RMS_ERROR_PIXELS:
        print(
            "Calibration rejected: RMS reprojection error is "
            f"{rms_error:.3f} pixels; maximum is "
            f"{CALIBRATION_MAX_RMS_ERROR_PIXELS:.1f}."
        )
        return False

    if (
        per_view_errors.size > 0
        and np.max(per_view_errors) > CALIBRATION_MAX_VIEW_ERROR_PIXELS
    ):
        print(
            "Calibration rejected: at least one view has "
            f"more than {CALIBRATION_MAX_VIEW_ERROR_PIXELS:.1f} pixels "
            "of reprojection error."
        )
        return False

    if signatures is not None and not calibration_views_have_coverage(
        signatures
    ):
        print(
            "Calibration rejected: capture more varied board positions "
            "and distances before saving."
        )
        return False

    calibration_path = get_calibration_path()
    board_columns, board_rows = CALIBRATION_BOARD_SIZE

    np.savez(
        calibration_path,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_width=np.int32(image_size[0]),
        image_height=np.int32(image_size[1]),
        rms_error=np.float64(rms_error),
        board_columns=np.int32(board_columns),
        board_rows=np.int32(board_rows),
        square_size_m=np.float64(CALIBRATION_SQUARE_SIZE_M),
        per_view_errors=per_view_errors
    )

    print()
    print("Camera calibration saved:")
    print(f"  File: {calibration_path}")
    print(f"  Resolution: {image_size[0]}x{image_size[1]}")
    print(f"  RMS reprojection error: {rms_error:.4f} pixels")
    print()

    return True


def run_camera_calibration():
    """Capture chessboard views and save camera_calibration.npz."""

    board_columns, board_rows = CALIBRATION_BOARD_SIZE
    object_template = np.zeros(
        (board_columns * board_rows, 3),
        dtype=np.float32
    )

    object_template[:, :2] = (
        np.mgrid[
            0:board_columns,
            0:board_rows
        ].T.reshape(-1, 2)
        * CALIBRATION_SQUARE_SIZE_M
    )

    cap = open_configured_camera()

    if cap is None:
        print(
            f"ERROR: Cannot open camera index {CAMERA_INDEX}"
        )
        return False

    object_points = []
    image_points = []
    view_signatures = []
    last_capture_time = 0.0
    image_size = None
    saved = False

    print()
    print("CAMERA CALIBRATION")
    print("===================")
    print(
        "Board inner corners: "
        f"{board_columns}x{board_rows}"
    )
    print(
        "Square size: "
        f"{CALIBRATION_SQUARE_SIZE_M * MILLIMETRES_PER_METRE:.1f} mm"
    )
    print(
        f"Capture {CALIBRATION_REQUIRED_VIEWS} varied views. "
        "Press SPACE to capture."
    )
    print("Press S to save early after 10 views, or Q to cancel.")
    print()

    try:
        while True:
            cap, frame, ret = read_frame_with_recovery(cap)

            if not ret:
                print("ERROR: Cannot read camera frame")
                break

            height, width = frame.shape[:2]
            image_size = (width, height)

            gray = cv.cvtColor(
                frame,
                cv.COLOR_BGR2GRAY
            )

            found, corners = find_calibration_corners(gray)
            display = frame.copy()

            if found:
                cv.drawChessboardCorners(
                    display,
                    CALIBRATION_BOARD_SIZE,
                    corners,
                    found
                )

                status = "BOARD FOUND - press SPACE"
                status_color = (0, 255, 0)
            else:
                status = "BOARD NOT FOUND"
                status_color = (0, 0, 255)

            cv.putText(
                display,
                status,
                (20, 35),
                cv.FONT_HERSHEY_SIMPLEX,
                0.75,
                status_color,
                2
            )

            cv.putText(
                display,
                f"Views: {len(object_points)}/{CALIBRATION_REQUIRED_VIEWS}",
                (20, 70),
                cv.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            cv.imshow(
                "Camera Calibration",
                display
            )

            key = cv.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == 32 and found:
                now = time.monotonic()

                if now - last_capture_time < 0.5:
                    continue

                signature = calibration_view_signature(
                    corners,
                    image_size
                )

                if calibration_view_is_duplicate(
                    signature,
                    view_signatures
                ):
                    print("Skipped near-duplicate calibration view.")
                    last_capture_time = now
                    continue

                object_points.append(
                    object_template.copy()
                )

                image_points.append(
                    corners.copy()
                )

                view_signatures.append(signature)

                last_capture_time = now

                print(
                    f"Captured view "
                    f"{len(object_points)}/{CALIBRATION_REQUIRED_VIEWS}"
                )

                if len(object_points) >= CALIBRATION_REQUIRED_VIEWS:
                    saved = save_camera_calibration(
                        object_points,
                        image_points,
                        image_size,
                        view_signatures
                    )
                    break

            if key == ord("s"):
                if len(object_points) < 10:
                    print("Capture at least 10 views before saving.")
                    continue

                if not calibration_views_have_coverage(view_signatures):
                    print(
                        "Capture more varied board positions and distances "
                        "before saving."
                    )
                    continue

                saved = save_camera_calibration(
                    object_points,
                    image_points,
                    image_size,
                    view_signatures
                )
                break
    finally:
        if cap is not None:
            cap.release()
        cv.destroyAllWindows()

    if not saved:
        print("Calibration was not saved.")

    return saved
