"""Interactive 6D pose-estimation application loop."""

import time

import cv2 as cv
import numpy as np

from .aruco import (
    create_aruco_detector, detect_aruco_markers, estimate_marker_pose,
    find_object_for_marker, rotation_matrix_to_euler,
)
from .calibration import load_camera_calibration
from .camera import open_configured_camera, read_frame_with_recovery
from .cube_detection import detect_cube
from .cube_pose import cube_pose_diagnostics
from .drawing import (
    cube_text_position, draw_cube_pose, draw_pose_guide,
    draw_pose_text_lines, format_cube_geometry_diagnostics,
    put_pose_text,
)
from .settings import (
    AXIS_X_COLOR, AXIS_Y_COLOR, AXIS_Z_COLOR, CAMERA_FPS,
    CAMERA_INDEX, CUBE_MAX_HOLD_SECONDS, CUBE_SIZE_M, MARKER_SIZE_M,
    MILLIMETRES_PER_METRE, POSE_TEXT_COLOR, POSE_TEXT_FONT_SCALE,
    POSE_TEXT_THICKNESS,
)
from .shapes import detect_shapes, draw_detected_object
from .tracking import CubeTracker

def main():
    # --------------------------------------------------------
    # Open external webcam
    # --------------------------------------------------------

    cap = open_configured_camera()

    if cap is None:
        print(
            f"ERROR: Cannot open camera index {CAMERA_INDEX}"
        )
        return

    # --------------------------------------------------------
    # Read first frame
    # --------------------------------------------------------

    cap, frame, ret = read_frame_with_recovery(cap)

    if not ret:
        print("ERROR: Cannot read camera frame")
        if cap is not None:
            cap.release()
        return

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    camera_matrix, dist_coeffs, calibration_loaded = (
        load_camera_calibration(
            width,
            height
        )
    )

    # --------------------------------------------------------
    # ArUco
    # --------------------------------------------------------

    aruco_detector = create_aruco_detector()

    print()
    print("======================================")
    print("6D POSE ESTIMATION")
    print("Pure OpenCV / Digital Image Processing")
    print("======================================")
    print(f"Camera index : {CAMERA_INDEX}")
    print(
        "Marker size  : "
        f"{MARKER_SIZE_M * MILLIMETRES_PER_METRE:.1f} mm"
    )
    print(
        "Cube size    : "
        f"{CUBE_SIZE_M * MILLIMETRES_PER_METRE:.1f} mm"
    )
    print(
        "Calibration  : "
        f"{'calibrated' if calibration_loaded else 'approximate'}"
    )
    print()
    print("Press Q to quit.")
    print()

    cube_tracker = CubeTracker()

    while True:
        frame_time = time.monotonic()
        cap, frame, ret = read_frame_with_recovery(cap)

        if not ret:
            print("ERROR: Camera did not recover after all retries.")
            break

        display = frame.copy()

        if not calibration_loaded:
            cv.putText(
                display,
                "UNCALIBRATED - APPROXIMATE POSE",
                (20, 35),
                cv.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

        draw_pose_guide(display)

        # ====================================================
        # 1. DETECT SHAPES
        # ====================================================

        objects, edge_image = detect_shapes(frame)

        for obj in objects:
            draw_detected_object(
                display,
                obj
            )

        # ====================================================
        # 1B. DETECT THE UNMARKED CUBE
        # ====================================================

        cube_detection, cube_mask = detect_cube(
            frame,
            cube_tracker.reference_pose
        )

        track_result = cube_tracker.update(
            cube_detection,
            camera_matrix,
            dist_coeffs,
            frame_time,
        )
        cube_pose_to_draw = track_result["pose"]
        cube_pose_diagnostics = track_result.get("diagnostics")
        cube_rejection_reason = track_result.get("rejection_reason")
        cube_text_color = (255, 0, 255)
        accepted = bool(track_result.get("accepted"))
        if accepted:
            cube_status = "CUBE"
        elif track_result.get("held"):
            held_seconds = track_result.get("held_seconds", 0.0)
            cube_status = (
                "CUBE OCCLUDED - HELD/STALE "
                f"({held_seconds:.1f}s)"
                if cube_rejection_reason is None
                else f"POSE REJECTED: {cube_rejection_reason} - HELD/STALE ({held_seconds:.1f}s)"
            )
            cube_text_color = (0, 255, 255)
        else:
            cube_status = None

        if cube_detection is not None:
            cv.drawContours(
                display,
                [cube_detection["contour"]],
                -1,
                (255, 0, 255) if accepted else (0, 0, 255),
                2,
            )

        if cube_pose_to_draw is not None:
            draw_cube_pose(
                display,
                cube_pose_to_draw,
                camera_matrix,
                dist_coeffs
            )

            center_tvec = cube_pose_to_draw["center_tvec"]
            x_mm = float(center_tvec[0][0]) * MILLIMETRES_PER_METRE
            y_mm = float(center_tvec[1][0]) * MILLIMETRES_PER_METRE
            z_mm = float(center_tvec[2][0]) * MILLIMETRES_PER_METRE
            range_mm = float(np.linalg.norm(center_tvec)) * (
                MILLIMETRES_PER_METRE
            )

            rotation_matrix, _ = cv.Rodrigues(
                cube_pose_to_draw["rvec"]
            )

            roll, pitch, yaw = (
                rotation_matrix_to_euler(
                    rotation_matrix
                )
            )

            text_x, text_y = cube_text_position(
                display,
                cube_detection["contour"]
                if cube_detection is not None else None,
                cube_pose_to_draw
            )

            if cube_pose_to_draw.get("orientation_ambiguous"):
                cube_status = "CUBE — ORIENTATION AMBIGUOUS"
                cube_text_color = (0, 191, 255)

            observation_mode = cube_pose_to_draw.get(
                "observation_mode",
                "unknown"
            ).upper()

            cube_text = [
                (cube_status, cube_text_color),
                (f"Mode:{observation_mode}", POSE_TEXT_COLOR),
                (f"X:{x_mm:+.1f} mm", AXIS_X_COLOR),
                (f"Y:{y_mm:+.1f} mm", AXIS_Y_COLOR),
                (f"Z:{z_mm:+.1f} mm", AXIS_Z_COLOR),
                (f"Range:{range_mm:.1f} mm", POSE_TEXT_COLOR),
                (f"Roll (X) relative:{roll:+.1f} deg", AXIS_X_COLOR),
                (f"Pitch (Y) relative:{pitch:+.1f} deg", AXIS_Y_COLOR),
                (f"Yaw (Z) relative:{yaw:+.1f} deg", AXIS_Z_COLOR)
            ]

            draw_pose_text_lines(
                display,
                cube_text,
                text_x,
                text_y
            )
        elif cube_detection is not None:
            cube_x, cube_y, _, _ = cv.boundingRect(
                cube_detection["contour"]
            )

            cv.putText(
                display,
                f"POSE REJECTED: {cube_rejection_reason}",
                (cube_x, max(20, cube_y - 10)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

        if cube_rejection_reason == "GEOMETRY":
            diagnostic_text = format_cube_geometry_diagnostics(
                cube_pose_diagnostics
            )

            if diagnostic_text is not None:
                put_pose_text(
                    display,
                    diagnostic_text,
                    (15, display.shape[0] - 15),
                    (0, 165, 255),
                    0.6,
                    2
                )

        # ====================================================
        # 2. DETECT ARUCO
        # ====================================================

        gray = cv.cvtColor(
            frame,
            cv.COLOR_BGR2GRAY
        )

        corners, ids, rejected = detect_aruco_markers(
            aruco_detector,
            gray
        )

        # ====================================================
        # 3. ESTIMATE 6D POSE
        # ====================================================

        if ids is not None:
            cv.aruco.drawDetectedMarkers(
                display,
                corners,
                ids
            )

            ids = np.asarray(ids).flatten()

            for i, marker_corners in enumerate(corners):
                marker_id = int(ids[i])

                print("Detected marker:", marker_id)

                success, rvec, tvec = (
                    estimate_marker_pose(
                        marker_corners,
                        camera_matrix,
                        dist_coeffs
                )
            )

                if not success:
                    continue

                # --------------------------------------------
                # Marker center
                # --------------------------------------------

                marker_points = marker_corners.reshape(
                    4,
                    2
                )

                center_x = int(
                    np.mean(marker_points[:, 0])
                )

                center_y = int(
                    np.mean(marker_points[:, 1])
                )

                # --------------------------------------------
                # Find which shape contains this marker
                # --------------------------------------------

                obj = find_object_for_marker(
                    (center_x, center_y),
                    objects
                )

                if obj is not None:
                    shape_name = obj["shape"]
                else:
                    shape_name = "UNKNOWN"

                # --------------------------------------------
                # Draw XYZ axes
                # --------------------------------------------

                cv.drawFrameAxes(
                    display,
                    camera_matrix,
                    dist_coeffs,
                    rvec,
                    tvec,
                    MARKER_SIZE_M * 0.75,
                    2
                )

                # --------------------------------------------
                # Translation
                # --------------------------------------------

                x = float(tvec[0][0])
                y = float(tvec[1][0])
                z = float(tvec[2][0])

                x_mm = x * MILLIMETRES_PER_METRE
                y_mm = y * MILLIMETRES_PER_METRE
                z_mm = z * MILLIMETRES_PER_METRE

                # --------------------------------------------
                # Rotation
                # --------------------------------------------

                rotation_matrix, _ = cv.Rodrigues(
                    rvec
                )

                roll, pitch, yaw = (
                    rotation_matrix_to_euler(
                        rotation_matrix
                    )
                )

                # --------------------------------------------
                # Display
                # --------------------------------------------

                text_x = center_x + 20
                text_y = center_y

                put_pose_text(
                    display,
                    f"ID:{marker_id} {shape_name}",
                    (text_x, text_y),
                    (255, 255, 0),
                    POSE_TEXT_FONT_SCALE,
                    POSE_TEXT_THICKNESS
                )

                draw_pose_text_lines(
                    display,
                    [
                        (f"X:{x_mm:+.1f} mm", AXIS_X_COLOR),
                        (f"Y:{y_mm:+.1f} mm", AXIS_Y_COLOR),
                        (f"Z:{z_mm:+.1f} mm", AXIS_Z_COLOR),
                        (f"Roll (X):{roll:+.1f} deg", AXIS_X_COLOR),
                        (f"Pitch (Y):{pitch:+.1f} deg", AXIS_Y_COLOR),
                        (f"Yaw (Z):{yaw:+.1f} deg", AXIS_Z_COLOR)
                    ],
                    text_x,
                    text_y + 22
                )

                # Terminal output
                print(
                    f"ID={marker_id:2d} "
                    f"Shape={shape_name:10s} | "
                    f"XYZ=({x_mm:+.1f}, {y_mm:+.1f}, {z_mm:+.1f}) mm | "
                    f"RPY=({roll:+.1f}, "
                    f"{pitch:+.1f}, "
                    f"{yaw:+.1f}) deg"
                )

        # ====================================================
        # SHOW WINDOWS
        # ====================================================

        cv.imshow(
            "6D Pose Estimation",
            display
        )

        cv.imshow(
            "DIP Edge Detection",
            edge_image
        )

        cv.imshow(
            "Cube Segmentation",
            cube_mask
        )

        key = cv.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    if cap is not None:
        cap.release()
    cv.destroyAllWindows()
