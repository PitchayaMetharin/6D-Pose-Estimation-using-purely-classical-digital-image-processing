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
from .ball_detection import (
    claimed_contour, detect_balls_detailed, load_ball_color_profiles,
    reset_ball_color_profile, sample_ball_color_profile,
    save_ball_color_profiles,
)
from .cube_detection import detect_cube, detect_cube_partial
from .cube_pose import cube_pose_diagnostics
from .drawing import (
    ball_text_lines, cube_text_position, draw_ball_pose, draw_cube_pose, draw_pose_guide,
    draw_pose_text_lines, format_cube_geometry_diagnostics,
    put_pose_text,
)
from .settings import (
    AXIS_X_COLOR, AXIS_Y_COLOR, AXIS_Z_COLOR, CAMERA_FPS,
    BALL_PROFILE_IDS, CAMERA_INDEX,
    CUBE_MAX_HOLD_SECONDS, CUBE_SIZE_M, HANDHELD_TRACKING_ENABLED, MARKER_SIZE_M,
    MILLIMETRES_PER_METRE, POSE_TEXT_COLOR, POSE_TEXT_FONT_SCALE,
    POSE_TEXT_THICKNESS,
)
from .shapes import detect_shapes, draw_detected_object
from .tracking import BallTracker, CubeTracker

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
    profiles = load_ball_color_profiles()
    ball_trackers = {
        profile_id: BallTracker(
            profile_id,
            profiles[profile_id].radius_m,
            enabled=profiles[profile_id].enabled,
        )
        for profile_id in BALL_PROFILE_IDS
    }
    selected_profile_id = BALL_PROFILE_IDS[0]
    handheld_enabled = bool(HANDHELD_TRACKING_ENABLED)
    hsv_for_click = None

    def on_mouse(event, x, y, _flags, _userdata):
        if event != cv.EVENT_LBUTTONDOWN or hsv_for_click is None:
            return
        try:
            current = profiles[selected_profile_id]
            profiles[selected_profile_id] = sample_ball_color_profile(
                hsv_for_click,
                (x, y),
                selected_profile_id,
                radius_m=current.radius_m,
            )
            ball_trackers[selected_profile_id].radius_m = profiles[selected_profile_id].radius_m
            ball_trackers[selected_profile_id].enabled = True
            # A new colour profile is a new identity observation.  Clear only
            # that identity so the other ball keeps its trusted pose.
            ball_trackers[selected_profile_id].reset()
            print(f"Calibrated {selected_profile_id} from HSV patch at ({x}, {y}). Press S to save.")
        except ValueError as error:
            print(f"Ball color sample rejected: {error}")

    cv.namedWindow("6D Pose Estimation")
    cv.setMouseCallback("6D Pose Estimation", on_mouse)

    print("Controls:")
    print("  1          Select the small_ball colour profile.")
    print("  2          Select the large_ball colour profile.")
    print("  Left click Calibrate the selected profile from a solid ball area.")
    print("  S          Save both ball colour profiles.")
    print("  R          Reset and disable the selected ball profile.")
    print("  H          Toggle handheld partial tracking.")
    print("  Q          Quit the application.")
    print("Ball profiles: " + ", ".join(
        f"{profile_id}={'enabled' if profiles[profile_id].enabled else 'disabled'} "
        f"({profiles[profile_id].radius_m * MILLIMETRES_PER_METRE:.1f} mm)"
        for profile_id in BALL_PROFILE_IDS
    ))
    print()

    cube_tracker = CubeTracker()

    while True:
        frame_time = time.monotonic()
        cap, frame, ret = read_frame_with_recovery(cap)

        if not ret:
            print("ERROR: Camera did not recover after all retries.")
            break

        display = frame.copy()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        hsv_for_click = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

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
        put_pose_text(
            display,
            f"HANDHELD FALLBACK: {'ON' if handheld_enabled else 'OFF'}",
            (display.shape[1] - 440, 35),
            (0, 220, 220) if handheld_enabled else (130, 130, 130),
            0.55,
            2,
        )
        selected_profile = profiles[selected_profile_id]
        put_pose_text(
            display,
            f"BALL PROFILE: {selected_profile_id}  R={selected_profile.radius_m * MILLIMETRES_PER_METRE:.1f} mm",
            (display.shape[1] - 440, 57),
            (255, 255, 255),
            0.55,
            2,
        )

        # ====================================================
        # 1. DETECT BALLS AND GENERIC SHAPES
        # ====================================================

        ball_reports, ball_masks, _ = detect_balls_detailed(
            frame,
            profiles,
            camera_matrix,
            dist_coeffs,
            previous_poses={
                profile_id: ball_trackers[profile_id].reference_pose
                for profile_id in BALL_PROFILE_IDS
            },
            hsv=hsv_for_click,
            allow_partial=handheld_enabled,
        )
        ball_detections = [
            report["candidate"]
            for report in ball_reports.values()
            if report.get("candidate") is not None
        ]
        ball_by_id = {
            detection["profile_id"]: detection
            for detection in ball_detections
        }
        objects, edge_image = detect_shapes(frame, gray=gray)
        claimed = claimed_contour(objects, ball_detections)
        for object_index, obj in enumerate(objects):
            if object_index not in claimed:
                draw_detected_object(display, obj)

        ball_colors = {
            "small_ball": (0, 255, 255),
            "large_ball": (255, 128, 0),
        }
        ball_text_y = 70
        for profile_id in BALL_PROFILE_IDS:
            profile = profiles[profile_id]
            tracker = ball_trackers[profile_id]
            tracker.enabled = profile.enabled
            tracker.radius_m = profile.radius_m
            if not tracker.enabled:
                put_pose_text(
                    display,
                    f"{profile_id}: DISABLED (click to calibrate)",
                    (display.shape[1] - 440, ball_text_y),
                    (130, 130, 130),
                    0.55,
                    2,
                )
                ball_text_y += 7 * 22
                continue
            result = tracker.update(
                ball_by_id.get(profile_id),
                camera_matrix,
                dist_coeffs,
                frame_time,
                diagnostics=ball_reports[profile_id].get("diagnostics"),
            )
            pose = result.get("pose")
            if pose is not None:
                draw_ball_pose(
                    display,
                    pose,
                    camera_matrix,
                    dist_coeffs,
                    ball_colors[profile_id],
                    result.get("status", "BALL"),
                )
                draw_pose_text_lines(
                    display,
                    ball_text_lines(result, pose, MILLIMETRES_PER_METRE),
                    display.shape[1] - 440,
                    ball_text_y,
                    line_height=20,
                    font_scale=0.55,
                    thickness=2,
                )
            else:
                report = ball_reports[profile_id]
                range_m = report.get("range_m")
                status = result.get("status", report.get("status", "LOST"))
                status_text = status
                if range_m is not None and status in {"TOO CLOSE", "TOO FAR"}:
                    status_text = (
                        f"{status} ({float(range_m) * MILLIMETRES_PER_METRE:.0f} mm; "
                        "valid 250-750 mm)"
                    )
                put_pose_text(
                    display,
                    f"{profile_id} ({profile.radius_m * MILLIMETRES_PER_METRE:.1f} mm): {status_text}",
                    (display.shape[1] - 440, ball_text_y),
                    ball_colors[profile_id],
                    0.55,
                    2,
                )
            ball_text_y += 7 * 22

        # ====================================================
        # 1B. DETECT THE UNMARKED CUBE
        # ====================================================

        combined_ball_exclusion = np.zeros(gray.shape, dtype=np.uint8)
        for profile_id in BALL_PROFILE_IDS:
            if profiles[profile_id].enabled:
                combined_ball_exclusion = cv.bitwise_or(
                    combined_ball_exclusion,
                    ball_masks[profile_id],
                )

        cube_detection, cube_mask = detect_cube(
            frame,
            cube_tracker.reference_pose,
            gray=gray,
            allow_partial=False,
            exclusion_mask=combined_ball_exclusion,
        )

        track_result = cube_tracker.update(
            cube_detection,
            camera_matrix,
            dist_coeffs,
            frame_time,
        )
        # If the normal full silhouette was absent or failed its geometry
        # gates, make one conservative local fallback attempt.  It can only
        # use the still-trusted previous pose and therefore cannot acquire a
        # cube from a partial outline.
        if (
            handheld_enabled
            and not track_result.get("accepted")
            and cube_tracker.reference_pose is not None
        ):
            partial_detection = detect_cube_partial(
                frame,
                cube_tracker.reference_pose,
                gray=gray,
                exclusion_mask=combined_ball_exclusion,
            )
            if partial_detection is not None:
                partial_result = cube_tracker.update(
                    partial_detection,
                    camera_matrix,
                    dist_coeffs,
                    frame_time,
                )
                if partial_result.get("accepted") or not track_result.get("held"):
                    track_result = partial_result
        cube_pose_to_draw = track_result["pose"]
        cube_pose_diagnostics = track_result.get("diagnostics")
        cube_rejection_reason = track_result.get("rejection_reason")
        cube_text_color = (255, 0, 255)
        accepted = bool(track_result.get("accepted"))
        if accepted:
            cube_status = "CUBE — PARTIAL/HANDHELD" if track_result.get("partial") else "CUBE"
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

        if cube_detection is not None and cube_detection.get("contour") is not None:
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
                cube_status = (
                    "CUBE — PARTIAL/HANDHELD — ORIENTATION AMBIGUOUS"
                    if cube_pose_to_draw.get("partial")
                    else "CUBE — ORIENTATION AMBIGUOUS"
                )
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
        elif cube_detection is not None and cube_detection.get("contour") is not None:
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
        else:
            put_pose_text(
                display,
                f"CUBE: {track_result.get('status', 'LOST')}",
                (15, display.shape[0] - 15),
                (0, 0, 255) if track_result.get("status") == "LOST" else cube_text_color,
                0.7,
                2,
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

        selected_report = ball_reports[selected_profile_id]
        selected_mask = selected_report["mask"]
        ball_segmentation = cv.cvtColor(selected_mask, cv.COLOR_GRAY2BGR)
        cv.putText(
            ball_segmentation,
            f"{selected_profile_id}  R={profiles[selected_profile_id].radius_m * MILLIMETRES_PER_METRE:.1f} mm",
            (12, 28),
            cv.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )
        cv.putText(
            ball_segmentation,
            selected_report.get("status", "LOST"),
            (12, 56),
            cv.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 220, 255) if selected_report.get("accepted") else (0, 0, 255),
            2,
            cv.LINE_AA,
        )
        cv.imshow("Ball Segmentation", ball_segmentation)

        key = cv.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key in (ord("1"), ord("2")):
            selected_profile_id = BALL_PROFILE_IDS[key - ord("1")]
            print(f"Selected ball profile: {selected_profile_id}")
        elif key in (ord("s"), ord("S")):
            try:
                print(f"Saved ball profiles to {save_ball_color_profiles(profiles)}")
            except (OSError, ValueError) as error:
                print(f"Could not save ball profiles: {error}")
        elif key in (ord("r"), ord("R")):
            profiles[selected_profile_id] = reset_ball_color_profile(
                selected_profile_id,
                profiles,
            )
            ball_trackers[selected_profile_id].enabled = False
            ball_trackers[selected_profile_id].reset()
            print(f"Reset ball profile: {selected_profile_id}")
        elif key in (ord("h"), ord("H")):
            handheld_enabled = not handheld_enabled
            print(f"Handheld tracking: {'enabled' if handheld_enabled else 'disabled'}")

    if cap is not None:
        cap.release()
    cv.destroyAllWindows()
