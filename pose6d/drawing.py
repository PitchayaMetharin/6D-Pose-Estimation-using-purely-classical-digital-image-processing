"""Pose overlays and user-visible diagnostics."""

import cv2 as cv
import numpy as np

from .settings import (
    AXIS_X_COLOR, AXIS_Y_COLOR, AXIS_Z_COLOR, CUBE_SIZE_M,
    POSE_GUIDE_FONT_SCALE, POSE_GUIDE_LINE_HEIGHT, POSE_TEXT_COLOR,
    POSE_TEXT_FONT_SCALE, POSE_TEXT_LINE_HEIGHT,
    POSE_TEXT_SHADOW_COLOR, POSE_TEXT_THICKNESS,
)

def draw_cube_pose(
    frame,
    pose,
    camera_matrix,
    dist_coeffs
):
    projected_points = np.rint(
        pose["projected_points"]
    ).astype(np.int32)

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7)
    ]

    for start, end in edges:
        cv.line(
            frame,
            tuple(projected_points[start]),
            tuple(projected_points[end]),
            (255, 0, 255),
            2
        )

    for index, point in enumerate(projected_points):
        point_tuple = tuple(point)

        cv.circle(
            frame,
            point_tuple,
            4,
            (0, 165, 255),
            -1
        )

        cv.putText(
            frame,
            str(index + 1),
            (point_tuple[0] + 5, point_tuple[1] - 5),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2
        )

    cv.drawFrameAxes(
        frame,
        camera_matrix,
        dist_coeffs,
        pose["rvec"],
        pose["center_tvec"],
        CUBE_SIZE_M * 0.75,
        2
    )


def put_pose_text(
    frame,
    text,
    position,
    color,
    font_scale=POSE_TEXT_FONT_SCALE,
    thickness=POSE_TEXT_THICKNESS
):
    """Draw readable pose text with a dark outline."""

    cv.putText(
        frame,
        text,
        position,
        cv.FONT_HERSHEY_SIMPLEX,
        font_scale,
        POSE_TEXT_SHADOW_COLOR,
        thickness + 2,
        cv.LINE_AA
    )

    cv.putText(
        frame,
        text,
        position,
        cv.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv.LINE_AA
    )


def draw_pose_text_lines(
    frame,
    lines,
    text_x,
    text_y,
    line_height=POSE_TEXT_LINE_HEIGHT,
    font_scale=POSE_TEXT_FONT_SCALE,
    thickness=POSE_TEXT_THICKNESS
):
    for line_number, (text, color) in enumerate(lines):
        put_pose_text(
            frame,
            text,
            (text_x, text_y + line_number * line_height),
            color,
            font_scale,
            thickness
        )


def draw_pose_guide(frame):
    """Show the pose reference frame and actions that change each value."""

    frame_height, frame_width = frame.shape[:2]
    panel_x = 15
    panel_y = 58
    line_height = POSE_GUIDE_LINE_HEIGHT

    guide_lines = [
        ("POSE GUIDE", POSE_TEXT_COLOR),
        ("Position: cube center relative to camera", POSE_TEXT_COLOR),
        ("Origin: camera optical center | units: mm", POSE_TEXT_COLOR),
        ("X  + right / - left", AXIS_X_COLOR),
        ("Y  + down / - up", AXIS_Y_COLOR),
        ("Z  + forward / - toward camera", AXIS_Z_COLOR),
        ("Range = straight-line distance to cube center", POSE_TEXT_COLOR),
        ("Rotation: cube axes relative to camera axes", POSE_TEXT_COLOR),
        ("Roll   rotate around X axis", AXIS_X_COLOR),
        ("Pitch  rotate around Y axis", AXIS_Y_COLOR),
        ("Yaw    rotate around Z axis", AXIS_Z_COLOR),
        ("Unmarked cube: RPY has 24 symmetry-equivalent views", (220, 220, 220)),
        ("Balls: centre/range only; orientation is not observable", (220, 220, 220)),
        ("1: select the small_ball colour profile", (220, 220, 220)),
        ("2: select the large_ball colour profile", (220, 220, 220)),
        ("Left click ball: calibrate the selected profile", (220, 220, 220)),
        ("S: save both ball colour profiles", (220, 220, 220)),
        ("R: reset/disable the selected ball profile", (220, 220, 220)),
        ("H: toggle handheld partial tracking", (220, 220, 220)),
        ("Q: quit the application", (220, 220, 220)),
    ]

    panel_width = min(500, frame_width - 20)
    panel_height = line_height * len(guide_lines) + 14
    panel_bottom = min(
        frame_height - 8,
        panel_y + panel_height
    )

    overlay = frame.copy()

    cv.rectangle(
        overlay,
        (panel_x, panel_y - 16),
        (panel_x + panel_width, panel_bottom),
        (0, 0, 0),
        -1
    )

    cv.addWeighted(
        overlay,
        0.62,
        frame,
        0.38,
        0,
        frame
    )

    cv.rectangle(
        frame,
        (panel_x, panel_y - 16),
        (panel_x + panel_width, panel_bottom),
        (100, 100, 100),
        1
    )

    draw_pose_text_lines(
        frame,
        guide_lines,
        panel_x + 8,
        panel_y,
        line_height,
        POSE_GUIDE_FONT_SCALE,
        2
    )


def cube_text_position(frame, contour, pose):
    """Place cube telemetry in the fixed bottom-left corner."""

    frame_height = frame.shape[0]
    text_height = 9 * POSE_TEXT_LINE_HEIGHT

    text_x = 15
    text_y = max(25, frame_height - text_height - 15)

    return int(text_x), int(text_y)


def format_cube_geometry_diagnostics(diagnostics):
    if diagnostics is None:
        return None

    values = (
        diagnostics.get("reprojection_rms"),
        diagnostics.get("reprojection_limit"),
        diagnostics.get("hull_edge_error"),
        diagnostics.get("hull_edge_limit"),
        diagnostics.get("hull_iou"),
        diagnostics.get("hull_iou_minimum")
    )

    if any(value is None for value in values):
        return None

    solve_mode = diagnostics.get("solve_mode", "global").upper()
    point_count = diagnostics.get("selected_point_count", diagnostics.get("point_count", "?"))
    area_ratio = diagnostics.get("area_ratio")
    area_ratio = float(area_ratio) if area_ratio is not None else float("nan")
    return (
        f"{solve_mode} Pts {point_count} | "
        f"Reproj {values[0]:.1f}/{values[1]:.1f} px | "
        f"Hull {values[2]:.1f}/{values[3]:.1f} px | "
        f"IoU {values[4]:.2f}/{values[5]:.2f} | "
        f"Area {area_ratio:.2f}"
    )


def draw_ball_pose(frame, pose, camera_matrix, dist_coeffs, color=(0, 255, 255),
                   status="BALL"):
    """Draw only the spherical outline and centre (never axes or RPY)."""
    projected = pose.get("projected_contour")
    if projected is not None:
        contour = np.rint(np.asarray(projected).reshape(-1, 1, 2)).astype(np.int32)
        cv.polylines(frame, [contour], True, color, 2, cv.LINE_AA)
    center_tvec = np.asarray(pose["center_tvec"], dtype=np.float64).reshape(3, 1)
    projected_center, _ = cv.projectPoints(
        np.zeros((1, 3), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        center_tvec,
        camera_matrix,
        dist_coeffs,
    )
    center = tuple(np.rint(projected_center.reshape(2)).astype(int))
    cv.circle(frame, center, 5, color, -1, cv.LINE_AA)
    cv.putText(
        frame,
        str(status),
        (center[0] + 8, center[1] - 8),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv.LINE_AA,
    )
    return center


def ball_text_lines(result, pose, millimetres_per_metre=1000.0):
    """Return compact ball telemetry lines for the app overlay."""
    if pose is None:
        return []
    center = np.asarray(pose["center_tvec"], dtype=np.float64).reshape(3)
    radius_mm = float(pose.get("radius_m", 0.0)) * millimetres_per_metre
    range_mm = float(np.linalg.norm(center)) * millimetres_per_metre
    status = result.get("status", "LOST") if result is not None else "LOST"
    return [
        (f"{pose.get('profile_id', 'ball')}: {status}", POSE_TEXT_COLOR),
        (f"R:{radius_mm:.1f} mm", POSE_TEXT_COLOR),
        (f"X:{center[0] * millimetres_per_metre:+.1f} mm", AXIS_X_COLOR),
        (f"Y:{center[1] * millimetres_per_metre:+.1f} mm", AXIS_Y_COLOR),
        (f"Z:{center[2] * millimetres_per_metre:+.1f} mm", AXIS_Z_COLOR),
        (f"Range:{range_mm:.1f} mm", POSE_TEXT_COLOR),
    ]
