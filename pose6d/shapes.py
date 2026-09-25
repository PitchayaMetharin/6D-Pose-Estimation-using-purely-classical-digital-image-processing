"""Classical contour-based shape detection and drawing."""

import math

import cv2 as cv
import numpy as np

from .settings import (
    CIRCLE_MAX_AXIS_RATIO, CIRCLE_MIN_CIRCULARITY,
    MAX_RIGHT_ANGLE_COSINE, MIN_OBJECT_AREA, SQUARE_ASPECT_RATIO_MAX,
)

def calculate_circularity(contour):
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)

    if perimeter == 0:
        return 0

    circularity = 4 * math.pi * area / (perimeter * perimeter)

    return circularity


def has_right_angles(approx):
    if len(approx) != 4 or not cv.isContourConvex(approx):
        return False

    points = approx.reshape(4, 2).astype(np.float64)

    for i in range(4):
        previous_point = points[(i - 1) % 4]
        current_point = points[i]
        next_point = points[(i + 1) % 4]

        first_vector = previous_point - current_point
        second_vector = next_point - current_point

        denominator = (
            np.linalg.norm(first_vector)
            * np.linalg.norm(second_vector)
        )

        if denominator == 0:
            return False

        cosine = abs(
            np.dot(first_vector, second_vector) / denominator
        )

        if cosine > MAX_RIGHT_ANGLE_COSINE:
            return False

    return True


def classify_shape(contour):
    area = cv.contourArea(contour)

    if area < MIN_OBJECT_AREA:
        return None

    perimeter = cv.arcLength(contour, True)

    epsilon = 0.02 * perimeter

    approx = cv.approxPolyDP(
        contour,
        epsilon,
        True
    )

    vertices = len(approx)

    circularity = calculate_circularity(contour)

    # --------------------------------------------------------
    # Square / Rectangle
    # --------------------------------------------------------

    if vertices == 4:
        if not has_right_angles(approx):
            return None

        rect = cv.minAreaRect(contour)

        width, height = rect[1]

        if width == 0 or height == 0:
            return None

        ratio = max(width, height) / min(width, height)

        if ratio <= SQUARE_ASPECT_RATIO_MAX:
            return "SQUARE"

        return "RECTANGLE"

    # --------------------------------------------------------
    # Circle
    # --------------------------------------------------------

    if vertices >= 6 and circularity > CIRCLE_MIN_CIRCULARITY:
        if len(contour) < 5:
            return None

        try:
            _, axes, _ = cv.fitEllipse(contour)
        except cv.error:
            return None

        axis_a, axis_b = axes

        if axis_a == 0 or axis_b == 0:
            return None

        axis_ratio = max(axis_a, axis_b) / min(axis_a, axis_b)

        if axis_ratio <= CIRCLE_MAX_AXIS_RATIO:
            return "CIRCLE"

    return None


def detect_shapes(frame, gray=None):
    """Detect generic shapes, optionally reusing a caller's gray frame."""
    if gray is None:
        gray = cv.cvtColor(
            frame,
            cv.COLOR_BGR2GRAY
        )

    blurred = cv.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv.Canny(
        blurred,
        50,
        150
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    edges = cv.morphologyEx(
        edges,
        cv.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv.findContours(
        edges,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    objects = []

    for contour in contours:
        shape = classify_shape(contour)

        if shape is None:
            continue

        area = cv.contourArea(contour)

        M = cv.moments(contour)

        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        objects.append({
            "shape": shape,
            "contour": contour,
            "center": (cx, cy),
            "area": area
        })

    return objects, edges

def draw_detected_object(frame, obj):
    contour = obj["contour"]
    shape = obj["shape"]
    center = obj["center"]

    cv.drawContours(
        frame,
        [contour],
        -1,
        (0, 255, 0),
        2
    )

    cv.circle(
        frame,
        center,
        5,
        (0, 0, 255),
        -1
    )

    x, y, w, h = cv.boundingRect(contour)

    cv.putText(
        frame,
        shape,
        (x, y - 10),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
