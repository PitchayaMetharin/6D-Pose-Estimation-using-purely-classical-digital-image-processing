"""Camera opening, configuration, and recovery."""

import time

import cv2 as cv

from .settings import (
    CAMERA_FORMAT, CAMERA_FPS, CAMERA_INDEX, CAMERA_READ_RETRIES,
    CAMERA_REOPEN_ATTEMPTS, CAMERA_RETRY_DELAY_S, FRAME_HEIGHT,
    FRAME_WIDTH,
)

def configure_camera(cap):
    requested_fourcc = cv.VideoWriter_fourcc(*CAMERA_FORMAT)

    cap.set(
        cv.CAP_PROP_FOURCC,
        requested_fourcc
    )

    cap.set(
        cv.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    cap.set(
        cv.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )

    cap.set(
        cv.CAP_PROP_FPS,
        CAMERA_FPS
    )

    actual_width = int(round(cap.get(cv.CAP_PROP_FRAME_WIDTH)))
    actual_height = int(round(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))

    actual_fourcc_value = int(cap.get(cv.CAP_PROP_FOURCC))
    actual_fourcc = "".join(
        chr((actual_fourcc_value >> (8 * i)) & 0xFF)
        for i in range(4)
    )

    print(
        "Camera format: "
        f"{actual_width}x{actual_height} "
        f"{actual_fourcc or 'unknown'}"
    )

    if (actual_width, actual_height) != (FRAME_WIDTH, FRAME_HEIGHT):
        print(
            "WARNING: Camera did not accept the requested "
            f"resolution {FRAME_WIDTH}x{FRAME_HEIGHT}."
        )
        print(
            "Captured frames will be resized to the calibrated working "
            f"size {FRAME_WIDTH}x{FRAME_HEIGHT}."
        )


def normalize_frame(frame):
    """Return frames at the calibrated working resolution."""

    if frame is None:
        return None

    frame_height, frame_width = frame.shape[:2]

    if (frame_width, frame_height) == (FRAME_WIDTH, FRAME_HEIGHT):
        return frame

    return cv.resize(
        frame,
        (FRAME_WIDTH, FRAME_HEIGHT),
        interpolation=cv.INTER_AREA
    )


def open_configured_camera():
    cap = cv.VideoCapture(
        CAMERA_INDEX,
        cv.CAP_V4L2
    )

    if not cap.isOpened():
        cap.release()
        cap = cv.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        cap.release()
        return None

    configure_camera(cap)
    return cap


def read_frame_with_recovery(cap):
    for _ in range(CAMERA_READ_RETRIES + 1):
        ret, frame = cap.read()

        if ret:
            return cap, normalize_frame(frame), True

        time.sleep(CAMERA_RETRY_DELAY_S)

    print("Camera frame lost; attempting to reopen the camera.")

    for attempt in range(CAMERA_REOPEN_ATTEMPTS):
        if cap is not None:
            cap.release()
        cap = open_configured_camera()

        if cap is None:
            print(
                f"Camera reopen attempt {attempt + 1}/"
                f"{CAMERA_REOPEN_ATTEMPTS} failed."
            )
            time.sleep(CAMERA_RETRY_DELAY_S)
            continue

        for _ in range(CAMERA_READ_RETRIES + 1):
            ret, frame = cap.read()

            if ret:
                print("Camera recovered.")
                return cap, normalize_frame(frame), True

            time.sleep(CAMERA_RETRY_DELAY_S)

        print(
            f"Camera reopen attempt {attempt + 1}/"
            f"{CAMERA_REOPEN_ATTEMPTS} returned no frame."
        )

    return cap, None, False
