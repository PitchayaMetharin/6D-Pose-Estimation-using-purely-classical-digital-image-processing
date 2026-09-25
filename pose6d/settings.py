"""User-adjustable settings for the 6D pose application."""

CAMERA_INDEX = "/dev/video4"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CAMERA_FORMAT = "MJPG"
CAMERA_FPS = 30

# Actual printed ArUco black-square side length in metres.
# Example: 40 mm marker = 0.040 m
MARKER_SIZE_M = 0.040

MIN_OBJECT_AREA = 5000

# The calibration file must contain camera_matrix, dist_coeffs,
# image_width, and image_height saved by np.savez.
CALIBRATION_FILE = "camera_calibration.npz"
CALIBRATION_BOARD_SIZE = (7, 7)
CALIBRATION_SQUARE_SIZE_M = 0.025
CALIBRATION_REQUIRED_VIEWS = 20

SQUARE_ASPECT_RATIO_MAX = 1.15
MAX_RIGHT_ANGLE_COSINE = 0.15
CIRCLE_MIN_CIRCULARITY = 0.78
CIRCLE_MAX_AXIS_RATIO = 1.15

# The current cube is a dark, unmarked 30 mm cube.  Its silhouette is
# therefore detected separately from the generic square/circle detector.
CUBE_SIZE_M = 0.030
CUBE_DARK_THRESHOLD = 100
CUBE_MIN_AREA = 1200
# Keep the historical coarse epsilon for compatibility, but detection now
# evaluates the four fractions below so one-axis transition silhouettes retain
# their five/six outer vertices.
CUBE_APPROX_EPSILON = 0.02
CUBE_APPROX_EPSILON_FRACTIONS = (0.005, 0.008, 0.012, 0.020)
CUBE_LOCAL_MAPPING_LIMIT = 3
CUBE_POSE_TIME_CONSTANT_S = 0.12
CUBE_MAX_HOLD_SECONDS = 0.75
CUBE_MAX_TRACK_ERROR_PIXELS = 60.0
CUBE_MAX_FIT_ERROR_PIXELS = 3.0
CUBE_MAX_FIT_ERROR_FRACTION = 0.03
CUBE_MAX_PLANAR_FIT_ERROR_PIXELS = 15.0
CUBE_MAX_CENTER_SPEED_MPS = 1.0
CUBE_MAX_Z_SPEED_MPS = 0.75
CUBE_MAX_ROTATION_SPEED_DEG_PER_S = 360.0
CUBE_MAX_PLANAR_ROTATION_SPEED_DEG_PER_S = 540.0
CUBE_TEXT_GAP_PIXELS = 60
MILLIMETRES_PER_METRE = 1000.0

# OpenCV draws frame axes as X=red, Y=green, Z=blue.
AXIS_X_COLOR = (0, 0, 255)
AXIS_Y_COLOR = (0, 255, 0)
AXIS_Z_COLOR = (255, 0, 0)
POSE_TEXT_COLOR = (255, 255, 255)
POSE_TEXT_SHADOW_COLOR = (0, 0, 0)
POSE_TEXT_FONT_SCALE = 0.9
POSE_TEXT_LINE_HEIGHT = 30
POSE_TEXT_THICKNESS = 2
POSE_GUIDE_FONT_SCALE = 0.6
POSE_GUIDE_LINE_HEIGHT = 22

CAMERA_READ_RETRIES = 5
CAMERA_REOPEN_ATTEMPTS = 3
CAMERA_RETRY_DELAY_S = 0.02

CALIBRATION_MAX_RMS_ERROR_PIXELS = 1.0
CALIBRATION_MAX_VIEW_ERROR_PIXELS = 2.0
CALIBRATION_DUPLICATE_CENTER_DISTANCE = 0.06
CALIBRATION_DUPLICATE_LOG_AREA_DISTANCE = 0.10
CALIBRATION_DUPLICATE_ANGLE_DEG = 6.0

# ---------------------------------------------------------------------------
# Dual spherical-ball tracking
# ---------------------------------------------------------------------------

# Profiles are deliberately persisted separately from camera calibration.  A
# missing profile disables only its corresponding tracker; it must not make
# cube or ArUco tracking unavailable.
BALL_COLOR_PROFILES_FILE = "ball_color_profiles.npz"
BALL_PROFILE_IDS = ("small_ball", "large_ball")
BALL_DEFAULT_RADII_M = {
    "small_ball": 0.020,
    "large_ball": 0.0375,
}
SMALL_BALL_RADIUS_M = BALL_DEFAULT_RADII_M["small_ball"]
LARGE_BALL_RADIUS_M = BALL_DEFAULT_RADII_M["large_ball"]
BALL_WORKING_DISTANCE_MIN_M = 0.250
BALL_WORKING_DISTANCE_MAX_M = 0.750
# Small numerical allowance for the Euclidean range recovered from a sphere
# rendered exactly at a depth boundary (off-axis centres add only a few mm).
# Objects materially outside the working range still fail the hard gate.
BALL_RANGE_MARGIN_M = 0.005
BALL_MAX_HOLD_SECONDS = 0.75
BALL_POSE_TIME_CONSTANT_S = 0.12
BALL_MAX_CENTER_SPEED_MPS = 1.0
BALL_MAX_Z_SPEED_MPS = 0.75
BALL_MIN_PARTIAL_ARC_DEG = 120.0
BALL_MAX_PARTIAL_RESIDUAL_PIXELS = 8.0
BALL_MAX_EDGE_ERROR_PIXELS = 8.0
BALL_MIN_AREA = 500.0
BALL_MIN_CIRCULARITY = 0.25
BALL_CIRCULARITY_APPROX_EPSILON_FRACTION = 0.05
BALL_MIN_SOLIDITY = 0.85
BALL_MAX_AXIS_RATIO = 4.0
BALL_MIN_CONTOUR_POINTS = 20
BALL_MASK_CLOSE_KERNEL = 5
BALL_MASK_OPEN_KERNEL = 3
BALL_MASK_CLOSE_ITERATIONS = 2
BALL_MASK_OPEN_ITERATIONS = 1
BALL_BORDER_MARGIN_PIXELS = 2
BALL_HSV_HUE_MARGIN = 8
BALL_HSV_SAT_MARGIN = 24
BALL_HSV_VALUE_MARGIN = 24
BALL_NEUTRAL_SATURATION = 35
BALL_PROFILE_PATCH_RADIUS = 6
BALL_PROFILE_MIN_PIXELS = 9
BALL_PARTIAL_LIMB_SAMPLES = 720
BALL_PARTIAL_LIMB_TOLERANCE_PIXELS = 6.0
BALL_PARTIAL_SUPPORT_RADIUS_PIXELS = 2
BALL_IOU_MINIMUM = 0.72
BALL_AREA_RATIO_MINIMUM = 0.55
BALL_AREA_RATIO_MAXIMUM = 1.55
HANDHELD_TRACKING_ENABLED = True
CUBE_PARTIAL_MIN_VERTICES = 3
CUBE_PARTIAL_ROI_MARGIN_PIXELS = 24
CUBE_PARTIAL_MAX_CORNER_ERROR_PIXELS = 28.0
CUBE_PARTIAL_MAX_REPROJECTION_ERROR_PIXELS = 8.0
# Physical-edge evidence is deliberately stricter than the historical corner
# detector.  It rejects soft shadows and textured background corners before
# any PnP solve can turn them into a plausible pose.
CUBE_EDGE_CANNY_LOW = 70
CUBE_EDGE_CANNY_HIGH = 150
CUBE_EDGE_SUPPORT_BAND_PIXELS = 3
CUBE_EDGE_MIN_TOTAL_SUPPORT = 0.60
CUBE_EDGE_MIN_PER_EDGE_SUPPORT = 0.35
CUBE_PARTIAL_MAX_ANGLE_ERROR_DEG = 12.0
CUBE_PARTIAL_MAX_PERP_DISTANCE_PIXELS = 6.0
CUBE_PARTIAL_MIN_EDGE_OVERLAP = 0.25
CUBE_PARTIAL_MIN_PERIMETER_COVERAGE = 0.35
CUBE_PARTIAL_MAX_VERTEX_ERROR_PIXELS = 10.0
CUBE_PARTIAL_MAX_EDGE_RESIDUAL_PIXELS = 3.0
CUBE_PARTIAL_DEBUG_DIM_VALUE = 70
CUBE_EXCLUSION_DILATION_PIXELS = 3

# Chromatic profiles need enough illumination headroom to keep both the
# highlight and shaded limb.  Neutral profiles intentionally retain the older
# symmetric bounded margins below.
BALL_CHROMATIC_SATURATION_MARGIN = 48
BALL_CHROMATIC_VALUE_MARGIN = 64
