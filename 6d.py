#!/usr/bin/env python3
"""Compatibility entry point for the modular 6D pose application."""

import sys

from pose6d import *
from pose6d.app import main
from pose6d.calibration import run_camera_calibration


if __name__ == "__main__":
    if "--calibrate" in sys.argv:
        run_camera_calibration()
    else:
        main()
