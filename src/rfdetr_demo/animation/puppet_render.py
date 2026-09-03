# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Compatibility facade for the historical mascot animation module."""

from __future__ import annotations

import sys

from rfdetr_demo.animation.puppet_cli import main as main
from rfdetr_demo.animation.puppet_cli import parse_args as parse_args
from rfdetr_demo.animation.puppet_renderer import LayeredMascotRenderer as LayeredMascotRenderer
from rfdetr_demo.animation.puppet_timeline import interpolate_pose, person_for_track, resampled_targets
from rfdetr_demo.animation.puppet_video import render_puppet_video as render_puppet_video

_person_for_track = person_for_track
_interpolate_pose = interpolate_pose
_resampled_targets = resampled_targets

if __name__ == "__main__":
    sys.exit(main())
