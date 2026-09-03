# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Compatibility facade for the canonical temporal filtering package."""

from rfdetr_demo.inference.temporal import (
    COCO_BONE_PAIRS,
    KeypointTemporalFilter,
    MotionFilterStats,
    MotionPlausibilitySettings,
    OneEuroFilter,
    RejectMode,
    apply_temporal_filter,
)

__all__ = [
    "COCO_BONE_PAIRS",
    "KeypointTemporalFilter",
    "MotionFilterStats",
    "MotionPlausibilitySettings",
    "OneEuroFilter",
    "RejectMode",
    "apply_temporal_filter",
]
