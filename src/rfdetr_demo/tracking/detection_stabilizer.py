# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Stabilize per-frame keypoint detections with IoU-NMS and short track hold.

Deprecated facade — prefer :class:`PersonTrackPipeline` from ``tracking.pipeline``.
"""

from __future__ import annotations

from rfdetr_demo.tracking.bbox import (
    detection_bbox,
    detection_confidence,
    nms_detection_indices,
)
from rfdetr_demo.tracking.keypoints_ops import (
    is_track_ghost,
    merge_key_points,
    partition_live_and_ghost,
    subset_key_points,
)
from rfdetr_demo.tracking.pipeline import PersonTrackPipeline
from rfdetr_demo.tracking.stabilizer import (
    DetectionStabilizer,
    DetectionStabilizerSettings,
    is_detection_stabilizer_enabled,
)
from rfdetr_demo.tracking.types import (
    TRACK_IS_GHOST_KEY,
    StabilizationResult,
    StabilizationStats,
    TrackDiagnostic,
    TrackPipelineResult,
    person_track_settings_from_env,
)

__all__ = [
    "DetectionStabilizer",
    "DetectionStabilizerSettings",
    "PersonTrackPipeline",
    "StabilizationResult",
    "StabilizationStats",
    "TRACK_IS_GHOST_KEY",
    "TrackDiagnostic",
    "TrackPipelineResult",
    "detection_bbox",
    "detection_confidence",
    "is_detection_stabilizer_enabled",
    "is_track_ghost",
    "merge_key_points",
    "nms_detection_indices",
    "partition_live_and_ghost",
    "person_track_settings_from_env",
    "subset_key_points",
]
