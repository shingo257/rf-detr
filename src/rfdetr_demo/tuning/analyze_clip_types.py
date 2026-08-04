# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Dataclasses for short-clip keypoint quality analysis."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class JointFrameMetric:
    """Per-joint metrics for one analyzed frame."""

    joint_index: int
    joint_name: str
    x: float
    y: float
    confidence: float
    speed_px_per_sec: float | None
    displacement_px: float | None
    covariance_trace: float | None


@dataclass
class FrameMetric:
    """Aggregated metrics for one analyzed frame."""

    frame_index: int
    person_count: int
    joints: list[JointFrameMetric] = field(default_factory=list)
    max_joint_speed: float | None = None
    low_confidence_joints: int = 0


@dataclass
class ClipAnalysis:
    """Summary of a short-clip keypoint quality analysis."""

    source: str
    duration_sec: float
    fps: float
    frame_stride: int
    frames_analyzed: int
    avg_persons: float
    motion_speed_rejections: int
    motion_oscillation_corrections: int
    issues: list[str] = field(default_factory=list)
    frame_metrics: list[FrameMetric] = field(default_factory=list)


__all__ = ["ClipAnalysis", "FrameMetric", "JointFrameMetric"]
