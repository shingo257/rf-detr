# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Configuration and statistics for temporal keypoint filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RejectMode = Literal["hold", "hide"]


@dataclass(frozen=True)
class MotionPlausibilitySettings:
    """Thresholds for temporal keypoint gating and smoothing."""

    enabled: bool = True
    max_speed_fraction_per_sec: float = 0.35
    ema_alpha: float = 0.55
    use_one_euro_filter: bool = True
    min_cutoff: float = 1.0
    beta: float = 0.05
    d_cutoff: float = 1.0
    suppress_oscillation: bool = True
    oscillation_flip_threshold: int = 3
    oscillation_min_speed_fraction: float = 0.08
    covariance_sigma_multiplier: float = 4.0
    use_covariance_gate: bool = True
    reject_mode: RejectMode = "hold"
    max_consecutive_holds: int = 4
    track_match_fraction: float = 0.12
    bone_length_constraint_enabled: bool = True
    bone_length_max_dev: float = 0.25


@dataclass
class MotionFilterStats:
    """Counters updated while filtering a sequence."""

    speed_rejections: int = 0
    covariance_rejections: int = 0
    oscillation_corrections: int = 0
    smoothed_joints: int = 0
    bone_corrections: int = 0


__all__ = ["MotionFilterStats", "MotionPlausibilitySettings", "RejectMode"]
