# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Mutable track/joint state and geometry helpers for temporal filtering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import supervision as sv

from rfdetr_demo.inference.temporal.one_euro import OneEuroFilter


@dataclass
class JointState:
    """Mutable temporal state for one tracked joint."""

    last_xy: np.ndarray | None = None
    last_velocity: np.ndarray | None = None
    flip_count: int = 0
    consecutive_holds: int = 0
    last_frame_index: int | None = None
    one_euro_x: OneEuroFilter | None = None
    one_euro_y: OneEuroFilter | None = None


@dataclass
class TrackState:
    """Mutable temporal state for one tracked person."""

    joints: list[JointState] = field(default_factory=list)
    last_centroid: np.ndarray | None = None
    last_frame_index: int = -1
    reference_bone_lengths: dict[tuple[int, int], float] = field(default_factory=dict)


def frame_diagonal(frame_width: int, frame_height: int) -> float:
    """Return the frame diagonal in pixels."""
    return float(np.hypot(frame_width, frame_height))


def detection_centroid(key_points: sv.KeyPoints, detection_index: int) -> np.ndarray | None:
    """Return a hip-biased centroid for one detection."""
    xy = key_points.xy[detection_index]
    visible = key_points.visible
    if visible is not None:
        mask = visible[detection_index]
        points = xy[mask]
    else:
        points = xy[~np.all(np.isclose(xy, 0), axis=1)]
    if len(points) == 0:
        return None
    hip_indices = (11, 12)
    hip_points = [xy[index] for index in hip_indices if index < len(xy) and not np.allclose(xy[index], 0)]
    if hip_points:
        return cast(
            np.ndarray,
            np.asarray(np.mean(np.asarray(hip_points, dtype=np.float64), axis=0), dtype=np.float64),
        )
    return cast(np.ndarray, np.asarray(np.mean(points.astype(np.float64), axis=0), dtype=np.float64))


def covariance_trace(key_points: sv.KeyPoints, detection_index: int, joint_index: int) -> float | None:
    """Return the covariance trace for one joint when metadata is valid."""
    covariance_raw = key_points.data.get("covariance")
    if covariance_raw is None:
        return None
    covariance = np.asarray(covariance_raw, dtype=np.float64)
    if covariance.shape[:2] != key_points.xy.shape[:2]:
        return None
    matrix = covariance[detection_index, joint_index]
    if not np.isfinite(matrix).all():
        return None
    return float(matrix[0, 0] + matrix[1, 1])


__all__ = [
    "JointState",
    "TrackState",
    "covariance_trace",
    "detection_centroid",
    "frame_diagonal",
]
