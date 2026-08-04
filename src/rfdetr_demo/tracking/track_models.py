# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Track snapshot model and geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import supervision as sv

from rfdetr_demo.tracking.bbox import detection_confidence
from rfdetr_demo.tracking.types import TrackDiagnostic


@dataclass
class TrackSnapshot:
    """One person track between frames."""

    track_id: int
    key_points: sv.KeyPoints
    box: np.ndarray
    missed: int = 0
    sticky: bool = False
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))


def box_center(box: np.ndarray) -> tuple[float, float]:
    """Return the center ``(cx, cy)`` of an axis-aligned box."""
    return float((box[0] + box[2]) / 2.0), float((box[1] + box[3]) / 2.0)


def center_x_range(frame_width: int, fraction: tuple[float, float]) -> tuple[float, float]:
    """Map a normalized x-fraction range onto pixel coordinates."""
    return fraction[0] * frame_width, fraction[1] * frame_width


def in_center_lane(cx: float, frame_width: int, fraction: tuple[float, float]) -> bool:
    """Return whether ``cx`` falls inside the configured center lane."""
    x_min, x_max = center_x_range(frame_width, fraction)
    return x_min <= cx <= x_max


def track_diagnostic(
    track: TrackSnapshot,
    *,
    is_ghost: bool,
    matched_this_frame: bool,
) -> TrackDiagnostic:
    """Build a per-track diagnostic row for pipeline output."""
    cx, cy = box_center(track.box)
    return TrackDiagnostic(
        track_id=track.track_id,
        cx=cx,
        cy=cy,
        confidence=detection_confidence(track.key_points, 0),
        is_ghost=is_ghost,
        missed=track.missed,
        matched_this_frame=matched_this_frame,
    )


__all__ = [
    "TrackSnapshot",
    "box_center",
    "center_x_range",
    "in_center_lane",
    "track_diagnostic",
]
