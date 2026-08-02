# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Annotation and lane helpers for tracking audits."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from rfdetr_demo.media.audit.tracking_types import (
    DEFAULT_CENTER_X_FRACTION,
    RawDetectionDiagnostic,
)
from rfdetr_demo.tracking.bbox import detection_bbox, detection_confidence
from rfdetr_demo.tracking.types import TrackDiagnostic


def center_x_range(frame_width: int) -> tuple[float, float]:
    """Return pixel x-range treated as the center person lane."""
    x_min_frac, x_max_frac = DEFAULT_CENTER_X_FRACTION
    return x_min_frac * frame_width, x_max_frac * frame_width


def in_center_lane(cx: float, frame_width: int) -> bool:
    """Return whether centroid ``cx`` lies in the center lane."""
    x_min, x_max = center_x_range(frame_width)
    return x_min <= cx <= x_max


def raw_diagnostics(key_points: Any, frame_width: int) -> list[RawDetectionDiagnostic]:
    """Build raw detection diagnostics for one keypoint prediction."""
    rows: list[RawDetectionDiagnostic] = []
    for index in range(len(key_points)):
        box = detection_bbox(key_points, index)
        if box is None:
            continue
        cx = float((box[0] + box[2]) / 2.0)
        cy = float((box[1] + box[3]) / 2.0)
        rows.append(
            RawDetectionDiagnostic(
                index=index,
                cx=cx,
                cy=cy,
                confidence=detection_confidence(key_points, index),
                in_center_lane=in_center_lane(cx, frame_width),
            ),
        )
    return rows


def find_center_track(
    diagnostics: list[TrackDiagnostic],
    frame_width: int,
) -> TrackDiagnostic | None:
    """Return the track closest to the center lane midpoint, if any."""
    x_min, x_max = center_x_range(frame_width)
    candidates = [row for row in diagnostics if x_min <= row.cx <= x_max]
    if not candidates:
        return None
    return min(candidates, key=lambda row: abs(row.cx - (x_min + x_max) / 2.0))


def annotate_track_labels(
    annotated_bgr: npt.NDArray[np.uint8],
    diagnostics: list[TrackDiagnostic],
) -> npt.NDArray[np.uint8]:
    """Draw track IDs and ghost markers onto an annotated frame."""
    output = annotated_bgr.copy()
    for track in diagnostics:
        label = f"T{track.track_id}"
        if track.is_ghost:
            label += f" ghost m{track.missed}"
        color = (0, 180, 255) if track.is_ghost else (0, 255, 0)
        cv2.circle(output, (int(track.cx), int(track.cy)), 6, color, 2)
        cv2.putText(
            output,
            label,
            (int(track.cx) + 8, int(track.cy) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return output
