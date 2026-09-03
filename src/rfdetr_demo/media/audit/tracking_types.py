# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Dataclasses for center-person tracking audits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rfdetr_demo.tracking.types import TrackDiagnostic

# Center lane for mn1-2.mov (~1280px wide); scaled if frame width differs.
DEFAULT_CENTER_X_FRACTION = (0.28, 0.48)


@dataclass(frozen=True)
class RawDetectionDiagnostic:
    """One raw model detection before stabilization."""

    index: int
    cx: float
    cy: float
    confidence: float
    in_center_lane: bool


@dataclass
class TrackingFrameRecord:
    """Metrics for one processed video frame."""

    frame_index: int
    processed_count: int
    raw_count: int
    nms_count: int
    active_count: int
    ghost_count: int
    center_present_raw: bool
    center_present_stabilized: bool
    center_track_id: int | None
    center_is_ghost: bool
    center_confidence: float | None
    image_saved: bool
    image_relpath: str | None = None
    raw_detections: list[RawDetectionDiagnostic] = field(default_factory=list)
    stabilized_tracks: list[TrackDiagnostic] = field(default_factory=list)


@dataclass
class TrackingAuditSummary:
    """Run-level analysis written under confidential/audit."""

    run_id: str
    source_relpath: str
    frames_processed: int
    frames_logged: int
    images_saved: int
    run_dir_relpath: str
    evaluation: dict[str, Any]
    records: list[TrackingFrameRecord] = field(default_factory=list)
