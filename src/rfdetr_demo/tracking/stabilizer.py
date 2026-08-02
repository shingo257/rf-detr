# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Compatibility-oriented detection stabilizer built on the canonical track store."""

from __future__ import annotations

from dataclasses import dataclass, field

import supervision as sv

from rfdetr_demo.tracking.track_store import TrackStore
from rfdetr_demo.tracking.types import (
    PersonTrackSettings,
    StabilizationResult,
    StabilizationStats,
    is_person_track_enabled,
)

DetectionStabilizerSettings = PersonTrackSettings


def is_detection_stabilizer_enabled() -> bool:
    """Return whether the person tracking pipeline is enabled."""
    return bool(is_person_track_enabled())


@dataclass
class DetectionStabilizer:
    """Apply IoU-NMS and short missed-frame hold to keypoint detections."""

    settings: PersonTrackSettings = field(default_factory=PersonTrackSettings)
    frame_width: int = 1280
    _store: TrackStore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build the track store for this stabilizer."""
        self._store = TrackStore(settings=self.settings, frame_width=self.frame_width)

    def reset(self) -> None:
        """Clear track history."""
        self._store.reset()

    def apply(self, key_points: sv.KeyPoints, frame_index: int) -> StabilizationResult:
        """Return stabilized detections for one frame."""
        result = self._store.apply(key_points, frame_index)
        return StabilizationResult(
            key_points=result.key_points,
            stats=StabilizationStats(
                raw_count=result.stats.raw_count,
                nms_count=result.stats.nms_count,
                active_track_count=result.stats.active_track_count,
                ghost_count=result.stats.ghost_count,
            ),
            diagnostics=result.diagnostics,
        )


__all__ = [
    "DetectionStabilizer",
    "DetectionStabilizerSettings",
    "is_detection_stabilizer_enabled",
]
