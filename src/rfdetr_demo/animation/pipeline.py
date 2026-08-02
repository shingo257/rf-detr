"""Bridge RF-DETR ``sv.KeyPoints`` output to torso animation controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv

from rfdetr_demo.animation.temporal import TorsoTemporalFilter, TorsoTemporalSettings
from rfdetr_demo.animation.torso import (
    COCO17_KEYPOINT_NAMES,
    ControlPoint,
    PointSource,
    TorsoControls,
    derive_torso_controls,
)
from rfdetr_demo.tracking.keypoints_ops import is_track_ghost, track_id_at


@dataclass(frozen=True, slots=True)
class TrackedTorsoControls:
    """Torso controls associated with one RF-DETR detection and track."""

    detection_index: int
    track_id: int
    controls: TorsoControls
    coco17: dict[str, ControlPoint]
    is_ghost: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "detection_index": self.detection_index,
            "track_id": self.track_id,
            "is_ghost": self.is_ghost,
            "coco17": {name: point.as_dict() for name, point in self.coco17.items()},
            **self.controls.as_dict(),
        }


class TorsoControlPipeline:
    """Derive and stabilize torso controls for every tracked person."""

    def __init__(
        self,
        *,
        temporal_settings: TorsoTemporalSettings | None = None,
        min_confidence: float = 0.15,
    ) -> None:
        self.min_confidence = min_confidence
        self.temporal_filter = TorsoTemporalFilter(temporal_settings)

    def reset(self) -> None:
        """Clear temporal history for all people."""
        self.temporal_filter.reset()

    def process(self, key_points: sv.KeyPoints) -> list[TrackedTorsoControls]:
        """Convert RF-DETR keypoints into per-person torso controls.

        Explicit tracking metadata is preferred. Detection indices are used as
        a fallback only for untracked single-frame input.
        """
        if key_points.xy.ndim != 3 or key_points.xy.shape[1] < 17:
            raise ValueError(f"key_points.xy must have shape (N, K>=17, 2), got {key_points.xy.shape}")

        results: list[TrackedTorsoControls] = []
        for detection_index in range(len(key_points)):
            track_id = track_id_at(key_points, detection_index)
            resolved_track_id = detection_index if track_id is None else track_id
            xy = np.asarray(key_points.xy[detection_index, :17], dtype=np.float64)
            confidence = (
                np.asarray(key_points.keypoint_confidence[detection_index, :17], dtype=np.float64)
                if key_points.keypoint_confidence is not None
                else None
            )
            visible = (
                np.asarray(key_points.visible[detection_index, :17], dtype=bool)
                if key_points.visible is not None
                else None
            )
            controls = derive_torso_controls(
                xy,
                confidence=confidence,
                visible=visible,
                min_confidence=self.min_confidence,
            )
            filtered = self.temporal_filter.apply(controls, track_id=resolved_track_id)
            ghost = is_track_ghost(key_points, detection_index)
            point_source: PointSource = "interpolated" if ghost else "detected"
            resolved_confidence = np.ones(17, dtype=np.float64) if confidence is None else confidence
            resolved_visible = np.ones(17, dtype=bool) if visible is None else visible
            coco17 = {
                name: ControlPoint(
                    xy[index].copy(),
                    float(resolved_confidence[index]),
                    bool(
                        resolved_visible[index]
                        and resolved_confidence[index] >= self.min_confidence
                        and np.isfinite(xy[index]).all()
                    ),
                    point_source,
                )
                for index, name in enumerate(COCO17_KEYPOINT_NAMES)
            }
            results.append(
                TrackedTorsoControls(
                    detection_index=detection_index,
                    track_id=resolved_track_id,
                    controls=filtered,
                    coco17=coco17,
                    is_ghost=ghost,
                )
            )
        return results


def torso_frame_as_dict(
    controls: list[TrackedTorsoControls],
    *,
    frame_index: int,
    timestamp_sec: float | None = None,
) -> dict[str, object]:
    """Build one JSON-serializable animation frame."""
    frame: dict[str, object] = {
        "frame_index": frame_index,
        "people": [item.as_dict() for item in controls],
    }
    if timestamp_sec is not None:
        frame["timestamp_sec"] = float(timestamp_sec)
    return frame
