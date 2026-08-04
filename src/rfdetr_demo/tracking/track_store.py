# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Track state store: association, hold, and ghost flags."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import supervision as sv

from rfdetr_demo.tracking.bbox import detection_bbox, detection_confidence, iou, nms_detection_indices
from rfdetr_demo.tracking.keypoints_ops import single_detection_key_points, subset_key_points
from rfdetr_demo.tracking.track_hold import TrackHoldSupport
from rfdetr_demo.tracking.track_match import match_tracks_to_detections
from rfdetr_demo.tracking.track_models import (
    TrackSnapshot,
    box_center,
    in_center_lane,
    track_diagnostic,
)
from rfdetr_demo.tracking.types import PersonTrackSettings, TrackDiagnostic, TrackPipelineResult, TrackPipelineStats


@dataclass
class TrackStore(TrackHoldSupport):
    """Own track snapshots, NMS, association, and short missed-frame hold."""

    settings: PersonTrackSettings = field(default_factory=PersonTrackSettings)
    frame_width: int = 1280
    _tracks: list[TrackSnapshot] = field(default_factory=list, init=False, repr=False)
    _next_track_id: int = field(default=0, init=False, repr=False)
    _sticky_track_id: int | None = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        """Clear track history."""
        self._tracks.clear()
        self._next_track_id = 0
        self._sticky_track_id = None

    def apply(self, key_points: sv.KeyPoints, frame_index: int) -> TrackPipelineResult:
        """Return stabilized detections for one frame."""
        del frame_index
        raw_count = len(key_points)

        if not self.settings.enabled:
            return TrackPipelineResult(
                key_points=key_points,
                stats=TrackPipelineStats(
                    raw_count=raw_count,
                    nms_count=raw_count,
                    active_track_count=raw_count,
                    ghost_count=0,
                ),
                diagnostics=[],
            )

        if raw_count == 0:
            return self._apply_empty_frame(raw_count)

        nms_indices = nms_detection_indices(key_points, self.settings.nms_iou_threshold)
        nms_key_points = subset_key_points(key_points, nms_indices)
        nms_count = len(nms_key_points)

        detection_boxes: list[np.ndarray] = []
        for detection_index in range(nms_count):
            box = detection_bbox(nms_key_points, detection_index)
            detection_boxes.append(
                box.copy() if box is not None else np.zeros(4, dtype=np.float64),
            )

        track_boxes = [self._predicted_box(track) for track in self._tracks]
        matched, unmatched_tracks, unmatched_detections = match_tracks_to_detections(
            track_boxes,
            detection_boxes,
            match_iou_threshold=self.settings.match_iou_threshold,
        )

        if self.settings.sticky_center_track and self._sticky_track_id is not None:
            sticky_index = next(
                (index for index, track in enumerate(self._tracks) if track.track_id == self._sticky_track_id),
                None,
            )
            if sticky_index is not None and sticky_index in unmatched_tracks and unmatched_detections:
                sticky_box = track_boxes[sticky_index]
                best_det: int | None = None
                best_iou = self.settings.match_iou_threshold
                for detection_index in unmatched_detections:
                    score = iou(sticky_box, detection_boxes[detection_index])
                    cx, _ = box_center(detection_boxes[detection_index])
                    if in_center_lane(cx, self.frame_width, self.settings.center_x_fraction) and score >= best_iou:
                        best_iou = score
                        best_det = detection_index
                if best_det is not None:
                    matched.append((sticky_index, best_det))
                    unmatched_tracks.discard(sticky_index)
                    unmatched_detections.discard(best_det)

        matched_track_indices = {track_index for track_index, _ in matched}
        output_parts: list[sv.KeyPoints] = []
        ghost_flags: list[bool] = []
        track_ids: list[int] = []
        diagnostics: list[TrackDiagnostic] = []

        for track_index, detection_index in matched:
            track = self._tracks[track_index]
            snapshot = single_detection_key_points(nms_key_points, detection_index)
            det_box = detection_boxes[detection_index].copy()
            self._update_velocity(track, det_box)
            track.key_points = snapshot
            track.box = det_box
            track.missed = 0
            self._maybe_mark_sticky(track)
            output_parts.append(snapshot)
            ghost_flags.append(False)
            track_ids.append(track.track_id)
            diagnostics.append(track_diagnostic(track, is_ghost=False, matched_this_frame=True))

        for detection_index in sorted(unmatched_detections):
            if len(self._tracks) >= self.settings.max_tracks:
                break
            if self.settings.hysteresis_enabled:
                confidence = detection_confidence(nms_key_points, detection_index)
                if confidence < self.settings.new_track_min_confidence:
                    continue
            snapshot = single_detection_key_points(nms_key_points, detection_index)
            track = TrackSnapshot(
                track_id=self._next_track_id,
                key_points=snapshot,
                box=detection_boxes[detection_index].copy(),
                missed=0,
            )
            self._next_track_id += 1
            self._maybe_mark_sticky(track)
            self._tracks.append(track)
            matched_track_indices.add(len(self._tracks) - 1)
            output_parts.append(snapshot)
            ghost_flags.append(False)
            track_ids.append(track.track_id)
            diagnostics.append(track_diagnostic(track, is_ghost=False, matched_this_frame=True))

        for track_index in sorted(unmatched_tracks):
            track = self._tracks[track_index]
            track.missed += 1
            if track.missed <= self._hold_limit_for(track, len(output_parts)):
                ghost_key_points = self._advance_ghost(track)
                output_parts.append(ghost_key_points)
                ghost_flags.append(True)
                track_ids.append(track.track_id)
                diagnostics.append(track_diagnostic(track, is_ghost=True, matched_this_frame=False))

        self._tracks = [
            track
            for track_index, track in enumerate(self._tracks)
            if track_index in matched_track_indices or track.missed <= self._hold_limit_for(track, len(output_parts))
        ]

        return self._finalize_output(
            output_parts,
            ghost_flags,
            track_ids,
            diagnostics,
            raw_count=raw_count,
            nms_count=nms_count,
        )

    def _apply_empty_frame(self, raw_count: int) -> TrackPipelineResult:
        ghost_parts: list[sv.KeyPoints] = []
        ghost_flags: list[bool] = []
        track_ids: list[int] = []
        diagnostics: list[TrackDiagnostic] = []
        surviving_tracks: list[TrackSnapshot] = []
        for track in self._tracks:
            track.missed += 1
            if track.missed <= self._hold_limit_for(track, len(ghost_parts)):
                ghost_parts.append(self._advance_ghost(track))
                ghost_flags.append(True)
                track_ids.append(track.track_id)
                surviving_tracks.append(track)
                diagnostics.append(track_diagnostic(track, is_ghost=True, matched_this_frame=False))
        self._tracks = surviving_tracks
        return self._finalize_output(
            ghost_parts,
            ghost_flags,
            track_ids,
            diagnostics,
            raw_count=raw_count,
            nms_count=0,
        )


__all__ = ["TrackSnapshot", "TrackStore"]
