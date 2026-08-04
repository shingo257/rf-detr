# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Hold limits, motion prediction, ghost advance, and output capping."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import supervision as sv

from rfdetr_demo.tracking.bbox import detection_confidence
from rfdetr_demo.tracking.keypoints_ops import merge_key_points, shift_key_points
from rfdetr_demo.tracking.track_models import TrackSnapshot, box_center, in_center_lane
from rfdetr_demo.tracking.types import TrackDiagnostic, TrackPipelineResult, TrackPipelineStats

if TYPE_CHECKING:
    from rfdetr_demo.tracking.types import PersonTrackSettings


class TrackHoldSupport:
    """Hold / motion / output helpers mixed into ``TrackStore``.

    Expects ``settings``, ``frame_width``, ``_tracks``, and ``_sticky_track_id``
    on ``self``.
    """

    settings: PersonTrackSettings
    frame_width: int
    _tracks: list[TrackSnapshot]
    _sticky_track_id: int | None

    def _max_missed_for(self, track: TrackSnapshot) -> int:
        if self.settings.sticky_center_track and track.sticky:
            return self.settings.sticky_max_missed
        return self.settings.max_missed

    def _expected_count(self) -> int:
        return max(0, self.settings.expected_person_count)

    def _hold_limit_for(self, track: TrackSnapshot, current_output_count: int) -> int:
        """Extend hold when output is below the expected person count."""
        base = self._max_missed_for(track)
        expected = self._expected_count()
        if expected > 0 and self.settings.fill_below_expected and current_output_count < expected:
            return base + self.settings.fill_extra_missed
        return base

    def _cap_output(
        self,
        output_parts: list[sv.KeyPoints],
        ghost_flags: list[bool],
        track_ids: list[int],
        diagnostics: list[TrackDiagnostic],
    ) -> tuple[list[sv.KeyPoints], list[bool], list[int], list[TrackDiagnostic]]:
        """Drop lowest-priority tracks when count exceeds ``expected_person_count``."""
        expected = self._expected_count()
        if expected <= 0 or len(output_parts) <= expected:
            return output_parts, ghost_flags, track_ids, diagnostics

        scored: list[tuple[int, int, float]] = []
        for index, (key_points, is_ghost) in enumerate(zip(output_parts, ghost_flags, strict=True)):
            scored.append(
                (
                    index,
                    0 if is_ghost else 1,
                    detection_confidence(key_points, 0),
                ),
            )
        scored.sort(key=lambda row: (row[1], row[2]), reverse=True)
        keep = {row[0] for row in scored[:expected]}
        dropped_ids = {track_ids[index] for index in range(len(track_ids)) if index not in keep}
        if dropped_ids:
            self._tracks = [track for track in self._tracks if track.track_id not in dropped_ids]

        trimmed_parts: list[sv.KeyPoints] = []
        trimmed_flags: list[bool] = []
        trimmed_ids: list[int] = []
        trimmed_diagnostics: list[TrackDiagnostic] = []
        for index in sorted(keep):
            trimmed_parts.append(output_parts[index])
            trimmed_flags.append(ghost_flags[index])
            trimmed_ids.append(track_ids[index])
            trimmed_diagnostics.append(diagnostics[index])
        return trimmed_parts, trimmed_flags, trimmed_ids, trimmed_diagnostics

    def _finalize_output(
        self,
        output_parts: list[sv.KeyPoints],
        ghost_flags: list[bool],
        track_ids: list[int],
        diagnostics: list[TrackDiagnostic],
        *,
        raw_count: int,
        nms_count: int,
    ) -> TrackPipelineResult:
        output_parts, ghost_flags, track_ids, diagnostics = self._cap_output(
            output_parts,
            ghost_flags,
            track_ids,
            diagnostics,
        )
        ghost_count = sum(1 for is_ghost in ghost_flags if is_ghost)
        stabilized = merge_key_points(output_parts, ghost_flags=ghost_flags, track_ids=track_ids)
        return TrackPipelineResult(
            key_points=stabilized,
            stats=TrackPipelineStats(
                raw_count=raw_count,
                nms_count=nms_count,
                active_track_count=len(stabilized),
                ghost_count=ghost_count,
            ),
            diagnostics=diagnostics,
        )

    def _predicted_box(self, track: TrackSnapshot) -> np.ndarray:
        """Return the track box advanced by one step of its velocity."""
        if not self.settings.motion_enabled:
            return track.box
        shift = np.array(
            [track.velocity[0], track.velocity[1], track.velocity[0], track.velocity[1]],
            dtype=np.float64,
        )
        return track.box + shift

    def _update_velocity(self, track: TrackSnapshot, new_box: np.ndarray) -> None:
        """Blend the observed center displacement into the track velocity (EMA)."""
        if not self.settings.motion_enabled:
            return
        old_cx, old_cy = box_center(track.box)
        new_cx, new_cy = box_center(new_box)
        measured = np.array([new_cx - old_cx, new_cy - old_cy], dtype=np.float64)
        beta = self.settings.motion_smoothing
        track.velocity = beta * track.velocity + (1.0 - beta) * measured
        max_speed = self.settings.motion_max_speed
        if max_speed > 0:
            np.clip(track.velocity, -max_speed, max_speed, out=track.velocity)

    def _advance_ghost(self, track: TrackSnapshot) -> sv.KeyPoints:
        """Move a held track forward by its velocity and return shifted keypoints."""
        if not self.settings.motion_enabled or not np.any(track.velocity):
            return track.key_points
        dx = float(track.velocity[0])
        dy = float(track.velocity[1])
        track.box = self._predicted_box(track)
        track.key_points = shift_key_points(track.key_points, dx, dy)
        return track.key_points

    def _maybe_mark_sticky(self, track: TrackSnapshot) -> None:
        if not self.settings.sticky_center_track:
            return
        cx, _ = box_center(track.box)
        if in_center_lane(cx, self.frame_width, self.settings.center_x_fraction):
            track.sticky = True
            self._sticky_track_id = track.track_id


__all__ = ["TrackHoldSupport"]
