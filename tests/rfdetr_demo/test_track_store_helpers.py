# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for track match and hold helpers."""

from __future__ import annotations

import numpy as np
import supervision as sv

from rfdetr_demo.tracking.track_hold import TrackHoldSupport
from rfdetr_demo.tracking.track_match import match_tracks_to_detections
from rfdetr_demo.tracking.track_models import TrackSnapshot, box_center, in_center_lane
from rfdetr_demo.tracking.types import PersonTrackSettings


def _empty_key_points() -> sv.KeyPoints:
    return sv.KeyPoints(
        xy=np.zeros((1, 17, 2), dtype=np.float32),
        visible=np.ones((1, 17), dtype=bool),
        keypoint_confidence=np.full((1, 17), 0.9, dtype=np.float32),
        detection_confidence=np.asarray([0.9], dtype=np.float32),
        data={"xyxy": np.asarray([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)},
    )


def test_match_tracks_to_detections_pairs_by_iou() -> None:
    track_boxes = [
        np.asarray([100.0, 100.0, 200.0, 300.0]),
        np.asarray([400.0, 100.0, 500.0, 300.0]),
    ]
    detection_boxes = [
        np.asarray([405.0, 105.0, 505.0, 305.0]),
        np.asarray([110.0, 110.0, 210.0, 310.0]),
    ]
    matched, unmatched_tracks, unmatched_dets = match_tracks_to_detections(
        track_boxes,
        detection_boxes,
        match_iou_threshold=0.15,
    )
    assert set(matched) == {(0, 1), (1, 0)}
    assert unmatched_tracks == set()
    assert unmatched_dets == set()


def test_match_empty_detections_marks_all_tracks_unmatched() -> None:
    track_boxes = [np.asarray([0.0, 0.0, 10.0, 10.0])]
    matched, unmatched_tracks, unmatched_dets = match_tracks_to_detections(
        track_boxes,
        [],
        match_iou_threshold=0.15,
    )
    assert matched == []
    assert unmatched_tracks == {0}
    assert unmatched_dets == set()


def test_box_center_and_center_lane() -> None:
    cx, cy = box_center(np.asarray([100.0, 50.0, 200.0, 150.0]))
    assert cx == 150.0
    assert cy == 100.0
    assert in_center_lane(400.0, 1280, (0.28, 0.48))
    assert not in_center_lane(100.0, 1280, (0.28, 0.48))


def test_hold_limit_extends_when_below_expected() -> None:
    support = TrackHoldSupport()
    support.settings = PersonTrackSettings(
        max_missed=2,
        expected_person_count=5,
        fill_below_expected=True,
        fill_extra_missed=3,
    )
    support.frame_width = 1280
    support._tracks = []
    support._sticky_track_id = None
    track = TrackSnapshot(
        track_id=0,
        key_points=_empty_key_points(),
        box=np.asarray([0.0, 0.0, 10.0, 10.0]),
    )
    assert support._hold_limit_for(track, current_output_count=3) == 5
    assert support._hold_limit_for(track, current_output_count=5) == 2


def test_sticky_hold_uses_sticky_max_missed() -> None:
    support = TrackHoldSupport()
    support.settings = PersonTrackSettings(
        max_missed=1,
        sticky_center_track=True,
        sticky_max_missed=4,
    )
    support.frame_width = 1280
    support._tracks = []
    support._sticky_track_id = None
    track = TrackSnapshot(
        track_id=1,
        key_points=_empty_key_points(),
        box=np.asarray([0.0, 0.0, 10.0, 10.0]),
        sticky=True,
    )
    assert support._max_missed_for(track) == 4
