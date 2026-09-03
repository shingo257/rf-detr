# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Hysteresis in the audit's center-track selection."""

from __future__ import annotations

import pytest

from rfdetr_demo.media.audit.tracking_annotate import find_center_track
from rfdetr_demo.tracking.types import TrackDiagnostic

FRAME_WIDTH = 1000  # center lane spans x in [280, 480], midpoint 380


def _diag(
    *,
    track_id: int,
    cx: float,
    is_ghost: bool = False,
    missed: int = 0,
) -> TrackDiagnostic:
    return TrackDiagnostic(
        track_id=track_id,
        cx=cx,
        cy=200.0,
        confidence=0.9,
        is_ghost=is_ghost,
        missed=missed,
        matched_this_frame=not is_ghost,
    )


def test_picks_closest_live_track_when_no_incumbent() -> None:
    tracks = [_diag(track_id=1, cx=470.0), _diag(track_id=2, cx=390.0)]
    result = find_center_track(tracks, FRAME_WIDTH)
    assert result is not None
    assert result.track_id == 2


def test_prefers_live_over_a_closer_ghost_when_no_incumbent() -> None:
    tracks = [_diag(track_id=1, cx=380.0, is_ghost=True), _diag(track_id=2, cx=420.0)]
    result = find_center_track(tracks, FRAME_WIDTH)
    assert result is not None
    assert result.track_id == 2


def test_keeps_incumbent_when_challenger_is_only_marginally_closer() -> None:
    tracks = [_diag(track_id=6, cx=400.0), _diag(track_id=4, cx=372.0)]
    result = find_center_track(tracks, FRAME_WIDTH, previous_track_id=6)
    assert result is not None
    assert result.track_id == 6


def test_switches_when_challenger_is_clearly_closer_to_midpoint() -> None:
    tracks = [_diag(track_id=6, cx=465.0), _diag(track_id=4, cx=380.0)]
    result = find_center_track(tracks, FRAME_WIDTH, previous_track_id=6)
    assert result is not None
    assert result.track_id == 4


def test_never_hands_role_from_live_incumbent_to_a_ghost_challenger() -> None:
    tracks = [_diag(track_id=6, cx=470.0), _diag(track_id=9, cx=380.0, is_ghost=True, missed=1)]
    result = find_center_track(tracks, FRAME_WIDTH, previous_track_id=6)
    assert result is not None
    assert result.track_id == 6


def test_yields_from_stale_ghost_incumbent_to_a_live_challenger() -> None:
    tracks = [
        _diag(track_id=6, cx=390.0, is_ghost=True, missed=3),
        _diag(track_id=4, cx=430.0),
    ]
    result = find_center_track(tracks, FRAME_WIDTH, previous_track_id=6)
    assert result is not None
    assert result.track_id == 4


def test_returns_none_when_no_track_in_lane() -> None:
    tracks = [_diag(track_id=1, cx=100.0), _diag(track_id=2, cx=900.0)]
    assert find_center_track(tracks, FRAME_WIDTH, previous_track_id=1) is None


def test_incumbent_that_left_the_lane_is_dropped_for_the_closest_remaining() -> None:
    tracks = [_diag(track_id=4, cx=300.0)]
    result = find_center_track(tracks, FRAME_WIDTH, previous_track_id=6)
    assert result is not None
    assert result.track_id == 4


@pytest.mark.parametrize(
    ("margin_fraction", "expected_id"),
    [
        pytest.param(0.1, 4, id="small-margin-switches"),
        pytest.param(1.5, 6, id="large-margin-holds"),
    ],
)
def test_switch_margin_fraction_controls_handover(margin_fraction: float, expected_id: int) -> None:
    tracks = [_diag(track_id=6, cx=410.0), _diag(track_id=4, cx=372.0)]
    result = find_center_track(
        tracks,
        FRAME_WIDTH,
        previous_track_id=6,
        switch_margin_fraction=margin_fraction,
    )
    assert result is not None
    assert result.track_id == expected_id
