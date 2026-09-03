# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for center-person tracking audit evaluation helpers."""

from __future__ import annotations

import pytest

from rfdetr_demo.media.audit.tracking import (
    TrackingFrameRecord,
    center_x_range,
    evaluate_center_tracking,
)


def _record(
    *,
    frame_index: int,
    center_present_stabilized: bool = True,
    center_present_raw: bool = True,
    center_track_id: int | None = 1,
    center_is_ghost: bool = False,
) -> TrackingFrameRecord:
    return TrackingFrameRecord(
        frame_index=frame_index,
        processed_count=frame_index + 1,
        raw_count=1,
        nms_count=1,
        active_count=1 if center_present_stabilized else 0,
        ghost_count=1 if center_is_ghost else 0,
        center_present_raw=center_present_raw,
        center_present_stabilized=center_present_stabilized,
        center_track_id=center_track_id,
        center_is_ghost=center_is_ghost,
        center_confidence=0.9 if center_present_stabilized else None,
        image_saved=False,
    )


def test_center_x_range_scales_with_width() -> None:
    x_min, x_max = center_x_range(1000)
    assert x_min == pytest.approx(280.0)
    assert x_max == pytest.approx(480.0)


def test_evaluate_center_tracking_ok_when_stable() -> None:
    records = [_record(frame_index=i) for i in range(5)]
    result = evaluate_center_tracking(records)
    assert "OK" in result["verdict"]
    assert result["center_missing_stabilized_count"] == 0
    assert result["track_id_change_count"] == 0


def test_evaluate_center_tracking_flags_id_switch() -> None:
    records = [
        _record(frame_index=0, center_track_id=1),
        _record(frame_index=1, center_track_id=2),
        _record(frame_index=2, center_track_id=2),
    ]
    result = evaluate_center_tracking(records)
    assert result["track_id_change_count"] == 1
    assert "ID 切替" in "".join(result["issues"])


def test_evaluate_center_tracking_empty() -> None:
    result = evaluate_center_tracking([])
    assert result["verdict"] == "データなし"
