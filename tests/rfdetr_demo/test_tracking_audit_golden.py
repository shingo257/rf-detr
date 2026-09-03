# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Golden fixtures for tracking audit regression (no confidential media required)."""

from __future__ import annotations

import json
from pathlib import Path

from rfdetr_demo.media.audit.tracking import TrackingFrameRecord, evaluate_center_tracking

GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "tracking_audit_baseline.json"


def _load_golden() -> dict[str, object]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_tracking_audit_baseline_fixture_is_stable() -> None:
    """Lock published baseline metrics and sticky improvement target."""
    golden = _load_golden()
    assert golden["run_id"] == "20260620T101049Z-ed8dabe5"
    assert golden["frame_count"] == 713
    metrics = golden["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["center_missing_stabilized_count"] == 68
    assert metrics["track_id_change_count"] == 31
    targets = golden["targets"]
    assert isinstance(targets, dict)
    assert targets["sticky_enabled_max_track_id_changes"] == 15


def test_evaluate_center_tracking_summary_keys_match_golden() -> None:
    """Synthetic records must expose the same summary keys as the golden schema."""
    golden = _load_golden()
    records = [
        TrackingFrameRecord(
            frame_index=0,
            processed_count=1,
            raw_count=1,
            nms_count=1,
            active_count=1,
            ghost_count=0,
            center_present_raw=True,
            center_present_stabilized=True,
            center_track_id=1,
            center_is_ghost=False,
            center_confidence=0.9,
            image_saved=False,
        ),
        TrackingFrameRecord(
            frame_index=1,
            processed_count=2,
            raw_count=1,
            nms_count=1,
            active_count=1,
            ghost_count=0,
            center_present_raw=True,
            center_present_stabilized=True,
            center_track_id=2,
            center_is_ghost=False,
            center_confidence=0.9,
            image_saved=False,
        ),
    ]
    result = evaluate_center_tracking(records)
    expected_keys = golden["summary_keys"]
    assert isinstance(expected_keys, list)
    for key in expected_keys:
        assert key in result
    assert result["track_id_change_count"] == 1


def test_sticky_target_is_stricter_than_baseline() -> None:
    """Ensure the sticky goal remains an improvement over the recorded baseline."""
    golden = _load_golden()
    metrics = golden["metrics"]
    targets = golden["targets"]
    assert isinstance(metrics, dict)
    assert isinstance(targets, dict)
    baseline_switches = int(metrics["track_id_change_count"])  # type: ignore[arg-type]
    sticky_cap = int(targets["sticky_enabled_max_track_id_changes"])  # type: ignore[arg-type]
    assert sticky_cap < baseline_switches
