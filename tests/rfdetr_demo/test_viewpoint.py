# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for camera-viewpoint estimation (overhead vs eye-level)."""

from __future__ import annotations

from rfdetr_demo.tracking.viewpoint import (
    ViewpointEstimate,
    estimate_camera_viewpoint,
    preset_for_viewpoint,
)


def _box(cx: float, cy: float, width: float, height: float) -> tuple[float, float, float, float]:
    return (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)


class TestEstimateCameraViewpoint:
    def test_overhead_boxes_are_squarish_with_uniform_size(self) -> None:
        # Bird's-eye view: near-square person silhouettes, size independent of y-position.
        boxes = [
            _box(cx=100.0, cy=50.0, width=40.0, height=44.0),
            _box(cx=300.0, cy=200.0, width=42.0, height=40.0),
            _box(cx=500.0, cy=350.0, width=38.0, height=42.0),
            _box(cx=700.0, cy=500.0, width=41.0, height=39.0),
            _box(cx=900.0, cy=650.0, width=39.0, height=43.0),
        ]

        estimate = estimate_camera_viewpoint(boxes, frame_height=720.0)

        assert estimate.viewpoint == "overhead"
        assert estimate.confidence == 1.0
        assert estimate.sample_count == 5

    def test_eye_level_boxes_are_tall_and_grow_with_y(self) -> None:
        # Ground-level view: very tall/narrow silhouettes, boxes near the bottom
        # (closer) are larger. Aspect ratio (3.0) and correlation both clearly
        # signal eye-level, so both signals agree.
        boxes = [
            _box(cx=100.0, cy=100.0, width=20.0, height=60.0),
            _box(cx=200.0, cy=250.0, width=35.0, height=105.0),
            _box(cx=300.0, cy=400.0, width=55.0, height=165.0),
            _box(cx=400.0, cy=550.0, width=80.0, height=240.0),
            _box(cx=500.0, cy=680.0, width=110.0, height=330.0),
        ]

        estimate = estimate_camera_viewpoint(boxes, frame_height=720.0)

        assert estimate.viewpoint == "eye_level"
        assert estimate.confidence == 1.0
        assert estimate.sample_count == 5

    def test_too_few_boxes_default_to_low_confidence_overhead(self) -> None:
        boxes = [_box(cx=100.0, cy=100.0, width=40.0, height=40.0)]

        estimate = estimate_camera_viewpoint(boxes, frame_height=720.0)

        assert estimate.viewpoint == "overhead"
        assert estimate.confidence == 0.0
        assert estimate.sample_count == 1

    def test_no_boxes_default_to_low_confidence_overhead(self) -> None:
        estimate = estimate_camera_viewpoint([], frame_height=720.0)

        assert estimate.viewpoint == "overhead"
        assert estimate.confidence == 0.0
        assert estimate.sample_count == 0

    def test_conflicting_signals_trust_correlation_over_aspect_ratio(self) -> None:
        # Tall-ish boxes (a weak, on-its-own-misleading eye-level signal from
        # aspect ratio) but size is symmetric around the frame center rather
        # than growing with y, so it does not correlate with vertical position
        # (a clear overhead signal). Real detector output on genuine overhead
        # footage keeps producing "tall" boxes (see TestRealWorldCalibration
        # below), so when the two signals disagree, correlation — the signal
        # that actually behaves as expected on real footage — wins.
        boxes = [
            _box(cx=100.0, cy=50.0, width=30.0, height=70.0),
            _box(cx=300.0, cy=200.0, width=34.0, height=68.0),
            _box(cx=500.0, cy=350.0, width=28.0, height=72.0),
            _box(cx=700.0, cy=500.0, width=34.0, height=68.0),
            _box(cx=900.0, cy=650.0, width=30.0, height=70.0),
        ]

        estimate = estimate_camera_viewpoint(boxes, frame_height=720.0)

        assert estimate.viewpoint == "overhead"
        assert estimate.confidence == 0.5

    def test_degenerate_zero_width_boxes_are_skipped(self) -> None:
        boxes = [
            (100.0, 100.0, 100.0, 140.0),  # zero width, ignored
            *[_box(cx=100.0 + i * 50, cy=100.0 + i * 100, width=40.0, height=42.0) for i in range(4)],
        ]

        estimate = estimate_camera_viewpoint(boxes, frame_height=720.0)

        assert estimate.sample_count == 4


class TestRealWorldCalibration:
    """Regression tests grounded in real footage sampled with the production detector.

    Six stock clips shot at different mounting heights were probed with
    ``rfdetr-demo probe-viewpoint`` (RF-DETR large): a chest-height handheld
    shot, a ground-level street shot, three shots elevated to varying degrees
    (a building floor, a traffic-light post, a CCTV pole over a dense crowd),
    and one true bird's-eye scramble-crossing shot. All six produced "tall"
    median aspect ratios (1.78–2.92) — even the crossing shot, which is about
    as overhead as it gets — because the detector still regresses tall boxes
    around people seen from directly above. Aspect ratio alone is therefore
    not a reliable overhead/eye-level discriminator with this detector; the
    size-vs-position correlation split cleanly instead (0.61-0.85 for the
    five non-overhead clips regardless of elevation, ~0.18 for the one true
    overhead shot). These cases encode that finding so the heuristic can't
    regress back to leaning on aspect ratio alone.
    """

    def test_true_overhead_clip_profile_classifies_as_overhead(self) -> None:
        # ~1.78 median aspect ratio (still "tall", not square) but weak
        # size-vs-position correlation, matching the scramble-crossing clip.
        boxes = [
            _box(cx=100.0, cy=50.0, width=50.0, height=89.0),
            _box(cx=300.0, cy=200.0, width=52.0, height=93.0),
            _box(cx=500.0, cy=350.0, width=48.0, height=85.0),
            _box(cx=700.0, cy=500.0, width=51.0, height=91.0),
            _box(cx=900.0, cy=650.0, width=49.0, height=87.0),
        ]

        estimate = estimate_camera_viewpoint(boxes, frame_height=1920.0)

        assert estimate.viewpoint == "overhead"
        assert estimate.confidence == 1.0

    def test_moderate_elevation_clip_profile_classifies_as_eye_level(self) -> None:
        # ~2.8 median aspect ratio and strong size-vs-position correlation,
        # matching the moderately-elevated pedestrian-street clip.
        boxes = [
            _box(cx=100.0, cy=100.0, width=22.0, height=62.0),
            _box(cx=200.0, cy=250.0, width=34.0, height=95.0),
            _box(cx=300.0, cy=400.0, width=52.0, height=146.0),
            _box(cx=400.0, cy=550.0, width=74.0, height=207.0),
            _box(cx=500.0, cy=680.0, width=100.0, height=280.0),
        ]

        estimate = estimate_camera_viewpoint(boxes, frame_height=1920.0)

        assert estimate.viewpoint == "eye_level"
        assert estimate.confidence == 1.0


class TestPresetForViewpoint:
    def test_overhead_maps_to_overhead_preset(self) -> None:
        estimate = ViewpointEstimate(
            viewpoint="overhead",
            confidence=1.0,
            median_aspect_ratio=1.0,
            size_position_correlation=0.0,
            sample_count=5,
        )

        assert preset_for_viewpoint(estimate) == "overhead"

    def test_eye_level_maps_to_eye_level_preset(self) -> None:
        estimate = ViewpointEstimate(
            viewpoint="eye_level",
            confidence=1.0,
            median_aspect_ratio=2.0,
            size_position_correlation=0.8,
            sample_count=5,
        )

        assert preset_for_viewpoint(estimate) == "eye-level"
