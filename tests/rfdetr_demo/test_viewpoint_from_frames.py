# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for rfdetr_demo.tracking.viewpoint.estimate_viewpoint_from_frames."""

from __future__ import annotations

import numpy as np
import supervision as sv

from rfdetr_demo.tracking.viewpoint import estimate_viewpoint_from_frames

_PERSON = 1
_NOT_PERSON = 2


class _FakeDetectionModel:
    """Returns a fixed set of (box, class_id) detections for every frame it's asked to predict."""

    def __init__(self, detections_per_call: list[list[tuple[tuple[float, float, float, float], int]]]) -> None:
        self._detections_per_call = detections_per_call
        self.calls = 0

    def predict(self, _frame_rgb: object, **_kwargs: object) -> sv.Detections:
        detections = self._detections_per_call[self.calls]
        self.calls += 1
        num_detections = len(detections)
        xyxy = np.asarray([box for box, _ in detections], dtype=np.float32).reshape(num_detections, 4)
        class_id = np.asarray([cls for _, cls in detections], dtype=np.int64)
        return sv.Detections(
            xyxy=xyxy,
            confidence=np.full((num_detections,), 0.9, dtype=np.float32),
            class_id=class_id,
        )


def _blank_frame(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


class TestEstimateViewpointFromFrames:
    def test_aggregates_person_boxes_across_frames(self) -> None:
        model = _FakeDetectionModel(
            [
                [((0.0, 0.0, 40.0, 44.0), _PERSON), ((300.0, 200.0, 342.0, 240.0), _PERSON)],
                [((500.0, 350.0, 538.0, 392.0), _PERSON)],
            ],
        )
        frames = [_blank_frame(720, 1280), _blank_frame(720, 1280)]

        estimate = estimate_viewpoint_from_frames(frames, model, threshold=0.4)

        assert model.calls == 2
        assert estimate.sample_count == 3

    def test_non_person_detections_are_filtered_out(self) -> None:
        model = _FakeDetectionModel(
            [
                [((0.0, 0.0, 40.0, 44.0), _PERSON), ((300.0, 200.0, 342.0, 240.0), _NOT_PERSON)],
            ],
        )
        frames = [_blank_frame(720, 1280)]

        estimate = estimate_viewpoint_from_frames(frames, model, threshold=0.4)

        assert estimate.sample_count == 1

    def test_empty_frame_list_returns_low_confidence_estimate(self) -> None:
        model = _FakeDetectionModel([])

        estimate = estimate_viewpoint_from_frames([], model, threshold=0.4)

        assert estimate.sample_count == 0
        assert estimate.confidence == 0.0
        assert model.calls == 0

    def test_frames_without_boxes_are_skipped(self) -> None:
        model = _FakeDetectionModel([[], []])
        frames = [_blank_frame(480, 640), _blank_frame(480, 640)]

        estimate = estimate_viewpoint_from_frames(frames, model, threshold=0.4)

        assert model.calls == 2
        assert estimate.sample_count == 0
