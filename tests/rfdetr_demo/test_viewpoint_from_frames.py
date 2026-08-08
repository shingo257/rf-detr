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


class _FakeKeypointModel:
    """Returns a fixed set of person boxes for every frame it's asked to predict."""

    def __init__(self, boxes_per_call: list[list[tuple[float, float, float, float]]]) -> None:
        self._boxes_per_call = boxes_per_call
        self.calls = 0

    def predict(self, _frame_rgb: object, **_kwargs: object) -> sv.KeyPoints:
        boxes = self._boxes_per_call[self.calls]
        self.calls += 1
        num_persons = len(boxes)
        xy = np.zeros((num_persons, 17, 2), dtype=np.float32)
        return sv.KeyPoints(
            xy=xy,
            visible=np.zeros((num_persons, 17), dtype=bool),
            keypoint_confidence=np.zeros((num_persons, 17), dtype=np.float32),
            detection_confidence=np.full((num_persons,), 0.9, dtype=np.float32),
            data={"xyxy": np.asarray(boxes, dtype=np.float32).reshape(num_persons, 4)},
        )


def _blank_frame(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


class TestEstimateViewpointFromFrames:
    def test_aggregates_boxes_across_frames_and_uses_frame_height(self) -> None:
        model = _FakeKeypointModel(
            [
                [(0.0, 0.0, 40.0, 44.0), (300.0, 200.0, 342.0, 240.0)],
                [(500.0, 350.0, 538.0, 392.0)],
            ],
        )
        frames = [_blank_frame(720, 1280), _blank_frame(720, 1280)]

        estimate = estimate_viewpoint_from_frames(frames, model, threshold=0.4)

        assert model.calls == 2
        assert estimate.sample_count == 3

    def test_empty_frame_list_returns_low_confidence_estimate(self) -> None:
        model = _FakeKeypointModel([])

        estimate = estimate_viewpoint_from_frames([], model, threshold=0.4)

        assert estimate.sample_count == 0
        assert estimate.confidence == 0.0
        assert model.calls == 0

    def test_frames_without_boxes_are_skipped(self) -> None:
        model = _FakeKeypointModel([[], []])
        frames = [_blank_frame(480, 640), _blank_frame(480, 640)]

        estimate = estimate_viewpoint_from_frames(frames, model, threshold=0.4)

        assert model.calls == 2
        assert estimate.sample_count == 0
