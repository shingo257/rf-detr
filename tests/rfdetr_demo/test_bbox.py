# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for bounding-box geometry helpers (tracking/bbox.py)."""

from __future__ import annotations

import numpy as np

from rfdetr_demo.tracking.bbox import containment_ratio, suppress_contained_detections


class TestContainmentRatio:
    def test_identical_boxes_have_full_containment(self) -> None:
        box = np.array([0.0, 0.0, 10.0, 10.0])

        assert containment_ratio(box, box) == 1.0

    def test_smaller_box_fully_inside_larger_box_is_fully_contained(self) -> None:
        larger = np.array([0.0, 0.0, 100.0, 100.0])
        smaller = np.array([40.0, 40.0, 60.0, 60.0])

        assert containment_ratio(smaller, larger) == 1.0
        # Symmetric: containment is defined relative to the smaller box's area
        # regardless of argument order.
        assert containment_ratio(larger, smaller) == 1.0

    def test_disjoint_boxes_have_zero_containment(self) -> None:
        box_a = np.array([0.0, 0.0, 10.0, 10.0])
        box_b = np.array([100.0, 100.0, 110.0, 110.0])

        assert containment_ratio(box_a, box_b) == 0.0

    def test_partial_overlap_is_fraction_of_smaller_area(self) -> None:
        # box_a: 10x10 = 100 area. box_b: 10x10 = 100 area, overlapping by 5x10 = 50.
        box_a = np.array([0.0, 0.0, 10.0, 10.0])
        box_b = np.array([5.0, 0.0, 15.0, 10.0])

        assert containment_ratio(box_a, box_b) == 0.5

    def test_zero_area_box_has_zero_containment(self) -> None:
        degenerate = np.array([5.0, 5.0, 5.0, 10.0])
        other = np.array([0.0, 0.0, 10.0, 10.0])

        assert containment_ratio(degenerate, other) == 0.0


class TestSuppressContainedDetections:
    def test_keeps_all_boxes_when_none_overlap(self) -> None:
        xyxy = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [100.0, 100.0, 110.0, 110.0],
            ],
        )
        confidence = np.array([0.9, 0.8])

        keep = suppress_contained_detections(xyxy, confidence)

        assert sorted(keep) == [0, 1]

    def test_suppresses_lower_confidence_box_contained_in_higher_confidence_box(self) -> None:
        # A tile-boundary split: a full-body box from one tile and a smaller,
        # lower-confidence partial-body box from the adjacent tile, mostly
        # contained within the full box but with IoU well below a 0.5 NMS gate.
        xyxy = np.array(
            [
                [0.0, 0.0, 40.0, 100.0],  # full box, high confidence
                [0.0, 0.0, 40.0, 30.0],  # partial box, contained, low confidence
            ],
        )
        confidence = np.array([0.9, 0.4])

        keep = suppress_contained_detections(xyxy, confidence, containment_threshold=0.8)

        assert keep == [0]

    def test_does_not_suppress_below_containment_threshold(self) -> None:
        xyxy = np.array(
            [
                [0.0, 0.0, 40.0, 100.0],
                # Only its bottom-right corner overlaps box 0: intersection is
                # 20x20 = 400 out of a 40x40 = 1600 area, i.e. 0.25 containment.
                [20.0, 80.0, 60.0, 120.0],
            ],
        )
        confidence = np.array([0.9, 0.4])

        keep = suppress_contained_detections(xyxy, confidence, containment_threshold=0.8)

        assert sorted(keep) == [0, 1]

    def test_empty_input_returns_empty_list(self) -> None:
        xyxy = np.zeros((0, 4))
        confidence = np.zeros((0,))

        assert suppress_contained_detections(xyxy, confidence) == []

    def test_single_box_is_kept(self) -> None:
        xyxy = np.array([[0.0, 0.0, 10.0, 10.0]])
        confidence = np.array([0.5])

        assert suppress_contained_detections(xyxy, confidence) == [0]
