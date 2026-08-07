# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for detection-based tracking (Stage 1 counter)."""

from __future__ import annotations

import numpy as np
import supervision as sv

from rfdetr_demo.inference.callbacks import make_detection_track_callback
from rfdetr_demo.tracking.keypoints_ops import detections_to_key_points, track_ids_from_key_points
from rfdetr_demo.tracking.pipeline import PersonTrackPipeline
from rfdetr_demo.tracking.types import PersonTrackSettings


def _person_detections(boxes: list[tuple[float, float, float, float]]) -> sv.Detections:
    xyxy = np.asarray(boxes, dtype=np.float32).reshape(len(boxes), 4)
    return sv.Detections(
        xyxy=xyxy,
        confidence=np.full((len(boxes),), 0.9, dtype=np.float32),
        class_id=np.ones((len(boxes),), dtype=np.int64),  # COCO_PERSON_CLASS_ID = 1
    )


def test_detections_to_key_points_carries_boxes() -> None:
    detections = _person_detections([(10.0, 20.0, 30.0, 60.0), (100.0, 100.0, 140.0, 200.0)])
    key_points = detections_to_key_points(detections)

    assert len(key_points) == 2
    assert key_points.data["xyxy"].shape == (2, 4)
    np.testing.assert_allclose(key_points.data["xyxy"][1], [100.0, 100.0, 140.0, 200.0])
    # Joints are absent so appearance ReID falls back to the box.
    assert key_points.visible is not None
    assert not key_points.visible.any()


class _FakeDetectionModel:
    def __init__(self, per_frame: list[sv.Detections]) -> None:
        self._per_frame = per_frame
        self._index = 0

    def predict(self, _frame_rgb: object, **_kwargs: object) -> sv.Detections:
        detections = self._per_frame[self._index]
        self._index += 1
        return detections


def test_detection_track_callback_assigns_stable_ids() -> None:
    frame0 = _person_detections([(100.0, 100.0, 160.0, 300.0), (400.0, 100.0, 460.0, 300.0)])
    frame1 = _person_detections([(110.0, 100.0, 170.0, 300.0), (410.0, 100.0, 470.0, 300.0)])
    model = _FakeDetectionModel([frame0, frame1])
    pipeline = PersonTrackPipeline(settings=PersonTrackSettings(enabled=True), frame_width=640)
    stats: dict[str, int] = {"processed_frames": 0, "total_detections": 0}

    captured: list[sv.KeyPoints] = []
    original = PersonTrackPipeline.apply

    def _spy(self: PersonTrackPipeline, key_points: sv.KeyPoints, frame_index: int, frame: object = None):
        result = original(self, key_points, frame_index, frame)
        captured.append(result.key_points)
        return result

    PersonTrackPipeline.apply = _spy  # type: ignore[method-assign]
    try:
        callback = make_detection_track_callback(model, 0.5, True, stats, pipeline)
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        callback(blank, 0)
        callback(blank, 1)
    finally:
        PersonTrackPipeline.apply = original  # type: ignore[method-assign]

    ids_frame0 = sorted(t for t in track_ids_from_key_points(captured[0]) if t is not None)
    ids_frame1 = sorted(t for t in track_ids_from_key_points(captured[1]) if t is not None)
    assert ids_frame0 == [0, 1]
    # The two people barely moved, so ids persist across frames.
    assert ids_frame1 == [0, 1]
    assert stats["frame_active_tracks"] == 2
    # Two stable people over two frames => two distinct ids total.
    assert stats["unique_track_ids"] == 2


def test_detection_track_callback_accumulates_live_detections_excluding_ghosts() -> None:
    # Frame 0: two people detected. Frame 1: only one is detected, so the
    # tracker holds the missing one as a ghost (default max_missed=2) rather
    # than dropping it immediately.
    frame0 = _person_detections([(100.0, 100.0, 160.0, 300.0), (400.0, 100.0, 460.0, 300.0)])
    frame1 = _person_detections([(110.0, 100.0, 170.0, 300.0)])
    model = _FakeDetectionModel([frame0, frame1])
    pipeline = PersonTrackPipeline(settings=PersonTrackSettings(enabled=True), frame_width=640)
    stats: dict[str, int] = {
        "processed_frames": 0,
        "total_detections": 0,
        "total_live_detections": 0,
    }

    callback = make_detection_track_callback(model, 0.5, True, stats, pipeline)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    callback(blank, 0)
    callback(blank, 1)

    # Frame 0: 2 live, 0 ghosts. Frame 1: 1 live + 1 held ghost.
    assert stats["frame_ghost_tracks"] == 1
    assert stats["frame_live_tracks"] == 1
    # total_detections (the legacy metric) includes the held ghost: 2 + 2 = 4.
    assert stats["total_detections"] == 4
    # total_live_detections excludes ghost holds: 2 + 1 = 3.
    assert stats["total_live_detections"] == 3


def test_tiled_detector_runs_model_per_tile_and_merges() -> None:
    from rfdetr_demo.inference.callbacks import _build_person_detector

    class _OneBoxPerTileModel:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, image: np.ndarray, **_kwargs: object) -> sv.Detections:
            self.calls += 1
            height, width = image.shape[:2]
            return sv.Detections(
                xyxy=np.asarray([[2.0, 2.0, width - 2.0, height - 2.0]], dtype=np.float32),
                confidence=np.asarray([0.9], dtype=np.float32),
                class_id=np.ones((1,), dtype=np.int64),
            )

    model = _OneBoxPerTileModel()
    detect = _build_person_detector(model, 0.25, tile_size=256, tile_overlap=64)
    frame_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
    detections = detect(frame_rgb)

    # A 512x512 frame with 256px tiles is split into multiple tiles (model called > once).
    assert model.calls > 1
    assert len(detections) >= 1


def test_tile_boundary_duplicate_survives_iou_nms_but_not_containment_pass() -> None:
    from rfdetr_demo.inference.callbacks import _suppress_tile_boundary_duplicates

    # Simulates a person straddling a tile boundary: a full-body box merged
    # from one tile and a much smaller, lower-confidence partial-body box left
    # over from the adjacent tile. Their IoU (0.04) is far below a typical 0.5
    # merge-NMS threshold — the union is dominated by the large box — but the
    # smaller box is 100% inside the larger one (containment ratio 1.0).
    detections = sv.Detections(
        xyxy=np.asarray(
            [
                [100.0, 100.0, 300.0, 300.0],  # full body, high confidence
                [100.0, 100.0, 140.0, 140.0],  # partial body, fully contained
            ],
            dtype=np.float32,
        ),
        confidence=np.asarray([0.9, 0.4], dtype=np.float32),
        class_id=np.ones((2,), dtype=np.int64),
    )

    merged = _suppress_tile_boundary_duplicates(detections)

    assert len(merged) == 1
    assert merged.confidence[0] == np.float32(0.9)


def test_containment_pass_keeps_genuinely_distinct_people() -> None:
    from rfdetr_demo.inference.callbacks import _suppress_tile_boundary_duplicates

    detections = sv.Detections(
        xyxy=np.asarray(
            [
                [100.0, 100.0, 140.0, 200.0],
                [300.0, 100.0, 340.0, 200.0],  # a different person, no overlap
            ],
            dtype=np.float32,
        ),
        confidence=np.asarray([0.9, 0.85], dtype=np.float32),
        class_id=np.ones((2,), dtype=np.int64),
    )

    merged = _suppress_tile_boundary_duplicates(detections)

    assert len(merged) == 2


class _FakePoseModel:
    def __init__(self) -> None:
        self.crop_sizes: list[tuple[int, int]] = []

    def predict(self, frame_rgb: np.ndarray, **_kwargs: object) -> sv.KeyPoints:
        self.crop_sizes.append((frame_rgb.shape[0], frame_rgb.shape[1]))
        xy = np.zeros((1, 17, 2), dtype=np.float32)
        xy[0, 5] = (5.0, 5.0)
        xy[0, 6] = (10.0, 5.0)
        return sv.KeyPoints(
            xy=xy,
            visible=np.ones((1, 17), dtype=bool),
            detection_confidence=np.asarray([0.9], dtype=np.float32),
        )


def test_pose_subset_runs_on_top_k_largest_boxes() -> None:
    # Three people of different box sizes; pose_topk=2 should crop the 2 largest.
    frame0 = _person_detections(
        [(10.0, 10.0, 30.0, 40.0), (100.0, 100.0, 200.0, 400.0), (300.0, 100.0, 340.0, 180.0)],
    )
    model = _FakeDetectionModel([frame0])
    pose_model = _FakePoseModel()
    pipeline = PersonTrackPipeline(settings=PersonTrackSettings(enabled=True), frame_width=640)
    stats: dict[str, int] = {"processed_frames": 0, "total_detections": 0}
    callback = make_detection_track_callback(
        model,
        0.5,
        True,
        stats,
        pipeline,
        keypoint_model=pose_model,
        pose_topk=2,
    )
    callback(np.zeros((480, 640, 3), dtype=np.uint8), 0)

    # Pose ran on exactly the two largest boxes.
    assert len(pose_model.crop_sizes) == 2
    # The largest box is cropped first (crops carry a ~15% pad for full limbs).
    assert pose_model.crop_sizes[0][0] > pose_model.crop_sizes[1][0]
    assert pose_model.crop_sizes[0][0] >= 300


def test_detection_track_keeps_low_confidence_when_hysteresis_off() -> None:
    # Detection thresholding already gates; the tracker must not re-gate on
    # confidence or it collapses the count (regression: 28 -> 15 people).
    low_conf = sv.Detections(
        xyxy=np.asarray([[100.0, 100.0, 160.0, 300.0]], dtype=np.float32),
        confidence=np.asarray([0.35], dtype=np.float32),
        class_id=np.ones((1,), dtype=np.int64),
    )
    model = _FakeDetectionModel([low_conf])
    pipeline = PersonTrackPipeline(
        settings=PersonTrackSettings(enabled=True, hysteresis_enabled=False),
        frame_width=640,
    )
    stats: dict[str, int] = {"processed_frames": 0, "total_detections": 0}
    callback = make_detection_track_callback(model, 0.25, True, stats, pipeline)
    callback(np.zeros((480, 640, 3), dtype=np.uint8), 0)

    assert stats["frame_active_tracks"] == 1
