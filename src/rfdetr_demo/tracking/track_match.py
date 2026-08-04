# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Hungarian IoU matching between tracks and detections."""

from __future__ import annotations

import numpy as np

from rfdetr_demo.tracking.bbox import hungarian_maximize, iou


def match_tracks_to_detections(
    track_boxes: list[np.ndarray],
    detection_boxes: list[np.ndarray],
    *,
    match_iou_threshold: float,
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    """Return ``(track_idx, det_idx)`` pairs plus unmatched track/det indices.

    ``track_boxes`` are the per-track boxes to match against, which may be
    motion-predicted rather than the last observed position.
    """
    if not track_boxes or not detection_boxes:
        return [], set(range(len(track_boxes))), set(range(len(detection_boxes)))

    cost = np.zeros((len(track_boxes), len(detection_boxes)), dtype=np.float64)
    for track_index, track_box in enumerate(track_boxes):
        for detection_index, det_box in enumerate(detection_boxes):
            cost[track_index, detection_index] = iou(track_box, det_box)

    pairs = hungarian_maximize(cost)
    matched: list[tuple[int, int]] = []
    used_tracks: set[int] = set()
    used_detections: set[int] = set()
    for track_index, detection_index in pairs:
        if cost[track_index, detection_index] < match_iou_threshold:
            continue
        matched.append((track_index, detection_index))
        used_tracks.add(track_index)
        used_detections.add(detection_index)
    unmatched_tracks = set(range(len(track_boxes))) - used_tracks
    unmatched_detections = set(range(len(detection_boxes))) - used_detections
    return matched, unmatched_tracks, unmatched_detections


__all__ = ["match_tracks_to_detections"]
