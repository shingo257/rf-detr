# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for clip analysis issue derivation."""

from __future__ import annotations

from rfdetr_demo.tuning.analyze_clip_issues import derive_issues
from rfdetr_demo.tuning.analyze_clip_types import ClipAnalysis, FrameMetric, JointFrameMetric


def _analysis(
    *,
    frames: list[FrameMetric],
    avg_persons: float = 1.0,
    fps: float = 30.0,
    speed_rejections: int = 0,
    oscillation_corrections: int = 0,
) -> ClipAnalysis:
    return ClipAnalysis(
        source="clip.mp4",
        duration_sec=1.0,
        fps=fps,
        frame_stride=1,
        frames_analyzed=len(frames),
        avg_persons=avg_persons,
        motion_speed_rejections=speed_rejections,
        motion_oscillation_corrections=oscillation_corrections,
        frame_metrics=frames,
    )


def test_derive_issues_empty_frames() -> None:
    issues = derive_issues(_analysis(frames=[]), max_speed_px=100.0, fps=30.0)
    assert any("解析フレームが 0" in issue for issue in issues)


def test_derive_issues_person_count_varies() -> None:
    frames = [
        FrameMetric(frame_index=0, person_count=1),
        FrameMetric(frame_index=1, person_count=3),
    ]
    issues = derive_issues(_analysis(frames=frames), max_speed_px=100.0, fps=30.0)
    assert any("検出人数が変動" in issue for issue in issues)


def test_derive_issues_speed_over_limit() -> None:
    frames = [
        FrameMetric(
            frame_index=0,
            person_count=1,
            joints=[
                JointFrameMetric(
                    joint_index=0,
                    joint_name="nose",
                    x=0.0,
                    y=0.0,
                    confidence=0.9,
                    speed_px_per_sec=500.0,
                    displacement_px=50.0,
                    covariance_trace=None,
                ),
            ],
        ),
    ]
    issues = derive_issues(_analysis(frames=frames), max_speed_px=100.0, fps=30.0)
    assert any("人体上限" in issue for issue in issues)
