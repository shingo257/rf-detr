# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for temporal quality improvements (1EuroFilter, bone length invariance)."""

from __future__ import annotations

import numpy as np
import supervision as sv

from rfdetr_demo.animation.dynamics import MascotMotionDynamics
from rfdetr_demo.animation.retarget import MascotRigPose
from rfdetr_demo.inference.temporal import (
    KeypointTemporalFilter,
    MotionPlausibilitySettings,
    OneEuroFilter,
)
from rfdetr_demo.tracking.keypoints_ops import compute_joint_rms_jitter


def test_one_euro_filter_reduces_jitter() -> None:
    """Verify that OneEuroFilter dampens high-frequency noise while responding to true movement."""
    fps = 30.0
    dt = 1.0 / fps
    filt = OneEuroFilter(min_cutoff=1.0, beta=0.05, d_cutoff=1.0)

    # Generate synthetic noisy trajectory (sine wave + Gaussian noise)
    t = np.linspace(0, 2.0, 60)
    true_signal = 100.0 + 20.0 * np.sin(2 * np.pi * 0.5 * t)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 3.0, size=t.shape)
    noisy_signal = true_signal + noise

    filtered_signal = []
    for val in noisy_signal:
        filtered_val = filt.filter(val, dt)
        filtered_signal.append(filtered_val)

    noisy_jitter = compute_joint_rms_jitter(noisy_signal[:, None, None])
    filtered_jitter = compute_joint_rms_jitter(np.array(filtered_signal)[:, None, None])

    # Expect significant reduction in RMS jitter (> 50% reduction)
    assert filtered_jitter < noisy_jitter * 0.5
    # Expect endpoint to closely match true signal
    assert abs(filtered_signal[-1] - true_signal[-1]) < 5.0


def test_keypoint_temporal_filter_with_one_euro() -> None:
    """Verify KeypointTemporalFilter equipped with OneEuroFilter filters sv.KeyPoints sequence."""
    settings = MotionPlausibilitySettings(
        enabled=True,
        use_one_euro_filter=True,
        min_cutoff=1.0,
        beta=0.05,
        bone_length_constraint_enabled=True,
    )
    filt = KeypointTemporalFilter(
        settings,
        frame_width=640,
        frame_height=480,
        fps=30.0,
        frame_stride=1,
    )

    rng = np.random.default_rng(123)
    num_frames = 30
    num_joints = 17
    base_xy = np.tile(np.linspace(100, 300, num_joints)[:, None], (1, 2)).astype(np.float32)

    raw_frames = []
    filtered_frames = []

    for idx in range(num_frames):
        noise = rng.normal(0, 4.0, size=(num_joints, 2)).astype(np.float32)
        frame_xy = base_xy + noise
        raw_frames.append(frame_xy)

        kp = sv.KeyPoints(
            xy=frame_xy[None, ...],
            visible=np.ones((1, num_joints), dtype=bool),
            data={"track_id": np.array([1], dtype=np.int64)},
        )
        filtered_kp = filt.apply(kp, frame_index=idx)
        filtered_frames.append(filtered_kp.xy[0])

    raw_jitter = compute_joint_rms_jitter(np.array(raw_frames))
    filtered_jitter = compute_joint_rms_jitter(np.array(filtered_frames))

    assert filt.stats.smoothed_joints > 0
    assert filtered_jitter < raw_jitter * 0.6


def test_bone_length_invariance_constraint() -> None:
    """Verify that bone length constraint prevents artificial joint stretching."""
    settings = MotionPlausibilitySettings(
        enabled=True,
        use_one_euro_filter=False,
        ema_alpha=1.0,  # disable EMA to test bone length constraint exclusively
        max_speed_fraction_per_sec=5.0,  # allow higher speed for test jump
        bone_length_constraint_enabled=True,
        bone_length_max_dev=0.10,  # tight max dev of 10%
    )
    filt = KeypointTemporalFilter(
        settings,
        frame_width=640,
        frame_height=480,
        fps=30.0,
        frame_stride=1,
    )

    # Establish baseline COCO keypoints (left shoulder=5, left elbow=7)
    base_xy = np.zeros((17, 2), dtype=np.float32)
    base_xy[5] = [200.0, 200.0]
    base_xy[7] = [200.0, 250.0]  # distance = 50px

    kp1 = sv.KeyPoints(
        xy=base_xy[None, ...],
        visible=np.ones((1, 17), dtype=bool),
        data={"track_id": np.array([1], dtype=np.int64)},
    )
    filt.apply(kp1, frame_index=0)

    # Stretch elbow joint significantly in frame 2 (distance = 90px -> 80% deviation)
    stretched_xy = base_xy.copy()
    stretched_xy[7] = [200.0, 290.0]
    kp2 = sv.KeyPoints(
        xy=stretched_xy[None, ...],
        visible=np.ones((1, 17), dtype=bool),
        data={"track_id": np.array([1], dtype=np.int64)},
    )
    filtered_kp2 = filt.apply(kp2, frame_index=1)

    p_pos = filtered_kp2.xy[0, 5]
    c_pos = filtered_kp2.xy[0, 7]
    corrected_dist = float(np.linalg.norm(c_pos - p_pos))

    # Corrected distance should be constrained well below 90px
    assert corrected_dist < 80.0
    assert filt.stats.bone_corrections > 0



def test_mascot_motion_dynamics_with_one_euro() -> None:
    """Verify MascotMotionDynamics smoothly transitions with 1EuroFilter enabled."""
    dynamics = MascotMotionDynamics(fps=30.0, use_one_euro=True)

    pose1 = MascotRigPose(confidence=1.0, root_x_px=0.0, root_y_px=0.0)
    out1 = dynamics.apply(pose1)
    assert out1.root_x_px == 0.0

    # Step change target
    pose2 = MascotRigPose(confidence=1.0, root_x_px=50.0, root_y_px=0.0)
    outputs = []
    for _ in range(10):
        outputs.append(dynamics.apply(pose2).root_x_px)

    # Output should monotonically move towards target without sharp spikes
    assert outputs[0] < outputs[5] < 50.0
