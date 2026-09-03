# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for animation-oriented torso controls."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import supervision as sv
from PIL import Image

from rfdetr_demo.animation import puppet_render as legacy_puppet_render
from rfdetr_demo.animation.dynamics import MascotMotionDynamics
from rfdetr_demo.animation.metrics import evaluate_torso_sequence
from rfdetr_demo.animation.pipeline import TorsoControlPipeline, torso_frame_as_dict
from rfdetr_demo.animation.puppet_assets import load_mascot_rig_assets
from rfdetr_demo.animation.puppet_cli import parse_args
from rfdetr_demo.animation.puppet_continuous import ContinuousMeshCompositor
from rfdetr_demo.animation.puppet_layered import LayeredSpriteCompositor
from rfdetr_demo.animation.puppet_mesh import (
    add_mesh_rotation,
    anchor_angle_deg,
    apply_rigid_prop_constraints,
    mesh_weight,
)
from rfdetr_demo.animation.puppet_renderer import LayeredMascotRenderer
from rfdetr_demo.animation.puppet_timeline import interpolate_pose, resampled_targets
from rfdetr_demo.animation.puppet_video import render_puppet_video
from rfdetr_demo.animation.retarget import FukkachanRetargeter, MascotRigPose
from rfdetr_demo.animation.temporal import TorsoTemporalFilter, TorsoTemporalSettings
from rfdetr_demo.animation.torso import TORSO_CONTROL_NAMES, derive_torso_controls
from rfdetr_demo.animation.video_export import render_torso_controls, run_torso_animation_export


def _pose(*, x_offset: float = 0.0, confidence: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    xy = np.zeros((17, 2), dtype=np.float64)
    xy[0] = [150.0 + x_offset, 70.0]
    xy[1] = [135.0 + x_offset, 60.0]
    xy[2] = [165.0 + x_offset, 60.0]
    xy[3] = [120.0 + x_offset, 65.0]
    xy[4] = [180.0 + x_offset, 65.0]
    xy[5] = [100.0 + x_offset, 100.0]
    xy[6] = [200.0 + x_offset, 100.0]
    xy[7] = [85.0 + x_offset, 165.0]
    xy[8] = [215.0 + x_offset, 165.0]
    xy[9] = [75.0 + x_offset, 225.0]
    xy[10] = [225.0 + x_offset, 225.0]
    xy[11] = [110.0 + x_offset, 300.0]
    xy[12] = [190.0 + x_offset, 300.0]
    xy[13] = [120.0 + x_offset, 400.0]
    xy[14] = [180.0 + x_offset, 400.0]
    xy[15] = [115.0 + x_offset, 500.0]
    xy[16] = [185.0 + x_offset, 500.0]
    return xy, np.full(17, confidence, dtype=np.float64)


def test_neutral_pose_derives_expected_torso_geometry() -> None:
    xy, confidence = _pose()
    controls = derive_torso_controls(xy, confidence=confidence)

    assert tuple(controls.points) == TORSO_CONTROL_NAMES
    np.testing.assert_allclose(controls.points["neck_base"].xy, [150.0, 100.0])
    np.testing.assert_allclose(controls.points["pelvis_center"].xy, [150.0, 300.0])
    np.testing.assert_allclose(controls.points["spine_mid"].xy, [150.0, 200.0])
    assert controls.parameters["spine_curvature"] == pytest.approx(0.0)
    assert controls.parameters["torso_height_px"] == pytest.approx(200.0)
    assert controls.confidence == pytest.approx(1.0)


def test_counter_rotated_shoulders_and_hips_create_signed_s_curve() -> None:
    xy, confidence = _pose()
    xy[5] = [90.0, 120.0]
    xy[6] = [210.0, 80.0]
    xy[11] = [100.0, 280.0]
    xy[12] = [200.0, 320.0]

    controls = derive_torso_controls(xy, confidence=confidence)

    assert controls.parameters["torso_twist_2d_deg"] < 0.0
    assert controls.parameters["spine_curvature"] < 0.0
    linear_upper = controls.points["neck_base"].xy * 0.75 + controls.points["pelvis_center"].xy * 0.25
    linear_lower = controls.points["neck_base"].xy * 0.25 + controls.points["pelvis_center"].xy * 0.75
    upper_offset = controls.points["spine_upper"].xy - linear_upper
    lower_offset = controls.points["spine_lower"].xy - linear_lower
    np.testing.assert_allclose(upper_offset, -lower_offset)
    assert np.linalg.norm(upper_offset) > 0.0


def test_missing_required_joint_marks_all_derived_points_invisible() -> None:
    xy, confidence = _pose()
    confidence[11] = 0.01

    controls = derive_torso_controls(xy, confidence=confidence)

    assert controls.confidence == 0.0
    assert not any(point.visible for point in controls.points.values())


def test_temporal_filter_smooths_motion_and_fills_short_gap() -> None:
    settings = TorsoTemporalSettings(ema_alpha=0.5, max_gap_frames=2, gap_confidence_decay=0.5)
    temporal_filter = TorsoTemporalFilter(settings)
    first_xy, first_confidence = _pose(confidence=0.5)
    second_xy, second_confidence = _pose(x_offset=20.0, confidence=0.5)

    temporal_filter.apply(derive_torso_controls(first_xy, confidence=first_confidence))
    second = temporal_filter.apply(derive_torso_controls(second_xy, confidence=second_confidence))
    assert 150.0 < second.points["pelvis_center"].xy[0] < 170.0

    missing_confidence = second_confidence.copy()
    missing_confidence[5:7] = 0.0
    missing_confidence[11:13] = 0.0
    held = temporal_filter.apply(derive_torso_controls(second_xy, confidence=missing_confidence))
    assert held.points["pelvis_center"].visible
    assert held.points["pelvis_center"].source == "interpolated"
    assert held.points["pelvis_center"].confidence == pytest.approx(0.25)


def test_temporal_filter_expires_gap_and_keeps_track_states_separate() -> None:
    temporal_filter = TorsoTemporalFilter(TorsoTemporalSettings(max_gap_frames=1))
    xy, confidence = _pose()
    temporal_filter.apply(derive_torso_controls(xy, confidence=confidence), track_id=1)
    shifted_xy, shifted_confidence = _pose(x_offset=100.0)
    independent = temporal_filter.apply(
        derive_torso_controls(shifted_xy, confidence=shifted_confidence),
        track_id=2,
    )
    assert independent.points["pelvis_center"].xy[0] == pytest.approx(250.0)

    confidence[[5, 6, 11, 12]] = 0.0
    missing = derive_torso_controls(xy, confidence=confidence)
    assert temporal_filter.apply(missing, track_id=1).points["pelvis_center"].visible
    assert not temporal_filter.apply(missing, track_id=1).points["pelvis_center"].visible


def test_sequence_metrics_report_normalized_jitter_and_dropout() -> None:
    frames = []
    for x_offset in (0.0, 10.0, 0.0):
        xy, confidence = _pose(x_offset=x_offset)
        frames.append(derive_torso_controls(xy, confidence=confidence))

    metrics = evaluate_torso_sequence(frames)

    assert metrics.frame_count == 3
    assert metrics.dropout_rate == 0.0
    assert metrics.jitter_rms_fraction == pytest.approx(0.1)
    assert metrics.mean_confidence == pytest.approx(1.0)


def test_invalid_shapes_fail_with_actionable_message() -> None:
    with pytest.raises(ValueError, match="shape \\(17, 2\\)"):
        derive_torso_controls(np.zeros((16, 2), dtype=np.float64))


def test_pipeline_accepts_supervision_keypoints_and_exports_frame() -> None:
    xy, confidence = _pose()
    key_points = sv.KeyPoints(
        xy=xy[np.newaxis, ...],
        keypoint_confidence=confidence[np.newaxis, ...],
        visible=np.ones((1, 17), dtype=bool),
    )

    tracked = TorsoControlPipeline().process(key_points)
    frame = torso_frame_as_dict(tracked, frame_index=12, timestamp_sec=0.4)

    assert tracked[0].track_id == 0
    assert frame["frame_index"] == 12
    assert frame["timestamp_sec"] == pytest.approx(0.4)
    assert frame["people"][0]["points"]["pelvis_center"]["xy"] == [150.0, 300.0]
    assert frame["people"][0]["coco17"]["left_shoulder"]["xy"] == [100.0, 100.0]
    assert frame["people"][0]["coco17"]["left_shoulder"]["source"] == "detected"


def test_torso_overlay_changes_visible_control_pixels() -> None:
    xy, confidence = _pose()
    key_points = sv.KeyPoints(
        xy=xy[np.newaxis, ...],
        keypoint_confidence=confidence[np.newaxis, ...],
        visible=np.ones((1, 17), dtype=bool),
    )
    people = TorsoControlPipeline().process(key_points)
    frame = np.zeros((400, 320, 3), dtype=np.uint8)

    annotated = render_torso_controls(frame, people)

    assert np.count_nonzero(annotated) > 0


class _FakeKeypointModel:
    def predict(self, frame_rgb: np.ndarray, **_: object) -> sv.KeyPoints:
        xy, confidence = _pose()
        visible = np.zeros((1, 17), dtype=bool)
        visible[0, [5, 6, 11, 12]] = True
        return sv.KeyPoints(
            xy=xy[np.newaxis, ...],
            keypoint_confidence=confidence[np.newaxis, ...],
            visible=visible,
            detection_confidence=np.asarray([0.99], dtype=np.float32),
        )


def _write_test_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 400))
    assert writer.isOpened()
    for value in (0, 20, 40):
        writer.write(np.full((400, 320, 3), value, dtype=np.uint8))
    writer.release()


def test_video_export_writes_json_metrics_and_overlay(tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"
    json_path = tmp_path / "controls.json"
    overlay_path = tmp_path / "overlay.mp4"
    _write_test_video(source)
    progress: list[tuple[int, int]] = []

    summary = run_torso_animation_export(
        source_path=source,
        json_path=json_path,
        overlay_path=overlay_path,
        max_frames=2,
        model=_FakeKeypointModel(),
        progress_callback=lambda completed, total: progress.append((completed, total)),
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert summary["processed_frames"] == 2
    assert overlay_path.stat().st_size > 0
    assert len(payload["frames"]) == 2
    assert payload["metrics_by_track"]["0"]["frame_count"] == 2
    assert payload["source_frame_count"] == 3
    assert payload["source_frames_read"] == 2
    assert payload["complete_source"] is False
    assert progress == [(1, 2), (2, 2)]


def _person_dict() -> dict[str, object]:
    xy, confidence = _pose()
    key_points = sv.KeyPoints(
        xy=xy[np.newaxis, ...],
        keypoint_confidence=confidence[np.newaxis, ...],
        visible=np.ones((1, 17), dtype=bool),
    )
    return TorsoControlPipeline().process(key_points)[0].as_dict()


def test_retarget_reference_frame_is_neutral_and_motion_is_bounded() -> None:
    reference = _person_dict()
    retargeter = FukkachanRetargeter(reference)

    neutral = retargeter.apply(reference)
    assert neutral.root_x_px == pytest.approx(0.0)
    assert neutral.body_angle_deg == pytest.approx(0.0)
    assert neutral.left_arm_angle_deg == pytest.approx(0.0)

    moved = deepcopy(reference)
    moved["coco17"]["left_wrist"]["xy"] = [80.0, 180.0]
    moved["points"]["pelvis_center"]["xy"] = [250.0, 300.0]
    pose = retargeter.apply(moved)
    assert pose.left_arm_angle_deg != 0.0
    assert pose.root_x_px == pytest.approx(74.75)
    assert abs(pose.left_arm_angle_deg) <= 55.0


def test_retarget_detects_raised_foot_and_contact_state() -> None:
    reference = _person_dict()
    retargeter = FukkachanRetargeter(reference)
    moved = deepcopy(reference)
    moved["coco17"]["left_ankle"]["xy"][1] -= 60.0

    pose = retargeter.apply(moved)

    assert pose.left_foot_lift_px == pytest.approx(37.95)
    assert pose.left_foot_contact == 0.0
    assert pose.right_foot_contact == 1.0


def test_mesh_weight_applies_side_and_lower_body_masks() -> None:
    """Mesh weights should combine radial, side, and lower-body masks."""
    grid_y, grid_x = np.mgrid[0:5, 0:5]

    weight = mesh_weight(
        grid_x.astype(np.float32),
        grid_y.astype(np.float32),
        width=5,
        height=5,
        pivot=(2.0, 2.0),
        radius_x=2.0,
        radius_y=2.0,
        side="left",
        lower_bias=0.2,
    )

    assert weight.dtype == np.float32
    assert weight[2, 2] == pytest.approx(1.0)
    assert weight[0, 2] == pytest.approx(0.0)
    assert weight[2, 4] == pytest.approx(0.0)


def test_add_mesh_rotation_updates_displacements_about_pivot() -> None:
    """A weighted quarter turn should rotate offsets around the pivot."""
    grid_y, grid_x = np.mgrid[0:3, 0:3]
    displacement_x = np.zeros((3, 3), dtype=np.float32)
    displacement_y = np.zeros((3, 3), dtype=np.float32)

    add_mesh_rotation(
        displacement_x,
        displacement_y,
        grid_x=grid_x.astype(np.float32),
        grid_y=grid_y.astype(np.float32),
        pivot=(1.0, 1.0),
        angle_deg=90.0,
        weight=np.ones((3, 3), dtype=np.float32),
    )

    assert displacement_x[1, 1] == pytest.approx(0.0)
    assert displacement_y[1, 1] == pytest.approx(0.0)
    assert displacement_x[1, 2] == pytest.approx(-1.0)
    assert displacement_y[1, 2] == pytest.approx(1.0)


def test_anchor_angle_uses_mirrored_pose_and_profile_gain() -> None:
    """Left sprite anchors should follow the mirrored right-side pose."""
    pose = MascotRigPose(right_arm_angle_deg=24.0, left_arm_angle_deg=-9.0)

    angle = anchor_angle_deg("Left Arm", pose, {"arm_angle_gain": 0.5})

    assert angle == pytest.approx(12.0)


def test_rigid_prop_constraints_replace_local_soft_deformation() -> None:
    """Rigid props should share center translation without changing distant pixels."""
    grid_y, grid_x = np.mgrid[0:21, 0:21]
    grid_x = grid_x.astype(np.float32)
    grid_y = grid_y.astype(np.float32)
    displacement_x = grid_x * 0.2
    displacement_y = np.zeros((21, 21), dtype=np.float32)
    original_x = displacement_x.copy()

    apply_rigid_prop_constraints(
        displacement_x,
        displacement_y,
        grid_x=grid_x,
        grid_y=grid_y,
        width=21,
        height=21,
        rigid_props=[
            {
                "center": [0.5, 0.5],
                "radius": [0.1, 0.1],
                "feather": 0.2,
                "rotate": False,
            }
        ],
        pose=MascotRigPose(),
        mesh_profile={},
    )

    assert displacement_x[10, 9] > original_x[10, 9]
    assert displacement_x[10, 9] <= displacement_x[10, 10]
    assert displacement_x[0, 0] == pytest.approx(original_x[0, 0])
    assert np.count_nonzero(displacement_y) == 0


def test_mascot_rig_assets_normalize_profiles_layers_and_pivots(tmp_path: Path) -> None:
    """Rig assets should expose normalized renderer inputs in z-order."""
    rig_dir = tmp_path / "asset_rig"
    rig_dir.mkdir()
    full_body = Image.new("RGBA", (12, 10), (20, 180, 80, 255))
    part = Image.new("RGBA", (4, 4), (40, 120, 220, 255))
    full_body.save(rig_dir / "full_body.png")
    part.save(rig_dir / "body.png")
    part.save(rig_dir / "head.png")
    manifest = {
        "canvas": {"width": 12, "height": 10},
        "torso_profile": {"center_x": 0.44},
        "rigid_props": [{"name": "fan", "anchor": "Left Arm"}, "ignored"],
        "full_body": {"file": "full_body.png", "bbox": [0, 0, 12, 10]},
        "parts": [
            {
                "name": "Body",
                "file": "body.png",
                "z": 20,
                "bbox": [4, 4, 8, 8],
                "center": [6, 6],
            },
            {
                "name": "Head",
                "file": "head.png",
                "z": 10,
                "bbox": [2, 1, 6, 5],
                "center": [4, 3],
                "pivot": [3.5, 4.5],
            },
        ],
    }
    (rig_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assets = load_mascot_rig_assets(rig_dir)

    assert (assets.width, assets.height) == (12, 10)
    assert assets.render_mode == "layered"
    assert assets.torso_profile["protect_enabled"] is False
    assert assets.torso_profile["center_x"] == pytest.approx(0.44)
    assert assets.torso_profile["center_y"] == pytest.approx(0.58)
    assert assets.rigid_props == [{"name": "fan", "anchor": "Left Arm"}]
    assert [part_metadata["name"] for part_metadata, _ in assets.layers] == ["Head", "Body"]
    assert all(layer.shape == (10, 12, 4) for _, layer in assets.layers)
    assert assets.full_body_layer.shape == (10, 12, 4)
    assert assets.expression_base_layer.shape == (10, 12, 4)
    assert assets.residual_layer.shape == (10, 12, 4)
    assert assets.grid_x.dtype == np.float32
    assert assets.grid_y.dtype == np.float32
    assert assets.pivots["Head"] == pytest.approx((3.5, 4.5))


def test_continuous_mesh_assets_do_not_require_part_crops(tmp_path: Path) -> None:
    """Continuous mesh mode should load from the full-body sprite alone."""
    rig_dir = tmp_path / "continuous_asset_rig"
    rig_dir.mkdir()
    Image.new("RGBA", (8, 8), (20, 180, 80, 255)).save(rig_dir / "full_body.png")
    manifest = {
        "canvas": {"width": 8, "height": 8},
        "render_mode": "continuous_mesh",
        "full_body": {"file": "full_body.png", "bbox": [0, 0, 8, 8]},
        "parts": [
            {
                "name": "Body",
                "file": "missing-part.png",
                "z": 1,
                "bbox": [0, 0, 8, 8],
                "center": [4, 4],
            }
        ],
    }
    (rig_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assets = load_mascot_rig_assets(rig_dir)

    assert assets.render_mode == "continuous_mesh"
    assert len(assets.layers) == 1


def test_layered_renderer_produces_expected_frame_shape(tmp_path: Path) -> None:
    rig_dir = tmp_path / "rig"
    rig_dir.mkdir()
    rgba = Image.new("RGBA", (20, 20), (20, 180, 80, 255))
    rgba.save(rig_dir / "full_body.png")
    rgba.save(rig_dir / "body.png")
    manifest = {
        "canvas": {"width": 20, "height": 20},
        "full_body": {"file": "full_body.png", "bbox": [0, 0, 20, 20]},
        "parts": [
            {"name": "Body", "file": "body.png", "z": 10, "bbox": [0, 0, 20, 20], "center": [10, 10]},
        ],
    }
    (rig_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    renderer = LayeredMascotRenderer(rig_dir)
    frame = renderer.render(MascotRigPose(body_angle_deg=5.0))

    assert isinstance(renderer.compositor, LayeredSpriteCompositor)
    assert frame.shape == (20, 20, 3)
    assert np.count_nonzero(frame) > 0


def test_continuous_mesh_renderer_keeps_single_sprite_surface(tmp_path: Path) -> None:
    rig_dir = tmp_path / "mesh_rig"
    rig_dir.mkdir()
    rgba = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    for y in range(8, 35):
        for x in range(8, 33):
            rgba.putpixel((x, y), (20, 180, 80, 255))
    rgba.save(rig_dir / "full_body.png")
    names = ["Body", "Head", "Left Arm", "Right Arm", "Left Leg", "Right Leg"]
    parts = []
    for index, name in enumerate(names):
        filename = f"part_{index}.png"
        rgba.save(rig_dir / filename)
        parts.append(
            {
                "name": name,
                "file": filename,
                "z": index,
                "bbox": [0, 0, 40, 40],
                "center": [20, 20],
                "pivot": [20, 20],
            }
        )
    manifest = {
        "canvas": {"width": 40, "height": 40},
        "render_mode": "continuous_mesh",
        "use_residual_layer": False,
        "full_body": {"file": "full_body.png", "bbox": [0, 0, 40, 40]},
        "parts": parts,
    }
    (rig_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    pose = MascotRigPose(head_angle_deg=12.0, left_arm_angle_deg=18.0, spine_curve_px=3.0)
    renderer = LayeredMascotRenderer(rig_dir)
    frame = renderer.render(pose)

    assert isinstance(renderer.compositor, ContinuousMeshCompositor)
    assert frame.shape == (40, 40, 3)
    assert np.count_nonzero(np.any(frame < 245, axis=2)) > 100


def test_rigid_props_preserve_prop_aspect_under_arm_swing(tmp_path: Path) -> None:
    """Held props should rotate with the arm without soft-mesh squash."""
    rig_dir = tmp_path / "prop_rig"
    rig_dir.mkdir()
    rgba = Image.new("RGBA", (80, 80), (0, 0, 0, 0))
    for y in range(20, 70):
        for x in range(25, 60):
            rgba.putpixel((x, y), (40, 160, 70, 255))
    # Distinct pink rectangle used as a held prop near the left arm.
    for y in range(40, 58):
        for x in range(8, 24):
            rgba.putpixel((x, y), (240, 120, 160, 255))
    rgba.save(rig_dir / "full_body.png")
    names = ["Body", "Head", "Left Arm", "Right Arm", "Left Leg", "Right Leg"]
    parts = []
    for index, name in enumerate(names):
        filename = f"part_{index}.png"
        rgba.save(rig_dir / filename)
        parts.append(
            {
                "name": name,
                "file": filename,
                "z": index,
                "bbox": [0, 0, 80, 80],
                "center": [40.0, 40.0],
                "pivot": [40.0, 40.0] if name != "Left Arm" else [30.0, 45.0],
            }
        )

    def _pink_aspect(frame: np.ndarray) -> float:
        mask = (frame[:, :, 2] > 180) & (frame[:, :, 1] < 150) & (frame[:, :, 0] > 100) & (frame[:, :, 0] < 200)
        ys, xs = np.where(mask)
        assert len(xs) > 20
        return float((xs.max() - xs.min() + 1) / max(ys.max() - ys.min() + 1, 1))

    base_manifest = {
        "canvas": {"width": 80, "height": 80},
        "render_mode": "continuous_mesh",
        "use_residual_layer": False,
        "mesh_profile": {
            "arm_radius_x": 0.35,
            "arm_radius_y": 0.35,
            "arm_angle_gain": 1.0,
        },
        "full_body": {"file": "full_body.png", "bbox": [0, 0, 80, 80]},
        "parts": parts,
    }
    pose = MascotRigPose(right_arm_angle_deg=40.0)

    soft_dir = tmp_path / "soft"
    soft_dir.mkdir()
    for path in rig_dir.iterdir():
        (soft_dir / path.name).write_bytes(path.read_bytes())
    (soft_dir / "manifest.json").write_text(json.dumps(base_manifest), encoding="utf-8")
    soft_aspect = _pink_aspect(LayeredMascotRenderer(soft_dir).render(pose))

    rigid_manifest = {
        **base_manifest,
        "rigid_props": [
            {
                "name": "fan",
                "anchor": "Left Arm",
                "center": [0.20, 0.60],
                "radius": [0.14, 0.14],
                "feather": 0.08,
            }
        ],
    }
    (rig_dir / "manifest.json").write_text(json.dumps(rigid_manifest), encoding="utf-8")
    rigid_aspect = _pink_aspect(LayeredMascotRenderer(rig_dir).render(pose))

    # Source pink block is 16x18 → aspect ≈ 0.89. Soft warp should drift farther.
    assert abs(rigid_aspect - 16 / 18) < abs(soft_aspect - 16 / 18)


def test_motion_dynamics_preserves_first_pose_and_smooths_step_change() -> None:
    dynamics = MascotMotionDynamics(fps=30.0)
    neutral = MascotRigPose()
    assert dynamics.apply(neutral) == neutral

    target = replace(
        neutral,
        left_arm_angle_deg=40.0,
        head_angle_deg=12.0,
        mouth_open=1.0,
        confidence=0.8,
    )
    first = dynamics.apply(target)
    sequence = [first]
    for _ in range(29):
        sequence.append(dynamics.apply(target))

    assert 0.0 < first.left_arm_angle_deg < target.left_arm_angle_deg
    assert sequence[-1].left_arm_angle_deg > first.left_arm_angle_deg
    assert abs(sequence[-1].left_arm_angle_deg - target.left_arm_angle_deg) < 2.0
    assert 0.0 < first.mouth_open < target.mouth_open
    assert sequence[-1].mouth_open > 0.95
    assert sequence[-1].confidence == pytest.approx(0.8)


def test_motion_dynamics_reset_reanchors_without_lag() -> None:
    dynamics = MascotMotionDynamics(fps=30.0)
    dynamics.apply(MascotRigPose())
    dynamics.apply(MascotRigPose(root_x_px=80.0))
    dynamics.reset()
    target = MascotRigPose(root_x_px=-30.0)
    assert dynamics.apply(target) == target


def test_pose_interpolation_uses_shortest_angle_and_nearest_contact() -> None:
    start = MascotRigPose(head_angle_deg=170.0, left_foot_contact=1.0)
    end = MascotRigPose(head_angle_deg=-170.0, left_foot_contact=0.0)

    before_midpoint = interpolate_pose(start, end, 0.25)
    after_midpoint = interpolate_pose(start, end, 0.75)

    assert before_midpoint.head_angle_deg == pytest.approx(175.0)
    assert after_midpoint.head_angle_deg == pytest.approx(185.0)
    assert before_midpoint.left_foot_contact == 1.0
    assert after_midpoint.left_foot_contact == 0.0


def test_sparse_pose_resampling_holds_last_keyframe_to_source_end() -> None:
    keyframes = [
        (0, MascotRigPose(root_x_px=0.0)),
        (3, MascotRigPose(root_x_px=30.0)),
    ]

    output = resampled_targets(keyframes, source_frame_count=5)

    assert [index for index, _ in output] == [0, 1, 2, 3, 4]
    assert [pose.root_x_px for _, pose in output] == pytest.approx([0.0, 10.0, 20.0, 30.0, 30.0])


def test_legacy_puppet_render_facade_reexports_canonical_api() -> None:
    assert legacy_puppet_render.LayeredMascotRenderer is LayeredMascotRenderer
    assert legacy_puppet_render._interpolate_pose is interpolate_pose
    assert legacy_puppet_render._resampled_targets is resampled_targets
    assert legacy_puppet_render.render_puppet_video is render_puppet_video


def test_puppet_video_rejects_empty_control_sequence(tmp_path: Path) -> None:
    controls_path = tmp_path / "empty_controls.json"
    controls_path.write_text(json.dumps({"frames": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="Control JSON contains no frames"):
        render_puppet_video(
            controls_json=controls_path,
            rig_dir=tmp_path / "rig",
            output_path=tmp_path / "puppet.mp4",
        )


def test_puppet_cli_parses_rendering_options() -> None:
    args = parse_args(
        [
            "--controls",
            "controls.json",
            "--rig",
            "rig",
            "--output",
            "puppet.mp4",
            "--track-id",
            "4",
            "--no-dynamics",
            "--native-keyframes",
        ]
    )

    assert args.controls == Path("controls.json")
    assert args.rig == Path("rig")
    assert args.output == Path("puppet.mp4")
    assert args.track_id == 4
    assert args.no_dynamics is True
    assert args.native_keyframes is True
