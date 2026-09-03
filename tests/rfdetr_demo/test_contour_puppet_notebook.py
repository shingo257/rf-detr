# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[2] / "docs" / "cookbooks" / "rfdetr_keypoint_contour_puppet_colab_v1.ipynb"
)


def _load_notebook() -> dict:
    """Load the contour-puppet cookbook notebook."""
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _code_sources() -> list[str]:
    """Return complete source strings for every code cell."""
    return ["".join(cell.get("source", [])) for cell in _load_notebook()["cells"] if cell.get("cell_type") == "code"]


def _helper_namespace(mode: str) -> dict[str, object]:
    """Execute the self-contained geometry cell with lightweight test settings."""
    source = next(source for source in _code_sources() if "class TrackState" in source)
    namespace: dict[str, object] = {
        "cv2": cv2,
        "np": np,
        "dataclass": dataclass,
        "TARGET_TRACK_ID": None,
        "KEYPOINT_CONFIDENCE": 0.20,
        "POSE_SMOOTHING_ALPHA": 0.62,
        "BBOX_SMOOTHING_ALPHA": 0.32,
        "MAX_HOLD_FRAMES": 4,
        "MASK_THRESHOLD": 0.45,
        "MASK_CLOSE_RATIO": 0.008,
        "POSE_CONFIDENCE": 0.20,
        "SEG_CONFIDENCE": 0.25,
        "INFERENCE_IMAGE_SIZE": 960,
        "device_arg": "cpu",
        "PUPPET_SCALE": 0.72,
        "PUPPET_SIDE": "left",
        "PUPPET_GAP_IN_HEIGHT": 0.12,
        "PUPPET_MODE": mode,
        "PUPPET_STYLE": "slim_cute",
        "SLIM_ARM_WIDTH_RATIO": 0.042,
        "SLIM_LEG_WIDTH_RATIO": 0.060,
        "SLIM_SHOULDER_HALF_RATIO": 0.105,
        "SLIM_HIP_HALF_RATIO": 0.080,
        "CUTE_HEAD_RADIUS_X_RATIO": 0.165,
        "CUTE_HEAD_RADIUS_Y_RATIO": 0.145,
        "DRAW_EARS": True,
        "FILL_BGR": (250, 250, 245),
        "OUTLINE_BGR": (20, 20, 20),
        "OUTLINE_WIDTH": 4,
        "PART_OUTLINE_WIDTH": 3,
        "PRESERVE_PART_OUTLINES": True,
        "DRAW_INNER_GESTURE_LINES": False,
        "DRAW_FACE": True,
        "OVERLAY_OPACITY": 0.97,
    }
    exec(compile(source, "contour-puppet-helper-cell", "exec"), namespace)
    return namespace


def test_contour_puppet_notebook_code_cells_compile() -> None:
    """Ensure every notebook code cell is valid Python."""
    for index, source in enumerate(_code_sources()):
        compile(source, f"contour-puppet-cell-{index}", "exec")


@pytest.mark.parametrize(
    "required_marker",
    [
        pytest.param("pose_model.track(", id="tracked-pose"),
        pytest.param("seg_model.predict(", id="person-segmentation"),
        pytest.param("cv2.distanceTransform(", id="silhouette-width"),
        pytest.param("cv2.findContours(", id="line-contour"),
        pytest.param("cv2.warpAffine(", id="silhouette-retarget"),
        pytest.param("TARGET_TRACK_ID", id="person-selection"),
        pytest.param("PUPPET_MODE", id="render-mode"),
        pytest.param("PRESERVE_PART_OUTLINES", id="overlap-safe-part-lines"),
        pytest.param("_canonical_pose_from_bbox", id="missing-joint-completion"),
        pytest.param("part_line_mask", id="independent-part-line-layer"),
        pytest.param("contour_puppet_overlay.mp4", id="overlay-output"),
        pytest.param("contour_puppet_only.mp4", id="puppet-only-output"),
    ],
)
def test_contour_puppet_notebook_contains_required_pipeline(
    required_marker: str,
) -> None:
    """Keep the pose, silhouette, styling, and output stages in the cookbook."""
    assert required_marker in "\n".join(_code_sources())


def test_contour_puppet_notebook_is_clean_for_distribution() -> None:
    """Prevent local execution state or generated output from entering Git."""
    notebook = _load_notebook()
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        assert cell.get("execution_count") is None
        assert cell.get("outputs") == []


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param("capsule", id="pose-width-body"),
        pytest.param("silhouette", id="retargeted-mask"),
        pytest.param("hybrid", id="combined-body"),
    ],
)
def test_contour_puppet_notebook_renders_synthetic_person(mode: str) -> None:
    """Exercise the notebook renderer without model downloads or a private video."""
    namespace = _helper_namespace(mode)
    frame = np.full((512, 640, 3), 210, dtype=np.uint8)
    mask = np.zeros((512, 640), dtype=np.uint8)
    cv2.ellipse(mask, (320, 92), (45, 55), 0, 0, 360, 255, -1)
    cv2.fillConvexPoly(
        mask,
        np.asarray([[275, 140], [365, 140], [350, 300], [290, 300]], dtype=np.int32),
        255,
    )
    for point_a, point_b, width in (
        ((280, 150), (245, 285), 28),
        ((360, 150), (400, 285), 28),
        ((300, 285), (280, 475), 42),
        ((340, 285), (365, 475), 42),
    ):
        cv2.line(mask, point_a, point_b, 255, width)
    pose = np.asarray(
        [
            [320, 90],
            [304, 80],
            [336, 80],
            [285, 88],
            [355, 88],
            [280, 150],
            [360, 150],
            [255, 220],
            [385, 220],
            [245, 285],
            [400, 285],
            [300, 285],
            [340, 285],
            [290, 385],
            [355, 385],
            [280, 475],
            [365, 475],
        ],
        dtype=np.float32,
    )
    state_class = namespace["TrackState"]
    state = state_class(
        track_id=1,
        pose_xy=pose,
        pose_conf=np.ones(17, dtype=np.float32),
        bbox=np.asarray([230, 40, 410, 495], dtype=np.float32),
        mask=mask,
    )

    rendered, puppet_mask, target_pose, _ = namespace["_render_character"](
        frame.copy(),
        state,
        "overlay",
    )

    assert np.count_nonzero(puppet_mask) > 2_000
    assert np.any(rendered != frame)
    assert target_pose.shape == (17, 2)


def test_overlapping_limbs_keep_internal_part_lines() -> None:
    """Keep arm boundaries visible when both arms overlap the torso."""
    namespace = _helper_namespace("hybrid")
    frame = np.full((512, 640, 3), 210, dtype=np.uint8)
    mask = np.zeros((512, 640), dtype=np.uint8)
    cv2.ellipse(mask, (320, 92), (45, 55), 0, 0, 360, 255, -1)
    cv2.fillConvexPoly(
        mask,
        np.asarray([[270, 140], [370, 140], [355, 310], [285, 310]], dtype=np.int32),
        255,
    )
    cv2.line(mask, (295, 285), (285, 475), 255, 42)
    cv2.line(mask, (345, 285), (355, 475), 255, 42)
    pose = np.asarray(
        [
            [320, 90],
            [304, 80],
            [336, 80],
            [285, 88],
            [355, 88],
            [290, 150],
            [350, 150],
            [305, 220],
            [335, 220],
            [312, 285],
            [328, 285],
            [300, 285],
            [340, 285],
            [292, 385],
            [348, 385],
            [285, 475],
            [355, 475],
        ],
        dtype=np.float32,
    )
    state_class = namespace["TrackState"]
    state = state_class(
        track_id=1,
        pose_xy=pose,
        pose_conf=np.ones(17, dtype=np.float32),
        bbox=np.asarray([230, 40, 410, 495], dtype=np.float32),
        mask=mask,
    )

    rendered, _, target_pose, _ = namespace["_render_character"](
        frame.copy(),
        state,
        "stage",
    )

    outline_color = np.asarray(namespace["OUTLINE_BGR"], dtype=np.uint8)
    dark_pixels = np.all(rendered == outline_color, axis=2)
    for joint_index in (7, 8, 9, 10):
        x, y = np.rint(target_pose[joint_index]).astype(int)
        patch = dark_pixels[max(0, y - 12) : y + 13, max(0, x - 12) : x + 13]
        assert np.count_nonzero(patch) >= 8


def test_missing_first_frame_joints_receive_complete_body_pose() -> None:
    """Create every required joint even when the first detection has no keypoint confidence."""
    namespace = _helper_namespace("hybrid")
    state = namespace["TrackState"]()
    empty_xy = np.zeros((1, 17, 2), dtype=np.float32)
    empty_conf = np.zeros((1, 17), dtype=np.float32)
    bbox = np.asarray([[100, 40, 300, 440]], dtype=np.float32)
    selected = (
        0,
        empty_xy,
        empty_conf,
        bbox,
        np.asarray([0.9], dtype=np.float32),
        [7],
    )

    assert namespace["_update_pose_state"](state, selected)
    assert np.isfinite(state.pose_xy).all()
    assert state.pose_xy.shape == (17, 2)
    assert np.ptp(state.pose_xy[:, 0]) > 0
    assert np.ptp(state.pose_xy[:, 1]) > 0
    assert np.all(state.pose_conf >= namespace["KEYPOINT_CONFIDENCE"] * 0.50)


def test_auto_overlay_side_stays_fixed_after_first_resolution() -> None:
    """Prevent the overlay puppet from jumping across the dancer."""
    namespace = _helper_namespace("capsule")
    namespace["PUPPET_SIDE"] = "auto"
    state = namespace["TrackState"]()
    frame_shape = (720, 720, 3)

    first_bbox = np.asarray([250.0, 80.0, 470.0, 680.0], dtype=np.float32)
    resolved_side = namespace["_resolve_overlay_side"](
        first_bbox,
        frame_shape,
        state.placement_side,
    )
    state.placement_side = resolved_side

    moved_bbox = np.asarray([80.0, 80.0, 300.0, 680.0], dtype=np.float32)
    assert (
        namespace["_resolve_overlay_side"](
            moved_bbox,
            frame_shape,
            state.placement_side,
        )
        == resolved_side
    )


def test_locked_track_does_not_fall_back_to_another_person() -> None:
    """Hold the selected dancer when ByteTrack temporarily reports only another id."""
    namespace = _helper_namespace("capsule")
    state = namespace["TrackState"](
        track_id=1,
        bbox=np.asarray([250.0, 80.0, 470.0, 680.0], dtype=np.float32),
    )
    result = SimpleNamespace(
        orig_shape=(720, 720),
        keypoints=SimpleNamespace(
            xy=np.zeros((1, 17, 2), dtype=np.float32),
            conf=np.ones((1, 17), dtype=np.float32),
        ),
        boxes=SimpleNamespace(
            xyxy=np.asarray([[100.0, 80.0, 300.0, 680.0]], dtype=np.float32),
            conf=np.asarray([0.95], dtype=np.float32),
            id=np.asarray([2.0], dtype=np.float32),
        ),
    )

    assert namespace["_select_pose_person"](result, state) is None


def test_slim_cute_profile_renders_head_wider_than_torso() -> None:
    """Keep the default character cute and slender instead of copying baggy clothes."""
    namespace = _helper_namespace("capsule")
    bbox = np.asarray([230.0, 40.0, 410.0, 495.0], dtype=np.float32)
    pose = namespace["_canonical_pose_from_bbox"](bbox)
    pose[7] = [295.0, 240.0]
    pose[9] = [295.0, 330.0]
    pose[8] = [345.0, 240.0]
    pose[10] = [345.0, 330.0]

    source_mask = np.zeros((512, 640), dtype=np.uint8)
    cv2.ellipse(source_mask, (320, 92), (45, 55), 0, 0, 360, 255, -1)
    cv2.fillConvexPoly(
        source_mask,
        np.asarray([[275, 140], [365, 140], [350, 300], [290, 300]], dtype=np.int32),
        255,
    )
    state = namespace["TrackState"](
        track_id=1,
        pose_xy=pose,
        pose_conf=np.ones(17, dtype=np.float32),
        bbox=bbox,
        mask=source_mask,
    )
    stage = np.full((512, 640, 3), 240, dtype=np.uint8)
    _, puppet_mask, target_pose, _ = namespace["_render_character"](stage, state, "stage")

    def row_width(y_coordinate: float) -> int:
        occupied = np.flatnonzero(puppet_mask[int(round(y_coordinate))] > 0)
        return 0 if occupied.size == 0 else int(occupied[-1] - occupied[0] + 1)

    head_width = row_width(target_pose[0, 1])
    torso_y = float(np.mean(target_pose[[5, 6, 11, 12], 1]))
    torso_width = row_width(torso_y)
    assert head_width >= torso_width * 1.15
