# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Mascot video orchestration built on the renderer and pure timeline helpers."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import cv2

from rfdetr_demo.animation.dynamics import MascotMotionDynamics
from rfdetr_demo.animation.puppet_renderer import LayeredMascotRenderer
from rfdetr_demo.animation.puppet_timeline import person_for_track, resampled_targets
from rfdetr_demo.animation.retarget import FukkachanRetargeter, MascotRigPose


def render_puppet_video(
    *,
    controls_json: Path,
    rig_dir: Path,
    output_path: Path,
    track_id: int = 0,
    pose_json: Path | None = None,
    dynamics_enabled: bool = True,
    resample_to_source_fps: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Render a mascot preview video and pose sidecar from control JSON."""
    data = json.loads(controls_json.read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    if not frames:
        raise ValueError("Control JSON contains no frames")
    reference_person = person_for_track(frames[0], track_id)
    if reference_person is None:
        raise ValueError(f"Track {track_id} is missing from the first frame")

    renderer = LayeredMascotRenderer(rig_dir)
    retargeter = FukkachanRetargeter(reference_person)
    frame_stride = max(int(data.get("frame_stride", 1)), 1)
    should_resample = bool(resample_to_source_fps and frame_stride > 1 and data.get("complete_source", False))
    output_fps = float(data["fps"]) if should_resample else float(data["fps"]) / frame_stride
    dynamics = MascotMotionDynamics(fps=output_fps) if dynamics_enabled else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")
    writer = cv2.VideoWriter(
        str(partial),
        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
        output_fps,
        (renderer.width, renderer.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {partial}")

    keyframes: list[tuple[int, MascotRigPose]] = []
    last_target = MascotRigPose()
    for frame in frames:
        person = person_for_track(frame, track_id)
        if person is not None:
            last_target = retargeter.apply(person)
        keyframes.append((int(frame["frame_index"]), last_target))
    targets = (
        resampled_targets(keyframes, source_frame_count=int(data.get("source_frame_count", 0)))
        if should_resample
        else keyframes
    )

    poses: list[dict[str, object]] = []
    try:
        for completed, (frame_index, target_pose) in enumerate(targets, start=1):
            pose = dynamics.apply(target_pose) if dynamics is not None else target_pose
            writer.write(renderer.render(pose))
            poses.append({"frame_index": frame_index, **pose.as_dict()})
            if progress_callback is not None:
                progress_callback(completed, len(targets))
    finally:
        writer.release()
    if output_path.exists():
        output_path.unlink()
    partial.replace(output_path)

    resolved_pose_json = pose_json or output_path.with_suffix(".poses.json")
    resolved_pose_json.write_text(json.dumps({"track_id": track_id, "poses": poses}, indent=2), encoding="utf-8")
    return {
        "output": str(output_path.resolve()),
        "poses": str(resolved_pose_json.resolve()),
        "frames": len(poses),
        "track_id": track_id,
        "dynamics": dynamics_enabled,
        "resampled": should_resample,
        "fps": output_fps,
    }
