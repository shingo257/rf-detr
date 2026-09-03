# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Load and normalize mascot rig assets for the renderer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

UInt8Array: TypeAlias = NDArray[np.uint8]
Float32Array: TypeAlias = NDArray[np.float32]
PartMetadata: TypeAlias = dict[str, Any]
Layer: TypeAlias = tuple[PartMetadata, UInt8Array]


@dataclass(slots=True)
class MascotRigAssets:
    """Normalized manifest metadata and image arrays used during rendering."""

    rig_dir: Path
    manifest: dict[str, Any]
    render_mode: str
    mesh_profile: dict[str, Any]
    face_profile: dict[str, Any]
    torso_profile: dict[str, Any]
    rigid_props: list[dict[str, Any]]
    width: int
    height: int
    layers: list[Layer]
    full_body_layer: UInt8Array
    expression_base_layer: UInt8Array
    grid_x: Float32Array
    grid_y: Float32Array
    residual_layer: UInt8Array
    pivots: dict[str, tuple[float, float]]


def _load_rgba_canvas(
    rig_dir: Path,
    metadata: dict[str, Any],
    *,
    width: int,
    height: int,
) -> UInt8Array:
    """Load one cropped RGBA asset into a full-canvas array."""
    canvas: UInt8Array = np.zeros((height, width, 4), dtype=np.uint8)
    with Image.open(rig_dir / str(metadata["file"])) as image:
        crop: UInt8Array = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    x1, y1, x2, y2 = (int(value) for value in metadata["bbox"])
    canvas[y1:y2, x1:x2] = crop[: y2 - y1, : x2 - x1]
    return canvas


def _expression_base(
    full_body_layer: UInt8Array,
    face_profile: dict[str, Any],
    *,
    width: int,
    height: int,
) -> UInt8Array:
    """Create a mouth-neutral source layer for expression rendering."""
    expression_base_layer = full_body_layer.copy()
    mouth_profile = face_profile.get("mouth")
    if not isinstance(mouth_profile, dict):
        return expression_base_layer

    mouth_x = int(float(mouth_profile["center"][0]) * width)
    mouth_y = int(float(mouth_profile["center"][1]) * height)
    cover_rx = max(2, int(float(mouth_profile["cover_radius"][0]) * width))
    cover_ry = max(2, int(float(mouth_profile["cover_radius"][1]) * height))
    inpaint_mask: UInt8Array = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(inpaint_mask, (mouth_x, mouth_y), (cover_rx, cover_ry), 0, 0, 360, 255, -1)
    local_pixels = full_body_layer[inpaint_mask > 0]
    skin_pixels = local_pixels[
        (local_pixels[:, 3] > 240)
        & (local_pixels[:, 0] > 200)
        & (local_pixels[:, 1] > 130)
        & (local_pixels[:, 2] > 170)
    ]
    if skin_pixels.size:
        sampled_skin = np.median(skin_pixels[:, :3], axis=0).astype(np.uint8)
        expression_base_layer[inpaint_mask > 0, :3] = sampled_skin
    return expression_base_layer


def _expanded_layers_and_residual(layers: list[Layer], full_body: UInt8Array) -> tuple[list[Layer], UInt8Array]:
    """Expand layer seams and return the remaining full-body residual."""
    expanded_layers: list[Layer] = []
    seam_kernel: UInt8Array = np.ones((7, 7), dtype=np.uint8)
    for part, layer in layers:
        expanded_alpha = cv2.dilate(layer[:, :, 3], seam_kernel, iterations=1)
        expanded_alpha = np.minimum(expanded_alpha, full_body[:, :, 3])
        expanded = full_body.copy()
        expanded[:, :, 3] = expanded_alpha
        expanded_layers.append((part, expanded))

    union_alpha = np.maximum.reduce([layer[:, :, 3] for _, layer in expanded_layers])
    residual = full_body.copy()
    residual[:, :, 3] = (full_body[:, :, 3].astype(np.float32) * (1.0 - union_alpha.astype(np.float32) / 255.0)).astype(
        np.uint8
    )
    return expanded_layers, residual


def _default_pivots(width: int, height: int) -> dict[str, tuple[float, float]]:
    """Scale the historical 760x777 rig pivots to the requested canvas."""
    scale_x = width / 760.0
    scale_y = height / 777.0
    return {
        "Body": (380.0 * scale_x, 600.0 * scale_y),
        "Head": (380.0 * scale_x, 410.0 * scale_y),
        "Left Arm": (240.0 * scale_x, 500.0 * scale_y),
        "Left Hand": (240.0 * scale_x, 500.0 * scale_y),
        "Right Arm": (520.0 * scale_x, 500.0 * scale_y),
        "Right Hand": (520.0 * scale_x, 500.0 * scale_y),
        "Left Leg": (300.0 * scale_x, 625.0 * scale_y),
        "Left Foot": (300.0 * scale_x, 625.0 * scale_y),
        "Right Leg": (460.0 * scale_x, 625.0 * scale_y),
        "Right Foot": (460.0 * scale_x, 625.0 * scale_y),
        "Bell": (380.0 * scale_x, 600.0 * scale_y),
    }


def load_mascot_rig_assets(rig_dir: Path) -> MascotRigAssets:
    """Load a rig manifest and prepare all renderer-owned image arrays."""
    manifest: dict[str, Any] = json.loads((rig_dir / "manifest.json").read_text(encoding="utf-8"))
    render_mode = str(manifest.get("render_mode", "layered"))
    mesh_profile = dict(manifest.get("mesh_profile", {}))
    face_profile = dict(manifest.get("face_profile", {}))
    torso_profile: dict[str, Any] = {
        "protect_enabled": False,
        "center_x": 0.50,
        "center_y": 0.58,
        "radius_x": 0.26,
        "radius_y": 0.32,
        "feather": 0.20,
    }
    if "torso_profile" in manifest:
        torso_profile.update(manifest["torso_profile"])
    rigid_props = [dict(item) for item in manifest.get("rigid_props", []) if isinstance(item, dict)]
    width = int(manifest["canvas"]["width"])
    height = int(manifest["canvas"]["height"])

    parts = cast(list[PartMetadata], manifest["parts"])
    layers: list[Layer] = []
    for part in sorted(parts, key=lambda item: int(item["z"])):
        layer: UInt8Array = np.zeros((height, width, 4), dtype=np.uint8)
        if render_mode != "continuous_mesh":
            layer = _load_rgba_canvas(rig_dir, part, width=width, height=height)
        layers.append((part, layer))

    full_meta = cast(dict[str, Any], manifest["full_body"])
    full_body = _load_rgba_canvas(rig_dir, full_meta, width=width, height=height)
    full_body_layer = full_body.copy()
    expression_base_layer = _expression_base(full_body_layer, face_profile, width=width, height=height)
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    expanded_layers, residual_layer = _expanded_layers_and_residual(layers, full_body)

    pivots = _default_pivots(width, height)
    for part, _ in expanded_layers:
        if "pivot" in part:
            pivots[str(part["name"])] = cast(tuple[float, float], tuple(float(value) for value in part["pivot"]))

    return MascotRigAssets(
        rig_dir=rig_dir,
        manifest=manifest,
        render_mode=render_mode,
        mesh_profile=mesh_profile,
        face_profile=face_profile,
        torso_profile=torso_profile,
        rigid_props=rigid_props,
        width=width,
        height=height,
        layers=expanded_layers,
        full_body_layer=full_body_layer,
        expression_base_layer=expression_base_layer,
        grid_x=grid_x.astype(np.float32),
        grid_y=grid_y.astype(np.float32),
        residual_layer=residual_layer,
        pivots=pivots,
    )
