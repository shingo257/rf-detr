# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Layered-sprite composition for mascot rendering."""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from PIL import Image

from rfdetr_demo.animation.puppet_assets import MascotRigAssets, UInt8Array
from rfdetr_demo.animation.retarget import MascotRigPose


class LayeredSpriteCompositor:
    """Compose independently transformed mascot layers onto a BGR frame."""

    def __init__(
        self,
        assets: MascotRigAssets,
        *,
        background: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        """Initialize composition state from normalized rig assets."""
        self.manifest = assets.manifest
        self.width = assets.width
        self.height = assets.height
        self.background = background
        self.layers = assets.layers
        self.residual_layer = assets.residual_layer
        self.pivots = assets.pivots

    def _s_curve(self, layer: UInt8Array, amount_px: float) -> UInt8Array:
        """Apply a bounded horizontal S-curve to a full-canvas layer."""
        if abs(amount_px) < 0.05:
            return layer
        yy, xx = np.mgrid[0 : self.height, 0 : self.width]
        upper = self.height * 0.38
        lower = self.height * 0.90
        t = np.clip((yy - upper) / max(lower - upper, 1.0), 0.0, 1.0)
        shift = amount_px * np.sin(2.0 * np.pi * t)
        map_x = (xx - shift).astype(np.float32)
        map_y = yy.astype(np.float32)
        remapped: UInt8Array = cv2.remap(
            layer,
            map_x,
            map_y,
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return remapped

    def _transform(
        self,
        layer: UInt8Array,
        *,
        pivot: tuple[float, float],
        angle_deg: float,
        translation: tuple[float, float],
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> UInt8Array:
        """Apply one pivot-relative affine transform to a layer."""
        px, py = pivot
        radians = np.radians(angle_deg)
        cosine = float(np.cos(radians))
        sine = float(np.sin(radians))
        linear = np.asarray(
            [[cosine * scale_x, -sine * scale_y], [sine * scale_x, cosine * scale_y]],
            dtype=np.float64,
        )
        pivot_array = np.asarray([px, py], dtype=np.float64)
        offset = pivot_array + np.asarray(translation, dtype=np.float64) - linear @ pivot_array
        matrix = np.column_stack((linear, offset)).astype(np.float32)
        transformed: UInt8Array = cv2.warpAffine(
            layer,
            matrix,
            (self.width, self.height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        return transformed

    @staticmethod
    def _part_transform(name: str, pose: MascotRigPose) -> tuple[float, float, float]:
        """Resolve angle and scale controls for one named sprite layer."""
        body_follow = pose.body_angle_deg * 0.35
        if name in {"Left Arm", "Left Hand"}:
            return pose.right_arm_angle_deg + body_follow, 1.0, 1.0
        if name in {"Right Arm", "Right Hand"}:
            return pose.left_arm_angle_deg + body_follow, 1.0, 1.0
        if name in {"Left Leg", "Left Foot"}:
            extra = pose.right_foot_angle_deg if name == "Left Foot" else 0.0
            return pose.right_leg_angle_deg + extra, 1.0, 1.0
        if name in {"Right Leg", "Right Foot"}:
            extra = pose.left_foot_angle_deg if name == "Right Foot" else 0.0
            return pose.left_leg_angle_deg + extra, 1.0, 1.0
        if name == "Head":
            return pose.head_angle_deg + pose.body_angle_deg * 0.25, 1.0, 1.0
        if name in {"Body", "Bell"}:
            return pose.body_angle_deg, pose.body_scale_x, pose.body_scale_y
        if name == "Pouch":
            return pose.body_angle_deg * 0.45 + pose.right_arm_angle_deg * 0.18, 1.0, 1.0
        return 0.0, 1.0, 1.0

    @staticmethod
    def _part_translation(name: str, pose: MascotRigPose) -> tuple[float, float]:
        """Resolve root and foot-lift translation for one named layer."""
        root_x = pose.root_x_px
        root_y = pose.root_y_px
        if name in {"Left Leg", "Left Foot"}:
            extra_y = -root_y if pose.right_foot_contact >= 0.5 else -pose.right_foot_lift_px
            return root_x, root_y + extra_y
        if name in {"Right Leg", "Right Foot"}:
            extra_y = -root_y if pose.left_foot_contact >= 0.5 else -pose.left_foot_lift_px
            return root_x, root_y + extra_y
        return root_x, root_y

    def render(self, pose: MascotRigPose) -> UInt8Array:
        """Render one layered BGR frame."""
        canvas = Image.new("RGBA", (self.width, self.height), (*self.background, 255))
        translation = (pose.root_x_px, pose.root_y_px)
        if bool(self.manifest.get("use_residual_layer", True)):
            residual = self._s_curve(self.residual_layer, pose.spine_curve_px)
            residual = self._transform(
                residual,
                pivot=self.pivots["Body"],
                angle_deg=pose.body_angle_deg,
                translation=translation,
                scale_x=pose.body_scale_x,
                scale_y=pose.body_scale_y,
            )
            canvas = Image.alpha_composite(canvas, Image.fromarray(residual, "RGBA"))
        for part, raw_layer in self.layers:
            name = str(part["name"])
            layer = self._s_curve(raw_layer, pose.spine_curve_px) if name in {"Body", "Bell"} else raw_layer
            angle, scale_x, scale_y = self._part_transform(name, pose)
            fallback_pivot = cast(tuple[float, float], tuple(float(value) for value in part["center"]))
            transformed = self._transform(
                layer,
                pivot=self.pivots.get(name, fallback_pivot),
                angle_deg=angle,
                translation=self._part_translation(name, pose),
                scale_x=scale_x,
                scale_y=scale_y,
            )
            canvas = Image.alpha_composite(canvas, Image.fromarray(transformed, "RGBA"))
        rgb: UInt8Array = np.asarray(canvas.convert("RGB"), dtype=np.uint8)
        bgr: UInt8Array = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr
