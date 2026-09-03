# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Continuous-mesh composition for mascot rendering."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from rfdetr_demo.animation.puppet_assets import Float32Array, MascotRigAssets, UInt8Array
from rfdetr_demo.animation.puppet_mesh import add_mesh_rotation, apply_rigid_prop_constraints, mesh_weight
from rfdetr_demo.animation.retarget import MascotRigPose


class ContinuousMeshCompositor:
    """Warp one uncut mascot sprite with smoothly blended joint handles."""

    def __init__(
        self,
        assets: MascotRigAssets,
        *,
        background: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        """Initialize composition state from normalized rig assets."""
        self.width = assets.width
        self.height = assets.height
        self.background = background
        self.mesh_profile = assets.mesh_profile
        self.face_profile = assets.face_profile
        self.torso_profile = assets.torso_profile
        self.rigid_props = assets.rigid_props
        self.pivots = assets.pivots
        self.grid_x = assets.grid_x
        self.grid_y = assets.grid_y
        self.full_body_layer = assets.full_body_layer
        self.expression_base_layer = assets.expression_base_layer

    def render(self, pose: MascotRigPose) -> UInt8Array:
        """Render one continuous-mesh BGR frame."""
        displacement_x: Float32Array = np.full(
            (self.height, self.width),
            pose.root_x_px,
            dtype=np.float32,
        )
        displacement_y: Float32Array = np.full(
            (self.height, self.width),
            pose.root_y_px,
            dtype=np.float32,
        )

        head_pivot = self.pivots["Head"]
        head_cutoff = float(self.mesh_profile.get("head_cutoff", 0.78))
        head_blend = float(self.mesh_profile.get("head_blend", 0.20))
        head_weight: Float32Array = np.clip(
            (self.height * head_cutoff - self.grid_y) / (self.height * head_blend),
            0.0,
            1.0,
        ).astype(np.float32)
        add_mesh_rotation(
            displacement_x,
            displacement_y,
            grid_x=self.grid_x,
            grid_y=self.grid_y,
            pivot=head_pivot,
            angle_deg=(pose.head_angle_deg + pose.body_angle_deg * 0.20)
            * float(self.mesh_profile.get("head_angle_gain", 1.0)),
            weight=head_weight,
        )
        face_displacement_x = displacement_x.copy()
        face_displacement_y = displacement_y.copy()

        torso_pivot = self.pivots["Body"]
        torso_weight = mesh_weight(
            self.grid_x,
            self.grid_y,
            width=self.width,
            height=self.height,
            pivot=torso_pivot,
            radius_x=self.width * float(self.mesh_profile.get("torso_radius_x", 0.30)),
            radius_y=self.height * float(self.mesh_profile.get("torso_radius_y", 0.25)),
        )
        add_mesh_rotation(
            displacement_x,
            displacement_y,
            grid_x=self.grid_x,
            grid_y=self.grid_y,
            pivot=torso_pivot,
            angle_deg=pose.body_angle_deg * float(self.mesh_profile.get("body_angle_gain", 1.0)),
            weight=torso_weight,
        )
        torso_t = np.clip((self.grid_y - self.height * 0.58) / (self.height * 0.30), 0.0, 1.0)
        displacement_x += (
            pose.spine_curve_px
            * float(self.mesh_profile.get("spine_curve_gain", 1.0))
            * np.sin(2.0 * np.pi * torso_t)
            * torso_weight
        )
        displacement_x += (pose.body_scale_x - 1.0) * (self.grid_x - torso_pivot[0]) * torso_weight
        displacement_y += (pose.body_scale_y - 1.0) * (self.grid_y - torso_pivot[1]) * torso_weight

        torso_displacement_x = displacement_x.copy()
        torso_displacement_y = displacement_y.copy()

        arm_specs = (
            ("Left Arm", pose.right_arm_angle_deg, "left"),
            ("Right Arm", pose.left_arm_angle_deg, "right"),
        )
        for name, angle, side in arm_specs:
            pivot = self.pivots[name]
            weight = mesh_weight(
                self.grid_x,
                self.grid_y,
                width=self.width,
                height=self.height,
                pivot=pivot,
                radius_x=self.width * float(self.mesh_profile.get("arm_radius_x", 0.17)),
                radius_y=self.height * float(self.mesh_profile.get("arm_radius_y", 0.18)),
                side=side,
                lower_bias=float(self.mesh_profile.get("arm_lower_bias", 0.63)),
            )
            add_mesh_rotation(
                displacement_x,
                displacement_y,
                grid_x=self.grid_x,
                grid_y=self.grid_y,
                pivot=pivot,
                angle_deg=angle * float(self.mesh_profile.get("arm_angle_gain", 1.0)),
                weight=weight,
            )

        leg_specs = (
            (
                "Left Leg",
                pose.right_leg_angle_deg + pose.right_foot_angle_deg * 0.35,
                "left",
                pose.right_foot_lift_px,
                pose.right_foot_contact,
            ),
            (
                "Right Leg",
                pose.left_leg_angle_deg + pose.left_foot_angle_deg * 0.35,
                "right",
                pose.left_foot_lift_px,
                pose.left_foot_contact,
            ),
        )
        for name, angle, side, lift, contact in leg_specs:
            pivot = self.pivots[name]
            weight = mesh_weight(
                self.grid_x,
                self.grid_y,
                width=self.width,
                height=self.height,
                pivot=pivot,
                radius_x=self.width * float(self.mesh_profile.get("leg_radius_x", 0.13)),
                radius_y=self.height * float(self.mesh_profile.get("leg_radius_y", 0.16)),
                side=side,
                lower_bias=float(self.mesh_profile.get("leg_lower_bias", 0.82)),
            )
            add_mesh_rotation(
                displacement_x,
                displacement_y,
                grid_x=self.grid_x,
                grid_y=self.grid_y,
                pivot=pivot,
                angle_deg=angle * float(self.mesh_profile.get("leg_angle_gain", 1.0)),
                weight=weight,
            )
            displacement_y += (-pose.root_y_px if contact >= 0.5 else -lift) * weight

        if self.torso_profile and self.torso_profile.get("protect_enabled", True):
            center_x = self.width * float(self.torso_profile.get("center_x", 0.50))
            center_y = self.height * float(self.torso_profile.get("center_y", 0.58))
            radius_x = self.width * float(self.torso_profile.get("radius_x", 0.26))
            radius_y = self.height * float(self.torso_profile.get("radius_y", 0.32))
            feather = max(float(self.torso_profile.get("feather", 0.20)), 1e-3)

            cx_idx = int(np.clip(center_x, 0, self.width - 1))
            cy_idx = int(np.clip(center_y, 0, self.height - 1))
            protected_cx = center_x + float(torso_displacement_x[cy_idx, cx_idx])
            protected_cy = center_y + float(torso_displacement_y[cy_idx, cx_idx])
            distance = ((self.grid_x - protected_cx) / radius_x) ** 2 + ((self.grid_y - protected_cy) / radius_y) ** 2
            torso_protection: Float32Array = np.clip(
                (1.0 + feather - distance) / feather,
                0.0,
                1.0,
            ).astype(np.float32)
            torso_protection = cv2.GaussianBlur(torso_protection, (0, 0), 1.5)
            displacement_x = displacement_x * (1.0 - torso_protection) + torso_displacement_x * torso_protection
            displacement_y = displacement_y * (1.0 - torso_protection) + torso_displacement_y * torso_protection

        apply_rigid_prop_constraints(
            displacement_x,
            displacement_y,
            grid_x=self.grid_x,
            grid_y=self.grid_y,
            width=self.width,
            height=self.height,
            rigid_props=self.rigid_props,
            pose=pose,
            mesh_profile=self.mesh_profile,
        )

        if self.face_profile.get("protect_shape") == "ellipse":
            center_x, center_y = self.face_profile.get("center", [0.5, 0.43])
            radius_x, radius_y = self.face_profile.get("radius", [0.25, 0.28])
            source_face_x = int(np.clip(self.width * float(center_x), 0, self.width - 1))
            source_face_y = int(np.clip(self.height * float(center_y), 0, self.height - 1))
            protected_center_x = source_face_x + float(face_displacement_x[source_face_y, source_face_x])
            protected_center_y = source_face_y + float(face_displacement_y[source_face_y, source_face_x])
            distance = ((self.grid_x - protected_center_x) / (self.width * float(radius_x))) ** 2 + (
                (self.grid_y - protected_center_y) / (self.height * float(radius_y))
            ) ** 2
            feather = max(float(self.face_profile.get("feather", 0.18)), 1e-3)
            face_weight: Float32Array = np.clip(
                (1.0 + feather - distance) / feather,
                0.0,
                1.0,
            ).astype(np.float32)
            for feature in self.face_profile.get("protected_features", []):
                feature_x, feature_y = feature["center"]
                feature_rx, feature_ry = feature["radius"]
                source_x = int(np.clip(self.width * float(feature_x), 0, self.width - 1))
                source_y = int(np.clip(self.height * float(feature_y), 0, self.height - 1))
                moved_x = source_x + float(face_displacement_x[source_y, source_x])
                moved_y = source_y + float(face_displacement_y[source_y, source_x])
                feature_distance = ((self.grid_x - moved_x) / (self.width * float(feature_rx))) ** 2 + (
                    (self.grid_y - moved_y) / (self.height * float(feature_ry))
                ) ** 2
                feature_weight = np.clip((1.0 + feather - feature_distance) / feather, 0.0, 1.0)
                face_weight = np.maximum(face_weight, feature_weight)
            face_weight = cv2.GaussianBlur(face_weight.astype(np.float32), (0, 0), 1.2)
            displacement_x = displacement_x * (1.0 - face_weight) + face_displacement_x * face_weight
            displacement_y = displacement_y * (1.0 - face_weight) + face_displacement_y * face_weight

            reaction = self.face_profile.get("mouth_reaction", {})
            eye_lift = self.height * float(reaction.get("eye_lift", 0.0)) * pose.mouth_open
            eye_scale = 1.0 - float(reaction.get("eye_shrink", 0.0)) * pose.mouth_open
            for feature in self.face_profile.get("protected_features", []):
                if not str(feature.get("name", "")).endswith("eye"):
                    continue
                feature_x, feature_y = feature["center"]
                feature_rx, feature_ry = feature["radius"]
                source_x = int(np.clip(self.width * float(feature_x), 0, self.width - 1))
                source_y = int(np.clip(self.height * float(feature_y), 0, self.height - 1))
                moved_x = source_x + float(face_displacement_x[source_y, source_x])
                moved_y = source_y + float(face_displacement_y[source_y, source_x])
                eye_distance = ((self.grid_x - moved_x) / (self.width * float(feature_rx))) ** 2 + (
                    (self.grid_y - moved_y) / (self.height * float(feature_ry))
                ) ** 2
                eye_weight = np.clip((1.0 + feather - eye_distance) / feather, 0.0, 1.0)
                displacement_x += (eye_scale - 1.0) * (self.grid_x - moved_x) * eye_weight
                displacement_y += ((eye_scale - 1.0) * (self.grid_y - moved_y) - eye_lift) * eye_weight

        map_x = (self.grid_x - displacement_x).astype(np.float32)
        map_y = (self.grid_y - displacement_y).astype(np.float32)
        source_layer = self.expression_base_layer if pose.mouth_open >= 0.08 else self.full_body_layer
        warped: UInt8Array = cv2.remap(
            source_layer,
            map_x,
            map_y,
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        self._draw_mouth_expression(warped, pose, displacement_x, displacement_y)
        canvas = Image.new("RGBA", (self.width, self.height), (*self.background, 255))
        canvas = Image.alpha_composite(canvas, Image.fromarray(warped, "RGBA"))
        rgb: UInt8Array = np.asarray(canvas.convert("RGB"), dtype=np.uint8)
        bgr: UInt8Array = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr

    def _draw_mouth_expression(
        self,
        layer: UInt8Array,
        pose: MascotRigPose,
        displacement_x: Float32Array,
        displacement_y: Float32Array,
    ) -> None:
        """Replace the neutral mouth with a clean Kirby-style open-mouth shape."""
        profile = self.face_profile.get("mouth")
        amount = float(np.clip(pose.mouth_open, 0.0, 1.0))
        if not isinstance(profile, dict) or amount < 0.08:
            return

        source_x = int(np.clip(float(profile["center"][0]) * self.width, 0, self.width - 1))
        source_y = int(np.clip(float(profile["center"][1]) * self.height, 0, self.height - 1))
        center = (
            int(round(source_x + float(displacement_x[source_y, source_x]))),
            int(round(source_y + float(displacement_y[source_y, source_x]))),
        )
        outline_color = tuple(int(value) for value in profile.get("outline_color_rgba", [205, 35, 112, 255]))
        mouth_color = tuple(int(value) for value in profile.get("mouth_color_rgba", [112, 24, 55, 255]))
        tongue_color = tuple(int(value) for value in profile.get("tongue_color_rgba", [245, 91, 142, 255]))

        size_scale_x, size_scale_y = profile.get("size_scale", [1.0, 1.0])
        radius_x = max(2, int(self.width * float(size_scale_x) * (0.025 + 0.055 * amount)))
        radius_y = max(3, int(self.height * float(size_scale_y) * (0.020 + 0.095 * amount)))
        thickness = max(1, int(round(self.width * float(size_scale_x) * 0.010)))
        cv2.ellipse(layer, center, (radius_x, radius_y), 0, 0, 360, mouth_color, -1, cv2.LINE_AA)
        cv2.ellipse(layer, center, (radius_x, radius_y), 0, 0, 360, outline_color, thickness, cv2.LINE_AA)
        if amount > 0.42:
            tongue_center = (center[0], center[1] + int(radius_y * 0.48))
            cv2.ellipse(
                layer,
                tongue_center,
                (max(2, int(radius_x * 0.68)), max(2, int(radius_y * 0.28))),
                0,
                0,
                360,
                tongue_color,
                -1,
                cv2.LINE_AA,
            )
