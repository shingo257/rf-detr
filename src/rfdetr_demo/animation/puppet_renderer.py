# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Select the canonical compositor for a mascot rig."""

from __future__ import annotations

from pathlib import Path

from rfdetr_demo.animation.puppet_assets import MascotRigAssets, UInt8Array, load_mascot_rig_assets
from rfdetr_demo.animation.puppet_continuous import ContinuousMeshCompositor
from rfdetr_demo.animation.puppet_layered import LayeredSpriteCompositor
from rfdetr_demo.animation.retarget import MascotRigPose


class LayeredMascotRenderer:
    """Compatibility router for layered and continuous mascot compositors."""

    def __init__(
        self,
        rig_dir: Path,
        *,
        background: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        """Load rig assets and select the matching composition strategy."""
        assets = load_mascot_rig_assets(rig_dir)
        self.assets: MascotRigAssets = assets
        self.rig_dir = assets.rig_dir
        self.manifest = assets.manifest
        self.render_mode = assets.render_mode
        self.mesh_profile = assets.mesh_profile
        self.face_profile = assets.face_profile
        self.torso_profile = assets.torso_profile
        self.rigid_props = assets.rigid_props
        self.width = assets.width
        self.height = assets.height
        self.background = background
        self.layers = assets.layers
        self.full_body_layer = assets.full_body_layer
        self.expression_base_layer = assets.expression_base_layer
        self.grid_x = assets.grid_x
        self.grid_y = assets.grid_y
        self.residual_layer = assets.residual_layer
        self.pivots = assets.pivots
        self.compositor: ContinuousMeshCompositor | LayeredSpriteCompositor
        if self.render_mode == "continuous_mesh":
            self.compositor = ContinuousMeshCompositor(assets, background=background)
        else:
            self.compositor = LayeredSpriteCompositor(assets, background=background)

    def render(self, pose: MascotRigPose) -> UInt8Array:
        """Render one BGR frame through the selected compositor."""
        return self.compositor.render(pose)
