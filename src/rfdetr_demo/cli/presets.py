# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""FlowCount preset definitions.

Named flag bundles matching the presets documented in
``ER-FlowScan/FlowCount/README.md``, so ``--preset overhead`` (etc.) sets
``--resolution``/``--threshold``/``--tile``/``--tile-overlap``/``--pose-topk``
in one shot instead of requiring users to copy the flag combination by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PresetName = Literal["overhead", "eye-level", "fast"]


@dataclass(frozen=True)
class PresetFlags:
    """Resolved detect/track flags for a named preset.

    Attributes:
        resolution: Model input resolution (``--resolution``).
        threshold: Detection confidence threshold (``--threshold``).
        tile_size: Tile size for tiled inference, ``0`` disables it (``--tile``).
        tile_overlap: Tile overlap in pixels (``--tile-overlap``).
        pose_topk: Number of foreground tracks to pose-estimate (``--pose-topk``).
    """

    resolution: int
    threshold: float
    tile_size: int
    tile_overlap: int
    pose_topk: int


PRESETS: dict[PresetName, PresetFlags] = {
    "overhead": PresetFlags(resolution=960, threshold=0.25, tile_size=640, tile_overlap=256, pose_topk=8),
    "eye-level": PresetFlags(resolution=960, threshold=0.4, tile_size=0, tile_overlap=128, pose_topk=3),
    "fast": PresetFlags(resolution=960, threshold=0.3, tile_size=0, tile_overlap=128, pose_topk=0),
}
