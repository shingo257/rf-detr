# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for FlowCount preset definitions."""

from __future__ import annotations

from rfdetr_demo.cli.presets import PRESETS


class TestPresets:
    def test_overhead_matches_documented_flags(self) -> None:
        preset = PRESETS["overhead"]

        assert preset.resolution == 960
        assert preset.threshold == 0.25
        assert preset.tile_size == 640
        assert preset.tile_overlap == 256
        assert preset.pose_topk == 8

    def test_eye_level_matches_documented_flags(self) -> None:
        preset = PRESETS["eye-level"]

        assert preset.resolution == 960
        assert preset.threshold == 0.4
        assert preset.tile_size == 0
        assert preset.pose_topk == 3

    def test_fast_matches_documented_flags(self) -> None:
        preset = PRESETS["fast"]

        assert preset.resolution == 960
        assert preset.threshold == 0.3
        assert preset.tile_size == 0
        assert preset.pose_topk == 0

    def test_only_documented_presets_exist(self) -> None:
        assert set(PRESETS) == {"overhead", "eye-level", "fast"}
