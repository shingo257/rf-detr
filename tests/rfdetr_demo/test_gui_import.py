# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Smoke test for GUI module imports."""

from __future__ import annotations

import importlib.util

import pytest

_HAS_TK = importlib.util.find_spec("tkinter") is not None


@pytest.mark.skipif(not _HAS_TK, reason="tkinter not installed")
def test_main_window_imports() -> None:
    from rfdetr_demo.gui.main_window import VideoDemoGuiApp

    assert VideoDemoGuiApp.__name__ == "VideoDemoGuiApp"
