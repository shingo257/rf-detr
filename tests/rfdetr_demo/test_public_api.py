# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the stable rfdetr_demo public API."""

from __future__ import annotations

import subprocess
import sys

import pytest

import rfdetr_demo
from rfdetr_demo.inference.runner import run_demo as concrete_run_demo


def test_run_demo_is_exported_from_package_root() -> None:
    from rfdetr_demo import run_demo

    assert run_demo is concrete_run_demo
    assert "run_demo" in rfdetr_demo.__all__


def test_run_demo_export_remains_lazy_in_fresh_interpreter() -> None:
    script = (
        "import sys; import rfdetr_demo; "
        "assert 'rfdetr_demo.inference.runner' not in sys.modules; "
        "assert callable(rfdetr_demo.run_demo); "
        "assert 'rfdetr_demo.inference.runner' in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_unknown_public_attribute_has_actionable_error() -> None:
    with pytest.raises(AttributeError, match="does_not_exist"):
        getattr(rfdetr_demo, "does_not_exist")
