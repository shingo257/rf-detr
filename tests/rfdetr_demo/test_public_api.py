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
from rfdetr_demo.public import PUBLIC_API
from rfdetr_demo.tracking.pipeline import PersonTrackPipeline as ConcretePersonTrackPipeline
from rfdetr_demo.tuning.auto_tune import DEFAULT_PARAMETERS as CONCRETE_DEFAULT_PARAMETERS
from rfdetr_demo.vast.cli import run_vast_cli as concrete_run_vast_cli


def test_package_version_is_0_2_0() -> None:
    """Demo layer version bump for the Phase 13 public API freeze."""
    assert rfdetr_demo.__version__ == "0.2.0"


def test_public_api_allowlist_matches_package_exports() -> None:
    """Root ``__all__`` must cover version plus every allowlisted symbol."""
    assert set(PUBLIC_API) <= set(rfdetr_demo.__all__)
    assert "PUBLIC_API" in rfdetr_demo.__all__
    assert "__version__" in rfdetr_demo.__all__


@pytest.mark.parametrize("name", sorted(PUBLIC_API))
def test_public_symbol_resolves_from_package_and_public_module(name: str) -> None:
    """Each allowlisted name is importable from both package root and ``public``."""
    from rfdetr_demo import public as public_module

    package_value = getattr(rfdetr_demo, name)
    public_value = getattr(public_module, name)
    assert package_value is public_value


def test_run_demo_is_exported_from_package_root() -> None:
    from rfdetr_demo import run_demo

    assert run_demo is concrete_run_demo
    assert "run_demo" in rfdetr_demo.__all__


def test_person_track_pipeline_is_exported() -> None:
    from rfdetr_demo import PersonTrackPipeline

    assert PersonTrackPipeline is ConcretePersonTrackPipeline


def test_default_parameters_and_vast_cli_are_exported() -> None:
    from rfdetr_demo import DEFAULT_PARAMETERS, run_vast_cli

    assert DEFAULT_PARAMETERS is CONCRETE_DEFAULT_PARAMETERS
    assert run_vast_cli is concrete_run_vast_cli


def test_run_demo_export_remains_lazy_in_fresh_interpreter() -> None:
    script = (
        "import sys; import rfdetr_demo; "
        "assert 'rfdetr_demo.inference.runner' not in sys.modules; "
        "assert callable(rfdetr_demo.run_demo); "
        "assert 'rfdetr_demo.inference.runner' in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_unknown_public_attribute_has_actionable_error() -> None:
    with pytest.raises(AttributeError, match="Stable public API"):
        getattr(rfdetr_demo, "does_not_exist")
