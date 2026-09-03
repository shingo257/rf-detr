# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Smoke tests for Phase 12b GUI helper modules (Tk-free)."""

from __future__ import annotations

import importlib.util

import pytest

from rfdetr_demo.gui.controllers.vast_offers import OfferSearchUiOutcome, build_offer_search_ui
from rfdetr_demo.gui.controllers.vast_progress_ui import progress_ui_state
from rfdetr_demo.vast.start_phases import VastJobPhase, VastProgressUpdate
from rfdetr_demo.vast.types import VastGpuOffer

_HAS_TK = importlib.util.find_spec("tkinter") is not None


def test_build_offer_search_ui_empty() -> None:
    outcome = build_offer_search_ui([])
    assert isinstance(outcome, OfferSearchUiOutcome)
    assert outcome.labels == []
    assert outcome.show_empty_info_dialog is True


def test_build_offer_search_ui_with_offers() -> None:
    offers = [
        VastGpuOffer(
            offer_id=1,
            gpu_name="RTX_4090",
            num_gpus=1,
            gpu_ram_gb=24.0,
            dph_total=0.4,
            reliability=0.99,
            cuda_max_good=12.4,
        ),
    ]
    outcome = build_offer_search_ui(offers)
    assert outcome.labels == [offers[0].label]
    assert outcome.default_label == outcome.labels[0]
    assert outcome.show_empty_info_dialog is False


def test_progress_ui_state_maps_percent() -> None:
    update = VastProgressUpdate(
        phase=VastJobPhase.UPLOADING,
        message="uploading",
        percent=42.0,
    )
    state = progress_ui_state(update)
    assert state.percent == 42.0
    assert state.show_progress_panel is True
    assert state.phase_log_line is not None


@pytest.mark.skipif(not _HAS_TK, reason="tkinter not installed")
def test_io_task_mixin_includes_sections() -> None:
    from rfdetr_demo.gui.panels.io_task import IoTaskPanelMixin
    from rfdetr_demo.gui.panels.io_task_sections import IoTaskSectionsMixin

    assert issubclass(IoTaskPanelMixin, IoTaskSectionsMixin)


@pytest.mark.skipif(not _HAS_TK, reason="tkinter not installed")
def test_compute_mixin_includes_vast_handlers() -> None:
    from rfdetr_demo.gui.panels.compute import ComputePanelMixin
    from rfdetr_demo.gui.panels.compute_vast import ComputeVastMixin

    assert issubclass(ComputePanelMixin, ComputeVastMixin)
