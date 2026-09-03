# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Vast.ai progress UI state for the GUI controller."""

from __future__ import annotations

from dataclasses import dataclass

from rfdetr_demo.vast.start_phases import VastJobPhase, VastProgressUpdate

_LOG_PHASES = frozenset(
    {
        VastJobPhase.REQUESTING,
        VastJobPhase.BOOTING,
        VastJobPhase.SSH_READY,
        VastJobPhase.UPLOADING,
        VastJobPhase.DOWNLOADING,
        VastJobPhase.CLEANUP,
        VastJobPhase.DONE,
        VastJobPhase.FAILED,
    },
)


@dataclass(frozen=True)
class VastProgressUiState:
    """Progress bar and status text derived from a Vast phase update."""

    percent: float
    progress_text: str
    status_message: str
    status_metrics: str
    show_progress_panel: bool
    phase_log_line: str | None = None


def progress_ui_state(update: VastProgressUpdate) -> VastProgressUiState:
    """Map a Vast progress update onto Tk status/progress widgets."""
    percent = min(100.0, max(0.0, update.percent))
    phase_log_line: str | None = None
    if update.phase in _LOG_PHASES:
        status_suffix = f" [{update.vast_status}]" if update.vast_status else ""
        extra = ""
        if update.ssh_port is not None and update.ssh_host:
            extra = f" | ssh -p {update.ssh_port} root@{update.ssh_host}"
            if update.dph_total is not None:
                extra += f" (~${update.dph_total:.2f}/h)"
        phase_log_line = f"[Vast:{update.phase.value}] {update.message}{status_suffix}{extra}"
    return VastProgressUiState(
        percent=percent,
        progress_text=f"{percent:.0f}%  ·  {update.message}",
        status_message="外部 GPU 実行中",
        status_metrics=update.message,
        show_progress_panel=update.phase != VastJobPhase.IDLE,
        phase_log_line=phase_log_line,
    )


__all__ = ["VastProgressUiState", "progress_ui_state"]
