# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Job runner lifecycle helpers (cancel, error, form enable, completion)."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox
from typing import Any

from rfdetr_demo.gui.controllers.run_controller import RunController, frame_audit_gui_lines
from rfdetr_demo.gui.controllers.tune_controller import TuneController
from rfdetr_demo.gui.panels._common import logger
from rfdetr_demo.gui.state.job_state import TuneJobState
from rfdetr_demo.gui.state.tune_parameters import build_tune_parameters
from rfdetr_demo.vast.start_phases import VastJobPhase, VastProgressUpdate


class JobRunnerLifecycleMixin:
    """Cancel / error / form-enable / completion handlers for the job runner panel."""

    def _log_frame_audit_summary(self, summary: dict[str, Any]) -> None:
        for line in frame_audit_gui_lines(summary):
            self._append_log(line, level="info")

    def _on_vast_error(self, error: Exception, hint: str) -> None:
        self._stop_vast_elapsed_timer()
        self.vast_progress_panel.show_failed(str(error), hint=hint)
        self._on_error(error)

    def _cancel_job(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            return
        self._cancel_event.set()
        self._set_status_phase("running", "キャンセル要求中…")
        self._append_log("キャンセルを要求しました。", level="warn")

    def _on_cancelled(self) -> None:
        self._stop_vast_elapsed_timer()
        self._reset_tune_state(clear_checkbox=False)
        self._set_running(False)
        self._status_phase = "cancelled"
        self._reset_insight_panel(status="キャンセルしました")
        self._set_status_phase("cancelled", "キャンセルしました")
        self.flipbook_panel.show_message("キャンセルしました")
        self._append_log("処理をキャンセルしました。", level="warn")

    def _on_error(self, error: Exception) -> None:
        self._reset_tune_state(clear_checkbox=False)
        self._set_running(False)
        self._status_phase = "error"
        self._set_status_phase("error", "エラー")
        self._append_log(f"エラー: {error}", level="error")
        messagebox.showerror("エラー", str(error))

    def _on_tune_preview_complete(self, summary: dict[str, Any]) -> None:
        """Pause after a tune preview segment so the user can adjust parameters."""
        self._stop_vast_elapsed_timer()
        self._set_running(False)
        try:
            self._tune_job_state = self._tune_job_state.transition_tune_complete()
        except ValueError:
            logger.warning("Tune complete in unexpected state: %s", self._tune_job_state.value)
            self._tune_job_state = TuneJobState.TUNE_PAUSED
        self.tune_retry_button.configure(state="normal")
        self.auto_tune_button.configure(state="normal")
        self._status_phase = "idle"
        self.progress_var.set(100.0)
        processed = summary.get("processed_frames", "?")
        cached = len(self._tune_cache.entries)
        self.insight_frames_var.set(f"{processed} フレーム")
        self.insight_eta_var.set("—")

        parameters = build_tune_parameters(self)
        plan = TuneController.plan_tune_preview_complete(
            summary,
            cache_count=cached,
            live_preview_enabled=bool(self._tune_live_preview_var.get()),
            auto_tune_enabled=bool(self.auto_tune_var.get()),
            task=parameters.task,
            parameters=parameters,
        )
        self.progress_text_var.set(plan.progress_text)
        self._set_status_phase("idle", plan.status_message, plan.status_metrics)
        self._update_start_button_label()
        for line in plan.log_lines:
            self._append_log(line.message, level=line.level)
        self._log_frame_audit_summary(summary)
        if plan.run_auto_tune:
            self._run_auto_tune(apply=True)
        if plan.refresh_live_preview:
            self._refresh_tune_live_preview()

    def _on_complete(self, summary: dict[str, Any]) -> None:
        if summary.get("tune_preview"):
            self._on_tune_preview_complete(summary)
            return

        if self._tune_job_state == TuneJobState.FULL_RUNNING:
            self._tune_job_state = self._tune_job_state.transition_done()

        self._reset_tune_state(clear_checkbox=False)
        self._stop_vast_elapsed_timer()
        self._set_running(False)
        self._status_phase = "done"
        self.progress_var.set(100.0)
        self.insight_eta_var.set("—")
        progress_text, status_metrics, is_vast, log_lines = RunController.complete_ui_plan(summary)
        if is_vast:
            self.vast_progress_panel.grid(**self._vast_progress_grid)
            self.vast_progress_panel.apply_update(
                VastProgressUpdate(
                    phase=VastJobPhase.DONE,
                    message="外部 GPU ジョブ完了",
                    percent=100.0,
                ),
            )
        else:
            self.insight_frames_var.set(f"{summary.get('processed_frames', '?')} フレーム")
        self.progress_text_var.set(progress_text)
        self._set_status_phase("done", "完了", status_metrics)
        for line in log_lines:
            self._append_log(line)
        self._log_frame_audit_summary(summary)
        messagebox.showinfo("完了", f"出力しました:\n{summary['target']}")

    def _set_form_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in self._form_widgets:
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass
        if enabled:
            self._update_task_controls()
            self._update_compute_controls()
        else:
            self.start_button.configure(state="disabled")
            self.cancel_button.configure(state="normal")

    def _set_running(self, running: bool) -> None:
        self._set_form_enabled(not running)
        if running:
            self.start_button.configure(state="disabled")
            self.cancel_button.configure(state="normal")
        else:
            self.cancel_button.configure(state="disabled")
            self._update_compute_controls()
