# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""GUI panel mixin — see ``VideoDemoGuiApp`` in ``main_window``."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

from rfdetr_demo.gui.controllers.tune_controller import TuneController
from rfdetr_demo.gui.panels.io_task_sections import IoTaskSectionsMixin
from rfdetr_demo.gui.state.job_state import TuneJobState
from rfdetr_demo.gui.state.tune_parameters import build_tune_parameters
from rfdetr_demo.gui.state.ui_bindings import parse_task
from rfdetr_demo.gui.theme import PAD_ROW, PAD_SECTION
from rfdetr_demo.inference.types import TaskName


class IoTaskPanelMixin(IoTaskSectionsMixin):
    """Mixin for I/O paths, task settings, and tune-preview controls."""

    def _build_left_column(self, scroll_parent: ttk.Frame) -> None:
        """Column 1: I/O paths and inference task settings (scrollable)."""
        padding = {"padx": 4, "pady": PAD_ROW // 2}

        self._build_io_section(scroll_parent, padding)
        self._build_task_section(scroll_parent, padding)
        self._build_tune_section(scroll_parent, padding)
        self._build_uncertainty_section(scroll_parent, padding)
        self._build_motion_section(scroll_parent, padding)
        scroll_parent.columnconfigure(0, weight=1)

    def _build_tune_section(
        self,
        scroll_parent: ttk.Frame,
        padding: dict[str, int],
    ) -> None:
        """Build local tune-preview controls."""
        tune_frame = ttk.LabelFrame(scroll_parent, text="試走＋調整（ローカル）", padding=8)
        tune_frame.grid(row=2, column=0, sticky="ew", pady=(PAD_SECTION, 0))
        tune_frame.columnconfigure(1, weight=1)
        self._register_form_widget(tune_frame)

        trow = 0
        self.tune_mode_check = ttk.Checkbutton(
            tune_frame,
            text="試走モードを有効化",
            variable=self.tune_mode_var,
            command=self._on_tune_mode_changed,
        )
        self.tune_mode_check.grid(row=trow, column=0, columnspan=2, sticky="w", **padding)
        trow += 1

        ttk.Label(tune_frame, text="試走時間（秒）").grid(row=trow, column=0, sticky="w", **padding)
        self.tune_seconds_spin = ttk.Spinbox(
            tune_frame,
            from_=0.5,
            to=600.0,
            increment=0.5,
            textvariable=self.tune_preview_seconds_var,
            width=8,
        )
        self.tune_seconds_spin.grid(row=trow, column=1, sticky="w", **padding)
        trow += 1

        self.tune_mode_hint = ttk.Label(
            tune_frame,
            text="先頭 N 秒を試走 → パラメータ調整 → 本番実行",
            style="Caption.TLabel",
            wraplength=280,
        )
        self.tune_mode_hint.grid(row=trow, column=0, columnspan=2, sticky="w", padx=6)
        trow += 1

        self.tune_live_check = ttk.Checkbutton(
            tune_frame,
            text="試走後にリアルタイムプレビュー（パラメータ変更で即反映）",
            variable=self._tune_live_preview_var,
        )
        self.tune_live_check.grid(row=trow, column=0, columnspan=2, sticky="w", **padding)
        trow += 1

        self.auto_tune_var_check = ttk.Checkbutton(
            tune_frame,
            text="試走完了後に自動調整を提案",
            variable=self.auto_tune_var,
        )
        self.auto_tune_var_check.grid(row=trow, column=0, columnspan=2, sticky="w", **padding)
        trow += 1

        tune_buttons = ttk.Frame(tune_frame)
        tune_buttons.grid(row=trow, column=0, columnspan=2, sticky="w", **padding)
        self.tune_retry_button = ttk.Button(
            tune_buttons,
            text="試走を再実行",
            command=self._retry_tune_preview,
            state="disabled",
        )
        self.tune_retry_button.grid(row=0, column=0, padx=(0, 8))
        self.auto_tune_button = ttk.Button(
            tune_buttons,
            text="自動調整を適用",
            command=lambda: self._run_auto_tune(apply=True),
            state="disabled",
        )
        self.auto_tune_button.grid(row=0, column=1)

        self._tune_mode_widgets.extend(
            [self.tune_mode_check, self.tune_seconds_spin, self.tune_mode_hint, self.tune_live_check],
        )

    def _browse_source(self) -> None:
        selected = filedialog.askopenfilename(
            title="入力動画を選択",
            filetypes=[
                ("Video", "*.mp4;*.mov;*.avi;*.mkv"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.source_var.set(selected)

    def _browse_output(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="出力 MP4 を指定",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4")],
        )
        if selected:
            self.output_var.set(selected)

    def _bind_tune_live_traces(self) -> None:
        live_vars = (
            self.threshold_var,
            self.uncertainty_style_var,
            self.ellipse_sigma_var,
            self.max_ellipse_axis_var,
            self.heatmap_opacity_var,
            self.heatmap_decay_var,
            self.vertex_radius_var,
            self.keypoint_threshold_var,
            self.keypoint_uncertainty_var,
            self.person_only_var,
            self.motion_filter_var,
            self.motion_max_speed_var,
            self.motion_ema_alpha_var,
            self.motion_oscillation_var,
        )
        for var in live_vars:
            var.trace_add("write", lambda *_: self._schedule_tune_live_refresh())

    def _schedule_tune_live_refresh(self) -> None:
        if self._tune_job_state != TuneJobState.TUNE_PAUSED or not self._tune_live_preview_var.get():
            return
        if self._tune_live_after_id is not None:
            self.root.after_cancel(self._tune_live_after_id)
        self._tune_live_after_id = self.root.after(120, self._refresh_tune_live_preview)

    def _cancel_tune_live_refresh(self) -> None:
        if self._tune_live_after_id is not None:
            self.root.after_cancel(self._tune_live_after_id)
            self._tune_live_after_id = None

    def _apply_proposed_parameters(self, proposed: Any) -> None:
        self.threshold_var.set(proposed.threshold)
        self.keypoint_threshold_var.set(proposed.keypoint_threshold)
        self.motion_max_speed_var.set(proposed.motion_max_speed_fraction)
        self.motion_ema_alpha_var.set(proposed.motion_ema_alpha)
        self.motion_filter_var.set(proposed.motion_filter_enabled)
        self.motion_oscillation_var.set(proposed.motion_oscillation_enabled)
        self.ellipse_sigma_var.set(proposed.ellipse_sigma)
        self.heatmap_opacity_var.set(proposed.heatmap_opacity)

    def _run_auto_tune(self, *, apply: bool) -> None:
        parameters = build_tune_parameters(self)
        outcome = TuneController.run_auto_tune(
            self._tune_cache,
            parameters,
            apply=apply,
        )
        for line in outcome.log_lines:
            self._append_log(line.message, level=line.level)
        if outcome.apply_recommended and outcome.proposed is not None:
            self._apply_proposed_parameters(outcome.proposed)
            self._refresh_tune_live_preview()

    def _refresh_tune_live_preview(self) -> None:
        self._tune_live_after_id = None
        if self._tune_job_state != TuneJobState.TUNE_PAUSED or not self._tune_cache.has_entries:
            return

        parameters = build_tune_parameters(self)
        rendered = TuneController.render_live_preview(
            self._tune_cache,
            parameters,
            video_fps=self._tune_video_fps,
            frame_stride=self._tune_frame_stride,
        )
        self.flipbook_panel.refresh_live_preview(rendered)
        self._set_status_phase(
            "idle",
            "試走完了 — リアルタイム調整中",
            TuneController.live_preview_status_metrics(parameters),
        )

    def _retry_tune_preview(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showwarning("実行中", "処理が既に実行中です。")
            return
        if not self.tune_mode_var.get():
            messagebox.showinfo("試走再実行", "試走＋調整モードを有効にしてください。")
            return
        self._reset_tune_state(clear_checkbox=False)
        self._start_job()

    def _on_tune_mode_changed(self) -> None:
        if not self.tune_mode_var.get():
            self._reset_tune_state(clear_checkbox=False)
        self._update_start_button_label()

    def _update_start_button_label(self) -> None:
        label = TuneController.start_button_label(
            tune_state=self._tune_job_state,
            tune_mode=bool(self.tune_mode_var.get()),
            compute_backend=str(self.compute_var.get()),
        )
        self.start_button.configure(text=label)

    def _reset_tune_state(self, *, clear_checkbox: bool) -> None:
        self._cancel_tune_live_refresh()
        self._tune_job_state = self._tune_job_state.transition_cancel()
        self._tune_cache.clear()
        self.tune_retry_button.configure(state="disabled")
        self.auto_tune_button.configure(state="disabled")
        if clear_checkbox:
            self.tune_mode_var.set(False)
        self._update_start_button_label()

    def _parse_task(self) -> TaskName:
        """Return the selected inference task name."""
        return parse_task(self.task_var.get())

    def _update_task_controls(self) -> None:
        task = self.task_var.get()
        is_keypoint = task == "keypoint"
        uses_model_size = task in {"detect", "segment"}
        self.model_combo.configure(state="readonly" if uses_model_size else "disabled")
        uncertainty_enabled = is_keypoint and self.keypoint_uncertainty_var.get()
        self.style_combo.configure(state="readonly" if uncertainty_enabled else "disabled")
        if self._person_only_check is not None:
            self._person_only_check.configure(state="normal" if uses_model_size else "disabled")
        is_keypoint_motion = is_keypoint
        for child in self._motion_frame.winfo_children():
            try:
                child.configure(state="normal" if is_keypoint_motion else "disabled")
            except tk.TclError:
                pass
        for child in self._uncertainty_frame.winfo_children():
            try:
                child.configure(state="normal" if is_keypoint else "disabled")
            except tk.TclError:
                pass
