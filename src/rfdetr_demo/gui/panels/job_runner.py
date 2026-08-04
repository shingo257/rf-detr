# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""GUI panel mixin — see ``VideoDemoGuiApp`` in ``main_window``."""

from __future__ import annotations

import threading
import time
from tkinter import messagebox
from typing import Any

from rfdetr_demo.gui.controllers.run_controller import (
    LocalJobCallbacks,
    RunController,
    VastJobCallbacks,
)
from rfdetr_demo.gui.controllers.vast_controller import VastController
from rfdetr_demo.gui.panels._common import logger
from rfdetr_demo.gui.panels.job_runner_lifecycle import JobRunnerLifecycleMixin
from rfdetr_demo.gui.state.job_state import StartJobError
from rfdetr_demo.gui.state.ui_bindings import build_run_config
from rfdetr_demo.inference.types import VideoProcessingCancelledError
from rfdetr_demo.inference.video_io import probe_video_size
from rfdetr_demo.vast.start_phases import VastJobPhase, VastProgressUpdate
from rfdetr_demo.vast.types import VastRunnerCancelledError, VastRunnerError


class JobRunnerMixin(JobRunnerLifecycleMixin):
    """Mixin for job runner panel."""

    def _start_job(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showwarning("実行中", "処理が既に実行中です。")
            return

        try:
            run_config = build_run_config(self)
        except ValueError as error:
            messagebox.showerror("入力エラー", str(error))
            return

        plan = RunController.prepare_start(run_config, tune_state=self._tune_job_state)
        if isinstance(plan, StartJobError):
            messagebox.showerror(plan.title, plan.message)
            return

        config = plan.config
        compute_backend = config.compute_backend

        if plan.is_full_run_after_tune:
            self._tune_job_state = self._tune_job_state.transition_full_start()
            self._update_start_button_label()

        offer = self._selected_vast_offer() if compute_backend == "vast" else None
        if compute_backend == "vast":
            if not self._confirm_vast_transfer(config.source_path):
                return
            start_error = VastController.validate_job_start(
                api_key=config.vast_api_key,
                offer_selected=config.vast_offer_id is not None,
            )
            if start_error is not None:
                messagebox.showerror(start_error.title, start_error.message)
                return

        self._cancel_event.clear()
        self._reset_insight_panel(status="準備中…")
        self._progress_total = 0
        self._preview_pending = None
        self._preview_flush_scheduled = False
        if plan.is_tune_preview_run:
            self._tune_cache.clear()
            self._tune_cache.task = config.task
            self._tune_cache.person_only = config.person_only
        try:
            _frame_w, _frame_h, video_fps = probe_video_size(config.source_path)
        except RuntimeError:
            video_fps = 30.0
        self._tune_video_fps = video_fps
        self._tune_frame_stride = config.frame_stride
        self._tune_cache.fps = video_fps
        self._tune_cache.frame_stride = config.frame_stride
        if not plan.is_full_run_after_tune:
            self.flipbook_panel.reset()
        self.vast_progress_panel.reset()
        self._set_status_phase("running", "準備中…")
        for level, message in RunController.startup_log_lines(plan):
            self._append_log(message, level=level)
        if plan.is_tune_preview_run:
            self._tune_job_state = self._tune_job_state.transition_tune_start()
        self._set_running(True)
        self._job_started_at = time.perf_counter()

        if compute_backend == "vast":
            assert offer is not None
            self.flipbook_panel.show_message(
                "外部 GPU で解析中（プレビューはローカル実行時のみ表示）",
            )
            self.vast_progress_panel.apply_update(
                VastProgressUpdate(
                    phase=VastJobPhase.REQUESTING,
                    message="外部 GPU ジョブを開始します…",
                    percent=5.0,
                ),
            )
            self._start_vast_elapsed_timer()
            vast_config = RunController.build_vast_config(plan)
            self._worker = threading.Thread(
                target=self._run_vast_job,
                kwargs={"plan": plan, "vast_config": vast_config},
                daemon=True,
            )
        else:
            local_callbacks = LocalJobCallbacks(
                progress_callback=self._on_progress_from_worker,
                preview_callback=self._on_preview_from_worker if config.preview_enabled else None,
                cancel_event=self._cancel_event,
                frame_audit_log_callback=lambda line: self.root.after(
                    0,
                    lambda msg=line: self._append_log(msg, level="info"),
                ),
            )
            job_kwargs = RunController.build_local_job_kwargs(
                plan,
                callbacks=local_callbacks,
                tune_cache=self._tune_cache,
            )
            self._worker = threading.Thread(
                target=self._run_local_job,
                kwargs={"job_kwargs": job_kwargs},
                daemon=True,
            )
        self._worker.start()

    def _run_local_job(self, *, job_kwargs: dict[str, Any]) -> None:
        try:
            summary = RunController.run_local(**job_kwargs)
        except VideoProcessingCancelledError:
            self.root.after(0, self._on_cancelled)
            return
        except Exception as error:
            logger.exception("Video demo failed")
            self.root.after(0, lambda e=error: self._on_error(e))
            return
        self.root.after(0, lambda: self._on_complete(summary))

    def _run_vast_job(self, *, plan: object, vast_config: object) -> None:
        del plan
        try:
            summary = RunController.run_vast(
                vast_config,
                callbacks=VastJobCallbacks(
                    cancel_event=self._cancel_event,
                    log_callback=lambda msg: self.root.after(0, lambda m=msg: self._append_log(m)),
                    phase_callback=self._on_vast_progress,
                ),
            )
        except VastRunnerCancelledError:
            self.root.after(0, self._on_cancelled)
            return
        except VastRunnerError as error:
            self.root.after(0, lambda e=error: self._on_vast_error(e, str(e)))
            return
        except Exception as error:
            logger.exception("Vast.ai video demo failed")
            hint = str(error)
            self.root.after(
                0,
                lambda e=error, h=hint: self._on_vast_error(e, h),
            )
            return
        self.root.after(0, lambda: self._on_complete(summary))
