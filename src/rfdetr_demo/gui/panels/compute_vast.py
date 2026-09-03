# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Vast.ai handlers for the compute panel mixin."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from tkinter import messagebox, ttk

from rfdetr_demo.gui.controllers.vast_controller import VastController
from rfdetr_demo.gui.panels._common import logger
from rfdetr_demo.vast.start_phases import VastProgressUpdate
from rfdetr_demo.vast.types import VastGpuOffer, VastRunnerCancelledError, VastRunnerError


class ComputeVastMixin:
    """Vast transfer, API key, preflight, offer search, and progress UI handlers."""

    def _confirm_vast_transfer(self, source_path: Path) -> bool:
        """Confirm Vast.ai upload policy for *source_path*."""
        if VastController.should_skip_transfer_prompt(source_path):
            return True
        if not messagebox.askokcancel(
            "Vast.ai 転送確認",
            VastController.build_transfer_prompt_message(source_path),
        ):
            return False
        if not VastController.persist_transfer_consent():
            logger.warning("Could not persist Vast consent file")
        return True

    def _load_vast_api_key_from_env(self, *, silent: bool = False) -> None:
        explicit = VastController.normalize_api_key_input(self.vast_api_key_var.get())
        outcome = VastController.load_api_key(explicit)
        if not outcome.success:
            self.vast_api_key_source_var.set(outcome.source_label)
            if not silent and outcome.error_message:
                messagebox.showwarning("API キー", outcome.error_message)
            self._refresh_vast_preflight()
            return
        if not self.vast_api_key_var.get().strip() and outcome.key:
            self.vast_api_key_var.set(outcome.key)
        self.vast_api_key_source_var.set(outcome.source_label)
        if not silent and outcome.log_line:
            self._append_log(outcome.log_line)
        self._refresh_vast_preflight()

    def _save_vast_api_key_local(self) -> None:
        try:
            source_label, log_line = VastController.save_api_key_local(self.vast_api_key_var.get())
        except ValueError as error:
            messagebox.showwarning("API キー", str(error))
            return
        self.vast_api_key_source_var.set(source_label)
        self._append_log(log_line)
        self._refresh_vast_preflight()

    def _refresh_vast_preflight(self) -> None:
        for child in self.preflight_checks_frame.winfo_children():
            child.destroy()
        checks = VastController.run_preflight(
            explicit_api_key=self._resolve_vast_api_key_input(),
            offer_selected=self._selected_vast_offer() is not None,
        )
        header, rows = VastController.build_preflight_view(checks)
        self.preflight_overall_var.set(header.overall_text)

        row = 0
        for entry in rows:
            ttk.Label(
                self.preflight_checks_frame,
                text=entry.line,
                style=entry.style,
            ).grid(row=row, column=0, sticky="w", pady=1)
            row += 1
            if entry.fix_hint:
                ttk.Label(
                    self.preflight_checks_frame,
                    text=entry.fix_hint,
                    style="Caption.TLabel",
                    wraplength=280,
                ).grid(row=row, column=0, sticky="w", padx=(16, 0), pady=(0, 2))
                row += 1

        self._update_compute_controls()

    def _start_vast_elapsed_timer(self) -> None:
        self._stop_vast_elapsed_timer()
        self._vast_elapsed_started_at = time.perf_counter()

        def tick() -> None:
            if self._vast_elapsed_started_at is None:
                return
            elapsed = int(time.perf_counter() - self._vast_elapsed_started_at)
            self.vast_progress_panel.set_elapsed(elapsed)
            self._vast_elapsed_after_id = self.root.after(1000, tick)

        tick()

    def _stop_vast_elapsed_timer(self) -> None:
        if self._vast_elapsed_after_id is not None:
            self.root.after_cancel(self._vast_elapsed_after_id)
            self._vast_elapsed_after_id = None
        self._vast_elapsed_started_at = None

    def _append_startup_info(self) -> None:
        for line in VastController.startup_log_lines():
            self._append_log(line)

    def _startup_vast_orphan_cleanup(self) -> None:
        for line in VastController.startup_orphan_cleanup(
            explicit_api_key=self._resolve_vast_api_key_input(),
        ):
            self.root.after(0, lambda msg=line: self._append_log(msg))

    def _selected_vast_offer(self) -> VastGpuOffer | None:
        return VastController.find_offer(self._vast_offers, self.vast_offer_var.get())

    def _refresh_vast_offers(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showwarning("実行中", "処理中は GPU 検索できません。")
            return

        def worker() -> None:
            try:
                offers = VastController.search_offers(
                    api_key=self._resolve_vast_api_key_input(),
                    max_dph=float(self.vast_max_dph_var.get()),
                    gpu_name=self.vast_gpu_filter_var.get(),
                )
            except (VastRunnerError, VastRunnerCancelledError) as error:
                self.root.after(0, lambda e=error: self._on_vast_search_error(str(e)))
                return
            self.root.after(0, lambda: self._apply_vast_offers(offers))

        self.vast_refresh_button.configure(state="disabled")
        self._set_status_phase("running", "Vast.ai GPU を検索中…")
        threading.Thread(target=worker, daemon=True).start()

    def _on_vast_search_error(self, message: str) -> None:
        self.vast_refresh_button.configure(state="normal")
        self._set_status_phase("idle", "待機中")
        self._append_log(message, level="error")
        messagebox.showerror("Vast.ai", message)

    def _apply_vast_offers(self, offers: list[VastGpuOffer]) -> None:
        self._vast_offers = offers
        ui = VastController.build_offer_search_ui(offers)
        self.vast_offer_combo["values"] = ui.labels
        self.vast_offer_var.set(ui.default_label)
        for level, message in ui.log_lines:
            self._append_log(message, level=level)
        if ui.show_empty_info_dialog:
            messagebox.showinfo(
                "Vast.ai",
                "条件に合う GPU が見つかりませんでした。\n最大 $/時間 や GPU フィルタを緩めてください。",
            )
        self.vast_refresh_button.configure(state="normal")
        self._set_status_phase("idle", "待機中")
        self._refresh_vast_preflight()

    def _resolve_vast_api_key_input(self) -> str | None:
        return VastController.normalize_api_key_input(self.vast_api_key_var.get())

    def _on_vast_progress(self, update: VastProgressUpdate) -> None:
        self.root.after(0, lambda: self._apply_vast_progress_ui(update))

    def _apply_vast_progress_ui(self, update: VastProgressUpdate) -> None:
        state = VastController.progress_ui_state(update)
        self.progress_var.set(state.percent)
        self.progress_text_var.set(state.progress_text)
        self._set_status_phase("running", state.status_message, state.status_metrics)
        if state.show_progress_panel:
            self.vast_progress_panel.grid(**self._vast_progress_grid)
        self.vast_progress_panel.apply_update(update)
        if state.phase_log_line:
            self._append_log(state.phase_log_line)
