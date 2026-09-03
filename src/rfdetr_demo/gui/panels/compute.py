# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""GUI panel mixin — see ``VideoDemoGuiApp`` in ``main_window``."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from rfdetr_demo.gui.controllers.vast_controller import VastController
from rfdetr_demo.gui.panels._common import VAST_GPU_FILTERS
from rfdetr_demo.gui.panels.compute_vast import ComputeVastMixin
from rfdetr_demo.gui.theme import PAD_ROW, PAD_SECTION
from rfdetr_demo.vast.start_progress import VastStartProgressPanel


class ComputePanelMixin(ComputeVastMixin):
    """Mixin for compute panel view and control enablement."""

    def _build_run_panel(self, scroll_parent: ttk.Frame) -> None:
        """Run controls and optional Vast.ai settings inside the settings column."""
        padding = {"padx": 4, "pady": PAD_ROW // 2}

        run_frame = ttk.LabelFrame(scroll_parent, text="実行", padding=4)
        run_frame.grid(row=6, column=0, sticky="ew", pady=(PAD_SECTION, 0))
        run_frame.columnconfigure(1, weight=1)
        self._register_form_widget(run_frame)

        compute_frame = ttk.Frame(run_frame)
        compute_frame.grid(row=0, column=0, columnspan=2, sticky="ew", **padding)
        ttk.Radiobutton(
            compute_frame,
            text="ローカル (CPU/GPU)",
            value="local",
            variable=self.compute_var,
        ).grid(row=0, column=0, sticky="w", padx=(0, 12))
        ttk.Radiobutton(
            compute_frame,
            text="外部 GPU (Vast.ai)",
            value="vast",
            variable=self.compute_var,
        ).grid(row=0, column=1, sticky="w")

        self.vast_frame = ttk.LabelFrame(scroll_parent, text="外部 GPU (Vast.ai)", padding=4)
        self._vast_frame_grid = {"row": 7, "column": 0, "sticky": "ew", "pady": (PAD_SECTION, 0)}
        self.vast_frame.columnconfigure(0, weight=1)
        self._register_form_widget(self.vast_frame)
        vast_row = 0

        ttk.Label(self.vast_frame, text="API キー").grid(row=vast_row, column=0, sticky="w", pady=2)
        ttk.Entry(self.vast_frame, textvariable=self.vast_api_key_var, show="*").grid(
            row=vast_row + 1,
            column=0,
            sticky="ew",
            pady=2,
        )
        key_btn_frame = ttk.Frame(self.vast_frame)
        key_btn_frame.grid(row=vast_row + 2, column=0, sticky="w", pady=2)
        ttk.Button(key_btn_frame, text="環境から読込", command=self._load_vast_api_key_from_env).grid(
            row=0,
            column=0,
            padx=(0, 6),
        )
        ttk.Button(key_btn_frame, text="ローカル保存", command=self._save_vast_api_key_local).grid(row=0, column=1)
        vast_row += 3

        ttk.Label(self.vast_frame, textvariable=self.vast_api_key_source_var, style="Caption.TLabel").grid(
            row=vast_row,
            column=0,
            sticky="w",
            pady=(0, 4),
        )
        vast_row += 1

        self.preflight_frame = ttk.LabelFrame(self.vast_frame, text="Preflight", padding=6)
        self.preflight_frame.grid(row=vast_row, column=0, sticky="ew", pady=4)
        self.preflight_frame.columnconfigure(0, weight=1)
        self.preflight_overall_var = tk.StringVar(value="")
        preflight_header = ttk.Frame(self.preflight_frame)
        preflight_header.grid(row=0, column=0, sticky="ew")
        preflight_header.columnconfigure(0, weight=1)
        ttk.Label(preflight_header, textvariable=self.preflight_overall_var).grid(row=0, column=0, sticky="w")
        ttk.Button(
            preflight_header, text="再チェック", style="Small.TButton", command=self._refresh_vast_preflight
        ).grid(
            row=0,
            column=1,
            sticky="e",
        )
        self.preflight_checks_frame = ttk.Frame(self.preflight_frame)
        self.preflight_checks_frame.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        vast_row += 1

        opts_frame = ttk.Frame(self.vast_frame)
        opts_frame.grid(row=vast_row, column=0, sticky="ew", pady=2)
        opts_frame.columnconfigure(1, weight=1)
        ttk.Label(opts_frame, text="最大 $/h").grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Spinbox(
            opts_frame,
            from_=0.10,
            to=5.00,
            increment=0.05,
            textvariable=self.vast_max_dph_var,
            width=8,
        ).grid(row=0, column=1, sticky="w")
        ttk.Label(opts_frame, text="GPU").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=(4, 0))
        ttk.Combobox(
            opts_frame,
            textvariable=self.vast_gpu_filter_var,
            values=list(VAST_GPU_FILTERS),
            state="readonly",
            width=14,
        ).grid(row=1, column=1, sticky="w", pady=(4, 0))
        vast_row += 1

        ttk.Label(self.vast_frame, text="オファー").grid(row=vast_row, column=0, sticky="w", pady=(4, 2))
        offer_frame = ttk.Frame(self.vast_frame)
        offer_frame.grid(row=vast_row + 1, column=0, sticky="ew", pady=2)
        offer_frame.columnconfigure(0, weight=1)
        self.vast_offer_combo = ttk.Combobox(
            offer_frame,
            textvariable=self.vast_offer_var,
            state="readonly",
        )
        self.vast_offer_combo.grid(row=0, column=0, sticky="ew")
        self.vast_refresh_button = ttk.Button(offer_frame, text="GPU 検索", command=self._refresh_vast_offers)
        self.vast_refresh_button.grid(row=0, column=1, padx=(6, 0))
        vast_row += 2

        ttk.Checkbutton(
            self.vast_frame,
            text="完了後にインスタンスを破棄（推奨）",
            variable=self.vast_destroy_var,
        ).grid(row=vast_row, column=0, sticky="w", pady=2)

        self.vast_progress_panel = VastStartProgressPanel(scroll_parent)
        self._vast_progress_grid = {"row": 8, "column": 0, "sticky": "ew", "pady": (PAD_SECTION, 0)}
        scroll_parent.columnconfigure(0, weight=1)

    def _update_compute_controls(self) -> None:
        use_vast = self.compute_var.get() == "vast"
        vast_state = "normal" if use_vast else "disabled"
        for child in self.vast_frame.winfo_children():
            try:
                child.configure(state=vast_state)
            except tk.TclError:
                pass
        self.vast_refresh_button.configure(state=vast_state)
        self.vast_offer_combo.configure(state="readonly" if use_vast else "disabled")

        if use_vast and not self._vast_visible:
            self.vast_frame.grid(**self._vast_frame_grid)
            self._vast_visible = True
        elif not use_vast and self._vast_visible:
            self.vast_frame.grid_remove()
            self.vast_progress_panel.grid_remove()
            self._vast_visible = False

        worker_idle = self._worker is None or not self._worker.is_alive()
        if use_vast and worker_idle:
            checks = VastController.run_preflight(
                explicit_api_key=self._resolve_vast_api_key_input(),
                offer_selected=self._selected_vast_offer() is not None,
            )
            header, _ = VastController.build_preflight_view(checks)
            self.start_button.configure(state="disabled" if header.blocks_start else "normal")
        elif not use_vast and worker_idle:
            self.start_button.configure(state="normal")

        tune_state = "disabled" if use_vast else "normal"
        for widget in self._tune_mode_widgets:
            try:
                widget.configure(state=tune_state)
            except tk.TclError:
                pass
        if use_vast and self.tune_mode_var.get():
            self._reset_tune_state(clear_checkbox=True)

        self._update_start_button_label()
