# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Form sections for the I/O task column (IO, task, uncertainty, motion)."""

from __future__ import annotations

from tkinter import ttk

from rfdetr_demo.gui.theme import PAD_SECTION


class IoTaskSectionsMixin:
    """Builds scrollable form sections for the left settings column."""

    def _build_io_section(
        self,
        scroll_parent: ttk.Frame,
        padding: dict[str, int],
    ) -> None:
        """Build the source/output path section."""
        io_frame = ttk.LabelFrame(scroll_parent, text="入出力", padding=8)
        io_frame.grid(row=0, column=0, sticky="ew")
        io_frame.columnconfigure(0, weight=1)
        self._register_form_widget(io_frame)

        ttk.Label(io_frame, text="入力動画").grid(row=0, column=0, sticky="w", **padding)
        source_frame = ttk.Frame(io_frame)
        source_frame.grid(row=1, column=0, sticky="ew", **padding)
        source_frame.columnconfigure(0, weight=1)
        ttk.Entry(source_frame, textvariable=self.source_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(source_frame, text="参照…", command=self._browse_source).grid(row=0, column=1, padx=(6, 0))

        ttk.Label(io_frame, text="出力 MP4（空=自動）").grid(row=2, column=0, sticky="w", **padding)
        output_frame = ttk.Frame(io_frame)
        output_frame.grid(row=3, column=0, sticky="ew", **padding)
        output_frame.columnconfigure(0, weight=1)
        ttk.Entry(output_frame, textvariable=self.output_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(output_frame, text="参照…", command=self._browse_output).grid(row=0, column=1, padx=(6, 0))

    def _build_task_section(
        self,
        scroll_parent: ttk.Frame,
        padding: dict[str, int],
    ) -> None:
        """Build task/model/threshold controls."""
        task_frame = ttk.LabelFrame(scroll_parent, text="タスク設定", padding=8)
        task_frame.grid(row=1, column=0, sticky="ew", pady=(PAD_SECTION, 0))
        task_frame.columnconfigure(1, weight=1)
        self._register_form_widget(task_frame)

        row = 0
        ttk.Label(task_frame, text="タスク").grid(row=row, column=0, sticky="w", **padding)
        radios = ttk.Frame(task_frame)
        radios.grid(row=row, column=1, sticky="w", **padding)
        ttk.Radiobutton(radios, text="キーポイント", value="keypoint", variable=self.task_var).grid(
            row=0,
            column=0,
            padx=(0, 8),
            sticky="w",
        )
        ttk.Radiobutton(radios, text="物体検出", value="detect", variable=self.task_var).grid(
            row=0,
            column=1,
            padx=(0, 8),
            sticky="w",
        )
        ttk.Radiobutton(radios, text="セグメント", value="segment", variable=self.task_var).grid(
            row=0,
            column=2,
            sticky="w",
        )
        row += 1

        ttk.Label(task_frame, text="モデル").grid(row=row, column=0, sticky="w", **padding)
        self.model_combo = ttk.Combobox(
            task_frame,
            textvariable=self.model_var,
            values=["nano", "small", "medium", "large"],
            state="readonly",
            width=12,
        )
        self.model_combo.grid(row=row, column=1, sticky="w", **padding)
        row += 1

        ttk.Label(task_frame, text="検出閾値").grid(row=row, column=0, sticky="w", **padding)
        ttk.Spinbox(
            task_frame,
            from_=0.05,
            to=0.95,
            increment=0.05,
            textvariable=self.threshold_var,
            width=8,
        ).grid(row=row, column=1, sticky="w", **padding)
        row += 1

        self._person_only_check = ttk.Checkbutton(
            task_frame,
            text="人物のみ（COCO person）",
            variable=self.person_only_var,
        )
        self._person_only_check.grid(row=row, column=0, columnspan=2, sticky="w", **padding)
        row += 1

        ttk.Label(task_frame, text="フレーム間隔").grid(row=row, column=0, sticky="w", **padding)
        ttk.Spinbox(
            task_frame,
            from_=1,
            to=30,
            textvariable=self.frame_stride_var,
            width=8,
        ).grid(row=row, column=1, sticky="w", **padding)
        row += 1

        ttk.Label(task_frame, text="最大フレーム（空=全編）").grid(row=row, column=0, sticky="w", **padding)
        ttk.Entry(task_frame, textvariable=self.max_frames_var, width=12).grid(row=row, column=1, sticky="w", **padding)

    def _build_uncertainty_section(
        self,
        scroll_parent: ttk.Frame,
        padding: dict[str, int],
    ) -> None:
        self._uncertainty_frame = ttk.LabelFrame(scroll_parent, text="不確実性（キーポイント）", padding=8)
        self._uncertainty_frame.grid(row=4, column=0, sticky="ew", pady=(PAD_SECTION, 0))
        self._uncertainty_frame.columnconfigure(1, weight=1)
        self._register_form_widget(self._uncertainty_frame)
        uncertainty_frame = self._uncertainty_frame

        urow = 0
        ttk.Checkbutton(
            uncertainty_frame,
            text="不確実性ゾーンを描画",
            variable=self.keypoint_uncertainty_var,
        ).grid(row=urow, column=0, columnspan=2, sticky="w", **padding)
        urow += 1

        ttk.Label(uncertainty_frame, text="スタイル").grid(row=urow, column=0, sticky="w", **padding)
        self.style_combo = ttk.Combobox(
            uncertainty_frame,
            textvariable=self.uncertainty_style_var,
            values=["heatmap", "magnitude", "halo", "ellipse", "outline", "cross", "filled"],
            state="readonly",
            width=14,
        )
        self.style_combo.grid(row=urow, column=1, sticky="w", **padding)
        urow += 1

        ttk.Label(uncertainty_frame, text="関節信頼度").grid(row=urow, column=0, sticky="w", **padding)
        ttk.Spinbox(
            uncertainty_frame,
            from_=0.0,
            to=0.95,
            increment=0.05,
            textvariable=self.keypoint_threshold_var,
            width=8,
        ).grid(row=urow, column=1, sticky="w", **padding)
        urow += 1

        ttk.Label(uncertainty_frame, text="σ").grid(row=urow, column=0, sticky="w", **padding)
        ttk.Spinbox(
            uncertainty_frame,
            from_=0.5,
            to=3.0,
            increment=0.1,
            textvariable=self.ellipse_sigma_var,
            width=8,
        ).grid(row=urow, column=1, sticky="w", **padding)
        urow += 1

        ttk.Label(uncertainty_frame, text="楕円最大半径").grid(row=urow, column=0, sticky="w", **padding)
        axis_frame = ttk.Frame(uncertainty_frame)
        axis_frame.grid(row=urow, column=1, sticky="w", **padding)
        ttk.Spinbox(
            axis_frame,
            from_=0,
            to=120,
            increment=2,
            textvariable=self.max_ellipse_axis_var,
            width=8,
        ).grid(row=0, column=0)
        ttk.Label(axis_frame, text="px（0=上限なし）", style="Caption.TLabel").grid(row=0, column=1, padx=(6, 0))
        urow += 1

        ttk.Label(uncertainty_frame, text="不透明度").grid(row=urow, column=0, sticky="w", **padding)
        ttk.Spinbox(
            uncertainty_frame,
            from_=0.05,
            to=1.0,
            increment=0.05,
            textvariable=self.heatmap_opacity_var,
            width=8,
        ).grid(row=urow, column=1, sticky="w", **padding)
        urow += 1

        ttk.Label(uncertainty_frame, text="減衰（ヒートマップ）").grid(row=urow, column=0, sticky="w", **padding)
        ttk.Spinbox(
            uncertainty_frame,
            from_=0.5,
            to=6.0,
            increment=0.5,
            textvariable=self.heatmap_decay_var,
            width=8,
        ).grid(row=urow, column=1, sticky="w", **padding)
        urow += 1

        ttk.Label(uncertainty_frame, text="関節ドット半径").grid(row=urow, column=0, sticky="w", **padding)
        ttk.Spinbox(
            uncertainty_frame,
            from_=1,
            to=12,
            textvariable=self.vertex_radius_var,
            width=8,
        ).grid(row=urow, column=1, sticky="w", **padding)

    def _build_motion_section(
        self,
        scroll_parent: ttk.Frame,
        padding: dict[str, int],
    ) -> None:
        self._motion_frame = ttk.LabelFrame(scroll_parent, text="時系列フィルタ（キーポイント）", padding=8)
        self._motion_frame.grid(row=5, column=0, sticky="ew", pady=(PAD_SECTION, 0))
        self._motion_frame.columnconfigure(1, weight=1)
        self._register_form_widget(self._motion_frame)
        motion_frame = self._motion_frame

        mrow = 0
        ttk.Checkbutton(
            motion_frame,
            text="非現実的な移動速度・振動を抑制",
            variable=self.motion_filter_var,
        ).grid(row=mrow, column=0, columnspan=2, sticky="w", **padding)
        mrow += 1

        ttk.Label(motion_frame, text="最大速度").grid(row=mrow, column=0, sticky="w", **padding)
        speed_frame = ttk.Frame(motion_frame)
        speed_frame.grid(row=mrow, column=1, sticky="w", **padding)
        ttk.Spinbox(
            speed_frame,
            from_=0.05,
            to=2.0,
            increment=0.05,
            textvariable=self.motion_max_speed_var,
            width=8,
        ).grid(row=0, column=0)
        ttk.Label(speed_frame, text="× 対角線/秒", style="Caption.TLabel").grid(row=0, column=1, padx=(6, 0))
        mrow += 1

        ttk.Label(motion_frame, text="平滑化 α").grid(row=mrow, column=0, sticky="w", **padding)
        ttk.Spinbox(
            motion_frame,
            from_=0.1,
            to=1.0,
            increment=0.05,
            textvariable=self.motion_ema_alpha_var,
            width=8,
        ).grid(row=mrow, column=1, sticky="w", **padding)
        mrow += 1

        ttk.Checkbutton(
            motion_frame,
            text="振動ハンチング（反復反転）を検出して抑制",
            variable=self.motion_oscillation_var,
        ).grid(row=mrow, column=0, columnspan=2, sticky="w", **padding)
        mrow += 1

        ttk.Label(
            motion_frame,
            text="フレーム間の移動量が人体上限を超える、または急激な反転を繰り返す関節を前フレーム位置で補正",
            style="Caption.TLabel",
            wraplength=280,
        ).grid(row=mrow, column=0, columnspan=2, sticky="w", padx=6)
