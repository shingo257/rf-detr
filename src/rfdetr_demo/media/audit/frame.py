# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Confidential per-frame inference audit (first N processed frames)."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from rfdetr_demo.media.audit.common import (
    append_jsonl,
    build_base_audit_line,
    make_run_id,
    repo_relpath,
    save_audit_image,
    write_summary_json,
)
from rfdetr_demo.paths import CONFIDENTIAL_AUDIT

logger = logging.getLogger(__name__)

DEFAULT_FRAME_AUDIT_COUNT = 20
_FRAME_AUDIT_ROOT = CONFIDENTIAL_AUDIT / "frame-runs"
_FRAME_AUDIT_JSONL = CONFIDENTIAL_AUDIT / "frame-audit.jsonl"


@dataclass(frozen=True)
class FrameAuditRecord:
    """One audited inference frame."""

    processed_count: int
    frame_index: int
    task: str
    instance_count: int
    image_relpath: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameAuditSummary:
    """Evaluation summary for a confidential frame audit run."""

    run_id: str
    source_relpath: str
    task: str
    frames_logged: int
    records: list[FrameAuditRecord] = field(default_factory=list)
    evaluation: dict[str, Any] = field(default_factory=dict)
    run_dir_relpath: str = ""

    def gui_log_lines(self) -> list[str]:
        """Return non-secret paths suitable for the GUI log panel."""
        lines = [
            f"[機密監査] 先頭 {self.frames_logged} フレームを confidential/audit に保存しました",
            f"  run_id={self.run_id}",
            f"  保存先: {self.run_dir_relpath}/",
        ]
        eval_block = self.evaluation
        verdict = eval_block.get("verdict", "—")
        lines.append(f"  評価: {verdict}")
        for note in eval_block.get("notes", []):
            lines.append(f"    · {note}")
        for issue in eval_block.get("issues", []):
            lines.append(f"    ! {issue}")
        for row in eval_block.get("frame_highlights", [])[:5]:
            lines.append(
                f"    #{row['processed_count']:02d} "
                f"src={row['frame_index']} "
                f"検出={row['instance_count']} "
                f"({row['image_relpath']})",
            )
        if self.frames_logged > 5:
            lines.append(f"    … 他 {self.frames_logged - 5} フレーム（詳細は監査 JSONL）")
        return lines


class ConfidentialFrameAuditLogger:
    """Persist annotated frames and metrics under ``confidential/audit/``."""

    def __init__(
        self,
        *,
        source_path: Path,
        task: str,
        max_frames: int = DEFAULT_FRAME_AUDIT_COUNT,
        run_id: str | None = None,
        enabled: bool = True,
    ) -> None:
        self._source_path = source_path.resolve()
        self._task = task
        self._max_frames = max(1, max_frames)
        self._enabled = enabled
        self._run_id = run_id or make_run_id()
        self._run_dir = _FRAME_AUDIT_ROOT / self._run_id
        self._records: list[FrameAuditRecord] = []
        self._finalized = False

    @property
    def run_id(self) -> str:
        return self._run_id

    def maybe_record(
        self,
        *,
        frame_bgr: npt.NDArray[np.uint8],
        annotated_bgr: npt.NDArray[np.uint8],
        frame_index: int,
        processed_count: int,
        instance_count: int,
        details: dict[str, Any] | None = None,
    ) -> FrameAuditRecord | None:
        """Save one frame when ``processed_count`` is within the audit limit."""
        del frame_bgr  # kept for API compatibility with inference callbacks
        if not self._enabled or processed_count > self._max_frames:
            return None

        image_name = f"frame_{processed_count:03d}_src{frame_index:05d}.jpg"
        image_path = self._run_dir / image_name
        if not save_audit_image(image_path, annotated_bgr):
            return None

        record = FrameAuditRecord(
            processed_count=processed_count,
            frame_index=frame_index,
            task=self._task,
            instance_count=instance_count,
            image_relpath=repo_relpath(image_path),
            details=details or {},
        )
        self._records.append(record)

        append_jsonl(
            _FRAME_AUDIT_JSONL,
            build_base_audit_line(
                audit_kind="frame",
                run_id=self._run_id,
                source_relpath=repo_relpath(self._source_path),
                frame_index=frame_index,
                processed_count=processed_count,
                image_relpath=record.image_relpath,
                task=self._task,
                instance_count=instance_count,
                details=record.details,
            ),
        )
        return record

    def finalize(self) -> FrameAuditSummary | None:
        """Write run summary and return evaluation for GUI / CLI."""
        if not self._enabled or self._finalized:
            return None
        self._finalized = True
        if not self._records:
            return None

        evaluation = evaluate_frame_audit(self._records, task=self._task)
        summary = FrameAuditSummary(
            run_id=self._run_id,
            source_relpath=repo_relpath(self._source_path),
            task=self._task,
            frames_logged=len(self._records),
            records=list(self._records),
            evaluation=evaluation,
            run_dir_relpath=repo_relpath(self._run_dir),
        )

        write_summary_json(
            self._run_dir / "summary.json",
            {
                "run_id": summary.run_id,
                "source_relpath": summary.source_relpath,
                "task": summary.task,
                "audit_kind": "frame",
                "frames_logged": summary.frames_logged,
                "evaluation": evaluation,
                "frames": [
                    {
                        "processed_count": record.processed_count,
                        "frame_index": record.frame_index,
                        "instance_count": record.instance_count,
                        "image_relpath": record.image_relpath,
                        "details": record.details,
                    }
                    for record in self._records
                ],
            },
        )
        logger.info(
            "frame_audit_complete run_id=%s frames=%s verdict=%s",
            self._run_id,
            summary.frames_logged,
            evaluation.get("verdict"),
        )
        return summary


def evaluate_frame_audit(records: list[FrameAuditRecord], *, task: str) -> dict[str, Any]:
    """Heuristic evaluation of whether inference looks reasonable."""
    counts = [record.instance_count for record in records]
    min_count = min(counts)
    max_count = max(counts)
    avg_count = sum(counts) / len(counts)
    zero_frames = sum(1 for count in counts if count == 0)
    spread = max_count - min_count

    issues: list[str] = []
    notes: list[str] = []

    if zero_frames == len(counts):
        issues.append("全フレームでインスタンス検出数が 0 です")
    elif zero_frames > 0:
        issues.append(f"{zero_frames}/{len(counts)} フレームで検出数 0")

    if task == "keypoint":
        if avg_count < 1.0:
            issues.append("キーポイント: 平均検出人数が 1 未満です")
        elif avg_count >= 1.0:
            notes.append(f"平均 {avg_count:.1f} 人/フレーム（キーポイント表示数）")
        if spread > 4:
            issues.append(f"検出人数のフレーム間変動が大きい ({min_count}〜{max_count})")
        elif spread <= 2:
            notes.append(f"検出人数は安定 ({min_count}〜{max_count})")
    else:
        if avg_count < 0.5:
            issues.append("物体検出/セグメント: 平均検出数が極端に低いです")
        else:
            notes.append(f"平均 {avg_count:.1f} インスタンス/フレーム")

    if not issues:
        verdict = "OK - 解析は概ね正常"
    elif zero_frames < len(counts) and avg_count >= 1.0:
        verdict = "概ね OK - 一部フレーム要確認"
    else:
        verdict = "要確認 - 解析結果に異常の可能性"

    frame_highlights = [
        {
            "processed_count": record.processed_count,
            "frame_index": record.frame_index,
            "instance_count": record.instance_count,
            "image_relpath": Path(record.image_relpath).name,
        }
        for record in records
    ]

    return {
        "verdict": verdict,
        "issues": issues,
        "notes": notes,
        "min_instances": min_count,
        "max_instances": max_count,
        "avg_instances": round(avg_count, 2),
        "zero_frame_count": zero_frames,
        "frame_highlights": frame_highlights,
    }


def wrap_callback_with_frame_audit(
    callback: Callable[[npt.NDArray[np.uint8], int], npt.NDArray[np.uint8]],
    *,
    audit: ConfidentialFrameAuditLogger,
    stats: dict[str, int],
    on_frame_logged: Callable[[FrameAuditRecord], None] | None = None,
) -> Callable[[npt.NDArray[np.uint8], int], npt.NDArray[np.uint8]]:
    """Wrap an inference callback to capture the first N annotated frames."""

    def wrapped(frame_bgr: npt.NDArray[np.uint8], index: int) -> npt.NDArray[np.uint8]:
        before_total = stats.get("total_detections", 0)
        annotated = callback(frame_bgr, index)
        processed_count = stats.get("processed_frames", 0)
        instance_count = stats.get("total_detections", 0) - before_total
        record = audit.maybe_record(
            frame_bgr=frame_bgr,
            annotated_bgr=annotated,
            frame_index=index,
            processed_count=processed_count,
            instance_count=max(0, instance_count),
            details={"stats_total_detections": stats.get("total_detections", 0)},
        )
        if on_frame_logged is not None and record is not None:
            on_frame_logged(record)
        return annotated

    return wrapped


def is_frame_audit_enabled() -> bool:
    """Return True unless frame audit is explicitly disabled."""
    raw = os.environ.get("RFDETR_FRAME_AUDIT", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}
