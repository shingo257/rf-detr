# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Evaluation helpers for center-person tracking audits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rfdetr_demo.media.audit.tracking_types import TrackingFrameRecord


def evaluate_center_tracking(records: list[TrackingFrameRecord]) -> dict[str, Any]:
    """Analyze when the center person is lost or only held as ghost."""
    if not records:
        return {"verdict": "データなし", "issues": ["フレーム記録がありません"]}

    total = len(records)
    center_missing_stabilized = [row for row in records if not row.center_present_stabilized]
    center_missing_raw = [row for row in records if not row.center_present_raw]
    center_ghost_only = [
        row for row in records if row.center_present_stabilized and row.center_is_ghost
    ]
    track_id_changes: list[dict[str, int | None]] = []
    prev_track_id: int | None = None
    for row in records:
        if row.center_track_id is None:
            prev_track_id = None
            continue
        if prev_track_id is not None and row.center_track_id != prev_track_id:
            track_id_changes.append(
                {
                    "frame_index": row.frame_index,
                    "from_track_id": prev_track_id,
                    "to_track_id": row.center_track_id,
                },
            )
        prev_track_id = row.center_track_id

    first_loss = center_missing_stabilized[0].frame_index if center_missing_stabilized else None
    issues: list[str] = []
    notes: list[str] = []

    if center_missing_stabilized:
        issues.append(
            f"中央人物なし（安定化後）: {len(center_missing_stabilized)}/{total} フレーム"
            f"（初回 frame_index={first_loss}）",
        )
    if center_missing_raw:
        notes.append(f"生検出で中央欠落: {len(center_missing_raw)}/{total} フレーム")
    if center_ghost_only:
        notes.append(f"中央人物がゴーストのみ: {len(center_ghost_only)}/{total} フレーム")
    if track_id_changes:
        issues.append(f"中央トラック ID 切替: {len(track_id_changes)} 回")

    if not center_missing_stabilized and not track_id_changes:
        verdict = "OK - 中央トラックは全フレームで維持"
    elif center_missing_stabilized and len(center_missing_stabilized) <= total * 0.05:
        verdict = "概ね OK - 中央トラックの短時間欠落あり"
    else:
        verdict = "要確認 - 中央トラックが途中で失われています"

    highlights = [
        {
            "frame_index": row.frame_index,
            "processed_count": row.processed_count,
            "raw": row.raw_count,
            "active": row.active_count,
            "center_track_id": row.center_track_id,
            "center_is_ghost": row.center_is_ghost,
            "center_present_stabilized": row.center_present_stabilized,
            "image_relpath": Path(row.image_relpath).name if row.image_relpath else None,
        }
        for row in records
        if (
            not row.center_present_stabilized
            or row.center_is_ghost
            or row.frame_index % 20 == 0
        )
    ][:30]

    return {
        "verdict": verdict,
        "issues": issues,
        "notes": notes,
        "frames_total": total,
        "center_missing_stabilized_count": len(center_missing_stabilized),
        "center_missing_raw_count": len(center_missing_raw),
        "center_ghost_only_count": len(center_ghost_only),
        "track_id_change_count": len(track_id_changes),
        "first_center_loss_frame_index": first_loss,
        "track_id_changes": track_id_changes[:20],
        "frame_highlights": highlights,
    }
