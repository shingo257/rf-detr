# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Issue derivation for short-clip keypoint quality analysis."""

from __future__ import annotations

import numpy as np

from rfdetr_demo.tuning.analyze_clip_types import ClipAnalysis


def derive_issues(analysis: ClipAnalysis, *, max_speed_px: float, fps: float) -> list[str]:
    """Return human-readable quality issues for a clip analysis."""
    issues: list[str] = []
    if analysis.frames_analyzed == 0:
        issues.append("解析フレームが 0 - 動画を開けなかったか、長さが不足しています。")
        return issues

    speeds = [
        joint.speed_px_per_sec
        for frame in analysis.frame_metrics
        for joint in frame.joints
        if joint.speed_px_per_sec is not None
    ]
    if speeds:
        peak_speed = max(speeds)
        over_limit = sum(1 for speed in speeds if speed > max_speed_px)
        if over_limit > 0:
            issues.append(
                f"人体上限 ({max_speed_px:.0f} px/s) を超える関節移動が {over_limit} 回 "
                f"(ピーク {peak_speed:.0f} px/s) - ハンチングや誤検出の可能性。"
            )

    if analysis.motion_speed_rejections > 0:
        issues.append(
            f"時系列フィルタが速度異常を {analysis.motion_speed_rejections} 回補正しました。"
        )
    if analysis.motion_oscillation_corrections > 0:
        issues.append(
            f"時系列フィルタが振動ハンチングを {analysis.motion_oscillation_corrections} 回抑制しました。"
        )

    person_counts = [frame.person_count for frame in analysis.frame_metrics]
    if max(person_counts) != min(person_counts):
        issues.append(
            f"フレーム間で検出人数が変動 ({min(person_counts)}-{max(person_counts)} 人) - "
            "トラッキング不安定または閾値調整が必要。"
        )
    if analysis.avg_persons < 0.5:
        issues.append("人物検出がほぼゼロ - 閾値を下げるか、入力解像度・画質を確認してください。")

    low_conf_total = sum(frame.low_confidence_joints for frame in analysis.frame_metrics)
    if low_conf_total > analysis.frames_analyzed * 3:
        issues.append(
            f"低信頼 (conf<0.5) 関節が多い ({low_conf_total} 件) - "
            "関節信頼度フィルタまたは σ/不透明度の調整を検討。"
        )

    confidences = [joint.confidence for frame in analysis.frame_metrics for joint in frame.joints]
    if confidences and float(np.mean(confidences)) < 0.55:
        issues.append(
            f"平均関節信頼度が低い ({np.mean(confidences):.2f}) - モデル・照明・被写体サイズを確認。"
        )

    traces = [
        joint.covariance_trace
        for frame in analysis.frame_metrics
        for joint in frame.joints
        if joint.covariance_trace
    ]
    if traces and float(np.max(traces)) > float(np.median(traces)) * 8:
        issues.append(
            "一部関節で共分散（不確実性）が他より極端に大きい - "
            "遮蔽・モーションブラー・フレーム外の可能性。"
        )

    if fps < 20:
        issues.append(f"FPS が低い ({fps:.1f}) - 速度推定の精度が落ちます。")

    if not issues:
        issues.append("1 秒 clip では重大な異常は検出されませんでした（より長い試走を推奨）。")
    return issues


__all__ = ["derive_issues"]
