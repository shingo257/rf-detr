# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Markdown / duration formatting for mzoo video benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def format_duration(seconds: float) -> str:
    """Format seconds as H:MM:SS or M:SS."""
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def render_markdown(report: dict[str, Any]) -> str:
    """Render benchmark JSON as a Japanese Markdown report."""
    env = report["environment"]
    source = report["source_video"]
    settings = report["settings"]
    model_names = ", ".join(f"`{row['job']}`" for row in report["results"])
    scope = "全フレーム推論" if settings.get("full_video") else "一部フレーム"
    if settings.get("max_frames"):
        scope = f"最大 {settings['max_frames']} フレーム"
    lines: list[str] = [
        "# mzoo.mov 動画解析ベンチマーク",
        "",
        f"ローカル環境で RF-DETR（{model_names}）を **{scope}** で実行した結果です。",
        "入力動画は Git に含めず、出力 MP4 も `artifacts/` 配下のローカル専用です。",
        "",
        f"- 計測開始 (UTC): `{report['started_at_utc']}`",
        f"- 計測終了 (UTC): `{report['finished_at_utc']}`",
        f"- 合計処理時間: **{format_duration(report['total_wall_clock_sec'])}** ({report['total_wall_clock_sec']} 秒)",
        "",
        "---",
        "",
        "## 実行環境",
        "",
        "| 項目 | 値 |",
        "|------|-----|",
        f"| ホスト | `{env['hostname']}` |",
        f"| OS | {env['platform']} |",
        f"| Python | {env['python_version']} |",
        f"| PyTorch | {env['pytorch_version']} |",
        f"| 推論デバイス | **{env['device']}** |",
        f"| CUDA | {'利用可' if env['cuda_available'] else '不可（CPU のみ）'} |",
    ]
    if env.get("gpu_name"):
        lines.append(f"| GPU | {env['gpu_name']} |")
    lines.extend(
        [
            f"| CPU | {env['cpu_name']} |",
            f"| CPU コア (論理 / 物理) | {env['cpu_logical_cores']} / {env.get('cpu_physical_cores') or '—'} |",
            f"| メモリ | {env['ram_total_gb'] if env.get('ram_total_gb') is not None else '—'} GB |",
            "",
            "---",
            "",
            "## 入力動画",
            "",
            "| 項目 | 値 |",
            "|------|-----|",
            f"| パス | `{source['path']}` |",
            f"| 解像度 | {source['width']}×{source['height']} |",
            f"| フレーム数 | {source['frame_count']} |",
            f"| FPS | {source['fps']} |",
            f"| 長さ | 約 {source['duration_sec']} 秒 |",
            f"| ファイルサイズ | {source['file_size_mb']} MB |",
            "",
            "---",
            "",
            "## 推論設定",
            "",
            "| 項目 | 値 |",
            "|------|-----|",
            "| タスク | 検出（COCO `person` のみ）+ キーポイント Preview |",
            f"| 信頼度しきい値 | {settings['threshold']} |",
            f"| frame_stride | {settings['frame_stride']} "
            f"({'全フレーム推論' if settings.get('full_video') else '間引き / 上限あり'}) |",
            f"| max_frames | {settings.get('max_frames') or 'なし（フル）'} |",
            f"| person_only | {settings['person_only']} |",
            "",
            f"対象モデル: {model_names}",
            "",
            "---",
            "",
            "## モデル別結果",
            "",
            "| モデル | 状態 | 推論フレーム | 検出数 | 処理時間 | 平均 FPS | 出力 MP4 | 出力サイズ |",
            "|--------|------|-------------:|-------:|---------:|---------:|----------|-----------:|",
        ],
    )

    for row in report["results"]:
        avg_fps = row.get("avg_fps")
        fps_text = f"{avg_fps:.2f}" if avg_fps is not None else "—"
        detections = row.get("total_detections", "—")
        frames = row.get("processed_frames", "—")
        elapsed = row.get("elapsed_sec", "—")
        output_name = Path(str(row.get("output", ""))).name or "—"
        size_mb = row.get("output_size_mb")
        size_text = f"{size_mb:.1f} MB" if size_mb is not None else "—"
        status = row.get("status", "unknown")
        if status != "success" and row.get("error"):
            status = f"{status}: {row['error'][:40]}"
        lines.append(
            f"| `{row['job']}` | {status} | {frames} | {detections} | "
            f"{elapsed}s | {fps_text} | `{output_name}` | {size_text} |",
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 出力ファイル（ローカル）",
            "",
            "```text",
            "artifacts/demo/",
        ],
    )
    for row in report["results"]:
        if str(row.get("status", "")).startswith("success"):
            lines.append(f"  {Path(str(row['output'])).name}")
    lines.extend(
        [
            "```",
            "",
            "## 再現コマンド",
            "",
            "```bat",
            "cd rf-detr",
            "uv run rfdetr-mzoo-benchmark ^",
            "  --only detect-nano --only keypoint-preview",
            "```",
            "",
            "個別実行例:",
            "",
            "```bat",
            "scripts\\run_demo_video.cmd --task detect --model nano --person-only --frame-stride 1",
            "scripts\\run_demo_video.cmd --task keypoint --frame-stride 1",
            "```",
            "",
        ],
    )
    return "\n".join(lines)


__all__ = ["format_duration", "render_markdown"]
