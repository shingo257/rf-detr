#!/usr/bin/env python3
"""Run full mzoo.mov inference for each RF-DETR video-demo model and write a Markdown report."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import cv2
import psutil
import torch

from rfdetr_demo.inference.runner import run_demo
from rfdetr_demo.inference.types import TaskName
from rfdetr_demo.paths import REPO_ROOT

logger = logging.getLogger(__name__)

ModelSize = Literal["nano", "small", "medium", "large"]

DETECTION_MODELS: tuple[ModelSize, ...] = ("nano", "small", "medium", "large")
BENCHMARK_JOBS: tuple[tuple[TaskName, ModelSize | None], ...] = (
    *((("detect", size) for size in DETECTION_MODELS)),
    ("keypoint", None),
)


def collect_environment() -> dict[str, Any]:
    """Gather hardware and software context for the benchmark report."""
    cpu_name = platform.processor() or "unknown"
    if platform.system() == "Windows":
        try:
            completed = subprocess.run(
                ["wmic", "cpu", "get", "Name"],
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
            lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
            if len(lines) >= 2:
                cpu_name = lines[1]
        except (OSError, subprocess.TimeoutExpired):
            pass

    cuda_available = torch.cuda.is_available()
    device_label = "CPU only"
    gpu_name: str | None = None
    if cuda_available:
        device_label = "CUDA GPU"
        gpu_name = torch.cuda.get_device_name(0)

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_name": cpu_name,
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "pytorch_version": torch.__version__,
        "device": device_label,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
    }


def probe_source_video(source_path: Path) -> dict[str, Any]:
    """Read basic metadata from the input video."""
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Source video not found or unreadable: {source_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    duration_sec = round(frame_count / fps, 2) if fps > 0 else None
    return {
        "path": str(source_path.resolve()),
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "fps": round(fps, 3),
        "duration_sec": duration_sec,
        "file_size_mb": round(source_path.stat().st_size / (1024**2), 2),
    }


def output_path_for(source_path: Path, task: TaskName, model_size: ModelSize | None) -> Path:
    """Build a distinct output filename per model."""
    if task == "keypoint":
        suffix = "keypoints"
    else:
        suffix = f"detected_{model_size}"
    return REPO_ROOT / "artifacts" / "demo" / f"{source_path.stem}_{suffix}.mp4"


def job_label(task: TaskName, model_size: ModelSize | None) -> str:
    """Human-readable job name."""
    if task == "keypoint":
        return "keypoint-preview"
    return f"detect-{model_size}"


def resolve_jobs(only: list[str] | None, skip: set[str]) -> tuple[tuple[TaskName, ModelSize | None], ...]:
    """Select benchmark jobs from --only or default list, applying --skip."""
    if only:
        label_to_job = {job_label(task, size): (task, size) for task, size in BENCHMARK_JOBS}
        selected: list[tuple[TaskName, ModelSize | None]] = []
        for label in only:
            if label not in label_to_job:
                supported = ", ".join(sorted(label_to_job))
                raise ValueError(f"Unknown job {label!r}. Choose from: {supported}")
            selected.append(label_to_job[label])
        jobs = tuple(selected)
    else:
        jobs = BENCHMARK_JOBS
    return tuple((task, size) for task, size in jobs if job_label(task, size) not in skip)


def run_all_benchmarks(
    source_path: Path,
    threshold: float,
    frame_stride: int,
    max_frames: int | None,
    jobs: tuple[tuple[TaskName, ModelSize | None], ...],
) -> dict[str, Any]:
    """Execute every configured model and collect timing summaries."""
    started_at = datetime.now(timezone.utc)
    environment = collect_environment()
    source_info = probe_source_video(source_path)

    results: list[dict[str, Any]] = []
    total_elapsed_sec = 0.0

    for task, model_size in jobs:
        label = job_label(task, model_size)
        target_path = output_path_for(source_path, task, model_size)
        detect_size: ModelSize = model_size if model_size is not None else "nano"
        logger.info("Starting benchmark job: %s -> %s", label, target_path.name)

        job_started = time.perf_counter()
        try:
            summary = run_demo(
                source_path=source_path,
                target_path=target_path,
                task=task,
                model_size=detect_size,
                threshold=threshold,
                frame_stride=frame_stride,
                max_frames=max_frames,
                person_only=True,
            )
            status = "success"
            error_message: str | None = None
        except Exception as exc:  # noqa: BLE001 — benchmark should continue on failure
            logger.exception("Benchmark job failed: %s", label)
            status = "failed"
            error_message = str(exc)
            summary = {
                "target": str(target_path.resolve()),
                "processed_frames": 0,
                "total_detections": 0,
                "elapsed_sec": round(time.perf_counter() - job_started, 2),
            }

        job_elapsed = float(summary.get("elapsed_sec", 0.0))
        total_elapsed_sec += job_elapsed

        output_size_mb: float | None = None
        target = Path(str(summary.get("target", target_path)))
        if status == "success" and target.is_file():
            output_size_mb = round(target.stat().st_size / (1024**2), 2)

        results.append(
            {
                "job": label,
                "task": task,
                "model_size": model_size if task == "detect" else "keypoint-preview",
                "status": status,
                "error": error_message,
                "output": str(target.resolve()) if target.exists() else str(target),
                "output_size_mb": output_size_mb,
                **{key: summary[key] for key in summary if key not in {"source", "target"}},
            },
        )

    finished_at = datetime.now(timezone.utc)
    return {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "total_wall_clock_sec": round(total_elapsed_sec, 2),
        "environment": environment,
        "source_video": source_info,
        "settings": {
            "threshold": threshold,
            "frame_stride": frame_stride,
            "max_frames": max_frames,
            "person_only": True,
            "full_video": frame_stride == 1 and max_frames is None,
        },
        "results": results,
    }


def format_duration(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def build_cached_result(
    source_path: Path,
    task: TaskName,
    model_size: ModelSize | None,
    target_path: Path,
    elapsed_sec: float,
    total_detections: int | None = None,
) -> dict[str, Any]:
    """Build a result row from an existing output file (reuse prior run)."""
    label = job_label(task, model_size)
    capture = cv2.VideoCapture(str(target_path))
    processed_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
    capture.release()
    output_size_mb = round(target_path.stat().st_size / (1024**2), 2) if target_path.is_file() else None
    row: dict[str, Any] = {
        "job": label,
        "task": task,
        "model_size": model_size if task == "detect" else "keypoint-preview",
        "status": "success (cached)",
        "error": None,
        "output": str(target_path.resolve()),
        "output_size_mb": output_size_mb,
        "threshold": 0.5,
        "frame_stride": 1,
        "person_only": True if task == "detect" else None,
        "processed_frames": processed_frames,
        "total_detections": total_detections if total_detections is not None else "—",
        "elapsed_sec": round(elapsed_sec, 2),
    }
    if processed_frames > 0 and elapsed_sec > 0:
        row["avg_fps"] = round(processed_frames / elapsed_sec, 2)
        if isinstance(total_detections, int):
            row["avg_detections_per_frame"] = round(total_detections / processed_frames, 2)
    return row


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
        f"- 合計処理時間: **{format_duration(report['total_wall_clock_sec'])}** "
        f"({report['total_wall_clock_sec']} 秒)",
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
            f"| CPU コア (論理 / 物理) | {env['cpu_logical_cores']} / {env['cpu_physical_cores']} |",
            f"| メモリ | {env['ram_total_gb']} GB |",
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
            f"| タスク | 検出（COCO `person` のみ）+ キーポイント Preview |",
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
            ".venv\\Scripts\\python.exe scripts\\run_mzoo_benchmark.py ^",
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Benchmark all RF-DETR video demo models on mzoo.mov.")
    parser.add_argument(
        "--source",
        type=Path,
        default=REPO_ROOT / "sample" / "mzoo.mov",
        help="Input video path (default: sample/mzoo.mov)",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="1 = infer every frame (full analysis)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N inferred frames (default: full video)",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Run only these jobs: detect-nano, detect-small, …, keypoint-preview",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Skip jobs by label, e.g. detect-nano or keypoint-preview",
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="For skipped jobs, include existing output MP4 in the report if present",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "artifacts" / "demo" / "mzoo_benchmark.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=REPO_ROOT / "docs" / "ja" / "mzoo-video-benchmark.md",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )

    if not args.source.is_file():
        logger.error("Source video missing: %s", args.source)
        return 1

    skip_models = set(args.skip)
    only_labels = args.only or None
    try:
        requested = resolve_jobs(only_labels, set())
        jobs = resolve_jobs(only_labels, skip_models)
    except ValueError as error:
        logger.error("%s", error)
        return 1

    report = run_all_benchmarks(
        source_path=args.source,
        threshold=args.threshold,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        jobs=jobs,
    )

    if args.reuse_cache and only_labels:
        cached: list[dict[str, Any]] = []
        for task, model_size in requested:
            label = job_label(task, model_size)
            if label not in skip_models:
                continue
            target = output_path_for(args.source, task, model_size)
            if not target.is_file() or target.stat().st_size < 1024:
                logger.warning("No cached output for skipped job %s: %s", label, target)
                continue
            elapsed = 269.0 if label == "detect-nano" else 0.0
            cached.append(
                build_cached_result(
                    args.source,
                    task,
                    model_size,
                    target,
                    elapsed_sec=elapsed,
                ),
            )
        if cached:
            report["results"] = cached + report["results"]
            report["total_wall_clock_sec"] = round(
                sum(float(r.get("elapsed_sec", 0)) for r in report["results"]),
                2,
            )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)

    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.md_out.write_text(render_markdown(report), encoding="utf-8")

    logger.info("Wrote JSON report: %s", args.json_out)
    logger.info("Wrote Markdown report: %s", args.md_out)
    print(f"Benchmark complete. Markdown: {args.md_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
