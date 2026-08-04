# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""CLI for MZoo RF-DETR video-demo model benchmarks."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from rfdetr_demo.benchmark.jobs import job_label, output_path_for, resolve_jobs
from rfdetr_demo.benchmark.report import render_markdown
from rfdetr_demo.benchmark.runner import build_cached_result, run_all_benchmarks
from rfdetr_demo.paths import REPO_ROOT

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the MZoo benchmark runner."""
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
    """Run selected MZoo benchmark jobs and write JSON / Markdown reports."""
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


__all__ = ["main", "parse_args"]
