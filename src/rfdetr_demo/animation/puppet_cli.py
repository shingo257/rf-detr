# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Command-line entry point for layered mascot video rendering."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rfdetr_demo.animation.puppet_video import render_puppet_video


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse mascot rendering command-line arguments."""
    parser = argparse.ArgumentParser(description="Render a layered Fukkachan animation preview.")
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--rig", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track-id", type=int, default=0)
    parser.add_argument("--poses", type=Path, default=None)
    parser.add_argument("--no-dynamics", action="store_true")
    parser.add_argument("--native-keyframes", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Render a puppet video and print its output summary."""
    args = parse_args(argv)
    try:
        summary = render_puppet_video(
            controls_json=args.controls,
            rig_dir=args.rig,
            output_path=args.output,
            track_id=args.track_id,
            pose_json=args.poses,
            dynamics_enabled=not args.no_dynamics,
            resample_to_source_fps=not args.native_keyframes,
        )
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
