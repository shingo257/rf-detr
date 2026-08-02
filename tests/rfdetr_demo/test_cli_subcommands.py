# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for rfdetr-demo CLI routing."""

from __future__ import annotations

from unittest.mock import patch

from rfdetr_demo.cli.main import SUBCOMMANDS, main


def test_default_routes_to_video_demo() -> None:
    with patch("rfdetr_demo.cli.main.video_demo_main", return_value=0) as video_main:
        assert main(["--help"]) == 0
        video_main.assert_called_once_with(["--help"])


def test_probe_count_subcommand() -> None:
    with patch("rfdetr_demo.cli.subcommands.probe_count.run", return_value=0) as probe_run:
        with patch("rfdetr_demo.cli.main._build_parser") as build_parser:
            namespace = type("NS", (), {"_handler": probe_run})()
            build_parser.return_value.parse_args.return_value = namespace
            assert main(["probe-count", "--frames", "5"]) == 0
            build_parser.return_value.parse_args.assert_called_once_with(
                ["probe-count", "--frames", "5"],
            )


def test_video_subcommand_passes_remaining_args() -> None:
    with patch("rfdetr_demo.cli.main.video_demo_main", return_value=0) as video_main:
        assert main(["video", "--task", "keypoint"]) == 0
        video_main.assert_called_once_with(["--task", "keypoint"])


def test_subcommand_names_are_stable() -> None:
    assert SUBCOMMANDS == frozenset(
        {"probe-count", "audit-tracking", "analyze-clip", "video"},
    )


def test_audit_tracking_sets_sticky_env(monkeypatch: object) -> None:
    import os
    from argparse import Namespace

    from rfdetr_demo.cli.subcommands import audit_tracking

    monkeypatch.delenv("RFDETR_STICKY_CENTER_TRACK", raising=False)
    monkeypatch.delenv("RFDETR_MAX_MISSED", raising=False)

    class _Summary:
        evaluation = {"verdict": "ok"}
        run_id = "test"
        images_saved = 0
        run_dir_relpath = "confidential/audit/tracking-runs/test"

    with patch(
        "rfdetr_demo.cli.subcommands.audit_tracking.run_center_tracking_audit",
        return_value=_Summary(),
    ):
        code = audit_tracking.run(
            Namespace(
                sticky_center_track=True,
                max_missed=3,
                source=None,
                interval=20,
                threshold=0.6,
                max_frames=1,
            ),
        )
    assert code == 0
    assert os.environ["RFDETR_STICKY_CENTER_TRACK"] == "1"
    assert os.environ["RFDETR_MAX_MISSED"] == "3"
