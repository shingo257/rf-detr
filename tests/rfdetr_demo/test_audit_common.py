# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for shared confidential audit helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rfdetr_demo.media.audit.common import (
    CLASSIFICATION,
    append_jsonl,
    build_base_audit_line,
    make_run_id,
    repo_relpath,
    save_audit_image,
    write_summary_json,
)


def test_build_base_audit_line_includes_shared_schema() -> None:
    line = build_base_audit_line(
        audit_kind="tracking",
        run_id="run-1",
        source_relpath="confidential/media/input/clip.mov",
        frame_index=3,
        processed_count=4,
        image_relpath="confidential/audit/x.jpg",
        raw_count=2,
    )
    assert line["classification"] == CLASSIFICATION
    assert line["audit_kind"] == "tracking"
    assert line["run_id"] == "run-1"
    assert line["source_relpath"].endswith("clip.mov")
    assert line["frame_index"] == 3
    assert line["processed_count"] == 4
    assert line["image_relpath"].endswith("x.jpg")
    assert line["raw_count"] == 2
    assert "timestamp" in line


def test_repo_relpath_falls_back_outside_repo(tmp_path: Path) -> None:
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"x")
    assert repo_relpath(outside, repo_root=tmp_path / "repo") == str(outside)


def test_append_jsonl_and_summary_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_root = tmp_path / "audit"
    monkeypatch.setattr("rfdetr_demo.media.audit.common.CONFIDENTIAL_AUDIT", audit_root)
    jsonl_path = audit_root / "events.jsonl"
    append_jsonl(
        jsonl_path,
        build_base_audit_line(
            audit_kind="frame",
            run_id="r1",
            source_relpath="a.mov",
            frame_index=0,
        ),
    )
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["audit_kind"] == "frame"

    summary_path = audit_root / "summary.json"
    write_summary_json(summary_path, {"run_id": "r1", "audit_kind": "frame"})
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["classification"] == CLASSIFICATION


def test_save_audit_image(tmp_path: Path) -> None:
    path = tmp_path / "frame.jpg"
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    assert save_audit_image(path, image) is True
    assert path.is_file()


def test_make_run_id_shape() -> None:
    run_id = make_run_id()
    assert "T" in run_id
    assert "-" in run_id
