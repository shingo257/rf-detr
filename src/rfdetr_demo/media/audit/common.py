# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared confidential audit I/O and JSONL schema helpers."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import numpy.typing as npt

from rfdetr_demo.paths import CONFIDENTIAL_AUDIT, REPO_ROOT

logger = logging.getLogger(__name__)

AuditKind = Literal["frame", "tracking"]
CLASSIFICATION = "CONFIDENTIAL"


def repo_relpath(path: Path, *, repo_root: Path | None = None) -> str:
    """Return a POSIX path relative to the repository root when possible."""
    root = repo_root if repo_root is not None else REPO_ROOT
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def make_run_id() -> str:
    """Return a UTC timestamped run identifier."""
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid.uuid4().hex[:8]}"


def build_base_audit_line(
    *,
    audit_kind: AuditKind,
    run_id: str,
    source_relpath: str,
    frame_index: int,
    processed_count: int | None = None,
    image_relpath: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build the shared confidential audit JSONL record fields."""
    line: dict[str, Any] = {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "classification": CLASSIFICATION,
        "audit_kind": audit_kind,
        "run_id": run_id,
        "source_relpath": source_relpath,
        "frame_index": frame_index,
    }
    if processed_count is not None:
        line["processed_count"] = processed_count
    if image_relpath is not None:
        line["image_relpath"] = image_relpath
    line.update(extra)
    return line


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON object as a line under ``confidential/audit``."""
    CONFIDENTIAL_AUDIT.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a pretty-printed confidential summary JSON file."""
    if "classification" not in payload:
        payload = {**payload, "classification": CLASSIFICATION}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_audit_image(path: Path, image_bgr: npt.NDArray[np.uint8]) -> bool:
    """Write a BGR audit image and return whether the write succeeded."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image_bgr):
        logger.error("audit_image_write_failed path=%s", path)
        return False
    return True
