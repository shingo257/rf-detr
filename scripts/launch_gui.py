#!/usr/bin/env python3
"""Windows-friendly RF-DETR demo GUI launcher with error logging."""

from __future__ import annotations

import ctypes
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = REPO_ROOT / "artifacts" / "gui_launch_error.log"


def _show_error(message: str) -> None:
    ctypes.windll.user32.MessageBoxW(  # type: ignore[attr-defined]
        0,
        message,
        "RF-DETR Demo GUI",
        0x00000010,
    )


def main() -> int:
    try:
        from rfdetr_demo.gui.main_window import main as gui_main
    except Exception:
        tb = traceback.format_exc()
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text(tb, encoding="utf-8")
        _show_error(
            "GUI の起動に失敗しました。\n\n"
            f"{tb[:1200]}\n\n"
            f"詳細ログ: {LOG_PATH}",
        )
        return 1

    try:
        return gui_main()
    except Exception:
        tb = traceback.format_exc()
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text(tb, encoding="utf-8")
        _show_error(
            "GUI 実行中にエラーが発生しました。\n\n"
            f"{tb[:1200]}\n\n"
            f"詳細ログ: {LOG_PATH}",
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
