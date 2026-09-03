"""Create a Colab bundle with portable POSIX ZIP member paths."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path, PurePosixPath


def build(source: Path, output: Path) -> None:
    files = sorted(path for path in source.rglob("*") if path.is_file())
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(source.parent)
            member_name = str(PurePosixPath(*relative.parts))
            archive.write(path, member_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
