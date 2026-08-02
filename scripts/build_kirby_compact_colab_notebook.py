# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Build the compact Colab notebook that accepts a separate runtime ZIP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BOOTSTRAP = """#@title Upload and initialize the Kirby runtime bundle
from pathlib import Path
import os
import shutil
import subprocess
import sys
import zipfile

COLAB_BOOTSTRAP_VERSION = '2026-07-12.compact-7'
print(f'Kirby Colab bootstrap: {COLAB_BOOTSTRAP_VERSION}')
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import files

    print('rf-detr-kirby-colab-bundle-v7.zip を選択してください。')
    uploaded = files.upload()
    zip_names = [name for name in uploaded if name.lower().endswith('.zip')]
    if len(zip_names) != 1:
        raise FileNotFoundError(f'ZIPを1個だけ選択してください。選択内容: {list(uploaded)}')

    bundle_name = zip_names[0]
    bundle_path = Path('/content') / bundle_name
    bundle_path.write_bytes(uploaded[bundle_name])
    if not zipfile.is_zipfile(bundle_path):
        raise zipfile.BadZipFile(f'有効なZIPではありません: {bundle_name}')

    extract_root = Path('/content/kirby-rfdetr-upload')
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True)

    with zipfile.ZipFile(bundle_path) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        for index, member in enumerate(members, 1):
            normalized_name = member.filename.replace(chr(92), '/')
            relative = Path(*[part for part in normalized_name.split('/') if part])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError(f'安全でないZIPパス: {member.filename}')
            target = extract_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open('wb') as destination:
                shutil.copyfileobj(source, destination)
            if index % 100 == 0 or index == len(members):
                print(f'展開: {index}/{len(members)}')

    package_dirs = sorted(
        path for path in extract_root.rglob('rfdetr_demo')
        if path.is_dir() and path.parent.name == 'src'
    )
    if not package_dirs:
        sample = [str(path.relative_to(extract_root)) for path in extract_root.rglob('*') if path.is_file()][:30]
        raise FileNotFoundError(
            '展開後に src/rfdetr_demo がありません。ZIP内容先頭: ' + repr(sample)
        )

    COLAB_ROOT = package_dirs[0].parent.parent
    required = [
        COLAB_ROOT / 'src' / 'rfdetr_demo' / 'animation' / 'puppet_video.py',
        COLAB_ROOT / 'カービィ.png',
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError('ZIPの必須ファイルが不足しています: ' + repr(missing))

    runtime_dependencies = [
        'requests', 'tqdm', 'pyyaml', 'scipy', 'pydantic>=2,<3',
        'transformers>=5.1.0,<6.0.0', 'supervision>=0.29.0',
        'pyDeprecate>=0.9,<0.10', 'opencv-python-headless>=4.8',
    ]
    subprocess.check_call(
        [
            sys.executable,
            '-m',
            'pip',
            'install',
            '--upgrade-strategy',
            'only-if-needed',
            *runtime_dependencies,
        ]
    )
    sys.path.insert(0, str(COLAB_ROOT / 'src'))
    os.environ['RFDETR_KIRBY_ROOT'] = str(COLAB_ROOT)
    print('Colab workspace:', COLAB_ROOT)
    print('rfdetr_demo:', COLAB_ROOT / 'src' / 'rfdetr_demo')
else:
    print('ローカルJupyterとして実行します。')
"""


def build(template: Path, output: Path) -> None:
    """Build a compact Colab notebook from an existing template.

    Args:
        template: Source notebook containing the replaceable bootstrap cell.
        output: Destination path for the generated notebook.
    """
    notebook = json.loads(template.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and "COLAB_BOOTSTRAP_VERSION" in source:
            cell["source"] = BOOTSTRAP.splitlines(keepends=True)
            cell["metadata"] = {**cell.get("metadata", {}), "cellView": "form"}
            cell["execution_count"] = None
            cell["outputs"] = []
            break
    else:
        raise ValueError("Bootstrap cell not found")

    notebook["cells"][0]["source"] = [
        "# RF-DETR人物追跡からカービィアニメーションを生成（コンパクトColab版）\n",
        "\n",
        "NotebookとランタイムZIPを分離し、Notebook本体を小さく保ちます。\n",
        "最初のセルで検証済みランタイムZIPを1個選択し、その後に処理対象動画を選択します。\n",
    ]
    notebook["cells"][1]["source"] = [
        "## Colab初期化\n",
        "\n",
        "`rf-detr-kirby-colab-bundle-v7.zip` を選択します。パス区切りをLinux形式へ正規化してから展開します。\n",
    ]
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    """Parse command-line arguments and build the compact notebook."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.template, args.output)


if __name__ == "__main__":
    main()
