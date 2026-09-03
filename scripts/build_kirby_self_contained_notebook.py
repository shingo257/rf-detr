"""Build a Colab notebook with the Kirby runtime bundle embedded as base64."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path


def _bootstrap_source(payload: str, digest: str) -> list[str]:
    chunks = [payload[index : index + 120] for index in range(0, len(payload), 120)]
    lines = [
        "#@title Initialize embedded Kirby runtime\n",
        "from pathlib import Path\n",
        "import base64\n",
        "import hashlib\n",
        "import io\n",
        "import os\n",
        "import shutil\n",
        "import subprocess\n",
        "import sys\n",
        "import zipfile\n",
        "\n",
        "COLAB_BOOTSTRAP_VERSION = '2026-07-12.self-contained-2'\n",
        "print(f'Kirby Colab bootstrap: {COLAB_BOOTSTRAP_VERSION}')\n",
        "IN_COLAB = 'google.colab' in sys.modules\n",
        "if IN_COLAB:\n",
        "    COLAB_ROOT = Path('/content/kirby-rfdetr-runtime')\n",
        "    encoded_bundle = (\n",
    ]
    lines.extend(f"        {chunk!r}\n" for chunk in chunks)
    lines.extend(
        [
            "    )\n",
            "    bundle_bytes = base64.b64decode(encoded_bundle)\n",
            f"    expected_digest = {digest!r}\n",
            "    actual_digest = hashlib.sha256(bundle_bytes).hexdigest().upper()\n",
            "    if actual_digest != expected_digest:\n",
            "        raise RuntimeError(f'Embedded bundle checksum mismatch: {actual_digest}')\n",
            "    print('embedded bundle SHA256:', actual_digest)\n",
            "    if COLAB_ROOT.exists():\n",
            "        shutil.rmtree(COLAB_ROOT)\n",
            "    package_dir = 'rfdetr' + chr(95) + 'demo'\n",
            "    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as archive:\n",
            "        names = archive.namelist()\n",
            "        source_tail = 'demo/animation/puppet' + chr(95) + 'render.py'\n",
            "        source_member = next(name for name in names if name.endswith(source_tail))\n",
            "        prefix = source_member.split('src/', 1)[0]\n",
            "        extracted = 0\n",
            "        for member in archive.infolist():\n",
            "            if member.is_dir() or not member.filename.startswith(prefix):\n",
            "                continue\n",
            "            relative = Path(member.filename[len(prefix):])\n",
            "            if '..' in relative.parts:\n",
            "                raise ValueError(f'Unsafe archive path: {member.filename}')\n",
            "            target = COLAB_ROOT / relative\n",
            "            target.parent.mkdir(parents=True, exist_ok=True)\n",
            "            with archive.open(member) as source, target.open('wb') as destination:\n",
            "                shutil.copyfileobj(source, destination)\n",
            "            extracted += 1\n",
            "    if not (COLAB_ROOT / 'src' / package_dir).exists():\n",
            "        raise FileNotFoundError('Embedded RF-DETR package extraction failed')\n",
            "    print('extracted files:', extracted)\n",
            "    print('bundle:', (COLAB_ROOT / 'COLAB_BUNDLE_VERSION.txt').read_text(encoding='ascii'))\n",
            "    runtime_dependencies = [\n",
            "        'requests', 'tqdm', 'pyyaml', 'scipy', 'pydantic>=2,<3',\n",
            "        'transformers>=5.1.0,<6.0.0', 'supervision>=0.29.0',\n",
            "        'pyDeprecate>=0.9,<0.10', 'opencv-python-headless>=4.8', 'pillow>=10.0',\n",
            "    ]\n",
            "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', *runtime_dependencies])\n",
            "    sys.path.insert(0, str(COLAB_ROOT / 'src'))\n",
            "    os.environ['RFDETR_KIRBY_ROOT'] = str(COLAB_ROOT)\n",
            "    print('Colab workspace:', COLAB_ROOT)\n",
            "else:\n",
            "    print('Running as local Jupyter')\n",
        ]
    )
    return lines


def build(template: Path, bundle: Path, output: Path) -> None:
    notebook = json.loads(template.read_text(encoding="utf-8"))
    bundle_bytes = bundle.read_bytes()
    payload = base64.b64encode(bundle_bytes).decode("ascii")
    digest = hashlib.sha256(bundle_bytes).hexdigest().upper()

    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and "COLAB_BOOTSTRAP_VERSION" in source:
            cell["source"] = _bootstrap_source(payload, digest)
            cell["metadata"] = {
                **cell.get("metadata", {}),
                "cellView": "form",
                "collapsed": True,
                "jupyter": {"source_hidden": True},
            }
            cell["execution_count"] = None
            cell["outputs"] = []
            break
    else:
        raise ValueError("Bootstrap cell not found")

    notebook["cells"][0]["source"] = [
        "# RF-DETR human tracking to Kirby animation (self-contained Colab)\n",
        "\n",
        "The RF-DETR runtime and original Kirby image are embedded in this notebook.\n",
        "Only the source video needs to be uploaded.\n",
    ]
    notebook["cells"][1]["source"] = [
        "## Colab initialization\n",
        "\n",
        "Run this collapsed cell to verify and extract the embedded runtime. No ZIP upload is required.\n",
    ]
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.template, args.bundle, args.output)


if __name__ == "__main__":
    main()
