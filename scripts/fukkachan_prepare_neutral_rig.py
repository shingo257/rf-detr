"""Split a neutral Fukkachan cutout into animation-ready raster layers."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True, slots=True)
class PartSpec:
    name: str
    file: str
    z: int
    pivot: tuple[float, float]
    shapes: tuple[tuple[str, tuple[float, ...]], ...]


def _scaled_shape_mask(size: tuple[int, int], shapes: tuple[tuple[str, tuple[float, ...]], ...]) -> np.ndarray:
    width, height = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for kind, values in shapes:
        coords = tuple(
            value * (width if index % 2 == 0 else height)
            for index, value in enumerate(values)
        )
        if kind == "ellipse":
            draw.ellipse(coords, fill=255)
        elif kind == "rectangle":
            draw.rectangle(coords, fill=255)
        elif kind == "polygon":
            draw.polygon(list(zip(coords[::2], coords[1::2], strict=True)), fill=255)
        else:
            raise ValueError(f"Unsupported mask shape: {kind}")
    return np.asarray(mask, dtype=np.uint8)


def _crop_layer(rgba: np.ndarray, alpha: np.ndarray, pad: int = 8) -> tuple[Image.Image, list[int]]:
    ys, xs = np.nonzero(alpha)
    if len(xs) == 0:
        raise RuntimeError("Part mask produced an empty layer")
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(rgba.shape[1], int(xs.max()) + pad + 1)
    y2 = min(rgba.shape[0], int(ys.max()) + pad + 1)
    layer = rgba.copy()
    layer[:, :, 3] = alpha
    return Image.fromarray(layer[y1:y2, x1:x2], "RGBA"), [x1, y1, x2, y2]


def prepare_neutral_rig(source: Path, output_dir: Path) -> dict[str, object]:
    image = Image.open(source).convert("RGBA")
    rgba = np.asarray(image).copy()
    foreground = rgba[:, :, 3]
    width, height = image.size

    specs = (
        PartSpec(
            "Head",
            "head.png",
            40,
            (0.50, 0.68),
            (
                ("rectangle", (0.10, 0.00, 0.90, 0.46)),
                ("ellipse", (0.24, 0.24, 0.80, 0.72)),
            ),
        ),
        PartSpec("Pouch", "pouch.png", 32, (0.28, 0.69), (("ellipse", (0.06, 0.45, 0.29, 0.68)),)),
        PartSpec(
            "Left Hand",
            "left_hand.png",
            35,
            (0.31, 0.75),
            (("ellipse", (0.23, 0.72, 0.34, 0.83)),),
        ),
        PartSpec(
            "Right Hand",
            "right_hand.png",
            35,
            (0.75, 0.75),
            (("ellipse", (0.72, 0.72, 0.82, 0.83)),),
        ),
        PartSpec(
            "Left Arm",
            "left_arm.png",
            25,
            (0.39, 0.70),
            (("polygon", (0.26, 0.67, 0.43, 0.66, 0.40, 0.84, 0.30, 0.85, 0.23, 0.76)),),
        ),
        PartSpec(
            "Right Arm",
            "right_arm.png",
            25,
            (0.65, 0.70),
            (("polygon", (0.62, 0.66, 0.78, 0.67, 0.82, 0.76, 0.73, 0.85, 0.64, 0.84)),),
        ),
        PartSpec(
            "Left Leg",
            "left_leg.png",
            18,
            (0.44, 0.86),
            (("ellipse", (0.34, 0.82, 0.51, 0.98)),),
        ),
        PartSpec(
            "Right Leg",
            "right_leg.png",
            18,
            (0.57, 0.86),
            (("ellipse", (0.53, 0.82, 0.69, 0.98)),),
        ),
        PartSpec(
            "Body",
            "body.png",
            10,
            (0.50, 0.86),
            (("polygon", (0.34, 0.65, 0.68, 0.65, 0.72, 0.87, 0.66, 0.91, 0.37, 0.91, 0.31, 0.84)),),
        ),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    image.save(output_dir / "full_body.png")
    claimed = np.zeros_like(foreground)
    parts: list[dict[str, object]] = []
    for spec in specs:
        geometric = _scaled_shape_mask(image.size, spec.shapes)
        alpha = np.minimum(foreground, geometric)
        # Keep semantic layers mostly disjoint while retaining a small overlap for antialiased seams.
        if spec.name != "Body":
            alpha = np.where(claimed > 245, 0, alpha).astype(np.uint8)
        claimed = np.maximum(claimed, alpha)
        cropped, bbox = _crop_layer(rgba, alpha)
        cropped.save(output_dir / spec.file)
        parts.append(
            {
                "name": spec.name,
                "file": spec.file,
                "z": spec.z,
                "bbox": bbox,
                "center": [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0],
                "pivot": [spec.pivot[0] * width, spec.pivot[1] * height],
            }
        )

    manifest: dict[str, object] = {
        "source": str(source),
        "canvas": {"width": width, "height": height},
        "render_mode": "continuous_mesh",
        "use_residual_layer": False,
        "full_body": {"file": "full_body.png", "bbox": [0, 0, width, height]},
        "parts": parts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the generated neutral Fukkachan rig.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_neutral_rig(args.input, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "parts": len(manifest["parts"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
