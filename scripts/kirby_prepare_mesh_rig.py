"""Create a continuous-mesh rig manifest for a transparent Kirby sprite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def prepare_kirby_mesh_rig(
    source: Path,
    output_dir: Path,
    *,
    padding_ratio: float = 0.25,
) -> dict[str, object]:
    image = Image.open(source).convert("RGBA")
    source_width, source_height = image.size
    padding_x = int(round(source_width * padding_ratio))
    padding_y = int(round(source_height * padding_ratio))
    width = source_width + padding_x * 2
    height = source_height + padding_y * 2
    width += width % 2
    height += height % 2
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    canvas.paste(image, (padding_x, padding_y))
    output_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(output_dir / "full_body.png")

    pivots = {
        "Body": (0.50, 0.68),
        "Head": (0.50, 0.66),
        "Left Arm": (0.16, 0.52),
        "Right Arm": (0.84, 0.52),
        "Left Leg": (0.33, 0.78),
        "Right Leg": (0.67, 0.78),
    }
    parts = [
        {
            "name": name,
            "z": index,
            "pivot": [padding_x + pivot[0] * source_width, padding_y + pivot[1] * source_height],
            "center": [padding_x + pivot[0] * source_width, padding_y + pivot[1] * source_height],
        }
        for index, (name, pivot) in enumerate(pivots.items())
    ]
    manifest: dict[str, object] = {
        "source": str(source),
        "source_placement": {
            "x": padding_x,
            "y": padding_y,
            "width": source_width,
            "height": source_height,
        },
        "canvas": {"width": width, "height": height},
        "render_mode": "continuous_mesh",
        "use_residual_layer": False,
        "full_body": {"file": "full_body.png", "bbox": [0, 0, width, height]},
        "mesh_profile": {
            "head_cutoff": 0.86,
            "head_blend": 0.38,
            "head_angle_gain": 0.65,
            "torso_radius_x": 0.42,
            "torso_radius_y": 0.38,
            "body_angle_gain": 0.60,
            "spine_curve_gain": 0.45,
            "arm_radius_x": 0.28,
            "arm_radius_y": 0.30,
            "arm_lower_bias": 0.28,
            "arm_angle_gain": 0.70,
            "leg_radius_x": 0.23,
            "leg_radius_y": 0.25,
            "leg_lower_bias": 0.60,
            "leg_angle_gain": 0.60,
        },
        "face_profile": {
            "protect_shape": "ellipse",
            "center": [
                (padding_x + source_width * 0.50) / width,
                (padding_y + source_height * 0.39) / height,
            ],
            "radius": [source_width * 0.25 / width, source_height * 0.29 / height],
            "feather": 0.22,
            "mouth_reaction": {
                "eye_lift": source_height * 0.020 / height,
                "eye_shrink": 0.055,
            },
            "protected_features": [
                {
                    "name": "left_eye",
                    "center": [
                        (padding_x + source_width * 0.41) / width,
                        (padding_y + source_height * 0.37) / height,
                    ],
                    "radius": [source_width * 0.075 / width, source_height * 0.19 / height],
                },
                {
                    "name": "right_eye",
                    "center": [
                        (padding_x + source_width * 0.59) / width,
                        (padding_y + source_height * 0.37) / height,
                    ],
                    "radius": [source_width * 0.075 / width, source_height * 0.19 / height],
                },
                {
                    "name": "left_cheek",
                    "center": [
                        (padding_x + source_width * 0.28) / width,
                        (padding_y + source_height * 0.47) / height,
                    ],
                    "radius": [source_width * 0.12 / width, source_height * 0.10 / height],
                },
                {
                    "name": "right_cheek",
                    "center": [
                        (padding_x + source_width * 0.72) / width,
                        (padding_y + source_height * 0.47) / height,
                    ],
                    "radius": [source_width * 0.12 / width, source_height * 0.10 / height],
                },
            ],
            "mouth": {
                "center": [
                    (padding_x + source_width * 0.50) / width,
                    (padding_y + source_height * 0.55) / height,
                ],
                "cover_radius": [source_width * 0.115 / width, source_height * 0.095 / height],
                "size_scale": [source_width / width, source_height / height],
                "face_color_rgba": [247, 184, 220, 255],
                "outline_color_rgba": [218, 31, 112, 255],
                "mouth_color_rgba": [91, 25, 55, 255],
                "tongue_color_rgba": [247, 89, 145, 255],
            },
        },
        "parts": parts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a Kirby continuous-mesh rig.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_kirby_mesh_rig(args.input, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "parts": len(manifest["parts"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()


