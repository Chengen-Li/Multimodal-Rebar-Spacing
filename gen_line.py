import os
import json
from pathlib import Path

# 假設你的 get_lines 在 line_extractor.py
from line_extractor import get_lines


def run_on_folder(
    image_dir,
    model_path,
    output_dir,
    threshold=10
):
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg",".png"}

    images = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in image_exts]
    )

    print(f"Found {len(images)} images")

    for img_path in images:
        print(f"[INFO] Processing {img_path.name}")

        try:
            line_json_str = get_lines(
                image_path=str(img_path),
                model_path=model_path,
                threshold=threshold
            )

            # 存成同名 json
            out_path = output_dir / f"{img_path.stem}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(line_json_str)

        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")


if __name__ == "__main__":
    run_on_folder(
        image_dir="getline_image",
        model_path="best.pt",
        output_dir="outputs/lines",
        threshold=10
    )
