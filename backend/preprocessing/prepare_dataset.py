"""Prepare and validate an OurImageModel image dataset.

Input images are scanned recursively and converted to a clean JSONL metadata
file. Invalid/corrupt images are skipped with a clear warning. Existing
captions can be supplied through a CSV or JSONL metadata file.

Expected output format:
{"image": "relative/path.jpg", "text": "product photo caption"}
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_external_metadata(path: Path | None) -> dict[str, str]:
    """Load optional captions keyed by image filename/path."""
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    result: dict[str, str] = {}
    suffix = path.suffix.lower()

    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                image = str(row.get("image", "")).strip()
                text = str(row.get("text", "")).strip()
                if image and text:
                    result[image] = text
        return result

    if suffix in {".jsonl", ".json"}:
        with path.open("r", encoding="utf-8") as file:
            if suffix == ".jsonl":
                records = [json.loads(line) for line in file if line.strip()]
            else:
                records = json.load(file)
                if isinstance(records, dict):
                    records = records.get("items", [])

        for row in records:
            if not isinstance(row, dict):
                continue
            image = str(row.get("image", "")).strip()
            text = str(row.get("text", row.get("caption", ""))).strip()
            if image and text:
                result[image] = text
        return result

    raise ValueError("External metadata must be CSV, JSONL, or JSON.")


def inspect_image(path: Path, min_resolution: int) -> tuple[bool, str]:
    """Verify that an image can be decoded and meets the minimum resolution."""
    try:
        with Image.open(path) as image:
            image.verify()

        with Image.open(path) as image:
            width, height = image.size
            if min(width, height) < min_resolution:
                return False, f"too small ({width}x{height})"
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return False, f"invalid image ({exc})"

    return True, "ok"


def default_caption(path: Path) -> str:
    """Create a conservative caption when no external caption exists."""
    name = path.stem.replace("_", " ").replace("-", " ").strip()
    return f"commercial product photograph of {name}" if name else "commercial product photograph"


def prepare_dataset(
    image_dir: Path,
    output_file: Path,
    external_metadata: Path | None = None,
    min_resolution: int = 512,
) -> dict[str, int]:
    """Scan images and write validated metadata JSONL."""
    image_dir = image_dir.resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    captions = load_external_metadata(external_metadata)
    files = sorted(
        path for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    stats = {"found": len(files), "valid": 0, "skipped": 0}

    with output_file.open("w", encoding="utf-8") as out:
        for path in files:
            relative = path.relative_to(image_dir).as_posix()
            ok, reason = inspect_image(path, min_resolution)
            if not ok:
                print(f"SKIP: {relative} -> {reason}")
                stats["skipped"] += 1
                continue

            caption = captions.get(relative) or captions.get(path.name) or default_caption(path)
            record: dict[str, Any] = {"image": relative, "text": caption}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["valid"] += 1

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare an OurImageModel dataset.")
    parser.add_argument("--image-dir", type=Path, default=Path("dataset/images"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/metadata/metadata.jsonl"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional CSV/JSON/JSONL file containing image captions.",
    )
    parser.add_argument("--min-resolution", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = prepare_dataset(
        image_dir=args.image_dir,
        output_file=args.output,
        external_metadata=args.metadata,
        min_resolution=args.min_resolution,
    )

    print(
        f"Dataset preparation complete: found={stats['found']}, "
        f"valid={stats['valid']}, skipped={stats['skipped']}"
    )


if __name__ == "__main__":
    main()
