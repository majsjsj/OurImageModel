"""Fast, dependency-light smoke tests for the OurImageModel codebase."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from backend.preprocessing.prepare_dataset import inspect_image, prepare_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_training_config_exists() -> None:
    config = ROOT / "configs" / "train.yaml"
    assert config.is_file()


def test_dataset_preparation_validates_images(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "product.png"
    Image.new("RGB", (512, 512), (128, 128, 128)).save(image_path)

    ok, reason = inspect_image(image_path, min_resolution=512)
    assert ok is True
    assert reason == "ok"

    output = tmp_path / "metadata.jsonl"
    stats = prepare_dataset(image_dir, output, min_resolution=512)

    assert stats == {"found": 1, "valid": 1, "skipped": 0}
    record = json.loads(output.read_text(encoding="utf-8").strip())
    assert record["image"] == "product.png"
    assert record["text"]


def test_small_images_are_rejected(tmp_path: Path) -> None:
    image_path = tmp_path / "small.png"
    Image.new("RGB", (256, 256), (128, 128, 128)).save(image_path)

    ok, reason = inspect_image(image_path, min_resolution=512)
    assert ok is False
    assert "too small" in reason
