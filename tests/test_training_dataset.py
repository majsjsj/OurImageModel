"""Tests for caption-aware training dataset utilities."""

from __future__ import annotations

import json

import pytest

from backend.training.dataset import Sample, load_metadata, split_samples, validate_image_paths


def test_load_metadata_requires_caption(tmp_path) -> None:
    path = tmp_path / "metadata.jsonl"
    path.write_text(json.dumps({"image": "product.png", "text": "luxury product photo"}) + "\n", encoding="utf-8")
    assert load_metadata(path) == [Sample("product.png", "luxury product photo")]


def test_split_is_deterministic() -> None:
    samples = [Sample(f"{i}.png", f"caption {i}") for i in range(20)]
    train_a, val_a = split_samples(samples, 0.2, seed=42)
    train_b, val_b = split_samples(samples, 0.2, seed=42)
    assert train_a == train_b
    assert val_a == val_b
    assert len(val_a) == 4
    assert {s.image for s in train_a}.isdisjoint({s.image for s in val_a})


def test_zero_validation_keeps_all_samples() -> None:
    samples = [Sample("a.png", "a"), Sample("b.png", "b")]
    train, val = split_samples(samples, 0.0)
    assert len(train) == 2
    assert val == []


def test_metadata_cannot_escape_dataset_directory(tmp_path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    with pytest.raises(ValueError, match="escapes dataset directory"):
        validate_image_paths([Sample("../secret.png", "caption")], image_dir)
