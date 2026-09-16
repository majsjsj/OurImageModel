"""Caption-aware dataset utilities for OurImageModel training.

The module intentionally does not import Diffusers or load model weights, so
its validation and split logic can be tested quickly in CI.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Sample:
    image: str
    text: str


def load_metadata(path: str | Path) -> list[Sample]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    samples: list[Sample] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on metadata line {line_number}.") from exc
        image = str(record.get("image", "")).strip()
        text = str(record.get("text", "")).strip()
        if not image:
            raise ValueError(f"Missing image on metadata line {line_number}.")
        if not text:
            raise ValueError(f"Missing text caption on metadata line {line_number}.")
        samples.append(Sample(image=image, text=text))

    if not samples:
        raise ValueError(f"Metadata file is empty: {path}")
    return samples


def split_samples(
    samples: list[Sample],
    validation_split: float = 0.05,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample]]:
    if not 0.0 <= validation_split < 1.0:
        raise ValueError("validation_split must be in [0, 1).")
    if len(samples) < 2 and validation_split > 0:
        raise ValueError("At least two samples are required for a validation split.")

    items = list(samples)
    random.Random(seed).shuffle(items)

    if validation_split == 0:
        return items, []

    validation_count = max(1, round(len(items) * validation_split))
    validation_count = min(validation_count, len(items) - 1)
    return items[validation_count:], items[:validation_count]


def validate_image_paths(samples: list[Sample], image_dir: str | Path) -> None:
    root = Path(image_dir).resolve()
    for sample in samples:
        candidate = (root / sample.image).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Metadata image escapes dataset directory: {sample.image}") from exc
        if not candidate.is_file():
            raise FileNotFoundError(f"Dataset image referenced by metadata not found: {candidate}")
