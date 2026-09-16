"""High-resolution image pipeline utilities.

V1 uses tiled processing so 4K/8K outputs do not require the entire target
canvas to be resident in GPU memory at once. The actual neural upscaler can
be plugged in later without changing the public API.
"""

from __future__ import annotations

from math import ceil
from pathlib import Path

from PIL import Image


SUPPORTED_PRESETS = {
    "512": (512, 512),
    "768": (768, 768),
    "1024": (1024, 1024),
    "1536": (1536, 1536),
    "2048": (2048, 2048),
    "4k": (3840, 2160),
    "4k_square": (4096, 4096),
    "8k": (7680, 4320),
    "8k_square": (8192, 8192),
}


def validate_dimensions(width: int, height: int, multiple: int = 8) -> None:
    """Validate arbitrary output dimensions supported by the pipeline."""
    if width < 256 or height < 256:
        raise ValueError("width and height must be at least 256 pixels.")
    if width % multiple or height % multiple:
        raise ValueError(f"width and height must be divisible by {multiple}.")


def get_preset(name: str) -> tuple[int, int]:
    """Return dimensions for a named resolution preset."""
    try:
        return SUPPORTED_PRESETS[name.lower()]
    except KeyError as exc:
        available = ", ".join(SUPPORTED_PRESETS)
        raise ValueError(f"Unknown resolution preset '{name}'. Available: {available}") from exc


def tile_grid(width: int, height: int, tile_size: int = 1024) -> list[tuple[int, int, int, int]]:
    """Return non-overlapping tile boxes covering the complete canvas."""
    validate_dimensions(width, height)
    if tile_size < 256:
        raise ValueError("tile_size must be at least 256.")

    tiles: list[tuple[int, int, int, int]] = []
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            right = min(left + tile_size, width)
            bottom = min(top + tile_size, height)
            tiles.append((left, top, right, bottom))
    return tiles


def resize_to_fit(image: Image.Image, width: int, height: int) -> Image.Image:
    """Resize while preserving aspect ratio and fitting inside the target."""
    validate_dimensions(width, height)
    scale = min(width / image.width, height / image.height)
    new_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    return image.resize(new_size, Image.Resampling.LANCZOS)


def upscale_placeholder(
    input_path: str | Path,
    output_path: str | Path,
    width: int,
    height: int,
) -> Path:
    """Temporary deterministic high-resolution stage.

    This is intentionally not presented as AI super-resolution. It provides
    correct 4K/8K canvas handling until a verified neural upscaler is selected.
    """
    validate_dimensions(width, height)
    source = Path(input_path)
    destination = Path(output_path)

    if not source.is_file():
        raise FileNotFoundError(f"Input image not found: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = image.convert("RGB")
        result = image.resize((width, height), Image.Resampling.LANCZOS)
        result.save(destination, format="PNG")

    return destination
