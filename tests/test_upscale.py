"""Tests for resolution and high-resolution utilities."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from backend.inference.upscale import get_preset, tile_grid, upscale_placeholder


def test_4k_and_8k_presets() -> None:
    assert get_preset("4k") == (3840, 2160)
    assert get_preset("8k") == (7680, 4320)
    assert get_preset("4K_SQUARE") == (4096, 4096)
    assert get_preset("8k_square") == (8192, 8192)


def test_tile_grid_covers_target() -> None:
    tiles = tile_grid(2048, 1536, tile_size=1024)
    assert len(tiles) == 4
    assert tiles[-1] == (1024, 1024, 2048, 1536)


def test_upscale_placeholder_preserves_exact_target(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    destination = tmp_path / "output.png"
    Image.new("RGB", (512, 512), (128, 128, 128)).save(source)

    result = upscale_placeholder(source, destination, 3840, 2160)

    assert result == destination
    with Image.open(result) as image:
        assert image.size == (3840, 2160)


def test_invalid_dimensions_are_rejected() -> None:
    with pytest.raises(ValueError):
        tile_grid(100, 100)

    with pytest.raises(ValueError):
        get_preset("12k")
