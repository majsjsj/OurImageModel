"""Model-agnostic LoRA training backend placeholder.

This module deliberately keeps foundation-model-specific code out of the
repository until the base model is selected and its training API is verified.
The public interface is stable: train(config).
"""

from __future__ import annotations

from typing import Any


def train(config: dict[str, Any]) -> None:
    """Start the configured LoRA training pipeline.

    The V1 launcher is intentionally explicit instead of silently attempting
    to train with an incompatible architecture. Once a foundation model is
    selected, its loader, target modules, scheduler, and dataset transforms
    should be implemented here and covered by tests.
    """
    model_name = config.get("model", {}).get("base_model")
    if not model_name:
        raise ValueError("Set model.base_model in configs/train.yaml first.")

    raise NotImplementedError(
        "Foundation-model-specific LoRA training is not wired yet. "
        f"Selected base model: {model_name}."
    )
