"""OurImageModel V1 training entry point.

Loads the YAML configuration, validates the training environment, and provides
one stable command-line entry point for the LoRA training pipeline.

The actual model-specific training implementation lives in
backend/training/train_lora.py so the launcher stays model-agnostic.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "train.yaml"


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the top-level YAML structure."""
    if not path.exists():
        raise FileNotFoundError(f"Training config not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError("Training config must contain a YAML mapping/object.")

    required_sections = ("project", "model", "training", "lora", "dataset")
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")

    return config


def validate_config(config: dict[str, Any]) -> None:
    """Fail early on settings that cannot produce a meaningful training run."""
    model = config["model"]
    training = config["training"]
    lora = config["lora"]

    if not model.get("base_model"):
        raise ValueError(
            "model.base_model is empty. Set it in configs/train.yaml "
            "before starting a training run."
        )

    if training.get("resolution", 0) < 256:
        raise ValueError("training.resolution must be at least 256.")

    if training.get("train_batch_size", 0) < 1:
        raise ValueError("training.train_batch_size must be at least 1.")

    if training.get("gradient_accumulation_steps", 0) < 1:
        raise ValueError("training.gradient_accumulation_steps must be at least 1.")

    if lora.get("rank", 0) < 1:
        raise ValueError("lora.rank must be at least 1.")


def check_runtime() -> None:
    """Print useful runtime information before importing heavy training code."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install the dependencies before training."
        ) from exc

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA is unavailable. Training a diffusion model on CPU is not recommended.")


def run_training(config: dict[str, Any]) -> None:
    """Delegate the run to the model-specific LoRA trainer."""
    try:
        trainer_module = importlib.import_module("backend.training.train_lora")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "backend/training/train_lora.py is not available yet. "
            "Create the training backend before starting a run."
        ) from exc

    train_function = getattr(trainer_module, "train", None)
    if not callable(train_function):
        raise RuntimeError(
            "backend.training.train_lora must expose a callable train(config) function."
        )

    train_function(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an OurImageModel LoRA adapter.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the YAML training configuration.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the environment and configuration without training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config

    config = load_config(config_path)
    validate_config(config)

    print(f"Project: {config['project'].get('name', 'OurImageModel')}")
    print(f"Config: {config_path}")
    print(f"Base model: {config['model'].get('base_model') or 'NOT SET'}")
    print(f"Method: {config['training'].get('method', 'unknown')}")

    check_runtime()

    if args.check_only:
        print("Configuration check passed.")
        return

    run_training(config)


if __name__ == "__main__":
    main()
