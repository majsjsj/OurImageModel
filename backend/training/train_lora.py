"""Qwen-Image LoRA training launcher.

The model-specific training loop is delegated to the maintained Hugging Face
Diffusers Qwen-Image LoRA example. This keeps the repository small while using
an implementation that tracks Qwen-Image's current transformer/VAE/text
conditioning stack.
"""

from __future__ import annotations

import argparse
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TRAINER_PATH = ROOT / "scripts" / "vendor" / "train_dreambooth_lora_qwen_image.py"
TRAINER_URL = (
    "https://raw.githubusercontent.com/huggingface/diffusers/main/"
    "examples/dreambooth/train_dreambooth_lora_qwen_image.py"
)


def ensure_trainer() -> Path:
    """Download the official trainer once; keep it outside the model repo logic."""
    TRAINER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if TRAINER_PATH.is_file() and TRAINER_PATH.stat().st_size > 10_000:
        return TRAINER_PATH
    try:
        urllib.request.urlretrieve(TRAINER_URL, TRAINER_PATH)
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(
            "Unable to download the official Qwen-Image LoRA trainer. "
            f"Source: {TRAINER_URL}"
        ) from exc
    if TRAINER_PATH.stat().st_size <= 10_000:
        raise RuntimeError("Downloaded Qwen-Image trainer is unexpectedly small.")
    return TRAINER_PATH


def build_command(config: dict[str, Any], trainer: Path) -> list[str]:
    model = config["model"]
    training = config["training"]
    lora = config["lora"]
    dataset = config["dataset"]
    checkpointing = config.get("checkpointing", {})
    logging = config.get("logging", {})
    validation = config.get("validation", {})

    base_model = model.get("base_model")
    if not base_model:
        raise ValueError("Set model.base_model in configs/train.yaml first.")

    image_dir = ROOT / dataset["image_dir"]
    output_dir = ROOT / training["output_dir"]
    prompt = str(validation.get("prompt") or "a premium commercial product")

    command = [
        "accelerate", "launch", str(trainer),
        "--pretrained_model_name_or_path", base_model,
        "--instance_data_dir", str(image_dir),
        "--output_dir", str(output_dir),
        "--mixed_precision", str(training.get("mixed_precision", "bf16")),
        "--instance_prompt", prompt,
        "--resolution", str(training.get("resolution", 1024)),
        "--train_batch_size", str(training.get("train_batch_size", 1)),
        "--gradient_accumulation_steps", str(training.get("gradient_accumulation_steps", 1)),
        "--learning_rate", str(training.get("learning_rate", 1e-4)),
        "--lr_scheduler", str(training.get("lr_scheduler", "constant")),
        "--lr_warmup_steps", str(training.get("lr_warmup_steps", 0)),
        "--num_train_epochs", str(training.get("num_train_epochs", 1)),
        "--save_steps", str(checkpointing.get("save_steps", 500)),
        "--checkpoints_total_limit", str(checkpointing.get("save_total_limit", 3)),
        "--seed", str(logging.get("seed", training.get("seed", 42))),
        "--report_to", str(logging.get("report_to", "none")),
        "--rank", str(lora.get("rank", 16)),
        "--validation_prompt", prompt,
        "--num_validation_images", str(validation.get("num_images", 1)),
    ]

    max_steps = training.get("max_train_steps")
    if max_steps is not None:
        command += ["--max_train_steps", str(max_steps)]
    if training.get("gradient_checkpointing", True):
        command.append("--gradient_checkpointing")
    if lora.get("target_modules"):
        command += ["--lora_layers", ",".join(lora["target_modules"])]
    resume = checkpointing.get("resume_from_checkpoint")
    if resume:
        command += ["--resume_from_checkpoint", str(resume)]
    return command


def train(config: dict[str, Any]) -> None:
    if config.get("training", {}).get("method") != "lora":
        raise ValueError("training.method must be 'lora'.")

    image_dir = ROOT / config["dataset"]["image_dir"]
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Training image directory not found: {image_dir}")
    if not any(image_dir.iterdir()):
        raise ValueError(f"Training image directory is empty: {image_dir}")

    trainer = ensure_trainer()
    command = build_command(config, trainer)
    print("Launching Qwen-Image LoRA training:")
    print(" ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OurImageModel with Qwen-Image LoRA.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train.yaml")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    trainer = ensure_trainer()
    if args.download_only:
        print(f"Trainer ready: {trainer}")
        return
    train(config)


if __name__ == "__main__":
    main()
