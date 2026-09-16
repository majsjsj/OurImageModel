"""Tests for the benchmark manifest and training command builder."""

from __future__ import annotations

from pathlib import Path

import yaml

from backend.training.train_lora import build_command

ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_suite_is_stable() -> None:
    suite = yaml.safe_load((ROOT / "benchmarks" / "prompts.yaml").read_text(encoding="utf-8"))
    assert suite["seed"] == 4242
    assert len(suite["prompts"]) == 8
    assert len({item["id"] for item in suite["prompts"]}) == 8


def test_qwen_training_command_uses_supported_checkpoint_flag() -> None:
    config = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    command = build_command(config, Path("trainer.py"))
    assert "--checkpointing_steps" in command
    assert "--save_steps" not in command
    assert "--rank" in command
    assert "--lora_alpha" in command
    assert "--lora_dropout" in command
