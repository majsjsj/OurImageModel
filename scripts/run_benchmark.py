"""Run a reproducible image-generation benchmark locally.

This script intentionally does not run in CI: Qwen-Image is a large model and
requires a suitable inference GPU. It records configuration and output paths
so Base vs adapter experiments can be compared later.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from backend.inference.generate import ImageGenerator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = ROOT / "benchmarks" / "prompts.yaml"


def load_suite(path: Path) -> dict:
    suite = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not suite.get("prompts"):
        raise ValueError("Benchmark suite must contain prompts.")
    return suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OurImageModel benchmark suite.")
    parser.add_argument("--model", required=True, help="Base model or local pipeline path.")
    parser.add_argument("--suite", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "benchmark")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter path to load after model initialization.")
    args = parser.parse_args()

    suite = load_suite(args.suite)
    settings = suite.get("settings", {})
    seed = int(suite.get("seed", 4242))
    args.output.mkdir(parents=True, exist_ok=True)

    generator = ImageGenerator(
        model_id=args.model,
        torch_dtype=args.dtype,
    )
    if args.adapter:
        generator.pipeline.load_lora_weights(args.adapter)

    results = []
    for item in suite["prompts"]:
        prompt_id = str(item["id"])
        destination = args.output / f"{prompt_id}.png"
        started = time.perf_counter()
        generator.generate(
            prompt=item["prompt"],
            output_path=destination,
            width=int(settings.get("width", 1024)),
            height=int(settings.get("height", 1024)),
            num_inference_steps=int(settings.get("steps", 30)),
            guidance_scale=float(settings.get("guidance", 5.0)),
            seed=seed,
        )
        results.append(
            {
                "id": prompt_id,
                "prompt": item["prompt"],
                "seed": seed,
                "output": str(destination.relative_to(ROOT)),
                "seconds": round(time.perf_counter() - started, 3),
            }
        )

    manifest = {
        "suite": str(args.suite.relative_to(ROOT)) if args.suite.is_relative_to(ROOT) else str(args.suite),
        "model": args.model,
        "adapter": args.adapter,
        "settings": settings,
        "results": results,
    }
    manifest_path = args.output / "results.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Benchmark complete: {manifest_path}")


if __name__ == "__main__":
    main()
