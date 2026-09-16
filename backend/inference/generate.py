"""Inference wrapper for OurImageModel.

The wrapper is intentionally foundation-model agnostic. A model can be loaded
from a local path or Hugging Face once the V1 foundation model is selected.
The same interface can later be used by the API layer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from diffusers import DiffusionPipeline


class ImageGenerator:
    """Load a text-to-image diffusion pipeline and generate images."""

    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        torch_dtype: str = "float16",
    ) -> None:
        if not model_id:
            raise ValueError("model_id must be provided.")

        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = self._resolve_dtype(torch_dtype, self.device)

        load_kwargs: dict[str, Any] = {"torch_dtype": self.dtype}
        if self.dtype == torch.float16:
            load_kwargs["use_safetensors"] = True

        self.pipeline = DiffusionPipeline.from_pretrained(model_id, **load_kwargs)
        self.pipeline = self.pipeline.to(self.device)

    @staticmethod
    def _resolve_dtype(dtype_name: str, device: str) -> torch.dtype:
        if dtype_name == "float16":
            if device == "cpu":
                return torch.float32
            return torch.float16
        if dtype_name == "bfloat16":
            return torch.bfloat16
        if dtype_name == "float32":
            return torch.float32
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        output_path: str | Path = "outputs/generated.png",
        negative_prompt: str | None = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        seed: int | None = 42,
    ) -> Path:
        """Generate one image and save it to disk."""
        if not prompt.strip():
            raise ValueError("prompt must not be empty.")
        if width < 256 or height < 256:
            raise ValueError("width and height must be at least 256.")
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be at least 1.")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        image = result.images[0]
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination)
        return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an image with OurImageModel.")
    parser.add_argument("--model", required=True, help="Foundation model or local pipeline path.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/generated.png"))
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = ImageGenerator(
        model_id=args.model,
        device=args.device,
        torch_dtype=args.dtype,
    )
    path = generator.generate(
        prompt=args.prompt,
        output_path=args.output,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
    )
    print(f"Image saved to: {path}")


if __name__ == "__main__":
    main()
