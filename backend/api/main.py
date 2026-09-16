"""FastAPI service for OurImageModel inference.

The API supports arbitrary output dimensions up to 8192px. Native generation
is used when requested; the explicit ``upscale`` mode provides a deterministic
high-resolution stage until a verified neural upscaler is wired in.
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from backend.inference.generate import ImageGenerator
from backend.inference.upscale import get_preset, upscale_placeholder, validate_dimensions


MAX_OUTPUT_SIZE = 8192

app = FastAPI(
    title="OurImageModel API",
    version="0.2.0",
    description="Text-to-image API with configurable high-resolution output.",
)

MODEL_ID = os.getenv("OURIMAGEMODEL_BASE_MODEL", "")
OUTPUT_DIR = Path(os.getenv("OURIMAGEMODEL_OUTPUT_DIR", "outputs/api"))

_generator: ImageGenerator | None = None
_generator_lock = Lock()


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    negative_prompt: str | None = Field(default=None, max_length=4000)
    width: int = Field(default=1024, ge=256, le=MAX_OUTPUT_SIZE)
    height: int = Field(default=1024, ge=256, le=MAX_OUTPUT_SIZE)
    resolution: str | None = Field(default=None, max_length=32)
    mode: Literal["native", "upscale"] = "native"
    base_width: int | None = Field(default=None, ge=256, le=4096)
    base_height: int | None = Field(default=None, ge=256, le=4096)
    steps: int = Field(default=30, ge=1, le=100)
    guidance: float = Field(default=5.0, ge=0.0, le=20.0)
    seed: int | None = Field(default=42, ge=0)

    @model_validator(mode="after")
    def validate_request(self) -> "GenerateRequest":
        if self.resolution:
            preset_width, preset_height = get_preset(self.resolution)
            if self.width == 1024 and self.height == 1024:
                self.width, self.height = preset_width, preset_height
            elif (self.width, self.height) != (preset_width, preset_height):
                raise ValueError("resolution preset conflicts with width/height.")

        validate_dimensions(self.width, self.height)

        if self.mode == "upscale":
            if self.base_width is None:
                self.base_width = min(self.width, 1536)
            if self.base_height is None:
                self.base_height = min(self.height, 1536)
            validate_dimensions(self.base_width, self.base_height)
            if self.base_width > self.width or self.base_height > self.height:
                raise ValueError("base dimensions cannot exceed target dimensions.")

        return self


class GenerateResponse(BaseModel):
    filename: str
    path: str
    seed: int | None
    width: int
    height: int
    mode: str
    high_resolution_stage: bool


def get_generator() -> ImageGenerator:
    """Initialize the generator once, on demand."""
    global _generator

    if _generator is not None:
        return _generator

    if not MODEL_ID:
        raise RuntimeError(
            "OURIMAGEMODEL_BASE_MODEL is not configured. "
            "Set it to the selected foundation model before generation."
        )

    with _generator_lock:
        if _generator is None:
            _generator = ImageGenerator(model_id=MODEL_ID)

    return _generator


@app.get("/health")
def health() -> dict[str, str]:
    """Simple service health check."""
    return {
        "status": "ok",
        "model_configured": "yes" if MODEL_ID else "no",
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    """Generate an image, optionally through the high-resolution stage."""
    try:
        generator = get_generator()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        filename = (
            f"generated_{request.width}x{request.height}_"
            f"{request.seed if request.seed is not None else 'random'}"
        )

        if request.mode == "native":
            output_path = generator.generate(
                prompt=request.prompt,
                output_path=OUTPUT_DIR / f"{filename}.png",
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                num_inference_steps=request.steps,
                guidance_scale=request.guidance,
                seed=request.seed,
            )
            high_resolution_stage = False
        else:
            # Generate at a manageable base size, then run the dedicated
            # high-resolution stage. The current stage is deterministic
            # resizing; a neural/tiled upscaler will replace it later.
            base_path = OUTPUT_DIR / f"{filename}_base.png"
            generator.generate(
                prompt=request.prompt,
                output_path=base_path,
                negative_prompt=request.negative_prompt,
                width=request.base_width or min(request.width, 1536),
                height=request.base_height or min(request.height, 1536),
                num_inference_steps=request.steps,
                guidance_scale=request.guidance,
                seed=request.seed,
            )
            output_path = upscale_placeholder(
                base_path,
                OUTPUT_DIR / f"{filename}.png",
                request.width,
                request.height,
            )
            high_resolution_stage = True

    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    return GenerateResponse(
        filename=output_path.name,
        path=str(output_path),
        seed=request.seed,
        width=request.width,
        height=request.height,
        mode=request.mode,
        high_resolution_stage=high_resolution_stage,
    )


@app.get("/image/{filename}")
def get_image(filename: str) -> FileResponse:
    """Return a generated PNG from the API output directory."""
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    path = OUTPUT_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")

    return FileResponse(path, media_type="image/png", filename=filename)
