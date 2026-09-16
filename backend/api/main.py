"""FastAPI service for OurImageModel inference.

The model is loaded lazily on the first generation request. The public API
accepts canvases from 256px up to 8192px; the inference backend can later route
4K/8K jobs through tiled or multi-stage high-resolution generation.
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.inference.generate import ImageGenerator


MAX_OUTPUT_SIZE = 8192

app = FastAPI(
    title="OurImageModel API",
    version="0.1.0",
    description="Text-to-image API for the OurImageModel project.",
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
    steps: int = Field(default=30, ge=1, le=100)
    guidance: float = Field(default=5.0, ge=0.0, le=20.0)
    seed: int | None = Field(default=42, ge=0)


class GenerateResponse(BaseModel):
    filename: str
    path: str
    seed: int | None
    width: int
    height: int


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
    """Generate an image from a text prompt."""
    try:
        generator = get_generator()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        filename = (
            f"generated_{request.width}x{request.height}_"
            f"{request.seed if request.seed is not None else 'random'}.png"
        )
        output_path = generator.generate(
            prompt=request.prompt,
            output_path=OUTPUT_DIR / filename,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance,
            seed=request.seed,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    return GenerateResponse(
        filename=output_path.name,
        path=str(output_path),
        seed=request.seed,
        width=request.width,
        height=request.height,
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
