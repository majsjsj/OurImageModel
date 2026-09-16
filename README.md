# OurImageModel

OurImageModel is an open development project focused on specialized image generation and editing for commercial product photography.

## V1 Goal

Build and evaluate a specialized model/adaptation for high-quality product advertising images, while keeping the system reproducible and practical to train with licensed or owned data.

The V1 foundation model is **Qwen/Qwen-Image**. OurImageModel is developed as a specialized LoRA/adaptation rather than attempting to train a 20B-class foundation model from scratch.

## Repository workflow

1. Prepare licensed or owned product images.
2. Validate the dataset and generate `metadata.jsonl`.
3. Run the configuration check:

```bash
python train.py --check-only
```

4. On a CUDA/Colab environment, download the maintained Qwen-Image LoRA trainer:

```bash
python backend/training/train_lora.py --download-only
```

5. Start training only after reviewing the dataset, VRAM, and training settings:

```bash
python train.py
```

Training outputs and model weights stay outside Git.

## Dataset

Put training images in:

```text
dataset/images/
```

Then prepare metadata:

```bash
python backend/preprocessing/prepare_dataset.py \
  --image-dir dataset/images \
  --output dataset/metadata/metadata.jsonl \
  --min-resolution 512
```

The dataset must contain images that are owned, properly licensed, or otherwise permitted for the intended training use. Keep source/license notes for every dataset source.

## High-resolution output

The inference API accepts output dimensions up to 8192px, but this does **not** mean the foundation model should generate an 8K image natively in one pass. The project is designed to support a later multi-stage/tiled high-resolution pipeline. The current placeholder resize stage is deterministic Pillow resizing, not AI super-resolution.

## Development roadmap

- [x] Repository foundation
- [x] Dataset validation and preprocessing
- [x] Inference wrapper
- [x] LoRA training launcher
- [x] Colab setup workflow
- [x] Basic automated tests
- [ ] Caption-aware training
- [ ] Benchmark suite: base model vs OurImageModel
- [ ] Verified neural upscaler / tiled high-resolution pipeline
- [ ] Product consistency and editing workflows
- [ ] API hardening and production deployment

## Testing

Run locally with:

```bash
python -m pytest -q
```

GitHub Actions also runs the dependency-light test suite on pushes and pull requests to `main`.
