#!/usr/bin/env bash
set -euo pipefail

# OurImageModel - Google Colab setup
# Run from the repository root after cloning.

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r requirements.txt

# Keep the CUDA-enabled PyTorch build supplied by the Colab runtime unless
# the runtime requires a specific wheel. Verify the environment instead of
# silently replacing a working CUDA installation.
"$PYTHON_BIN" - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA is unavailable. Model training is not recommended.")
PY

mkdir -p dataset/images dataset/metadata dataset/examples outputs checkpoints models

touch dataset/images/.gitkeep dataset/metadata/.gitkeep dataset/examples/.gitkeep

printf '\nOurImageModel Colab environment is ready.\n'
printf 'Next: set model.base_model in configs/train.yaml and run:\n'
printf '  python train.py --check-only\n'
