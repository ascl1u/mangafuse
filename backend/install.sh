#!/usr/bin/env bash
set -Eeuo pipefail

echo "Python: $(python --version 2>/dev/null || true)"
echo "Pip: $(pip --version 2>/dev/null || true)"

# Ensure latest packaging tools
python -m pip install --upgrade pip setuptools wheel

# Install CPU wheels for PyTorch first to avoid GPU/CUDA wheels
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Install remaining dependencies, excluding torch/torchvision to avoid reinstallation
tmp_req="$(mktemp)"
grep -vE '^(torch|torchvision)($|[=<>])' requirements.txt > "$tmp_req" || true
pip install -r "$tmp_req"
rm -f "$tmp_req"

echo "Dependency installation completed."


