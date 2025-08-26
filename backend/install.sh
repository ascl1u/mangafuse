#!/usr/bin/env bash
set -Eeuo pipefail

echo "Python: $(python --version 2>/dev/null || true)"
echo "Pip: $(pip --version 2>/dev/null || true)"

# Ensure latest packaging tools
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
tmp_req="$(mktemp)"
pip install -r "$tmp_req"
rm -f "$tmp_req"

echo "Dependency installation completed."


