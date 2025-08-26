#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status.
set -e

# 1. Install uv using pip
pip install uv

# 2. Use uv to install all dependencies from your requirements file.
# This is much faster than using pip directly.
uv pip install --system -r requirements-base.txt

echo "✅ Dependency installation with uv completed successfully."