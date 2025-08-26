#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status.
set -e

# Use pip to upgrade itself and install from the requirements file.
pip install --upgrade pip
pip install -r requirements-base.txt

echo "✅ Dependency installation with pip completed successfully."