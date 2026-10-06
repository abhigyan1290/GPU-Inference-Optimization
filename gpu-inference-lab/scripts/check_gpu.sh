#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run --locked python -m gpu_inference_lab.environment --require-cuda
