#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET="${1:-all}"
shift || true

python -m ragseg_dinov3.run_task_config \
  --config configs/dinov3_cod.json \
  --dataset "${DATASET}" \
  "$@"
