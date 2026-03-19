#!/bin/bash
set -euo pipefail

CKPT_PATH="${1:-./ckpt/train_DGM4_multicls_only/pytorch_model.pt}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"

CONFIGS=(
  "./fake_config/test_bbc_multicls.yaml"
  "./fake_config/test_guardian_multicls.yaml"
  "./fake_config/test_usa_today_multicls.yaml"
  "./fake_config/test_washington_post_multicls.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "========== Testing ${config} =========="
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python test_multiclass_only.py \
    --FKA_Owl_ckpt_path "${CKPT_PATH}" \
    --config "${config}"
done
