#!/bin/bash
set -euo pipefail

CKPT_PATH="${1:-./ckpt/train_DGM4_multicls_aux/pytorch_model.pt}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
IMAGEBIND_CKPT_PATH="${IMAGEBIND_CKPT_PATH:-../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth}"
VICUNA_CKPT_PATH="${VICUNA_CKPT_PATH:-../pretrained_ckpt/vicuna_ckpt/7b_v0/}"
DELTA_CKPT_PATH="${DELTA_CKPT_PATH:-../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt}"

CONFIGS=(
  "./fake_config/test_bbc_multicls.yaml"
  "./fake_config/test_guardian_multicls.yaml"
  "./fake_config/test_usa_today_multicls.yaml"
  "./fake_config/test_washington_post_multicls.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "========== Testing ${config} =========="
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python test_multiclass_aux.py \
    --FKA_Owl_ckpt_path "${CKPT_PATH}" \
    --config "${config}" \
    --imagebind_ckpt_path "${IMAGEBIND_CKPT_PATH}" \
    --vicuna_ckpt_path "${VICUNA_CKPT_PATH}" \
    --delta_ckpt_path "${DELTA_CKPT_PATH}"
done
