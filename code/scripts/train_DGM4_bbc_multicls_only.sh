#!/bin/bash
ROOT=../
export PYTHONPATH=$ROOT:$PYTHONPATH
IMAGEBIND_CKPT_PATH="${IMAGEBIND_CKPT_PATH:-../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth}"
VICUNA_CKPT_PATH="${VICUNA_CKPT_PATH:-../pretrained_ckpt/vicuna_ckpt/7b_v0/}"
DELTA_CKPT_PATH="${DELTA_CKPT_PATH:-../pandagpt_ckpt/7b/pytorch_model.pt}"

deepspeed --include localhost:0,1 --master_port 28401 train_DGM4.py \
    --model openllama_peft_multicls_only \
    --stage 1 \
    --imagebind_ckpt_path "${IMAGEBIND_CKPT_PATH}" \
    --vicuna_ckpt_path "${VICUNA_CKPT_PATH}" \
    --delta_ckpt_path "${DELTA_CKPT_PATH}" \
    --max_tgt_len 1024 \
    --save_path ./ckpt/train_DGM4_multicls_only/ \
    --log_path ./ckpt/train_DGM4_multicls_only/log_rest/ \
    --config ./fake_config/train_bbc_multicls.yaml
