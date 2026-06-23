#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

python build_faiss_index.py \
    --cod10k_root COD10K-v3 \
    --camo_root CAMO-V.1.0-CVIU2019 \
    --k 4096 \
    --niter 200 \
    --batch_size 16 \
    --gpu_id 0 \
    --output_index outputs/ragseg_index/sod_cod_rebuild.index \
    --output_scores outputs/ragseg_index/sod_cod_score_rebuild.index.npz
