#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

python evaluation.py \
    --dataset_name COD10K \
    --data_path /home/lsx/workplace/dino-sod/RAG-SEG-main/COD10K-v3 \
    --image_dir Test/Image \
    --gt_dir Test/GT_Object \
    --split Test/CAM_Instance_Test.json \
    --result_path outputs/ragseg_eval \
    --image_ext ".jpg" \
    --gt_ext ".png" \
    --gpu_id 0
