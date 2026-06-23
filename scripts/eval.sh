#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

python evaluation.py \
    --dataset_name PASCAL-S \
    --data_path /home/lsx/workplace/dino-sod/segdino-main/SOD/PASCAL-S \
    --image_dir test/image \
    --gt_dir test/mask \
    --result_path outputs/ragseg_eval \
    --image_ext ".jpg" \
    --gt_ext ".png" \
    --gpu_id 0 \
    --force
