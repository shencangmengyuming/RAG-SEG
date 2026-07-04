# RAG-SEG-DINOv3

Independent DINOv3 variant of RAG-SEG. The original DINOv2 scripts are left untouched.

## Local assets

The following files are intentionally not committed:

- the official DINOv3 repository, expected by default at `dinov3/`
- DINOv3 checkpoints, for example `dinov3/web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`
- datasets under `COD10K-v3/`, `CAMO-V.1.0-CVIU2019/`, and `CHAMELEON_TestingDataset/`
- generated FAISS indexes under `indexes/`
- generated predictions and metrics under `outputs/`

Clone or place the official DINOv3 repo locally, then put the DINOv3-S/16 checkpoint at the path above or pass `--dinov3_weights`.

## Build the DINOv3 retrieval index

Default: DINOv3-S/16 last-layer patch tokens, build images at `256x256`, `K=4096`.

```bash
python -m ragseg_dinov3.build_index_dinov3 \
  --cod10k_root COD10K-v3 \
  --camo_root CAMO-V.1.0-CVIU2019 \
  --image_size 256 \
  --k 4096 \
  --output_index indexes/dinov3/sod_cod_dinov3_vits16.index \
  --output_scores indexes/dinov3/sod_cod_score_dinov3_vits16.index.npz
```

## Evaluate

Default query size is `896x896`, matching the original RAG-SEG query token grid size:
`896 / 16 = 56`, analogous to DINOv2's `784 / 14 = 56`.

```bash
python -m ragseg_dinov3.evaluation_dinov3 \
  --dataset_name COD10K \
  --data_path COD10K-v3 \
  --image_dir Test/Image \
  --gt_dir Test/GT_Object \
  --split Info/CAM_test.txt \
  --result_path outputs/dinov3_last_eval \
  --image_ext ".jpg" \
  --gt_ext ".png" \
  --index_path indexes/dinov3/sod_cod_dinov3_vits16.index \
  --score_path indexes/dinov3/sod_cod_score_dinov3_vits16.index.npz
```

## SegDINO-style multi-layer ablation

Layer numbers are 1-based by default, so `3,6,9,12` maps to DINOv3-S blocks
`2,5,8,11` internally.

```bash
python -m ragseg_dinov3.build_index_dinov3 \
  --image_size 256 \
  --layers 3,6,9,12 \
  --fusion mean \
  --output_index indexes/dinov3/sod_cod_dinov3_vits16_l36912_mean.index \
  --output_scores indexes/dinov3/sod_cod_score_dinov3_vits16_l36912_mean.index.npz
```

Use the same `--layers` and `--fusion` values during evaluation.

## Current strongest local configuration

The strongest local DINOv3 run currently uses:

- DINOv3-S/16 last-layer patch tokens
- COD-only FAISS, `K=65536`
- L2-normalized query features and L2-normalized centroids
- retrieval top-5 mean
- `mask_thr=0.27`, `pos_thr=0.95`, `neg_thr=0.05`
- deterministic `farthest_confidence` point sampling

Example COD10K command:

```bash
python -m ragseg_dinov3.evaluation_dinov3 \
  --dataset_name COD10K \
  --data_path COD10K-v3 \
  --image_dir Test/Image \
  --gt_dir Test/GT_Object \
  --split Info/CAM_test.txt \
  --result_path outputs/dinov3_full_k65536_m27_farthest \
  --image_ext .jpg \
  --gt_ext .png \
  --image_size 896 \
  --l2_normalize_features \
  --index_path indexes/dinov3/sod_cod_dinov3_vits16_k65536_centroid_l2.index \
  --score_path indexes/dinov3/sod_cod_score_dinov3_vits16_k65536.index.npz \
  --retrieval_topk 5 \
  --rerank_mode mean \
  --mask_thr 0.27 \
  --pos_thr 0.95 \
  --neg_thr 0.05 \
  --point_sample_mode farthest_confidence \
  --point_farthest_weight 0.5 \
  --force
```

Local metrics from this configuration:

| Dataset | S_alpha | meanE | F_w_beta | MAE |
| --- | ---: | ---: | ---: | ---: |
| COD10K CAM test, 2026 images | 0.8595 | 0.9202 | 0.7992 | 0.02603 |
| CAMO test, 250 images | 0.8445 | 0.9040 | 0.8197 | 0.05582 |
| CHAMELEON test, 76 images | 0.8838 | 0.9448 | 0.8520 | 0.02517 |
