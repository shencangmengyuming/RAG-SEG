# DINOv3 RAG-SEG Improvement Campaign

## Goal

Improve the independent DINOv3 RAG-SEG variant on CHAMELEON until it matches or exceeds the local DINOv2 CHAMELEON baseline.

## Baseline Contract

- Dataset: `CHAMELEON_TestingDataset`
- Image dir: `image`
- GT dir: `mask`
- Metrics: `S_alpha`, `meanE`, `maxE`, `F_w_beta`, `MAE`
- DINOv2 baseline:
  - `S_alpha=0.8734`
  - `meanE=0.9139`
  - `maxE=0.9191`
  - `F_w_beta=0.8320`
  - `MAE=0.02447`

## Current DINOv3 Anchor

- DINOv3-S/16 last-layer index, `K=65536`, build size `256`, query size `896`
- L2-normalized query features with L2-normalized centroids
- Retrieval aggregation: top-5 mean
- Prompt thresholds: `mask_thr=0.30`, `pos_thr=0.95`, `neg_thr=0.05`
- CHAMELEON:
  - `S_alpha=0.8793`
  - `meanE=0.9379`
  - `maxE=0.9432`
  - `F_w_beta=0.8465`
  - `MAE=0.02541`

This exceeds the DINOv2 baseline on `S_alpha`, `meanE`, and `F_w_beta`, but still misses the DINOv2 `MAE=0.02447` target by about `0.00094`.

## Hypotheses

1. The current DINOv3 index is under-compressed relative to the DINOv2 local index: `4096` vs `69632` vectors.
2. DINOv3 centroid scores are softer than DINOv2 scores, so DINOv2 thresholds are not calibrated.
3. Last-layer DINOv3 features may be less stable for localization than SegDINO-style multi-layer features.
4. The remaining MAE gap is dominated by a small number of outlier images, especially `animal-36`; narrow global prompt-threshold sweeps may close the gap without adding non-original modules.

## Slice Order

1. Add feature-cache reuse to make large-K index builds cheap enough to iterate.
2. Build DINOv3 last-layer large index near DINOv2 capacity.
3. Evaluate CHAMELEON with default thresholds.
4. If below DINOv2, run prompt/threshold ablations.
5. If still below DINOv2, build and evaluate multi-layer DINOv3 index.
6. Avoid counting SP/MM/selector/adaptive variants as the final comparator unless the user explicitly expands beyond the original RAG-SEG pipeline.

## Stop Condition

Stop when a DINOv3 configuration reaches or exceeds the DINOv2 CHAMELEON baseline on the primary metrics, with special attention to `S_alpha`, `F_w_beta`, and `MAE`.
