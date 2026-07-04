# Checklist

- [x] Establish local DINOv2 CHAMELEON baseline.
- [x] Establish DINOv3 last-layer `K=4096` CHAMELEON baseline.
- [x] Diagnose initial gap.
- [x] Add DINOv3 feature-cache reuse.
- [x] Build DINOv3 last-layer large-K index. Completed: `K=65536`.
- [x] Evaluate DINOv3 large-K on CHAMELEON. `K=65536` improved most metrics but MAE remains high.
- [x] Pause non-original SP/MM ablations; user requested original-paper comparison only.
- [x] Build DINOv3 SOD/DUTS-train index segment with `K=65536`.
- [x] Merge DINOv3 COD `K=4096` + SOD `K=65536` into a `69632` vector index.
- [x] Evaluate merged DINOv3 COD+SOD index with original RAG-SEG pipeline. Result was worse than COD-only on CHAMELEON.
- [x] Diagnose current residual gap. Current best original-style DINOv3 exceeds DINOv2 on S/E/F, but misses MAE by about 0.00094; `animal-36` is the dominant outlier.
- [x] Run narrow original-pipeline threshold sweep around the current best top-5 mean configuration. Best moved slightly to `mask_thr=0.29`, `MAE=0.025383`; still below the DINOv2 MAE target.
- [x] Run original-pipeline prompt/query-resolution checks from the new `mask_thr=0.29` anchor. `pos_thr=0.93/0.94/0.96` and `image_size=1024` were worse.
- [x] Run multi-layer DINOv3 pilot index if needed. `layers=9,10,11,12`, mean fusion, L2, `K=16384` was much worse; do not expand this line.
- [x] Check whether remaining gap is caused by stochastic SAM point sampling. Seed sweep was worse than seed 42; deterministic farthest-confidence prompt sampling improved MAE but did not reach DINOv2.
- [x] Attempt COD-only `K=69632` capacity alignment. Stopped because FAISS KMeans fell back to CPU-like high-load behavior and would be too slow in the current route.
- [ ] Record final winning configuration.
