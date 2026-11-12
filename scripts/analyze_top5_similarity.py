#!/usr/bin/env python3
"""Analyze DINOv3 top-5 retrieval similarities for SOD datasets.

The script intentionally stops before SAM/metric evaluation. It measures the
raw FAISS similarities used by ``rerank_mode=softmax`` and summarizes both
dataset-level distributions and temperature-dependent weighting behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ragseg_dinov3.dinov3_extractor import DinoV3Extractor


DATASETS = {
    "PASCAL-S": {
        "data_path": "/home/lsx/workplace/dino-sod/segdino-main/SOD/PASCAL-S",
        "image_dir": "test/image",
        "image_ext": ".jpg",
    },
    "HKU-IS": {
        "data_path": "/home/lsx/workplace/dino-sod/segdino-main/SOD/HKU-IS",
        "image_dir": "test/image",
        "image_ext": ".png",
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs/top5_similarity_analysis")
    p.add_argument("--datasets", default="PASCAL-S,HKU-IS")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--image_size", type=int, default=896)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument(
        "--tokens_per_image",
        type=int,
        default=256,
        help="Number of patch tokens sampled per image before FAISS search; 0 uses all tokens.",
    )
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--hist_sample", type=int, default=250000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--taus",
        default="0.02,0.05,0.10,0.20,0.50,1.00",
        help="Comma-separated softmax temperatures.",
    )
    p.add_argument(
        "--index_path",
        default="indexes/dinov3/sod_k_ablation_20261019/duts_k4096_centroid_l2.index",
    )
    return p.parse_args()


def percentile_dict(values: np.ndarray) -> dict[str, float]:
    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    out = {}
    for q in qs:
        out[f"p{q}"] = float(np.percentile(values, q))
    return out


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_paths(cfg: dict, limit: int, seed: int):
    root = Path(cfg["data_path"]) / cfg["image_dir"]
    paths = sorted(root.glob(f"*{cfg['image_ext']}"))
    if not paths:
        raise FileNotFoundError(f"No images found in {root}")
    if limit > 0 and len(paths) > limit:
        rng = random.Random(seed)
        # Match the evaluator's deterministic first-N behavior while keeping
        # an explicit random seed available for future subsampling.
        paths = paths[:limit]
    return paths


def summarize_dataset(
    name: str,
    paths: list[Path],
    extractor: DinoV3Extractor,
    index,
    image_size: int,
    batch_size: int,
    tokens_per_image: int,
    taus: list[float],
    hist_sample: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    n_tokens = 0
    sum_s = np.zeros(5, dtype=np.float64)
    sum_s2 = np.zeros(5, dtype=np.float64)
    sum_gap_mean = 0.0
    sum_gap_last = 0.0
    sum_gap_mean2 = 0.0
    sum_gap_last2 = 0.0
    tau_sum = {str(t): {k: 0.0 for k in ("weighted_score", "entropy", "effective_k", "weight_top1") for _ in [0]} for t in taus}
    tau_sum2 = {str(t): {k: 0.0 for k in ("weighted_score", "entropy", "effective_k", "weight_top1")} for t in taus}
    sample = np.empty((0, 5), dtype=np.float32)

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        inputs = extractor.preprocess_paths(batch_paths, image_size)
        out = extractor.extract(inputs)
        token_batch = out.patch_tokens.detach().float().cpu().numpy()
        token_batch = token_batch.reshape(token_batch.shape[0], -1, token_batch.shape[-1])
        if tokens_per_image > 0 and token_batch.shape[1] > tokens_per_image:
            # Fixed stride sampling gives spatial coverage without retaining a
            # large query matrix. The same positions are used for each image.
            positions = np.linspace(0, token_batch.shape[1] - 1, tokens_per_image, dtype=np.int64)
            token_batch = token_batch[:, positions]
        tokens = token_batch.reshape(-1, token_batch.shape[-1]).astype("float32")
        distances, _ = index.search(tokens, 5)
        distances = distances.astype(np.float32)
        valid = distances >= 0
        if not valid.all():
            distances[~valid] = 0.0

        m = distances.shape[0]
        n_tokens += m
        sum_s += distances.sum(axis=0)
        sum_s2 += np.square(distances).sum(axis=0)
        gap_mean = distances[:, 0] - distances.mean(axis=1)
        gap_last = distances[:, 0] - distances[:, 4]
        sum_gap_mean += float(gap_mean.sum())
        sum_gap_last += float(gap_last.sum())
        sum_gap_mean2 += float(np.square(gap_mean).sum())
        sum_gap_last2 += float(np.square(gap_last).sum())

        if hist_sample > 0:
            take = min(hist_sample, m)
            if take < m:
                idx = rng.choice(m, size=take, replace=False)
                current = distances[idx]
            else:
                current = distances
            if sample.shape[0] < hist_sample:
                sample = np.concatenate([sample, current[: hist_sample - sample.shape[0]]], axis=0)
            else:
                replace = rng.choice(sample.shape[0], size=take, replace=False)
                sample[replace] = current[:take]

        for tau in taus:
            logits = distances / max(float(tau), 1e-6)
            logits -= logits.max(axis=1, keepdims=True)
            weights = np.exp(logits)
            weights /= weights.sum(axis=1, keepdims=True).clip(min=1e-12)
            # A monotonic similarity transform is used here so that the
            # weighted score remains interpretable even without mask scores.
            weighted = (weights * distances).sum(axis=1)
            entropy = -(weights * np.log(weights.clip(min=1e-12))).sum(axis=1)
            effective_k = np.exp(entropy)
            top1_weight = weights[:, 0]
            vals = {
                "weighted_score": weighted,
                "entropy": entropy,
                "effective_k": effective_k,
                "weight_top1": top1_weight,
            }
            for key, arr in vals.items():
                tau_sum[str(tau)][key] += float(arr.sum())
                tau_sum2[str(tau)][key] += float(np.square(arr).sum())

        if (start // batch_size) % 25 == 0:
            print(f"[{name}] {min(start + batch_size, len(paths))}/{len(paths)} images", flush=True)

    def mean_sd(total, total2):
        mean = total / n_tokens
        var = max(total2 / n_tokens - mean * mean, 0.0)
        return mean, math.sqrt(var)

    stats = {
        "dataset": name,
        "num_images": len(paths),
        "num_tokens": n_tokens,
        "top5_similarity": [],
        "gap_top1_minus_top5_mean": {},
        "gap_top1_minus_rank5": {},
        "tau": {},
    }
    for rank in range(5):
        mean, sd = mean_sd(sum_s[rank], sum_s2[rank])
        values = sample[:, rank] if sample.size else np.array([], dtype=np.float32)
        item = {"rank": rank + 1, "mean": mean, "std": sd}
        item.update(percentile_dict(values) if values.size else {})
        stats["top5_similarity"].append(item)
    mean, sd = mean_sd(sum_gap_mean, sum_gap_mean2)
    stats["gap_top1_minus_top5_mean"] = {"mean": mean, "std": sd}
    mean, sd = mean_sd(sum_gap_last, sum_gap_last2)
    stats["gap_top1_minus_rank5"] = {"mean": mean, "std": sd}
    for tau in taus:
        entry = {}
        for key in tau_sum[str(tau)]:
            mean, sd = mean_sd(tau_sum[str(tau)][key], tau_sum2[str(tau)][key])
            entry[key] = {"mean": mean, "std": sd}
        stats["tau"][str(tau)] = entry
    return stats, sample


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    index_path = Path(args.index_path)
    if not index_path.is_absolute():
        index_path = REPO_ROOT / index_path
    index = faiss.read_index(str(index_path))
    extractor = DinoV3Extractor(
        repo_root=REPO_ROOT,
        dinov3_repo="dinov3",
        weights="dinov3/web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        model_name="dinov3_vits16",
        fusion="last",
        l2_normalize=True,
        device=device,
    )
    if index.d != extractor.feature_dim:
        raise RuntimeError(f"Index dim {index.d} != extractor dim {extractor.feature_dim}")
    taus = [float(x) for x in args.taus.split(",") if x.strip()]
    out_dir = REPO_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    all_stats = []
    hist_data = {}
    for name in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        if name not in DATASETS:
            raise KeyError(f"Unknown dataset: {name}")
        paths = load_paths(DATASETS[name], args.limit, args.seed)
        stats, sample = summarize_dataset(
            name,
            paths,
            extractor,
            index,
            args.image_size,
            args.batch_size,
            args.tokens_per_image,
            taus,
            args.hist_sample,
            args.seed,
        )
        all_stats.append(stats)
        hist_data[name] = sample
        (out_dir / f"{name.replace('-', '_')}.json").write_text(json.dumps(stats, indent=2))

    (out_dir / "summary.json").write_text(json.dumps({"config": vars(args), "datasets": all_stats}, indent=2))
    rows = []
    tau_rows = []
    for stats in all_stats:
        for item in stats["top5_similarity"]:
            rows.append({"dataset": stats["dataset"], **item})
        for tau, item in stats["tau"].items():
            flat = {}
            for key, value_stats in item.items():
                for stat_name, value in value_stats.items():
                    flat[f"{key}_{stat_name}"] = value
            tau_rows.append({"dataset": stats["dataset"], "tau": tau, **flat})
    write_csv(out_dir / "top5_similarity_quantiles.csv", rows)
    write_csv(out_dir / "tau_sensitivity.csv", tau_rows)

    plt.figure(figsize=(10, 5))
    for name, sample in hist_data.items():
        if sample.size:
            plt.hist(sample[:, 0], bins=80, density=True, alpha=0.45, label=f"{name} top-1")
            plt.hist(sample[:, 4], bins=80, density=True, alpha=0.45, label=f"{name} top-5")
    plt.xlabel("FAISS cosine similarity")
    plt.ylabel("Density")
    plt.title("Top-1 vs top-5 retrieval similarity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "top1_top5_similarity_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for stats in all_stats:
        taus_sorted = sorted(float(t) for t in stats["tau"])
        y = [stats["tau"][str(t)]["effective_k"]["mean"] for t in taus_sorted]
        plt.plot(taus_sorted, y, marker="o", label=stats["dataset"])
    plt.xscale("log")
    plt.xlabel("tau")
    plt.ylabel("Effective number of neighbors exp(entropy)")
    plt.title("Temperature sensitivity of top-5 weighting")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "tau_effective_k.png", dpi=180)
    plt.close()
    print(json.dumps({"output": str(out_dir), "datasets": all_stats}, indent=2))


if __name__ == "__main__":
    main()
