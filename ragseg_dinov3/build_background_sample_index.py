import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a training-free background token FAISS index from cached DINOv3 features."
    )
    parser.add_argument(
        "--feature_cache",
        type=str,
        default="outputs/dinov3_index_l2/features_dinov3_vits16_last_256_4040.npy",
    )
    parser.add_argument(
        "--foreground_score_cache",
        type=str,
        default="outputs/dinov3_index_l2/scores_dinov3_vits16_last_256_4040.npy",
    )
    parser.add_argument("--foreground_thr", type=float, default=0.05)
    parser.add_argument("--num_prototypes", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_index",
        type=str,
        default="indexes/dinov3/cod_background_tokens_dinov3_vits16_k65536_l2.index",
    )
    parser.add_argument(
        "--output_scores",
        type=str,
        default="indexes/dinov3/cod_background_tokens_score_dinov3_vits16_k65536_l2.index.npz",
    )
    return parser.parse_args()


def resolve(repo_root: Path, path: str):
    path = Path(path).expanduser()
    return path if path.is_absolute() else repo_root / path


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    feature_cache = resolve(repo_root, args.feature_cache)
    foreground_score_cache = resolve(repo_root, args.foreground_score_cache)
    output_index = resolve(repo_root, args.output_index)
    output_scores = resolve(repo_root, args.output_scores)
    output_index.parent.mkdir(parents=True, exist_ok=True)
    output_scores.parent.mkdir(parents=True, exist_ok=True)

    features = np.load(feature_cache, mmap_mode="r")
    foreground_scores = np.load(foreground_score_cache, mmap_mode="r")
    if features.shape[0] != foreground_scores.shape[0]:
        raise RuntimeError(
            f"Feature rows {features.shape[0]} do not match score rows {foreground_scores.shape[0]}"
        )

    candidates = np.flatnonzero(foreground_scores <= args.foreground_thr)
    if len(candidates) < args.num_prototypes:
        raise RuntimeError(
            f"Only {len(candidates)} background candidates for {args.num_prototypes} prototypes"
        )
    rng = np.random.default_rng(args.seed)
    selected = np.sort(rng.choice(candidates, size=args.num_prototypes, replace=False))
    prototypes = np.asarray(features[selected], dtype="float32")

    index = faiss.IndexFlatIP(prototypes.shape[1])
    index.add(prototypes)
    faiss.write_index(index, str(output_index))

    scores = np.ones(args.num_prototypes, dtype="float32")
    np.savez_compressed(
        output_scores,
        scores=scores,
        counts=np.ones(args.num_prototypes, dtype=np.int64),
        metadata=json.dumps(
            {
                "score_mode": "background_token_dummy",
                "intended_negative_mode": "confidence",
                "feature_cache": str(feature_cache),
                "foreground_score_cache": str(foreground_score_cache),
                "foreground_thr": args.foreground_thr,
                "num_candidates": int(len(candidates)),
                "num_prototypes": int(args.num_prototypes),
                "seed": int(args.seed),
            },
            ensure_ascii=False,
        ),
    )
    print(f"Saved background index: {output_index}")
    print(f"Saved dummy score file: {output_scores}")
    print(
        f"selected {args.num_prototypes}/{len(candidates)} tokens "
        f"with foreground <= {args.foreground_thr}"
    )


if __name__ == "__main__":
    main()
