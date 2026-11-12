import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragseg_dinov3.build_index_dinov3 import assign_cluster_scores


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build training-free background scores for an existing DINOv3 FAISS index."
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default="indexes/dinov3/sod_cod_dinov3_vits16_k65536_centroid_l2.index",
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
    parser.add_argument(
        "--foreground_cluster_scores",
        type=str,
        default="indexes/dinov3/sod_cod_score_dinov3_vits16_k65536.index.npz",
    )
    parser.add_argument(
        "--output_scores",
        type=str,
        default="indexes/dinov3/sod_cod_background_score_dinov3_vits16_k65536.index.npz",
    )
    parser.add_argument("--batch_size", type=int, default=131072)
    parser.add_argument(
        "--reassign_from_cache",
        action="store_true",
        help="Recompute cluster assignment from token caches instead of reusing foreground cluster means.",
    )
    return parser.parse_args()


def resolve(repo_root: Path, path: str):
    path = Path(path).expanduser()
    return path if path.is_absolute() else repo_root / path


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    index_path = resolve(repo_root, args.index_path)
    feature_cache = resolve(repo_root, args.feature_cache)
    foreground_score_cache = resolve(repo_root, args.foreground_score_cache)
    foreground_cluster_scores = resolve(repo_root, args.foreground_cluster_scores)
    output_scores = resolve(repo_root, args.output_scores)
    output_scores.parent.mkdir(parents=True, exist_ok=True)

    metadata = {}
    source = None
    if foreground_cluster_scores.is_file():
        source = np.load(foreground_cluster_scores, allow_pickle=True)
        metadata = {key: to_jsonable(source[key]) for key in source.files if key != "scores"}
    if source is not None and not args.reassign_from_cache:
        foreground_cluster = source["scores"].astype("float32")
        counts = source["counts"] if "counts" in source.files else np.ones_like(foreground_cluster, dtype=np.int64)
        cluster_scores = np.zeros_like(foreground_cluster, dtype="float32")
        valid = counts > 0
        cluster_scores[valid] = 1.0 - foreground_cluster[valid]
    else:
        index = faiss.read_index(str(index_path))
        features = np.load(feature_cache, mmap_mode="r")
        foreground_scores = np.load(foreground_score_cache, mmap_mode="r")
        if features.shape[0] != foreground_scores.shape[0]:
            raise RuntimeError(
                f"Feature rows {features.shape[0]} do not match score rows {foreground_scores.shape[0]}"
            )
        if features.shape[1] != index.d:
            raise RuntimeError(f"Feature dim {features.shape[1]} does not match index dim {index.d}")

        background_scores = 1.0 - foreground_scores
        cluster_scores, counts = assign_cluster_scores(
            index=index,
            features=features,
            scores=background_scores,
            batch_size=args.batch_size,
        )
    metadata.update(
        {
            "score_mode": "background",
            "score_definition": "1 - foreground_mask_score",
            "index_path": str(index_path),
            "feature_cache": str(feature_cache),
            "foreground_score_cache": str(foreground_score_cache),
        }
    )

    np.savez_compressed(
        output_scores,
        scores=cluster_scores.astype("float32"),
        counts=counts,
        metadata=json.dumps(metadata, ensure_ascii=False),
    )
    print(f"Saved background scores: {output_scores}")
    print(
        "stats: "
        f"min={cluster_scores.min():.6f}, max={cluster_scores.max():.6f}, "
        f"mean={cluster_scores.mean():.6f}, nonempty={(counts > 0).sum()}/{len(counts)}"
    )


if __name__ == "__main__":
    main()
