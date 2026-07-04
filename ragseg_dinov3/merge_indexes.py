import argparse
import json
from pathlib import Path

import faiss
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Merge DINOv3 RAG-SEG FAISS indexes and score files.")
    parser.add_argument("--index_paths", nargs="+", required=True)
    parser.add_argument("--score_paths", nargs="+", required=True)
    parser.add_argument(
        "--component_limits",
        nargs="*",
        type=int,
        default=None,
        help="Optional per-component vector limits. Use 0 or a negative value to keep all vectors.",
    )
    parser.add_argument(
        "--l2_normalize",
        action="store_true",
        help="L2-normalize each component before adding to the merged IndexFlatIP.",
    )
    parser.add_argument("--output_index", required=True)
    parser.add_argument("--output_scores", required=True)
    return parser.parse_args()


def reconstruct_all(index):
    vectors = np.empty((index.ntotal, index.d), dtype="float32")
    index.reconstruct_n(0, index.ntotal, vectors)
    return vectors


def apply_limit(arr: np.ndarray, limit: int | None):
    if limit is None or limit <= 0 or limit >= arr.shape[0]:
        return arr
    return arr[:limit]


def l2_normalize_rows(arr: np.ndarray):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-6, None)


def main():
    args = parse_args()
    if len(args.index_paths) != len(args.score_paths):
        raise ValueError("--index_paths and --score_paths must have the same length")
    if args.component_limits is not None and len(args.component_limits) not in {0, len(args.index_paths)}:
        raise ValueError("--component_limits must be omitted or have one value per input index")
    limits = args.component_limits or [None] * len(args.index_paths)

    vectors = []
    scores = []
    counts = []
    metadata = []
    dim = None
    for index_path, score_path in zip(args.index_paths, args.score_paths):
        index = faiss.read_index(index_path)
        if dim is None:
            dim = index.d
        elif index.d != dim:
            raise ValueError(f"Index dim mismatch: {index_path} has {index.d}, expected {dim}")
        score_data = np.load(score_path, allow_pickle=True)
        score_arr = score_data["scores"].astype("float32")
        if score_arr.shape[0] != index.ntotal:
            raise ValueError(
                f"Score length mismatch for {score_path}: {score_arr.shape[0]} vs {index.ntotal}"
            )
        vector_arr = reconstruct_all(index)
        count_arr = (
            score_data["counts"].astype(np.int64)
            if "counts" in score_data.files
            else np.zeros(index.ntotal, dtype=np.int64)
        )
        limit = limits[len(metadata)]
        vector_arr = apply_limit(vector_arr, limit)
        score_arr = apply_limit(score_arr, limit)
        count_arr = apply_limit(count_arr, limit)
        if args.l2_normalize:
            vector_arr = l2_normalize_rows(vector_arr)
        vectors.append(vector_arr.astype("float32"))
        scores.append(score_arr.astype("float32"))
        counts.append(count_arr.astype(np.int64))
        metadata.append(
            {
                "index_path": str(index_path),
                "score_path": str(score_path),
                "ntotal": int(index.ntotal),
                "kept": int(vector_arr.shape[0]),
                "limit": None if limit is None else int(limit),
                "dim": int(index.d),
                "l2_normalized_on_merge": bool(args.l2_normalize),
                "keys": list(score_data.files),
                "metadata": str(score_data["metadata"].tolist()) if "metadata" in score_data.files else "",
            }
        )

    merged_vectors = np.concatenate(vectors, axis=0).astype("float32")
    merged_scores = np.concatenate(scores, axis=0).astype("float32")
    merged_counts = np.concatenate(counts, axis=0).astype(np.int64)
    merged_index = faiss.IndexFlatIP(dim)
    merged_index.add(merged_vectors)

    output_index = Path(args.output_index)
    output_scores = Path(args.output_scores)
    output_index.parent.mkdir(parents=True, exist_ok=True)
    output_scores.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(merged_index, str(output_index))
    np.savez(
        output_scores,
        scores=merged_scores,
        counts=merged_counts,
        component_sizes=np.array([item["ntotal"] for item in metadata], dtype=np.int64),
        feature_dim=np.array([dim], dtype=np.int64),
        metadata=json.dumps(metadata, indent=2),
    )
    print(f"Saved merged index: {output_index} ({merged_index.ntotal}, dim={dim})")
    print(f"Saved merged scores: {output_scores} ({merged_scores.shape[0]})")


if __name__ == "__main__":
    main()
