import argparse
import json
from pathlib import Path

import faiss
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Merge DINOv3 RAG-SEG FAISS indexes and score files.")
    parser.add_argument("--index_paths", nargs="+", required=True)
    parser.add_argument("--score_paths", nargs="+", required=True)
    parser.add_argument("--output_index", required=True)
    parser.add_argument("--output_scores", required=True)
    return parser.parse_args()


def reconstruct_all(index):
    vectors = np.empty((index.ntotal, index.d), dtype="float32")
    index.reconstruct_n(0, index.ntotal, vectors)
    return vectors


def main():
    args = parse_args()
    if len(args.index_paths) != len(args.score_paths):
        raise ValueError("--index_paths and --score_paths must have the same length")

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
        vectors.append(reconstruct_all(index))
        scores.append(score_arr)
        if "counts" in score_data.files:
            counts.append(score_data["counts"].astype(np.int64))
        else:
            counts.append(np.zeros(index.ntotal, dtype=np.int64))
        metadata.append(
            {
                "index_path": str(index_path),
                "score_path": str(score_path),
                "ntotal": int(index.ntotal),
                "dim": int(index.d),
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

