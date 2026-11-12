import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragseg_dinov3.dinov3_extractor import DinoV3Extractor


def log(message: str):
    print(message, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the independent DINOv3 RAG-SEG variant.")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--result_path", type=str, default="outputs/dinov3_eval")
    parser.add_argument("--split", type=str, default="")
    parser.add_argument("--image_ext", type=str, default=".jpg")
    parser.add_argument("--gt_ext", type=str, default=".png")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=896)
    parser.add_argument("--sam_size", type=int, default=1024)
    parser.add_argument("--dinov3_repo", type=str, default="dinov3")
    parser.add_argument(
        "--dinov3_weights",
        type=str,
        default="dinov3/web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    )
    parser.add_argument("--backbone", type=str, default="dinov3_vits16")
    parser.add_argument("--layers", type=str, default="")
    parser.add_argument("--fusion", type=str, default="last", choices=["last", "mean", "weighted", "concat"])
    parser.add_argument("--layer_weights", type=str, default="")
    parser.add_argument("--zero_based_layers", action="store_true")
    parser.add_argument("--l2_normalize_features", action="store_true")
    parser.add_argument("--debias_svd_components", type=int, default=0)
    parser.add_argument("--pos_thr", type=float, default=0.99)
    parser.add_argument("--neg_thr", type=float, default=0.05)
    parser.add_argument("--mask_thr", type=float, default=0.3)
    parser.add_argument("--disable_point_prompts", action="store_true")
    parser.add_argument("--disable_mask_prompt", action="store_true")
    parser.add_argument("--n_points", type=int, default=10)
    parser.add_argument(
        "--point_sample_mode",
        type=str,
        default="random",
        choices=["random", "confidence", "farthest_confidence"],
    )
    parser.add_argument("--point_farthest_weight", type=float, default=0.5)
    parser.add_argument("--index_path", type=str, default="indexes/dinov3/sod_cod_dinov3_vits16.index")
    parser.add_argument("--score_path", type=str, default="indexes/dinov3/sod_cod_score_dinov3_vits16.index.npz")
    parser.add_argument(
        "--source_index_paths",
        nargs="*",
        default=None,
        help="Optional per-source FAISS indexes. Enables separate source retrieval when provided.",
    )
    parser.add_argument(
        "--source_score_paths",
        nargs="*",
        default=None,
        help="Optional per-source score files. Must match --source_index_paths.",
    )
    parser.add_argument(
        "--source_fusion",
        type=str,
        default="fixed",
        choices=["fixed", "top1_conf", "agree_conf", "max_conf"],
    )
    parser.add_argument("--source_weights", type=str, default="")
    parser.add_argument("--source_fusion_temp", type=float, default=0.05)
    parser.add_argument("--source_agree_weight", type=float, default=0.2)
    parser.add_argument("--negative_index_path", type=str, default="")
    parser.add_argument("--negative_score_path", type=str, default="")
    parser.add_argument(
        "--negative_weight",
        type=float,
        default=0.0,
        help="Subtract a retrieved background/environment memory score before prior normalization.",
    )
    parser.add_argument(
        "--negative_mode",
        type=str,
        default="score",
        choices=["score", "confidence"],
        help="Use stored negative scores or retrieval similarity as the negative evidence.",
    )
    parser.add_argument(
        "--negative_confidence_mode",
        type=str,
        default="top1",
        choices=["top1", "mean", "max"],
    )
    parser.add_argument("--correspondence_feature_cache", type=str, default="")
    parser.add_argument("--correspondence_score_cache", type=str, default="")
    parser.add_argument("--correspondence_grid_size", type=int, default=16)
    parser.add_argument("--correspondence_topm", type=int, default=5)
    parser.add_argument("--correspondence_blend", type=float, default=0.0)
    parser.add_argument(
        "--correspondence_mode",
        type=str,
        default="blend",
        choices=["blend", "only"],
    )
    parser.add_argument("--correspondence_fg_power", type=float, default=2.0)
    parser.add_argument("--correspondence_temp", type=float, default=0.07)
    parser.add_argument("--retrieval_topk", type=int, default=1)
    parser.add_argument(
        "--rerank_mode",
        type=str,
        default="top1",
        choices=["top1", "mean", "max", "softmax", "gated_max", "blend_mean"],
    )
    parser.add_argument("--rerank_temperature", type=float, default=1.0)
    parser.add_argument("--rerank_lambda", type=float, default=0.7)
    parser.add_argument("--rerank_conf_thr", type=float, default=0.3)
    parser.add_argument("--rerank_conf_temp", type=float, default=0.1)
    parser.add_argument("--rerank_agree_temp", type=float, default=0.2)
    parser.add_argument(
        "--prior_mode",
        type=str,
        default="none",
        choices=["none", "pc", "fcpc", "region"],
    )
    parser.add_argument("--pc_seed_thr", type=float, default=0.5)
    parser.add_argument("--pc_expand_iter", type=int, default=2)
    parser.add_argument("--pc_close_kernel", type=int, default=5)
    parser.add_argument("--fcpc_seed_thr", type=float, default=0.7)
    parser.add_argument("--fcpc_low_thr", type=float, default=0.2)
    parser.add_argument("--fcpc_sim_thr", type=float, default=0.6)
    parser.add_argument("--fcpc_expand_iter", type=int, default=2)
    parser.add_argument("--fcpc_mix", type=float, default=0.6)
    parser.add_argument(
        "--local_propagation",
        type=str,
        default="off",
        choices=["off", "plain", "confidence"],
        help="Apply 8-neighbor feature propagation to the raw retrieval prior.",
    )
    parser.add_argument("--propagation_steps", type=int, default=1)
    parser.add_argument("--propagation_lambda", type=float, default=0.25)
    parser.add_argument("--propagation_feature_temp", type=float, default=0.1)
    parser.add_argument("--propagation_confidence_temp", type=float, default=0.15)
    parser.add_argument("--region_clusters", type=int, default=96)
    parser.add_argument("--region_kmeans_iter", type=int, default=6)
    parser.add_argument("--region_mix", type=float, default=0.35)
    parser.add_argument(
        "--region_score_mode",
        type=str,
        default="mean",
        choices=["mean", "max", "seed_mean"],
    )
    parser.add_argument("--region_seed_thr", type=float, default=0.65)
    parser.add_argument("--use_box_prompt", action="store_true")
    parser.add_argument("--box_pad_ratio", type=float, default=0.04)
    parser.add_argument("--multimask_voting", action="store_true")
    parser.add_argument("--sam_score_weight", type=float, default=0.15)
    parser.add_argument(
        "--consistency_mode",
        type=str,
        default="off",
        choices=["off", "filter", "blend"],
        help="Use horizontal-flip retrieval consistency to gate the coarse prior.",
    )
    parser.add_argument("--consistency_threshold", type=float, default=0.8)
    parser.add_argument(
        "--consistency_blend",
        type=float,
        default=0.5,
        help="Blend weight for the aligned flipped prior in consistency blend mode.",
    )
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--save_prior", action="store_true")
    parser.add_argument("--save_prior_float32", action="store_true")
    parser.add_argument("--save_sam_prior_float32", action="store_true")
    parser.add_argument("--table12_sweep", action="store_true")
    parser.add_argument("--table12_variant_ids", nargs="+", default=None)
    return parser.parse_args()


def resolve_dir(base: Path, maybe_relative: str):
    path = Path(maybe_relative).expanduser()
    return path if path.is_absolute() else base / path


def load_index(repo_root: Path, index_path: str, score_path: str):
    index_path = Path(index_path).expanduser()
    score_path = Path(score_path).expanduser()
    if not index_path.is_absolute():
        index_path = repo_root / index_path
    if not score_path.is_absolute():
        score_path = repo_root / score_path
    index = faiss.read_index(str(index_path))
    score_file = np.load(score_path, allow_pickle=True)
    scores = score_file["scores"].astype("float32")
    metadata = {key: score_file[key].tolist() for key in score_file.files if key != "scores"}
    return index, scores, metadata


def load_retrieval_sources(repo_root: Path, args):
    if args.source_index_paths or args.source_score_paths:
        if not args.source_index_paths or not args.source_score_paths:
            raise ValueError("--source_index_paths and --source_score_paths must be provided together")
        if len(args.source_index_paths) != len(args.source_score_paths):
            raise ValueError("--source_index_paths and --source_score_paths must have the same length")
        index_paths = args.source_index_paths
        score_paths = args.source_score_paths
    else:
        index_paths = [args.index_path]
        score_paths = [args.score_path]

    sources = []
    for source_id, (index_path, score_path) in enumerate(zip(index_paths, score_paths)):
        index, scores, metadata = load_index(repo_root, index_path, score_path)
        sources.append(
            {
                "source_id": source_id,
                "index_path": index_path,
                "score_path": score_path,
                "index": index,
                "scores": scores,
                "metadata": metadata,
            }
        )
    return sources


def load_negative_sources(repo_root: Path, args):
    if not args.negative_index_path and not args.negative_score_path:
        return []
    if not args.negative_index_path or not args.negative_score_path:
        raise ValueError("--negative_index_path and --negative_score_path must be provided together")
    index, scores, metadata = load_index(repo_root, args.negative_index_path, args.negative_score_path)
    return [
        {
            "source_id": 0,
            "index_path": args.negative_index_path,
            "score_path": args.negative_score_path,
            "index": index,
            "scores": scores,
            "metadata": metadata,
        }
    ]


def load_correspondence_memory(repo_root: Path, args):
    if not args.correspondence_feature_cache and not args.correspondence_score_cache:
        return None
    if not args.correspondence_feature_cache or not args.correspondence_score_cache:
        raise ValueError(
            "--correspondence_feature_cache and --correspondence_score_cache must be provided together"
        )
    feature_path = Path(args.correspondence_feature_cache).expanduser()
    score_path = Path(args.correspondence_score_cache).expanduser()
    if not feature_path.is_absolute():
        feature_path = repo_root / feature_path
    if not score_path.is_absolute():
        score_path = repo_root / score_path

    features = np.load(feature_path, mmap_mode="r")
    scores = np.load(score_path, mmap_mode="r")
    tokens_per_image = int(args.correspondence_grid_size) ** 2
    if features.shape[0] != scores.shape[0]:
        raise RuntimeError(
            f"Correspondence feature rows {features.shape[0]} do not match score rows {scores.shape[0]}"
        )
    if features.shape[0] % tokens_per_image:
        raise RuntimeError(
            f"Feature rows {features.shape[0]} are not divisible by tokens_per_image={tokens_per_image}"
        )
    num_images = features.shape[0] // tokens_per_image
    feature_dim = features.shape[1]
    prototypes = np.zeros((num_images, feature_dim), dtype=np.float32)
    for image_id in tqdm(range(num_images), desc="corr-proto", ncols=100):
        start = image_id * tokens_per_image
        end = start + tokens_per_image
        feats = np.asarray(features[start:end], dtype=np.float32)
        weights = np.asarray(scores[start:end], dtype=np.float32)
        weights = np.power(np.clip(weights, 0.0, 1.0), max(float(args.correspondence_fg_power), 0.0))
        if float(weights.sum()) <= 1e-6:
            proto = feats.mean(axis=0)
        else:
            proto = (feats * weights[:, None]).sum(axis=0) / weights.sum()
        proto = proto / max(float(np.linalg.norm(proto)), 1e-6)
        prototypes[image_id] = proto.astype(np.float32)

    return {
        "feature_path": str(feature_path),
        "score_path": str(score_path),
        "features": features,
        "scores": scores,
        "prototypes": prototypes,
        "tokens_per_image": tokens_per_image,
        "num_images": num_images,
        "feature_dim": feature_dim,
    }


def load_sam2_predictor(repo_root: Path, device: torch.device):
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != repo_root]
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    ckpt_path = repo_root / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"
    sam_model = build_sam2(
        config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
        ckpt_path=str(ckpt_path),
        device=device,
    )
    return SAM2ImagePredictor(sam_model)


def fuse_retrieval_scores(distances: np.ndarray, idxs: np.ndarray, scores: np.ndarray, args):
    valid = distances >= 0
    retrieved_scores = np.zeros_like(distances, dtype=np.float32)
    retrieved_scores[valid] = scores[idxs[valid]]

    if args.rerank_mode == "top1" or distances.shape[1] == 1:
        return retrieved_scores[:, 0]
    if args.rerank_mode == "mean":
        counts = valid.sum(axis=1).clip(min=1)
        return retrieved_scores.sum(axis=1) / counts
    if args.rerank_mode == "blend_mean":
        counts = valid.sum(axis=1).clip(min=1)
        mean_scores = retrieved_scores.sum(axis=1) / counts
        blend = float(np.clip(args.rerank_lambda, 0.0, 1.0))
        return (1.0 - blend) * retrieved_scores[:, 0] + blend * mean_scores
    if args.rerank_mode == "max":
        retrieved_scores[~valid] = 0
        return retrieved_scores.max(axis=1)
    if args.rerank_mode == "gated_max":
        retrieved_scores[~valid] = 0
        top1 = retrieved_scores[:, 0]
        topk_max = retrieved_scores.max(axis=1)
        topk_std = retrieved_scores.std(axis=1)
        conf_gate = 1.0 / (
            1.0 + np.exp(-(top1 - args.rerank_conf_thr) / max(args.rerank_conf_temp, 1e-6))
        )
        agree_gate = np.exp(-topk_std / max(args.rerank_agree_temp, 1e-6))
        gate = conf_gate * agree_gate
        return top1 + args.rerank_lambda * gate * (topk_max - top1)
    if args.rerank_mode == "softmax":
        temperature = max(args.rerank_temperature, 1e-6)
        logits = distances.astype(np.float32) / temperature
        logits[~valid] = -np.inf
        logits = logits - np.nanmax(logits, axis=1, keepdims=True)
        weights = np.exp(logits)
        weights[~valid] = 0
        weights_sum = weights.sum(axis=1, keepdims=True).clip(min=1e-6)
        weights = weights / weights_sum
        return (retrieved_scores * weights).sum(axis=1)
    raise ValueError(f"Unsupported rerank mode: {args.rerank_mode}")


def parse_source_weights(args, n_sources: int):
    if not args.source_weights:
        return np.full(n_sources, 1.0 / n_sources, dtype=np.float32)
    values = np.array([float(item) for item in args.source_weights.split(",")], dtype=np.float32)
    if values.shape[0] != n_sources:
        raise ValueError(f"--source_weights has {values.shape[0]} values for {n_sources} sources")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("--source_weights must sum to a positive value")
    return values / total


def softmax_rows(logits: np.ndarray, temperature: float):
    temperature = max(float(temperature), 1e-6)
    scaled = logits.astype(np.float32) / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    weights = np.exp(scaled)
    return weights / weights.sum(axis=1, keepdims=True).clip(min=1e-6)


def fuse_source_retrieval(source_results, args):
    if len(source_results) == 1:
        return source_results[0]["scores"]

    source_scores = np.stack([item["scores"] for item in source_results], axis=1)
    if args.source_fusion == "fixed":
        weights = parse_source_weights(args, len(source_results))
        return (source_scores * weights[None, :]).sum(axis=1)

    top1_conf = np.stack([item["distances"][:, 0] for item in source_results], axis=1)
    if args.source_fusion == "max_conf":
        winners = np.argmax(top1_conf, axis=1)
        return source_scores[np.arange(source_scores.shape[0]), winners]

    if args.source_fusion == "top1_conf":
        weights = softmax_rows(top1_conf, args.source_fusion_temp)
        return (source_scores * weights).sum(axis=1)

    if args.source_fusion == "agree_conf":
        confidence = []
        for item in source_results:
            retrieved = item["retrieved_scores"]
            score_std = retrieved.std(axis=1) if retrieved.shape[1] > 1 else np.zeros(retrieved.shape[0])
            confidence.append(item["distances"][:, 0] - args.source_agree_weight * score_std)
        weights = softmax_rows(np.stack(confidence, axis=1), args.source_fusion_temp)
        return (source_scores * weights).sum(axis=1)

    raise ValueError(f"Unsupported source_fusion: {args.source_fusion}")


def retrieve_from_source(flat_tokens: np.ndarray, source, args):
    distances, idxs = source["index"].search(flat_tokens, max(1, args.retrieval_topk))
    valid = distances >= 0
    retrieved_scores = np.zeros_like(distances, dtype=np.float32)
    retrieved_scores[valid] = source["scores"][idxs[valid]]
    return {
        "distances": distances,
        "idxs": idxs,
        "retrieved_scores": retrieved_scores,
        "scores": fuse_retrieval_scores(distances, idxs, source["scores"], args),
    }


def fuse_negative_confidence(source_results, args):
    values = []
    for item in source_results:
        distances = np.clip(item["distances"].astype(np.float32), 0.0, 1.0)
        valid = item["distances"] >= 0
        if args.negative_confidence_mode == "top1" or distances.shape[1] == 1:
            values.append(distances[:, 0])
        elif args.negative_confidence_mode == "mean":
            distances[~valid] = 0
            counts = valid.sum(axis=1).clip(min=1)
            values.append(distances.sum(axis=1) / counts)
        elif args.negative_confidence_mode == "max":
            distances[~valid] = 0
            values.append(distances.max(axis=1))
        else:
            raise ValueError(f"Unsupported negative_confidence_mode: {args.negative_confidence_mode}")
    return np.stack(values, axis=1).mean(axis=1)


def normalize01(x: np.ndarray):
    x = x.astype(np.float32)
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max - x_min < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def retrieval_confidence(source_results, args):
    """Combine nearest-neighbor similarity and Top-k mask-score agreement."""
    confidences = []
    for item in source_results:
        similarities = item["distances"].astype(np.float32)
        valid = similarities >= 0
        top1 = np.clip((similarities[:, 0] + 1.0) * 0.5, 0.0, 1.0)
        neighbor_scores = item["retrieved_scores"].astype(np.float32).copy()
        neighbor_scores[~valid] = 0.0
        counts = valid.sum(axis=1).clip(min=1)
        means = neighbor_scores.sum(axis=1) / counts
        variance = (((neighbor_scores - means[:, None]) ** 2) * valid).sum(axis=1) / counts
        disagreement = np.sqrt(variance)
        agree = np.exp(-disagreement / max(float(args.propagation_confidence_temp), 1e-6))
        confidences.append(top1 * agree)
    return np.mean(np.stack(confidences, axis=1), axis=1).astype(np.float32)


def local_spatial_propagation(prior: np.ndarray, token_grid: np.ndarray, confidence: np.ndarray, args):
    """Diffuse prior scores over 8-neighbor feature edges, optionally confidence-gated."""
    if token_grid is None or args.local_propagation == "off":
        return prior.astype(np.float32)

    h, w = prior.shape
    feats = token_grid.astype(np.float32)
    feats /= np.linalg.norm(feats, axis=-1, keepdims=True).clip(min=1e-6)
    confidence = np.clip(confidence.reshape(h, w), 0.0, 1.0)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    edge_weights = []
    neighbor_confidences = []
    for dy, dx in offsets:
        y0, y1 = max(0, -dy), min(h, h - dy)
        x0, x1 = max(0, -dx), min(w, w - dx)
        src = feats[y0:y1, x0:x1]
        dst = feats[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        similarity = np.sum(src * dst, axis=-1)
        weight = np.exp((similarity - 1.0) / max(float(args.propagation_feature_temp), 1e-6))
        full_weight = np.zeros((h, w), dtype=np.float32)
        full_conf = np.zeros((h, w), dtype=np.float32)
        full_weight[y0:y1, x0:x1] = weight
        full_conf[y0:y1, x0:x1] = confidence[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        edge_weights.append(full_weight)
        neighbor_confidences.append(full_conf)

    weights = np.stack(edge_weights, axis=0)
    neighbor_conf = np.stack(neighbor_confidences, axis=0)
    if args.local_propagation == "confidence":
        weights *= neighbor_conf
    weights_sum = weights.sum(axis=0).clip(min=1e-6)

    current = prior.astype(np.float32).copy()
    lam = float(np.clip(args.propagation_lambda, 0.0, 1.0))
    steps = max(0, int(args.propagation_steps))
    for _ in range(steps):
        neighbor_sum = np.zeros((h, w), dtype=np.float32)
        for weight, (dy, dx) in zip(weights, offsets):
            y0, y1 = max(0, -dy), min(h, h - dy)
            x0, x1 = max(0, -dx), min(w, w - dx)
            neighbor = current[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            neighbor_sum[y0:y1, x0:x1] += weight[y0:y1, x0:x1] * neighbor
        propagated = neighbor_sum / weights_sum
        if args.local_propagation == "confidence":
            step = lam * (1.0 - confidence)
        else:
            step = np.full((h, w), lam, dtype=np.float32)
        has_neighbors = weights.sum(axis=0) > 1e-6
        step *= has_neighbors
        current = (1.0 - step) * current + step * propagated
    return np.clip(current, 0.0, 1.0).astype(np.float32)


def dense_correspondence_prior(flat_tokens: np.ndarray, mask_vals: np.ndarray, memory, args):
    if memory is None:
        return mask_vals

    query_feats = flat_tokens.astype(np.float32)
    query_feats = query_feats / np.linalg.norm(query_feats, axis=1, keepdims=True).clip(min=1e-6)
    query_prior = normalize01(mask_vals)
    weights = np.power(
        np.clip(query_prior, 0.0, 1.0),
        max(float(args.correspondence_fg_power), 0.0),
    )
    if float(weights.sum()) <= 1e-6:
        thresh = float(np.quantile(query_prior, 0.9))
        weights = (query_prior >= thresh).astype(np.float32)
    query_proto = (query_feats * weights[:, None]).sum(axis=0) / max(float(weights.sum()), 1e-6)
    query_proto = query_proto / max(float(np.linalg.norm(query_proto)), 1e-6)

    source_logits = memory["prototypes"] @ query_proto.astype(np.float32)
    topm = max(1, min(int(args.correspondence_topm), memory["num_images"]))
    top_ids = np.argpartition(source_logits, -topm)[-topm:]
    top_ids = top_ids[np.argsort(source_logits[top_ids])[::-1]]
    logits = source_logits[top_ids].astype(np.float32)
    logits = logits / max(float(args.correspondence_temp), 1e-6)
    logits = logits - logits.max()
    source_weights = np.exp(logits)
    source_weights = source_weights / max(float(source_weights.sum()), 1e-6)

    corr = np.zeros(query_feats.shape[0], dtype=np.float32)
    tokens_per_image = memory["tokens_per_image"]
    for source_weight, source_id in zip(source_weights, top_ids):
        start = int(source_id) * tokens_per_image
        end = start + tokens_per_image
        source_feats = np.asarray(memory["features"][start:end], dtype=np.float32)
        source_feats = source_feats / np.linalg.norm(source_feats, axis=1, keepdims=True).clip(min=1e-6)
        source_scores = np.asarray(memory["scores"][start:end], dtype=np.float32)
        sims = query_feats @ source_feats.T
        nn_idx = np.argmax(sims, axis=1)
        nn_conf = np.clip(sims[np.arange(sims.shape[0]), nn_idx], 0.0, 1.0)
        transferred = source_scores[nn_idx] * nn_conf
        corr += float(source_weight) * transferred.astype(np.float32)

    corr = normalize01(corr)
    if args.correspondence_mode == "only":
        return corr
    blend = float(np.clip(args.correspondence_blend, 0.0, 1.0))
    return normalize01((1.0 - blend) * query_prior + blend * corr)


def part_composition_prior(prior: np.ndarray, args):
    seed = (prior >= args.pc_seed_thr).astype(np.uint8)
    if seed.sum() == 0:
        return prior
    close_kernel = max(1, int(args.pc_close_kernel))
    if close_kernel % 2 == 0:
        close_kernel += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    composed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel)
    expand_iter = max(0, int(args.pc_expand_iter))
    if expand_iter:
        expand_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        composed = cv2.dilate(composed, expand_kernel, iterations=expand_iter)
    soft = cv2.GaussianBlur(composed.astype(np.float32), (0, 0), sigmaX=1.0)
    enhanced = np.maximum(prior, prior * (1.0 - soft) + soft)
    return np.clip(enhanced, 0.0, 1.0).astype(np.float32)


def feature_constrained_part_composition_prior(prior: np.ndarray, token_grid: np.ndarray, args):
    if token_grid is None:
        return prior
    seed = prior >= args.fcpc_seed_thr
    if seed.sum() == 0:
        return prior
    feats = token_grid.astype(np.float32)
    feats = feats / np.linalg.norm(feats, axis=-1, keepdims=True).clip(min=1e-6)
    weights = (prior * seed).astype(np.float32)
    proto = (feats * weights[..., None]).sum(axis=(0, 1)) / weights.sum().clip(min=1e-6)
    proto = proto / np.linalg.norm(proto).clip(min=1e-6)
    sim = ((feats * proto).sum(axis=-1) + 1.0) * 0.5
    reachable = seed.astype(np.uint8)
    expand_iter = max(0, int(args.fcpc_expand_iter))
    if expand_iter:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        reachable = cv2.dilate(reachable, kernel, iterations=expand_iter)
    candidates = (
        reachable.astype(bool)
        & (prior >= args.fcpc_low_thr)
        & (sim >= args.fcpc_sim_thr)
    )
    enhanced = prior.copy()
    mixed = (1.0 - args.fcpc_mix) * prior + args.fcpc_mix * sim.astype(np.float32)
    enhanced[candidates] = np.maximum(enhanced[candidates], mixed[candidates])
    return np.clip(enhanced, 0.0, 1.0).astype(np.float32)


def kmeans_regions(feats: np.ndarray, n_clusters: int, n_iter: int):
    n_tokens, dim = feats.shape
    n_clusters = max(1, min(int(n_clusters), n_tokens))
    n_iter = max(1, int(n_iter))

    # Deterministic farthest-point initialization in feature space.
    centers = np.empty((n_clusters, dim), dtype=np.float32)
    centers[0] = feats[0]
    min_dist = np.full(n_tokens, np.inf, dtype=np.float32)
    for k in range(1, n_clusters):
        sim = feats @ centers[k - 1]
        dist = 1.0 - sim
        min_dist = np.minimum(min_dist, dist.astype(np.float32))
        centers[k] = feats[int(np.argmax(min_dist))]

    labels = np.zeros(n_tokens, dtype=np.int32)
    for _ in range(n_iter):
        labels = np.argmax(feats @ centers.T, axis=1).astype(np.int32)
        sums = np.zeros_like(centers)
        counts = np.bincount(labels, minlength=n_clusters).astype(np.float32)
        np.add.at(sums, labels, feats)
        valid = counts > 0
        centers[valid] = sums[valid] / counts[valid, None]
        centers[valid] /= np.linalg.norm(centers[valid], axis=1, keepdims=True).clip(min=1e-6)
    return labels, n_clusters


def region_aggregation_prior(prior: np.ndarray, token_grid: np.ndarray, args):
    if token_grid is None:
        return prior
    h, w = prior.shape
    feats = token_grid.reshape(-1, token_grid.shape[-1]).astype(np.float32)
    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True).clip(min=1e-6)
    labels, n_clusters = kmeans_regions(feats, args.region_clusters, args.region_kmeans_iter)

    flat_prior = prior.reshape(-1).astype(np.float32)
    region_scores = np.zeros(n_clusters, dtype=np.float32)
    for region_id in range(n_clusters):
        idx = labels == region_id
        if not idx.any():
            continue
        values = flat_prior[idx]
        if args.region_score_mode == "max":
            region_scores[region_id] = float(values.max())
        elif args.region_score_mode == "seed_mean":
            seeds = values[values >= args.region_seed_thr]
            region_scores[region_id] = float(seeds.mean()) if len(seeds) else float(values.mean())
        else:
            region_scores[region_id] = float(values.mean())

    region_prior = region_scores[labels].reshape(h, w)
    mix = float(np.clip(args.region_mix, 0.0, 1.0))
    enhanced = (1.0 - mix) * prior + mix * region_prior
    return normalize01(enhanced).astype(np.float32)


def enhance_prior(prior: np.ndarray, token_grid: np.ndarray, args):
    enhanced = normalize01(prior)
    if args.prior_mode == "pc":
        enhanced = part_composition_prior(enhanced, args)
    if args.prior_mode == "fcpc":
        enhanced = feature_constrained_part_composition_prior(enhanced, token_grid, args)
    if args.prior_mode == "region":
        enhanced = region_aggregation_prior(enhanced, token_grid, args)
    return enhanced.astype(np.float32)


def mask_to_box(mask: np.ndarray, pad_ratio: float):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pad = int(round(max(h, w) * max(0.0, pad_ratio)))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w - 1, x1 + pad)
    y1 = min(h - 1, y1 + pad)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def mask_iou(mask: np.ndarray, prior_binary: np.ndarray):
    prior = prior_binary.astype(bool)
    pred = mask.astype(bool)
    inter = np.logical_and(pred, prior).sum()
    union = np.logical_or(pred, prior).sum()
    return float(inter / union) if union else 0.0


def aligned_flip_prior_consistency(prior: np.ndarray, flipped_prior: np.ndarray, threshold: float):
    """Compare two retrieval priors after undoing the horizontal flip."""
    prior_binary = prior >= threshold
    flipped_binary = flipped_prior >= threshold
    union = np.logical_or(prior_binary, flipped_binary).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(prior_binary, flipped_binary).sum() / union)


def apply_consistency_gate(
    img: Image.Image,
    prior: np.ndarray,
    extractor: DinoV3Extractor,
    retrieval_sources,
    negative_sources,
    correspondence_memory,
    args,
):
    if args.consistency_mode == "off":
        return prior

    flipped = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    flipped_prior, _, _ = rag_coarse_mask(
        flipped,
        extractor,
        retrieval_sources,
        negative_sources,
        correspondence_memory,
        args,
    )
    aligned = np.fliplr(flipped_prior).astype(np.float32)
    consistency = aligned_flip_prior_consistency(
        normalize01(prior),
        normalize01(aligned),
        float(np.clip(args.mask_thr, 0.0, 1.0)),
    )

    if consistency >= float(args.consistency_threshold):
        return prior
    if args.consistency_mode == "blend":
        blend = float(np.clip(args.consistency_blend, 0.0, 1.0))
        return normalize01((1.0 - blend) * prior + blend * np.minimum(prior, aligned))
    return np.minimum(prior, aligned).astype(np.float32)


def select_sam_mask(masks: np.ndarray, sam_scores: np.ndarray, prior_binary: np.ndarray, args):
    if len(masks) == 1 or not args.multimask_voting:
        return masks[0]
    votes = []
    for i, mask in enumerate(masks):
        sam_score = float(sam_scores[i]) if i < len(sam_scores) else 0.0
        votes.append(mask_iou(mask, prior_binary) + args.sam_score_weight * sam_score)
    return masks[int(np.argmax(votes))]


def sample_prompt_points(indices: np.ndarray, values: np.ndarray, n_points: int, mode: str, high: bool, args):
    if len(indices) <= n_points or mode == "random":
        if len(indices) > n_points:
            return indices[np.random.choice(len(indices), n_points, replace=False)]
        return indices

    scores = values[indices[:, 0], indices[:, 1]].astype(np.float32)
    order = np.argsort(scores)
    if high:
        order = order[::-1]
    if mode == "confidence":
        return indices[order[:n_points]]

    ordered = indices[order]
    ordered_scores = scores[order]
    if not high:
        ordered_scores = 1.0 - ordered_scores
    score_min = float(ordered_scores.min())
    score_max = float(ordered_scores.max())
    if score_max - score_min > 1e-6:
        ordered_scores = (ordered_scores - score_min) / (score_max - score_min)
    else:
        ordered_scores = np.ones_like(ordered_scores, dtype=np.float32)

    selected = [0]
    coords = ordered.astype(np.float32)
    h, w = values.shape
    scale = np.array([max(h - 1, 1), max(w - 1, 1)], dtype=np.float32)
    coords = coords / scale
    min_dist = np.full(len(ordered), np.inf, dtype=np.float32)
    min_dist[0] = 0.0
    weight = float(np.clip(args.point_farthest_weight, 0.0, 1.0))
    for _ in range(1, min(n_points, len(ordered))):
        last = selected[-1]
        dist = np.linalg.norm(coords - coords[last], axis=1)
        min_dist = np.minimum(min_dist, dist.astype(np.float32))
        min_dist[selected] = -1.0
        dist_norm = min_dist / max(float(min_dist.max()), 1e-6)
        combined = (1.0 - weight) * ordered_scores + weight * dist_norm
        combined[selected] = -1.0
        selected.append(int(np.argmax(combined)))
    return ordered[selected]


def predict_from_prior(init_mask: np.ndarray, predictor, args):
    if args.disable_point_prompts:
        input_points = np.empty((0, 2), dtype=np.float32)
        input_labels = np.empty((0,), dtype=np.int32)
    else:
        pos_indices = np.argwhere(init_mask > args.pos_thr)
        neg_indices = np.argwhere(init_mask < args.neg_thr)
        pos_indices = sample_prompt_points(
            pos_indices, init_mask, args.n_points, args.point_sample_mode, True, args
        )
        neg_indices = sample_prompt_points(
            neg_indices, init_mask, args.n_points, args.point_sample_mode, False, args
        )
        pos_points = pos_indices[:, ::-1]
        neg_points = neg_indices[:, ::-1]
        if len(pos_points) == 0 and len(neg_points) == 0 and args.disable_mask_prompt:
            return None
        input_points = np.concatenate([pos_points, neg_points], axis=0).astype(np.float32)
        input_labels = np.concatenate(
            [np.ones(len(pos_points)), np.zeros(len(neg_points))]
        ).astype(np.int32)
    prior_binary = (init_mask > args.mask_thr).astype(np.float32)
    box = mask_to_box(prior_binary, args.box_pad_ratio) if args.use_box_prompt else None
    predict_kwargs = {
        "point_coords": input_points,
        "point_labels": input_labels,
        "box": box,
        "multimask_output": args.multimask_voting,
    }
    if not args.disable_mask_prompt:
        mask_input = cv2.resize(prior_binary, (256, 256), interpolation=cv2.INTER_NEAREST)
        predict_kwargs["mask_input"] = mask_input[None, :, :]
    masks, sam_scores, _ = predictor.predict(**predict_kwargs)
    return select_sam_mask(masks, sam_scores, prior_binary, args)


TABLE12_PROMPT_VARIANTS = [
    {"id": "mask_only", "pos_thr": 0.99, "neg_thr": 0.05, "disable_point_prompts": True},
    {"id": "p075_n0005", "pos_thr": 0.75, "neg_thr": 0.005},
    {"id": "p085_n0005", "pos_thr": 0.85, "neg_thr": 0.005},
    {"id": "p095_n0005", "pos_thr": 0.95, "neg_thr": 0.005},
    {
        "id": "p095_n0005_no_mask",
        "pos_thr": 0.95,
        "neg_thr": 0.005,
        "disable_mask_prompt": True,
    },
    {"id": "p095_n001", "pos_thr": 0.95, "neg_thr": 0.01},
    {"id": "p095_n005", "pos_thr": 0.95, "neg_thr": 0.05},
    {"id": "p095_n01", "pos_thr": 0.95, "neg_thr": 0.1},
    {"id": "p099_n0005", "pos_thr": 0.99, "neg_thr": 0.005},
    {"id": "p099_n001", "pos_thr": 0.99, "neg_thr": 0.01},
    {"id": "p099_n005", "pos_thr": 0.99, "neg_thr": 0.05},
    {"id": "p099_n01", "pos_thr": 0.99, "neg_thr": 0.1},
]


def predict_prompt_variants(
    init_mask: np.ndarray,
    predictor,
    args,
    image_rgb: np.ndarray,
    variants=TABLE12_PROMPT_VARIANTS,
):
    """Encode one image in SAM2, then decode all prompt-threshold variants."""
    predictor.set_image(image_rgb)
    outputs = {}
    for variant in variants:
        variant_args = copy.copy(args)
        variant_args.pos_thr = variant.get("pos_thr", args.pos_thr)
        variant_args.neg_thr = variant.get("neg_thr", args.neg_thr)
        variant_args.disable_point_prompts = variant.get("disable_point_prompts", False)
        variant_args.disable_mask_prompt = variant.get("disable_mask_prompt", False)
        np.random.seed(args.seed)
        random.seed(args.seed)
        mask = predict_from_prior(init_mask, predictor, variant_args)
        if mask is None:
            mask = np.zeros(init_mask.shape, dtype=np.float32)
        outputs[variant["id"]] = np.asarray(mask, dtype=np.float32)
    return outputs


def rag_coarse_mask(
    img: Image.Image,
    extractor: DinoV3Extractor,
    retrieval_sources,
    negative_sources,
    correspondence_memory,
    args,
):
    inputs = extractor.preprocess_image(img, args.image_size)
    if extractor.device.type == "cuda":
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            out = extractor.extract(inputs)
    else:
        with torch.inference_mode():
            out = extractor.extract(inputs)
    token_tensor = out.patch_tokens.squeeze(0).detach().float().cpu()
    token_vec = token_tensor.numpy()
    flat_tokens = token_vec.reshape(-1, token_vec.shape[-1]).astype("float32")
    source_results = [retrieve_from_source(flat_tokens, source, args) for source in retrieval_sources]
    mask_vals = fuse_source_retrieval(source_results, args)
    if negative_sources and args.negative_weight > 0:
        negative_results = [retrieve_from_source(flat_tokens, source, args) for source in negative_sources]
        if args.negative_mode == "confidence":
            negative_vals = fuse_negative_confidence(negative_results, args)
        else:
            negative_vals = fuse_source_retrieval(negative_results, args)
        mask_vals = mask_vals - float(args.negative_weight) * negative_vals
    if correspondence_memory is not None and (
        args.correspondence_mode == "only" or args.correspondence_blend > 0
    ):
        mask_vals = dense_correspondence_prior(flat_tokens, mask_vals, correspondence_memory, args)

    grid_h, grid_w = out.grid_size
    expected = grid_h * grid_w
    mask_vals = mask_vals[:expected]
    token_grid = flat_tokens[:expected].reshape(grid_h, grid_w, flat_tokens.shape[-1]).astype(np.float32)
    if args.local_propagation != "off":
        base_prior = normalize01(mask_vals.reshape(grid_h, grid_w))
        confidence = retrieval_confidence(source_results, args)[:expected]
        mask_vals = local_spatial_propagation(
            base_prior,
            token_grid,
            confidence,
            args,
        ).reshape(-1)
    saliency_map = None
    if out.cls_token is not None and out.cls_token.shape[-1] == flat_tokens.shape[-1]:
        cls = out.cls_token.squeeze(0).detach().float().cpu().numpy().astype(np.float32)
        cls = cls / np.linalg.norm(cls).clip(min=1e-6)
        tokens = flat_tokens[:expected]
        tokens = tokens / np.linalg.norm(tokens, axis=1, keepdims=True).clip(min=1e-6)
        saliency_map = normalize01((tokens @ cls).reshape(grid_h, grid_w))
    return mask_vals.reshape(grid_h, grid_w).astype(np.float32), token_grid, saliency_map


def predict_mask(
    img_path: Path,
    predictor,
    extractor: DinoV3Extractor,
    retrieval_sources,
    negative_sources,
    correspondence_memory,
    args,
    prior_dir: Path | None,
    prior_float32_dir: Path | None,
    sam_prior_float32_dir: Path | None,
):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    img = Image.open(img_path).convert("RGB")
    original_size = img.size
    coarse_prior, token_grid, _ = rag_coarse_mask(
        img, extractor, retrieval_sources, negative_sources, correspondence_memory, args
    )
    coarse_prior = apply_consistency_gate(
        img,
        coarse_prior,
        extractor,
        retrieval_sources,
        negative_sources,
        correspondence_memory,
        args,
    )
    if prior_dir is not None:
        prior_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(prior_dir / f"{img_path.stem}.png"), (normalize01(coarse_prior) * 255).astype(np.uint8))
    if prior_float32_dir is not None:
        prior_float32_dir.mkdir(parents=True, exist_ok=True)
        np.save(prior_float32_dir / f"{img_path.stem}.npy", coarse_prior.astype(np.float32))

    predictor.set_image(np.array(img.resize((args.sam_size, args.sam_size))))
    init_mask = enhance_prior(coarse_prior, token_grid, args)
    if sam_prior_float32_dir is not None:
        sam_prior_float32_dir.mkdir(parents=True, exist_ok=True)
        np.save(sam_prior_float32_dir / f"{img_path.stem}.npy", init_mask.astype(np.float32))
    init_mask = cv2.resize(init_mask, (args.sam_size, args.sam_size), interpolation=cv2.INTER_LINEAR)
    best_mask = predict_from_prior(init_mask, predictor, args)
    if best_mask is None:
        return np.zeros((original_size[1], original_size[0]), dtype=np.uint8)
    pred = (best_mask * 255).astype(np.uint8)
    return cv2.resize(pred, original_size, interpolation=cv2.INTER_LINEAR)


def read_split_names(split_path: Path, gt_ext: str):
    if split_path.suffix.lower() == ".json":
        data = json.loads(split_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "images" in data:
            return [Path(item["file_name"]).with_suffix(gt_ext).stem for item in data["images"]]
        if isinstance(data, list):
            return [Path(item.get("file_name", item)).with_suffix(gt_ext).stem for item in data]
        raise ValueError(f"Unsupported JSON split format: {split_path}")
    names = []
    for line in split_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            names.append(Path(line.split()[0]).with_suffix(gt_ext).stem)
    return names


def collect_names(image_dir: Path, gt_dir: Path, image_ext: str, gt_ext: str, split: str):
    image_stems = {p.stem for p in image_dir.iterdir() if p.suffix.lower() == image_ext.lower()}
    gt_stems = {p.stem for p in gt_dir.iterdir() if p.suffix.lower() == gt_ext.lower()}
    if split:
        split_names = read_split_names(Path(split).expanduser(), gt_ext)
        names = [name for name in split_names if name in image_stems and name in gt_stems]
        images_without_gt = sorted(name for name in split_names if name in image_stems and name not in gt_stems)
        gt_without_images = sorted(name for name in split_names if name in gt_stems and name not in image_stems)
        return names, images_without_gt, gt_without_images
    names = sorted(image_stems & gt_stems, key=lambda x: int(x) if x.isdigit() else x)
    return names, sorted(image_stems - gt_stems), sorted(gt_stems - image_stems)


def load_gray(path: Path):
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(path)
    return arr


def compute_metrics(pred_dir: Path, gt_dir: Path, names, gt_ext: str):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import py_sod_metrics

    sm = py_sod_metrics.Smeasure()
    em = py_sod_metrics.Emeasure()
    wfm = py_sod_metrics.WeightedFmeasure()
    mae = py_sod_metrics.MAE()
    for stem in tqdm(names, desc="metrics", ncols=100):
        pred = load_gray(pred_dir / f"{stem}.png")
        gt = load_gray(gt_dir / f"{stem}{gt_ext}")
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        sm.step(pred=pred, gt=gt)
        em.step(pred=pred, gt=gt)
        wfm.step(pred=pred, gt=gt)
        mae.step(pred=pred, gt=gt)
    em_result = em.get_results()["em"]
    return {
        "S_alpha": float(sm.get_results()["sm"]),
        "maxE": float(em_result["curve"].max()),
        "meanE": float(em_result["curve"].mean()),
        "adpE": float(em_result["adp"]),
        "F_w_beta": float(wfm.get_results()["wfm"]),
        "MAE": float(mae.get_results()["mae"]),
    }


def run_table12_sweep(
    args,
    data_path,
    image_dir,
    gt_dir,
    names,
    images_without_gt,
    gt_without_images,
    result_root,
    extractor,
    predictor,
    retrieval_sources,
    negative_sources,
    correspondence_memory,
    extractor_signature,
    index_metadata,
):
    variants = TABLE12_PROMPT_VARIANTS
    if args.table12_variant_ids:
        available = {item["id"] for item in variants}
        unknown = sorted(set(args.table12_variant_ids) - available)
        if unknown:
            raise ValueError(f"Unknown Table 12 variants: {unknown}")
        requested = set(args.table12_variant_ids)
        variants = [variant for variant in variants if variant["id"] in requested]

    cache_dir = result_root / "prior_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_config = vars(args).copy()
    non_prior_keys = {
        "result_path", "limit", "pos_thr", "neg_thr", "disable_point_prompts",
        "disable_mask_prompt", "n_points", "point_sample_mode", "point_farthest_weight",
        "use_box_prompt", "box_pad_ratio", "multimask_voting", "sam_score_weight",
        "force", "table12_sweep", "save_prior", "save_prior_float32",
        "save_sam_prior_float32", "eval_only",
        "table12_variant_ids",
    }
    cache_config = {key: value for key, value in cache_config.items() if key not in non_prior_keys}
    cache_config.update({"data_path": str(data_path), "image_dir": str(image_dir), "gt_dir": str(gt_dir)})
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("prior_config") != cache_config or existing.get("image_names") != names:
            raise RuntimeError(f"Prior cache does not match this evaluation contract: {cache_dir}")
    else:
        manifest_path.write_text(
            json.dumps({"prior_config": cache_config, "image_names": names}, indent=2),
            encoding="utf-8",
        )

    variant_dirs = {}
    for variant in variants:
        pred_dir = result_root / variant["id"] / args.dataset_name
        pred_dir.mkdir(parents=True, exist_ok=True)
        variant_dirs[variant["id"]] = pred_dir

    total_start = time.time()
    prior_compute_seconds = 0.0
    sam_sweep_seconds = 0.0
    for stem in tqdm(names, desc="shared prior + prompt sweep", ncols=100):
        missing_variants = [
            variant for variant in variants
            if not (variant_dirs[variant["id"]] / f"{stem}.png").exists()
        ]
        if not missing_variants:
            continue

        prior_path = cache_dir / f"{stem}.npy"
        init_mask = None
        if prior_path.exists():
            candidate = np.load(prior_path).astype(np.float32)
            if candidate.ndim == 2 and np.isfinite(candidate).all():
                init_mask = candidate

        img_path = image_dir / f"{stem}{args.image_ext}"
        img = Image.open(img_path).convert("RGB")
        if init_mask is None:
            prior_start = time.time()
            coarse_prior, token_grid, _ = rag_coarse_mask(
                img, extractor, retrieval_sources, negative_sources, correspondence_memory, args
            )
            coarse_prior = apply_consistency_gate(
                img, coarse_prior, extractor, retrieval_sources, negative_sources,
                correspondence_memory, args,
            )
            init_mask = enhance_prior(coarse_prior, token_grid, args).astype(np.float32)
            np.save(prior_path, init_mask)
            prior_compute_seconds += time.time() - prior_start

        init_mask = cv2.resize(init_mask, (args.sam_size, args.sam_size), interpolation=cv2.INTER_LINEAR)
        image_rgb = np.asarray(img.resize((args.sam_size, args.sam_size))).copy()
        sam_start = time.time()
        outputs = predict_prompt_variants(
            init_mask, predictor, args, image_rgb, variants=missing_variants
        )
        sam_sweep_seconds += time.time() - sam_start
        for variant in missing_variants:
            mask = (outputs[variant["id"]] * 255).clip(0, 255).astype(np.uint8)
            pred = cv2.resize(mask, img.size, interpolation=cv2.INTER_LINEAR)
            output_path = variant_dirs[variant["id"]] / f"{stem}.png"
            if not cv2.imwrite(str(output_path), pred):
                raise RuntimeError(f"Could not save prediction: {output_path}")

    shared_seconds = time.time() - total_start
    results = {}
    for variant in variants:
        variant_id = variant["id"]
        pred_dir = variant_dirs[variant_id]
        missing = [stem for stem in names if not (pred_dir / f"{stem}.png").exists()]
        if missing:
            raise RuntimeError(f"{variant_id}: {len(missing)} predictions missing; first={missing[0]}")
        metrics = compute_metrics(pred_dir, gt_dir, names, args.gt_ext)
        variant_config = vars(args).copy()
        variant_config.update(variant)
        variant_config["result_path"] = str(result_root / variant_id)
        result = {
            "dataset": args.dataset_name,
            "data_path": str(data_path),
            "image_dir": str(image_dir),
            "gt_dir": str(gt_dir),
            "split": "",
            "num_images": len(names),
            "images_without_gt": images_without_gt,
            "gt_without_images": gt_without_images,
            "prediction_dir": str(pred_dir),
            "seed": args.seed,
            "infer_seconds": shared_seconds,
            "config": variant_config,
            "extractor": extractor_signature,
            "index_metadata": index_metadata,
            "metrics": metrics,
            "sweep": {
                "shared_prior_cache": str(cache_dir),
                "shared_sam_image_embedding_per_image": True,
                "shared_sweep_seconds": shared_seconds,
                "prior_compute_seconds_this_pass": prior_compute_seconds,
                "sam_prompt_sweep_seconds_this_pass": sam_sweep_seconds,
            },
        }
        (pred_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        results[variant_id] = metrics

    sweep_manifest = {
        "dataset": args.dataset_name,
        "num_images": len(names),
        "variant_ids": [variant["id"] for variant in variants],
        "prior_cache": str(cache_dir),
        "prior_computations_per_image": 1,
        "sam_image_encodings_per_image": 1,
        "prompt_decoder_calls_per_image": len(variants),
        "shared_sweep_seconds": shared_seconds,
        "prior_compute_seconds_this_pass": prior_compute_seconds,
        "sam_prompt_sweep_seconds_this_pass": sam_sweep_seconds,
        "metrics": results,
    }
    (result_root / "sweep_manifest.json").write_text(
        json.dumps(sweep_manifest, indent=2), encoding="utf-8"
    )
    log(
        f"Table 12 sweep complete: images={len(names)}, variants={len(variants)}, "
        f"elapsed={shared_seconds / 60:.1f} min, prior_compute={prior_compute_seconds / 60:.1f} min, "
        f"SAM_sweep={sam_sweep_seconds / 60:.1f} min"
    )


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_path = Path(args.data_path).expanduser().resolve()
    image_dir = resolve_dir(data_path, args.image_dir).resolve()
    gt_dir = resolve_dir(data_path, args.gt_dir).resolve()
    result_root = Path(args.result_path).expanduser()
    result_root = result_root if result_root.is_absolute() else repo_root / result_root
    pred_dir = result_root / args.dataset_name
    pred_dir.mkdir(parents=True, exist_ok=True)
    prior_dir = result_root / f"{args.dataset_name}_prior" if args.save_prior else None
    prior_float32_dir = result_root / f"{args.dataset_name}_prior_float32" if args.save_prior_float32 else None
    sam_prior_float32_dir = (
        result_root / f"{args.dataset_name}_sam_prior_float32" if args.save_sam_prior_float32 else None
    )

    split = str(resolve_dir(data_path, args.split).resolve()) if args.split else ""
    names, images_without_gt, gt_without_images = collect_names(
        image_dir=image_dir,
        gt_dir=gt_dir,
        image_ext=args.image_ext,
        gt_ext=args.gt_ext,
        split=split,
    )
    if args.limit:
        names = names[: args.limit]
    if not names:
        raise RuntimeError("No image/GT pairs found for evaluation.")

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    log(f"Dataset: {args.dataset_name}")
    log(f"Images: {len(names)}")
    log(f"Image dir: {image_dir}")
    log(f"GT dir: {gt_dir}")
    log(f"Prediction dir: {pred_dir}")
    log(f"Device: {device}")

    infer_seconds = 0.0
    index_metadata = {}
    extractor_signature = None
    if not args.eval_only:
        log("Loading DINOv3 extractor...")
        extractor = DinoV3Extractor(
            repo_root=repo_root,
            dinov3_repo=args.dinov3_repo,
            weights=args.dinov3_weights,
            model_name=args.backbone,
            layers=args.layers,
            fusion=args.fusion,
            layer_weights=args.layer_weights,
            zero_based_layers=args.zero_based_layers,
            l2_normalize=args.l2_normalize_features,
            debias_svd_components=args.debias_svd_components,
            device=device,
        )
        extractor_signature = extractor.signature()
        log(f"Extractor: {extractor_signature}")
        log("Loading SAM2...")
        predictor = load_sam2_predictor(repo_root, device)
        log("Loading FAISS index sources...")
        retrieval_sources = load_retrieval_sources(repo_root, args)
        negative_sources = load_negative_sources(repo_root, args)
        correspondence_memory = load_correspondence_memory(repo_root, args)
        index_metadata = []
        all_sources = retrieval_sources + negative_sources
        for source in all_sources:
            index = source["index"]
            scores = source["scores"]
            if index.d != extractor.feature_dim:
                raise RuntimeError(
                    f"Index dim {index.d} does not match extractor dim {extractor.feature_dim} "
                    f"for {source['index_path']}. Rebuild the DINOv3 index with the same layers/fusion."
                )
            if index.ntotal != scores.shape[0]:
                raise RuntimeError(
                    f"Score length {scores.shape[0]} does not match index size {index.ntotal} "
                    f"for {source['score_path']}."
                )
            log(
                "FAISS source loaded: "
                f"{source['index_path']} ({index.ntotal} vectors, {scores.shape[0]} scores, dim={index.d})"
            )
            index_metadata.append(
                {
                    "source_id": source["source_id"],
                    "index_path": source["index_path"],
                    "score_path": source["score_path"],
                    "ntotal": int(index.ntotal),
                    "dim": int(index.d),
                    "metadata": source["metadata"],
                }
            )
        if correspondence_memory is not None:
            log(
                "Correspondence memory loaded: "
                f"{correspondence_memory['num_images']} images, "
                f"{correspondence_memory['tokens_per_image']} tokens/image, "
                f"dim={correspondence_memory['feature_dim']}"
            )
            index_metadata.append(
                {
                    "type": "dense_correspondence_memory",
                    "feature_path": correspondence_memory["feature_path"],
                    "score_path": correspondence_memory["score_path"],
                    "num_images": int(correspondence_memory["num_images"]),
                    "tokens_per_image": int(correspondence_memory["tokens_per_image"]),
                    "feature_dim": int(correspondence_memory["feature_dim"]),
                }
            )
        if args.table12_sweep:
            run_table12_sweep(
                args=args,
                data_path=data_path,
                image_dir=image_dir,
                gt_dir=gt_dir,
                names=names,
                images_without_gt=images_without_gt,
                gt_without_images=gt_without_images,
                result_root=result_root,
                extractor=extractor,
                predictor=predictor,
                retrieval_sources=retrieval_sources,
                negative_sources=negative_sources,
                correspondence_memory=correspondence_memory,
                extractor_signature=extractor_signature,
                index_metadata=index_metadata,
            )
            return
        log("Start inference...")
        start = time.time()
        for stem in tqdm(names, desc="infer", ncols=100):
            pred_path = pred_dir / f"{stem}.png"
            if pred_path.exists() and not args.force:
                continue
            img_path = image_dir / f"{stem}{args.image_ext}"
            pred = predict_mask(
                img_path=img_path,
                predictor=predictor,
                extractor=extractor,
                retrieval_sources=retrieval_sources,
                negative_sources=negative_sources,
                correspondence_memory=correspondence_memory,
                args=args,
                prior_dir=prior_dir,
                prior_float32_dir=prior_float32_dir,
                sam_prior_float32_dir=sam_prior_float32_dir,
            )
            cv2.imwrite(str(pred_path), pred)
        infer_seconds = time.time() - start

    missing = [stem for stem in names if not (pred_dir / f"{stem}.png").exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} predictions are missing, first: {missing[0]}")

    log("Computing metrics...")
    metrics = compute_metrics(pred_dir=pred_dir, gt_dir=gt_dir, names=names, gt_ext=args.gt_ext)
    result = {
        "dataset": args.dataset_name,
        "data_path": str(data_path),
        "image_dir": str(image_dir),
        "gt_dir": str(gt_dir),
        "split": split,
        "num_images": len(names),
        "images_without_gt": images_without_gt,
        "gt_without_images": gt_without_images,
        "prediction_dir": str(pred_dir),
        "seed": args.seed,
        "infer_seconds": infer_seconds,
        "config": vars(args),
        "extractor": extractor_signature,
        "index_metadata": index_metadata,
        "metrics": metrics,
    }
    (pred_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"{args.dataset_name} results:")
    log(
        "S_alpha:{S_alpha:.4f}, meanE:{meanE:.4f}, maxE:{maxE:.4f}, "
        "F_w_beta:{F_w_beta:.4f}, MAE:{MAE:.5f}".format(**metrics)
    )


if __name__ == "__main__":
    main()
