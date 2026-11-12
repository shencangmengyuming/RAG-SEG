import argparse
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


def log(message: str):
    print(message, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RAG-SEG on image/mask datasets.")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--result_path", type=str, default="outputs/eval")
    parser.add_argument("--split", type=str, default="")
    parser.add_argument("--image_ext", type=str, default=".jpg")
    parser.add_argument("--gt_ext", type=str, default=".png")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=784)
    parser.add_argument("--sam_size", type=int, default=1024)
    parser.add_argument(
        "--dinov2_checkpoint",
        type=str,
        default="dinov2/checkpoints/dinov2_vits14_pretrain.pth",
    )
    parser.add_argument("--pos_thr", type=float, default=0.99)
    parser.add_argument("--neg_thr", type=float, default=0.05)
    parser.add_argument("--mask_thr", type=float, default=0.3)
    parser.add_argument("--n_points", type=int, default=10)
    parser.add_argument("--index_path", type=str, default="sod_cod.index")
    parser.add_argument("--score_path", type=str, default="sod_cod_score.index.npz")
    parser.add_argument("--retrieval_topk", type=int, default=1)
    parser.add_argument(
        "--rerank_mode",
        type=str,
        default="top1",
        choices=["top1", "mean", "max", "softmax", "gated_max"],
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
        choices=["none", "pc", "sp", "pc_sp", "fcpc", "fcpc_sp"],
        help="Training-free prior enhancement: part composition, structural propagation, or both.",
    )
    parser.add_argument("--pc_seed_thr", type=float, default=0.5)
    parser.add_argument("--pc_expand_iter", type=int, default=2)
    parser.add_argument("--pc_close_kernel", type=int, default=5)
    parser.add_argument("--sp_iters", type=int, default=1)
    parser.add_argument("--sp_alpha", type=float, default=0.45)
    parser.add_argument("--sp_anchor", type=float, default=0.65)
    parser.add_argument("--fcpc_seed_thr", type=float, default=0.7)
    parser.add_argument("--fcpc_low_thr", type=float, default=0.2)
    parser.add_argument("--fcpc_sim_thr", type=float, default=0.6)
    parser.add_argument("--fcpc_expand_iter", type=int, default=2)
    parser.add_argument("--fcpc_mix", type=float, default=0.6)
    parser.add_argument("--use_box_prompt", action="store_true")
    parser.add_argument("--box_pad_ratio", type=float, default=0.04)
    parser.add_argument("--multimask_voting", action="store_true")
    parser.add_argument("--sam_score_weight", type=float, default=0.15)
    parser.add_argument("--selector_v2", action="store_true")
    parser.add_argument("--selector_area_weight", type=float, default=1.0)
    parser.add_argument("--selector_frag_weight", type=float, default=0.01)
    parser.add_argument("--selector_lcr_weight", type=float, default=0.5)
    parser.add_argument(
        "--selector_abs_frag_weight",
        type=float,
        default=0.0,
        help="Absolute fragmentation penalty for Selector v2 Mask Critic.",
    )
    parser.add_argument(
        "--selector_abs_lcr_weight",
        type=float,
        default=0.0,
        help="Absolute low-largest-component-ratio penalty for Selector v2 Mask Critic.",
    )
    parser.add_argument(
        "--selector_sp_margin",
        type=float,
        default=0.0,
        help="Selector v2 chooses the original RAG prior only if it beats SP by this margin.",
    )
    parser.add_argument(
        "--selector_isf_weight",
        type=float,
        default=0.0,
        help="Weight for TF-SSD-inspired DINO saliency contrast in Selector v2.",
    )
    parser.add_argument(
        "--selector_attn_isf_weight",
        type=float,
        default=0.0,
        help="Weight for true DINO last-layer CLS attention saliency contrast in Selector v2.",
    )
    parser.add_argument(
        "--selector_proto_weight",
        type=float,
        default=0.0,
        help="Weight for IPS-inspired DINO mask-prototype similarity in Selector v2.",
    )
    parser.add_argument(
        "--selector_proto_seed_thr",
        type=float,
        default=0.7,
        help="RAG prior threshold used to build the selector seed prototype.",
    )
    parser.add_argument(
        "--selector_trace",
        action="store_true",
        help="Write per-image selector scores and selected prior modes for diagnosis.",
    )
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Regenerate predictions even if files exist.")
    return parser.parse_args()


def prepare_dinov2(repo_root: Path, checkpoint: str, device: torch.device):
    dinov2_root = repo_root / "dinov2"
    if str(dinov2_root) not in sys.path:
        sys.path.insert(0, str(dinov2_root))
    from dinov2.hub.backbones import dinov2_vits14

    ckpt_path = Path(checkpoint).expanduser()
    if not ckpt_path.is_absolute():
        ckpt_path = repo_root / ckpt_path
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"DINOv2 checkpoint not found: {ckpt_path}")

    model = dinov2_vits14(pretrained=True, weights=str(ckpt_path)).to(device).eval()
    return model


def preprocess_dinov2_image(img: Image.Image, image_size: int, device: torch.device):
    arr = np.asarray(img.resize((image_size, image_size)), dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def load_index(repo_root: Path, index_path: str, score_path: str):
    index_path = Path(index_path).expanduser()
    score_path = Path(score_path).expanduser()
    if not index_path.is_absolute():
        index_path = repo_root / index_path
    if not score_path.is_absolute():
        score_path = repo_root / score_path
    index = faiss.read_index(str(index_path))
    scores = np.load(score_path)["scores"].astype("float32")
    return index, scores


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


def forward_dinov2_with_last_attention(model, inputs, device: torch.device):
    x = model.prepare_tokens_with_masks(inputs, None)
    last_attn_map = None

    for i, blk in enumerate(model.blocks):
        if i == len(model.blocks) - 1:
            normed = blk.norm1(x)
            qkv = blk.attn.qkv(normed)
            b, n, c3 = qkv.shape
            c = c3 // 3
            qkv = qkv.reshape(b, n, 3, blk.attn.num_heads, c // blk.attn.num_heads)
            q, k, _ = torch.unbind(qkv, 2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            attn = (q @ k.transpose(-2, -1)) * blk.attn.scale
            attn = attn.softmax(dim=-1)
            patch_start = 1 + getattr(model, "num_register_tokens", 0)
            cls_to_patch = attn[:, :, 0, patch_start:].mean(dim=1)
            last_attn_map = cls_to_patch.squeeze(0).detach().float().cpu().numpy()
        x = blk(x)

    x_norm = model.norm(x)
    return {
        "x_norm_clstoken": x_norm[:, 0],
        "x_norm_regtokens": x_norm[:, 1 : model.num_register_tokens + 1],
        "x_norm_patchtokens": x_norm[:, model.num_register_tokens + 1 :],
        "x_prenorm": x,
        "masks": None,
        "last_attn_map": last_attn_map,
    }


def fuse_retrieval_scores(distances: np.ndarray, idxs: np.ndarray, scores: np.ndarray, args):
    valid = distances >= 0
    retrieved_scores = np.zeros_like(distances, dtype=np.float32)
    retrieved_scores[valid] = scores[idxs[valid]]

    if args.rerank_mode == "top1" or distances.shape[1] == 1:
        return retrieved_scores[:, 0]
    if args.rerank_mode == "mean":
        counts = valid.sum(axis=1).clip(min=1)
        return retrieved_scores.sum(axis=1) / counts
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


def normalize01(x: np.ndarray):
    x = x.astype(np.float32)
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max - x_min < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


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


def structural_propagation_prior(prior: np.ndarray, token_grid: np.ndarray, args):
    if token_grid is None or args.sp_iters <= 0:
        return prior

    feats = token_grid.astype(np.float32)
    norms = np.linalg.norm(feats, axis=-1, keepdims=True).clip(min=1e-6)
    feats = feats / norms
    current = prior.astype(np.float32)
    anchor = current.copy()

    h, w = current.shape
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for _ in range(max(0, int(args.sp_iters))):
        accum = current.copy()
        weight_sum = np.ones_like(current, dtype=np.float32)
        for dy, dx in offsets:
            src_y0 = max(0, -dy)
            src_y1 = h - max(0, dy)
            src_x0 = max(0, -dx)
            src_x1 = w - max(0, dx)
            dst_y0 = max(0, dy)
            dst_y1 = h - max(0, -dy)
            dst_x0 = max(0, dx)
            dst_x1 = w - max(0, -dx)

            src_feat = feats[src_y0:src_y1, src_x0:src_x1]
            dst_feat = feats[dst_y0:dst_y1, dst_x0:dst_x1]
            affinity = ((src_feat * dst_feat).sum(axis=-1) + 1.0) * 0.5
            affinity = affinity.astype(np.float32)
            accum[dst_y0:dst_y1, dst_x0:dst_x1] += (
                affinity * current[src_y0:src_y1, src_x0:src_x1]
            )
            weight_sum[dst_y0:dst_y1, dst_x0:dst_x1] += affinity

        diffused = accum / weight_sum.clip(min=1e-6)
        current = (1.0 - args.sp_alpha) * current + args.sp_alpha * diffused
        current = np.maximum(current, args.sp_anchor * anchor)

    return np.clip(normalize01(current), 0.0, 1.0).astype(np.float32)


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
    reachable = reachable.astype(bool)

    candidates = reachable & (prior >= args.fcpc_low_thr) & (sim >= args.fcpc_sim_thr)
    enhanced = prior.copy()
    mixed = (1.0 - args.fcpc_mix) * prior + args.fcpc_mix * sim.astype(np.float32)
    enhanced[candidates] = np.maximum(enhanced[candidates], mixed[candidates])
    return np.clip(enhanced, 0.0, 1.0).astype(np.float32)


def enhance_prior(prior: np.ndarray, token_grid: np.ndarray, args, mode: str | None = None):
    prior_mode = args.prior_mode if mode is None else mode
    enhanced = normalize01(prior)
    if prior_mode in {"pc", "pc_sp"}:
        enhanced = part_composition_prior(enhanced, args)
    if prior_mode in {"fcpc", "fcpc_sp"}:
        enhanced = feature_constrained_part_composition_prior(enhanced, token_grid, args)
    if prior_mode in {"sp", "pc_sp", "fcpc_sp"}:
        enhanced = structural_propagation_prior(enhanced, token_grid, args)
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


def select_sam_mask(masks: np.ndarray, sam_scores: np.ndarray, prior_binary: np.ndarray, args):
    if len(masks) == 1 or not args.multimask_voting:
        sam_score = float(sam_scores[0]) if len(sam_scores) else 0.0
        return masks[0], mask_iou(masks[0], prior_binary), sam_score

    votes = []
    for i, mask in enumerate(masks):
        iou = mask_iou(mask, prior_binary)
        sam_score = float(sam_scores[i]) if i < len(sam_scores) else 0.0
        votes.append(iou + args.sam_score_weight * sam_score)
    best = int(np.argmax(votes))
    sam_score = float(sam_scores[best]) if best < len(sam_scores) else 0.0
    return masks[best], mask_iou(masks[best], prior_binary), sam_score


def mask_structure(mask: np.ndarray):
    binary = mask.astype(bool).astype(np.uint8)
    area = float(binary.mean())
    if binary.sum() == 0:
        return {"area": area, "components": 0.0, "largest_component_ratio": 0.0}
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        return {"area": area, "components": 0.0, "largest_component_ratio": 0.0}
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    return {
        "area": area,
        "components": float(n_labels - 1),
        "largest_component_ratio": float(component_areas.max() / binary.sum()),
    }


def mask_saliency_contrast(mask: np.ndarray, saliency_map: np.ndarray | None):
    if saliency_map is None:
        return 0.0
    binary = mask.astype(bool)
    if binary.sum() == 0 or binary.sum() == binary.size:
        return 0.0
    saliency = cv2.resize(
        saliency_map.astype(np.float32),
        (mask.shape[1], mask.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    inside = float(saliency[binary].mean())
    outside = float(saliency[~binary].mean())
    return inside - outside


def build_seed_prototype(
    coarse_prior: np.ndarray,
    token_grid: np.ndarray,
    seed_thr: float,
):
    if token_grid is None:
        return None
    prior = cv2.resize(
        normalize01(coarse_prior),
        (token_grid.shape[1], token_grid.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    seed = prior >= seed_thr
    if seed.sum() == 0:
        seed = prior >= float(prior.mean())
    if seed.sum() == 0:
        return None
    feats = token_grid.astype(np.float32)
    proto = feats[seed].mean(axis=0)
    norm = np.linalg.norm(proto)
    if norm < 1e-6:
        return None
    return (proto / norm).astype(np.float32)


def mask_prototype_similarity(
    mask: np.ndarray,
    token_grid: np.ndarray | None,
    seed_proto: np.ndarray | None,
):
    if token_grid is None or seed_proto is None:
        return 0.0
    mask_small = cv2.resize(
        mask.astype(np.float32),
        (token_grid.shape[1], token_grid.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    weights = np.clip(mask_small, 0.0, 1.0)
    if float(weights.sum()) < 1e-6:
        return 0.0
    feats = token_grid.astype(np.float32)
    proto = (feats * weights[..., None]).sum(axis=(0, 1)) / weights.sum()
    norm = np.linalg.norm(proto)
    if norm < 1e-6:
        return 0.0
    proto = proto / norm
    return float((proto * seed_proto).sum())


def predict_from_prior(
    init_mask: np.ndarray,
    predictor,
    args,
    saliency_map: np.ndarray | None = None,
    attn_map: np.ndarray | None = None,
    token_grid: np.ndarray | None = None,
    seed_proto: np.ndarray | None = None,
):
    pos_indices = np.argwhere(init_mask > args.pos_thr)
    neg_indices = np.argwhere(init_mask < args.neg_thr)
    pos_points = pos_indices[:, ::-1]
    neg_points = neg_indices[:, ::-1]
    if len(pos_points) > args.n_points:
        pos_points = pos_points[np.random.choice(len(pos_points), args.n_points, replace=False)]
    if len(neg_points) > args.n_points:
        neg_points = neg_points[np.random.choice(len(neg_points), args.n_points, replace=False)]

    if len(pos_points) == 0 and len(neg_points) == 0:
        return None

    input_points = np.concatenate([pos_points, neg_points], axis=0).astype(np.float32)
    input_labels = np.concatenate(
        [np.ones(len(pos_points)), np.zeros(len(neg_points))]
    ).astype(np.int32)

    prior_binary = (init_mask > args.mask_thr).astype(np.float32)
    box = mask_to_box(prior_binary, args.box_pad_ratio) if args.use_box_prompt else None
    mask_input = cv2.resize(prior_binary, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask_input = mask_input[None, :, :]
    masks, sam_scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        box=box,
        mask_input=mask_input,
        multimask_output=args.multimask_voting,
    )

    best_mask, prior_iou, sam_score = select_sam_mask(masks, sam_scores, prior_binary, args)
    return {
        "mask": best_mask,
        "prior_iou": prior_iou,
        "sam_score": sam_score,
        "structure": mask_structure(best_mask),
        "isf_score": mask_saliency_contrast(best_mask, saliency_map),
        "attn_isf_score": mask_saliency_contrast(best_mask, attn_map),
        "proto_score": mask_prototype_similarity(best_mask, token_grid, seed_proto),
    }


def selector_v2_score(candidate, reference, args):
    score = candidate["prior_iou"] + args.sam_score_weight * candidate["sam_score"]
    score += args.selector_isf_weight * candidate["isf_score"]
    score += args.selector_attn_isf_weight * candidate["attn_isf_score"]
    score += args.selector_proto_weight * candidate["proto_score"]
    cand = candidate["structure"]
    abs_frag_penalty = np.log1p(cand["components"]) / np.log(1001.0)
    abs_lcr_penalty = 1.0 - cand["largest_component_ratio"]
    score -= args.selector_abs_frag_weight * abs_frag_penalty
    score -= args.selector_abs_lcr_weight * abs_lcr_penalty
    if reference is None:
        return score
    ref = reference["structure"]
    area_penalty = max(0.0, cand["area"] - ref["area"])
    frag_penalty = max(0.0, cand["components"] - ref["components"]) / 100.0
    lcr_penalty = max(0.0, ref["largest_component_ratio"] - cand["largest_component_ratio"])
    return (
        score
        - args.selector_area_weight * area_penalty
        - args.selector_frag_weight * frag_penalty
        - args.selector_lcr_weight * lcr_penalty
    )


def predict_with_selector_v2(
    coarse_prior: np.ndarray,
    token_grid: np.ndarray,
    saliency_map: np.ndarray | None,
    attn_map: np.ndarray | None,
    predictor,
    args,
):
    candidates = []
    seed_proto = build_seed_prototype(coarse_prior, token_grid, args.selector_proto_seed_thr)
    for mode in ("none", "sp"):
        np.random.seed(args.seed)
        init_mask = enhance_prior(coarse_prior, token_grid, args, mode=mode)
        init_mask = cv2.resize(
            init_mask, (args.sam_size, args.sam_size), interpolation=cv2.INTER_LINEAR
        )
        candidate = predict_from_prior(
            init_mask,
            predictor,
            args,
            saliency_map=saliency_map,
            attn_map=attn_map,
            token_grid=token_grid,
            seed_proto=seed_proto,
        )
        if candidate is not None:
            candidate["mode"] = mode
            candidates.append(candidate)

    if not candidates:
        return None
    reference = next((item for item in candidates if item["mode"] == "none"), None)
    for candidate in candidates:
        candidate["selector_score"] = selector_v2_score(candidate, reference, args)

    by_mode = {item["mode"]: item for item in candidates}
    if "none" in by_mode and "sp" in by_mode:
        selected = by_mode["sp"]
        if by_mode["none"]["selector_score"] > by_mode["sp"]["selector_score"] + args.selector_sp_margin:
            selected = by_mode["none"]
    else:
        selected = max(candidates, key=lambda item: item["selector_score"])
    return selected["mask"], selected["mode"], candidates


def rag_coarse_mask(img: Image.Image, model, index, scores, device: torch.device, args):
    image_size = args.image_size
    inputs = preprocess_dinov2_image(img, image_size, device)
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                if args.selector_attn_isf_weight:
                    out = forward_dinov2_with_last_attention(model, inputs, device)
                else:
                    out = model.forward_features(inputs)
        else:
            if args.selector_attn_isf_weight:
                out = forward_dinov2_with_last_attention(model, inputs, device)
            else:
                out = model.forward_features(inputs)
    token_tensor = out["x_norm_patchtokens"].squeeze(0).detach().float().cpu()
    token_vec = token_tensor.numpy()
    flat_tokens = token_vec.reshape(-1, token_vec.shape[-1]).astype("float32")
    retrieval_topk = max(1, args.retrieval_topk)
    distances, idxs = index.search(flat_tokens, retrieval_topk)
    mask_vals = fuse_retrieval_scores(distances, idxs, scores, args)

    side = int(np.sqrt(flat_tokens.shape[0]))
    mask_vals = mask_vals[: side * side]
    token_grid = flat_tokens[: side * side].reshape(side, side, flat_tokens.shape[-1]).astype(np.float32)
    cls_token = out.get("x_norm_clstoken")
    saliency_map = None
    if cls_token is not None:
        cls = cls_token.squeeze(0).detach().float().cpu().numpy().astype(np.float32)
        cls = cls / np.linalg.norm(cls).clip(min=1e-6)
        tokens = flat_tokens[: side * side]
        tokens = tokens / np.linalg.norm(tokens, axis=1, keepdims=True).clip(min=1e-6)
        saliency_map = normalize01((tokens @ cls).reshape(side, side))
    attn_map = out.get("last_attn_map")
    if attn_map is not None:
        attn_map = normalize01(attn_map[: side * side].reshape(side, side))
    return mask_vals.reshape(side, side).astype(np.float32), token_grid, saliency_map, attn_map


def predict_mask(
    img_path: Path,
    predictor,
    model,
    index,
    scores,
    device: torch.device,
    args,
):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    img = Image.open(img_path).convert("RGB")
    original_size = img.size
    coarse_prior, token_grid, saliency_map, attn_map = rag_coarse_mask(
        img, model, index, scores, device, args
    )
    predictor.set_image(np.array(img.resize((args.sam_size, args.sam_size))))

    selector_info = None
    if args.selector_v2:
        selector_result = predict_with_selector_v2(
            coarse_prior, token_grid, saliency_map, attn_map, predictor, args
        )
        if selector_result is None:
            best_mask = None
        else:
            best_mask, selected_mode, candidates = selector_result
            selector_info = {
                "selected_mode": selected_mode,
                "candidates": [
                    {
                        "mode": item["mode"],
                        "selector_score": float(item["selector_score"]),
                        "prior_iou": float(item["prior_iou"]),
                        "sam_score": float(item["sam_score"]),
                        "isf_score": float(item["isf_score"]),
                        "attn_isf_score": float(item["attn_isf_score"]),
                        "proto_score": float(item["proto_score"]),
                        "structure": item["structure"],
                    }
                    for item in candidates
                ],
            }
    else:
        init_mask = enhance_prior(coarse_prior, token_grid, args)
        init_mask = cv2.resize(
            init_mask, (args.sam_size, args.sam_size), interpolation=cv2.INTER_LINEAR
        )
        result = predict_from_prior(
            init_mask,
            predictor,
            args,
            saliency_map=saliency_map,
            attn_map=attn_map,
            token_grid=token_grid,
        )
        best_mask = None if result is None else result["mask"]
    if best_mask is None:
        pred = np.zeros((original_size[1], original_size[0]), dtype=np.uint8)
        return pred, selector_info

    pred = (best_mask * 255).astype(np.uint8)
    pred = cv2.resize(pred, original_size, interpolation=cv2.INTER_LINEAR)
    return pred, selector_info


def resolve_dir(base: Path, maybe_relative: str):
    path = Path(maybe_relative).expanduser()
    return path if path.is_absolute() else base / path


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
        if not line.strip():
            continue
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
    repo_root = Path(__file__).resolve().parent
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


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    data_path = Path(args.data_path).expanduser().resolve()
    image_dir = resolve_dir(data_path, args.image_dir).resolve()
    gt_dir = resolve_dir(data_path, args.gt_dir).resolve()
    result_root = Path(args.result_path).expanduser()
    result_root = result_root if result_root.is_absolute() else repo_root / result_root
    pred_dir = result_root / args.dataset_name
    pred_dir.mkdir(parents=True, exist_ok=True)

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
    if not args.eval_only:
        log("Loading local DINOv2...")
        model = prepare_dinov2(repo_root, args.dinov2_checkpoint, device)
        log("Loading SAM2...")
        predictor = load_sam2_predictor(repo_root, device)
        log("Loading FAISS index...")
        index, scores = load_index(repo_root, args.index_path, args.score_path)
        log(f"FAISS index loaded: {index.ntotal} vectors, {scores.shape[0]} scores")
        log("Start inference...")

        start = time.time()
        trace_path = result_root / f"{args.dataset_name}_selector_trace.jsonl"
        if args.selector_trace and trace_path.exists() and args.force:
            trace_path.unlink()
        for stem in tqdm(names, desc="infer", ncols=100):
            pred_path = pred_dir / f"{stem}.png"
            if pred_path.exists() and not args.force:
                continue
            img_path = image_dir / f"{stem}{args.image_ext}"
            pred, selector_info = predict_mask(
                img_path=img_path,
                predictor=predictor,
                model=model,
                index=index,
                scores=scores,
                device=device,
                args=args,
            )
            cv2.imwrite(str(pred_path), pred)
            if args.selector_trace and selector_info is not None:
                selector_info["stem"] = stem
                with trace_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(selector_info) + "\n")
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
        "metrics": metrics,
    }
    (pred_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    log(f"{args.dataset_name} results:")
    log(
        "S_alpha:{S_alpha:.4f}, meanE:{meanE:.4f}, maxE:{maxE:.4f}, "
        "F_w_beta:{F_w_beta:.4f}, MAE:{MAE:.5f}".format(**metrics)
    )
    log(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
