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
    parser.add_argument("--pos_thr", type=float, default=0.99)
    parser.add_argument("--neg_thr", type=float, default=0.05)
    parser.add_argument("--mask_thr", type=float, default=0.3)
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
        choices=["none", "pc", "fcpc"],
    )
    parser.add_argument("--pc_seed_thr", type=float, default=0.5)
    parser.add_argument("--pc_expand_iter", type=int, default=2)
    parser.add_argument("--pc_close_kernel", type=int, default=5)
    parser.add_argument("--fcpc_seed_thr", type=float, default=0.7)
    parser.add_argument("--fcpc_low_thr", type=float, default=0.2)
    parser.add_argument("--fcpc_sim_thr", type=float, default=0.6)
    parser.add_argument("--fcpc_expand_iter", type=int, default=2)
    parser.add_argument("--fcpc_mix", type=float, default=0.6)
    parser.add_argument("--use_box_prompt", action="store_true")
    parser.add_argument("--box_pad_ratio", type=float, default=0.04)
    parser.add_argument("--multimask_voting", action="store_true")
    parser.add_argument("--sam_score_weight", type=float, default=0.15)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--save_prior", action="store_true")
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


def enhance_prior(prior: np.ndarray, token_grid: np.ndarray, args):
    enhanced = normalize01(prior)
    if args.prior_mode == "pc":
        enhanced = part_composition_prior(enhanced, args)
    if args.prior_mode == "fcpc":
        enhanced = feature_constrained_part_composition_prior(enhanced, token_grid, args)
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
    return select_sam_mask(masks, sam_scores, prior_binary, args)


def rag_coarse_mask(img: Image.Image, extractor: DinoV3Extractor, index, scores, args):
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
    distances, idxs = index.search(flat_tokens, max(1, args.retrieval_topk))
    mask_vals = fuse_retrieval_scores(distances, idxs, scores, args)

    grid_h, grid_w = out.grid_size
    expected = grid_h * grid_w
    mask_vals = mask_vals[:expected]
    token_grid = flat_tokens[:expected].reshape(grid_h, grid_w, flat_tokens.shape[-1]).astype(np.float32)
    saliency_map = None
    if out.cls_token is not None and out.cls_token.shape[-1] == flat_tokens.shape[-1]:
        cls = out.cls_token.squeeze(0).detach().float().cpu().numpy().astype(np.float32)
        cls = cls / np.linalg.norm(cls).clip(min=1e-6)
        tokens = flat_tokens[:expected]
        tokens = tokens / np.linalg.norm(tokens, axis=1, keepdims=True).clip(min=1e-6)
        saliency_map = normalize01((tokens @ cls).reshape(grid_h, grid_w))
    return mask_vals.reshape(grid_h, grid_w).astype(np.float32), token_grid, saliency_map


def predict_mask(img_path: Path, predictor, extractor: DinoV3Extractor, index, scores, args, prior_dir: Path | None):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    img = Image.open(img_path).convert("RGB")
    original_size = img.size
    coarse_prior, token_grid, _ = rag_coarse_mask(img, extractor, index, scores, args)
    if prior_dir is not None:
        prior_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(prior_dir / f"{img_path.stem}.png"), (normalize01(coarse_prior) * 255).astype(np.uint8))

    predictor.set_image(np.array(img.resize((args.sam_size, args.sam_size))))
    init_mask = enhance_prior(coarse_prior, token_grid, args)
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
            device=device,
        )
        extractor_signature = extractor.signature()
        log(f"Extractor: {extractor_signature}")
        log("Loading SAM2...")
        predictor = load_sam2_predictor(repo_root, device)
        log("Loading FAISS index...")
        index, scores, index_metadata = load_index(repo_root, args.index_path, args.score_path)
        if index.d != extractor.feature_dim:
            raise RuntimeError(
                f"Index dim {index.d} does not match extractor dim {extractor.feature_dim}. "
                "Rebuild the DINOv3 index with the same layers/fusion."
            )
        log(f"FAISS index loaded: {index.ntotal} vectors, {scores.shape[0]} scores, dim={index.d}")
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
                index=index,
                scores=scores,
                args=args,
                prior_dir=prior_dir,
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
