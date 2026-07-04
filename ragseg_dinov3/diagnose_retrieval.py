import argparse
import json
import sys
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
from ragseg_dinov3.evaluation_dinov3 import collect_names, load_index, normalize01, resolve_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose DINOv3 RAG retrieval priors without SAM.")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--image_ext", type=str, default=".jpg")
    parser.add_argument("--gt_ext", type=str, default=".png")
    parser.add_argument("--split", type=str, default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=896)
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
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--score_path", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    return parser.parse_args()


def component_boundaries(metadata, ntotal: int):
    sizes = metadata.get("component_sizes")
    if sizes is None:
        return [0, ntotal]
    if isinstance(sizes, (int, float)):
        sizes = [int(sizes)]
    sizes = [int(item) for item in sizes]
    bounds = [0]
    for size in sizes:
        bounds.append(bounds[-1] + size)
    if bounds[-1] != ntotal:
        return [0, ntotal]
    return bounds


def component_ids(idxs: np.ndarray, bounds: list[int]):
    comp = np.zeros_like(idxs, dtype=np.int32)
    for cid, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        comp[(idxs >= lo) & (idxs < hi)] = cid
    return comp


def mae(pred: np.ndarray, gt: np.ndarray):
    return float(np.mean(np.abs(pred.astype(np.float32) - gt.astype(np.float32))))


def binary_iou(pred: np.ndarray, gt: np.ndarray, thr: float):
    p = pred >= thr
    g = gt >= 0.5
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, g).sum() / union)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_path = Path(args.data_path).expanduser().resolve()
    image_dir = resolve_dir(data_path, args.image_dir).resolve()
    gt_dir = resolve_dir(data_path, args.gt_dir).resolve()
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
        raise RuntimeError("No image/GT pairs found for diagnosis.")

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
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
    index, scores, metadata = load_index(repo_root, args.index_path, args.score_path)
    if index.d != extractor.feature_dim:
        raise RuntimeError(f"Index dim {index.d} != extractor dim {extractor.feature_dim}")

    bounds = component_boundaries(metadata, index.ntotal)
    per_image = []
    all_scores = []
    all_distances = []
    all_components = []
    for stem in tqdm(names, desc="diagnose", ncols=100):
        img = Image.open(image_dir / f"{stem}{args.image_ext}").convert("RGB")
        inputs = extractor.preprocess_image(img, args.image_size)
        if extractor.device.type == "cuda":
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                out = extractor.extract(inputs)
        else:
            with torch.inference_mode():
                out = extractor.extract(inputs)
        flat = out.patch_tokens.squeeze(0).detach().float().cpu().numpy().astype("float32")
        dists, idxs = index.search(flat, 1)
        idxs = idxs[:, 0]
        dists = dists[:, 0]
        vals = scores[idxs].astype(np.float32)
        comps = component_ids(idxs, bounds)

        grid_h, grid_w = out.grid_size
        prior = vals.reshape(grid_h, grid_w)
        prior01 = normalize01(prior)
        gt = cv2.imread(str(gt_dir / f"{stem}{args.gt_ext}"), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise FileNotFoundError(gt_dir / f"{stem}{args.gt_ext}")
        gt01 = gt.astype(np.float32) / 255.0
        prior_up = cv2.resize(prior01, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        hist = np.bincount(comps, minlength=len(bounds) - 1).astype(np.int64)
        row = {
            "name": stem,
            "tokens": int(vals.shape[0]),
            "score_mean": float(vals.mean()),
            "score_std": float(vals.std()),
            "score_min": float(vals.min()),
            "score_max": float(vals.max()),
            "distance_mean": float(dists.mean()),
            "distance_std": float(dists.std()),
            "prior_mae": mae(prior_up, gt01),
            "prior_iou_030": binary_iou(prior_up, gt01, 0.30),
            "component_counts": hist.tolist(),
            "component_ratios": (hist / max(1, vals.shape[0])).tolist(),
        }
        per_image.append(row)
        all_scores.append(vals)
        all_distances.append(dists.astype(np.float32))
        all_components.append(comps)

    all_scores = np.concatenate(all_scores)
    all_distances = np.concatenate(all_distances)
    all_components = np.concatenate(all_components)
    comp_hist = np.bincount(all_components, minlength=len(bounds) - 1).astype(np.int64)
    summary = {
        "dataset": args.dataset_name,
        "num_images": len(names),
        "images_without_gt": images_without_gt,
        "gt_without_images": gt_without_images,
        "index_path": args.index_path,
        "score_path": args.score_path,
        "index_ntotal": int(index.ntotal),
        "index_dim": int(index.d),
        "component_boundaries": bounds,
        "component_counts": comp_hist.tolist(),
        "component_ratios": (comp_hist / max(1, all_components.shape[0])).tolist(),
        "score": {
            "mean": float(all_scores.mean()),
            "std": float(all_scores.std()),
            "min": float(all_scores.min()),
            "max": float(all_scores.max()),
            "p10": float(np.quantile(all_scores, 0.10)),
            "p50": float(np.quantile(all_scores, 0.50)),
            "p90": float(np.quantile(all_scores, 0.90)),
        },
        "distance": {
            "mean": float(all_distances.mean()),
            "std": float(all_distances.std()),
            "min": float(all_distances.min()),
            "max": float(all_distances.max()),
            "p10": float(np.quantile(all_distances, 0.10)),
            "p50": float(np.quantile(all_distances, 0.50)),
            "p90": float(np.quantile(all_distances, 0.90)),
        },
        "prior": {
            "mae_mean": float(np.mean([row["prior_mae"] for row in per_image])),
            "iou_030_mean": float(np.mean([row["prior_iou_030"] for row in per_image])),
        },
        "extractor": extractor.signature(),
        "per_image": per_image,
    }
    output_json = Path(args.output_json).expanduser()
    if not output_json.is_absolute():
        output_json = repo_root / output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["index_ntotal", "component_ratios", "score", "distance", "prior"]}, indent=2))


if __name__ == "__main__":
    main()
