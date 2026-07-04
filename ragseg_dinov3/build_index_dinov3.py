import argparse
import json
import sys
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragseg_dinov3.dinov3_extractor import DinoV3Extractor


def parse_args():
    parser = argparse.ArgumentParser(description="Build a DINOv3 RAG-SEG FAISS retrieval library.")
    parser.add_argument("--cod10k_root", type=str, default="COD10K-v3")
    parser.add_argument("--camo_root", type=str, default="CAMO-V.1.0-CVIU2019")
    parser.add_argument("--sod_root", type=str, default="")
    parser.add_argument("--sod_image_dir", type=str, default="train/image")
    parser.add_argument("--sod_mask_dir", type=str, default="train/mask")
    parser.add_argument("--no_cod10k", action="store_true")
    parser.add_argument("--no_camo", action="store_true")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--niter", type=int, default=200)
    parser.add_argument("--nredo", type=int, default=1)
    parser.add_argument("--min_points_per_centroid", type=int, default=1)
    parser.add_argument(
        "--max_points_per_centroid",
        type=int,
        default=0,
        help="FAISS KMeans training cap per centroid. 0 keeps the FAISS default.",
    )
    parser.add_argument(
        "--spherical_kmeans",
        action="store_true",
        help="Use spherical KMeans for L2-normalized feature spaces.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--cpu_faiss", action="store_true")
    parser.add_argument("--faiss_gpu", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
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
    parser.add_argument("--cache_dir", type=str, default="outputs/dinov3_index")
    parser.add_argument(
        "--reuse_cache",
        action="store_true",
        help="Reuse existing feature/score cache when shape and metadata match.",
    )
    parser.add_argument("--output_index", type=str, default="indexes/dinov3/sod_cod_dinov3_vits16.index")
    parser.add_argument(
        "--output_scores",
        type=str,
        default="indexes/dinov3/sod_cod_score_dinov3_vits16.index.npz",
    )
    return parser.parse_args()


def log(message: str):
    print(message, flush=True)


def resolve(root: Path, path: str):
    path = Path(path).expanduser()
    return path if path.is_absolute() else root / path


def collect_cod10k(root: Path):
    split_path = root / "Train" / "CAM_Instance_Train.json"
    image_dir = root / "Train" / "Image"
    gt_dir = root / "Train" / "GT_Object"
    data = json.loads(split_path.read_text(encoding="utf-8"))
    pairs = []
    for item in data["images"]:
        image_path = image_dir / item["file_name"]
        gt_path = gt_dir / Path(item["file_name"]).with_suffix(".png").name
        if image_path.is_file() and gt_path.is_file():
            pairs.append((image_path, gt_path))
    return pairs


def collect_camo(root: Path):
    image_dir = root / "Images" / "Train"
    gt_dir = root / "GT"
    pairs = []
    for image_path in sorted(image_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        gt_path = gt_dir / f"{image_path.stem}.png"
        if gt_path.is_file():
            pairs.append((image_path, gt_path))
    return pairs


def collect_image_mask_dir(root: Path, image_dir: str, mask_dir: str):
    image_root = root / image_dir
    mask_root = root / mask_dir
    pairs = []
    image_paths = []
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths.extend(image_root.glob(suffix))
    for image_path in sorted(image_paths):
        mask_path = mask_root / f"{image_path.stem}.png"
        if mask_path.is_file():
            pairs.append((image_path, mask_path))
    return pairs


def mask_scores(mask_path: Path, grid_size: tuple[int, int], image_size: int):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(np.float32) / 255.0
    mask = cv2.resize(mask, (grid_size[1], grid_size[0]), interpolation=cv2.INTER_AREA)
    return mask.reshape(-1).astype("float32")


def extract_feature_cache(args, repo_root: Path, pairs, extractor: DinoV3Extractor):
    if args.image_size % extractor.patch_size != 0:
        raise ValueError(f"--image_size must be divisible by patch size {extractor.patch_size}")
    grid_size = (args.image_size // extractor.patch_size, args.image_size // extractor.patch_size)
    tokens_per_image = grid_size[0] * grid_size[1]
    num_tokens = len(pairs) * tokens_per_image
    cache_dir = resolve(repo_root, args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.backbone}_{args.fusion}_{args.image_size}_{len(pairs)}"
    feature_path = cache_dir / f"features_{suffix}.npy"
    score_path = cache_dir / f"scores_{suffix}.npy"

    if args.reuse_cache and feature_path.is_file() and score_path.is_file():
        features = np.load(feature_path, mmap_mode="r")
        scores = np.load(score_path, mmap_mode="r")
        expected_feature_shape = (num_tokens, extractor.feature_dim)
        expected_score_shape = (num_tokens,)
        if features.shape == expected_feature_shape and scores.shape == expected_score_shape:
            log(f"Reusing feature cache: {feature_path} {features.shape}")
            log(f"Reusing score cache: {score_path} {scores.shape}")
            return feature_path, score_path, grid_size
        log(
            "Ignoring cache because shape does not match: "
            f"features {features.shape} vs {expected_feature_shape}, "
            f"scores {scores.shape} vs {expected_score_shape}"
        )

    features = np.lib.format.open_memmap(
        feature_path, mode="w+", dtype="float32", shape=(num_tokens, extractor.feature_dim)
    )
    scores = np.lib.format.open_memmap(score_path, mode="w+", dtype="float32", shape=(num_tokens,))

    offset = 0
    for start in tqdm(range(0, len(pairs), args.batch_size), desc="extract", ncols=100):
        batch_pairs = pairs[start : start + args.batch_size]
        image_paths = [item[0] for item in batch_pairs]
        inputs = extractor.preprocess_paths(image_paths, args.image_size)
        with torch.inference_mode():
            if extractor.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = extractor.extract(inputs)
            else:
                out = extractor.extract(inputs)
        token_vec = out.patch_tokens.detach().float().cpu().numpy().astype("float32")
        token_vec = token_vec.reshape(-1, token_vec.shape[-1])
        if token_vec.shape[0] != len(batch_pairs) * tokens_per_image:
            raise RuntimeError(f"Unexpected token count: {token_vec.shape[0]}")

        batch_scores = np.concatenate(
            [mask_scores(mask_path, grid_size, args.image_size) for _, mask_path in batch_pairs],
            axis=0,
        )
        end = offset + token_vec.shape[0]
        features[offset:end] = token_vec
        scores[offset:end] = batch_scores
        offset = end

    features.flush()
    scores.flush()
    return feature_path, score_path, grid_size


def faiss_gpu_available():
    return (
        hasattr(faiss, "StandardGpuResources")
        and hasattr(faiss, "get_num_gpus")
        and faiss.get_num_gpus() > 0
    )


def should_use_faiss_gpu(args):
    if args.cpu_faiss:
        return False
    return args.faiss_gpu or faiss_gpu_available()


def train_kmeans(args, features, use_gpu: bool):
    if args.k > features.shape[0]:
        raise ValueError(f"k={args.k} is larger than vector count: {features.shape[0]}")
    kmeans = faiss.Kmeans(
        d=features.shape[1],
        k=args.k,
        niter=args.niter,
        nredo=args.nredo,
        verbose=True,
        gpu=use_gpu,
        seed=args.seed,
        min_points_per_centroid=args.min_points_per_centroid,
        spherical=args.spherical_kmeans,
    )
    if args.max_points_per_centroid > 0:
        kmeans.cp.max_points_per_centroid = args.max_points_per_centroid
    kmeans.train(features)
    return kmeans.centroids.astype("float32")


def make_index(centroids):
    index = faiss.IndexFlatIP(centroids.shape[1])
    index.add(centroids.astype("float32"))
    return index


def assign_cluster_scores(index, features, scores, batch_size: int):
    sums = np.zeros(index.ntotal, dtype=np.float64)
    counts = np.zeros(index.ntotal, dtype=np.int64)
    for start in tqdm(range(0, features.shape[0], batch_size), desc="assign", ncols=100):
        end = min(start + batch_size, features.shape[0])
        _, idxs = index.search(features[start:end], 1)
        idxs = idxs[:, 0]
        np.add.at(sums, idxs, scores[start:end])
        np.add.at(counts, idxs, 1)

    cluster_scores = np.zeros(index.ntotal, dtype="float32")
    valid = counts > 0
    cluster_scores[valid] = (sums[valid] / counts[valid]).astype("float32")
    return cluster_scores, counts


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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

    pairs = []
    if not args.no_cod10k:
        pairs.extend(collect_cod10k(resolve(repo_root, args.cod10k_root)))
    if not args.no_camo:
        pairs.extend(collect_camo(resolve(repo_root, args.camo_root)))
    if args.sod_root:
        pairs.extend(
            collect_image_mask_dir(
                resolve(repo_root, args.sod_root),
                args.sod_image_dir,
                args.sod_mask_dir,
            )
        )
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise RuntimeError("No image/mask pairs found.")

    log(f"Images: {len(pairs)}")
    log(f"Backbone: {extractor.signature()}")
    log(f"Tokens per image: {(args.image_size // extractor.patch_size) ** 2}")
    log(f"Device: {device}")
    feature_path, score_path, grid_size = extract_feature_cache(args, repo_root, pairs, extractor)
    features = np.load(feature_path, mmap_mode="r")
    scores = np.load(score_path, mmap_mode="r")

    log(f"Feature cache: {feature_path} {features.shape}")
    log(f"Score cache: {score_path} {scores.shape}")
    use_faiss_gpu = should_use_faiss_gpu(args)
    log(f"FAISS GPU: {use_faiss_gpu}")
    log("Training FAISS KMeans...")
    centroids = train_kmeans(args, features, use_faiss_gpu)
    index = make_index(centroids)
    cluster_scores, counts = assign_cluster_scores(index, features, scores, args.batch_size * 4096)

    output_index = resolve(repo_root, args.output_index)
    output_scores = resolve(repo_root, args.output_scores)
    output_index.parent.mkdir(parents=True, exist_ok=True)
    output_scores.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_index))
    np.savez(
        output_scores,
        scores=cluster_scores,
        counts=counts,
        k=np.array([args.k], dtype=np.int64),
        image_size=np.array([args.image_size], dtype=np.int64),
        patch_size=np.array([extractor.patch_size], dtype=np.int64),
        grid_h=np.array([grid_size[0]], dtype=np.int64),
        grid_w=np.array([grid_size[1]], dtype=np.int64),
        feature_dim=np.array([extractor.feature_dim], dtype=np.int64),
        num_images=np.array([len(pairs)], dtype=np.int64),
        backbone=np.array([args.backbone]),
        fusion=np.array([args.fusion]),
        layers=np.array([args.layers]),
        l2_normalize=np.array([int(args.l2_normalize_features)], dtype=np.int64),
        niter=np.array([args.niter], dtype=np.int64),
        nredo=np.array([args.nredo], dtype=np.int64),
        min_points_per_centroid=np.array([args.min_points_per_centroid], dtype=np.int64),
        max_points_per_centroid=np.array([args.max_points_per_centroid], dtype=np.int64),
        spherical_kmeans=np.array([int(args.spherical_kmeans)], dtype=np.int64),
        metadata=json.dumps(extractor.signature()),
    )
    log(f"Saved index: {output_index}")
    log(f"Saved scores: {output_scores}")


if __name__ == "__main__":
    main()
