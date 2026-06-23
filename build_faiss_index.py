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


def parse_args():
    parser = argparse.ArgumentParser(description="Build a RAG-SEG FAISS retrieval library.")
    parser.add_argument("--cod10k_root", type=str, default="COD10K-v3")
    parser.add_argument("--camo_root", type=str, default="CAMO-V.1.0-CVIU2019")
    parser.add_argument("--no_cod10k", action="store_true")
    parser.add_argument("--no_camo", action="store_true")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--niter", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--cpu_faiss", action="store_true", help="Force FAISS KMeans/search on CPU.")
    parser.add_argument("--faiss_gpu", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--dinov2_checkpoint",
        type=str,
        default="dinov2/checkpoints/dinov2_vits14_pretrain.pth",
    )
    parser.add_argument("--cache_dir", type=str, default="outputs/ragseg_index")
    parser.add_argument("--output_index", type=str, default="outputs/ragseg_index/sod_cod_rebuild.index")
    parser.add_argument(
        "--output_scores",
        type=str,
        default="outputs/ragseg_index/sod_cod_score_rebuild.index.npz",
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
    return dinov2_vits14(pretrained=True, weights=str(ckpt_path)).to(device).eval()


def preprocess_images(paths, image_size: int, device: torch.device):
    batch = []
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    for path in paths:
        img = Image.open(path).convert("RGB").resize((image_size, image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - mean) / std
        batch.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(batch, dim=0).to(device)


def mask_scores(mask_path: Path, grid_size: int, image_size: int):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(np.float32) / 255.0
    mask = cv2.resize(mask, (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    return mask.reshape(-1).astype("float32")


def extract_feature_cache(args, repo_root: Path, pairs, device: torch.device):
    grid_size = args.image_size // args.patch_size
    tokens_per_image = grid_size * grid_size
    num_tokens = len(pairs) * tokens_per_image
    cache_dir = resolve(repo_root, args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_path = cache_dir / f"features_{args.image_size}_{len(pairs)}.npy"
    score_path = cache_dir / f"scores_{args.image_size}_{len(pairs)}.npy"

    features = np.lib.format.open_memmap(
        feature_path, mode="w+", dtype="float32", shape=(num_tokens, 384)
    )
    scores = np.lib.format.open_memmap(
        score_path, mode="w+", dtype="float32", shape=(num_tokens,)
    )

    model = prepare_dinov2(repo_root, args.dinov2_checkpoint, device)
    offset = 0
    for start in tqdm(range(0, len(pairs), args.batch_size), desc="extract", ncols=100):
        batch_pairs = pairs[start : start + args.batch_size]
        image_paths = [item[0] for item in batch_pairs]
        inputs = preprocess_images(image_paths, args.image_size, device)
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model.forward_features(inputs)
            else:
                out = model.forward_features(inputs)
        token_vec = out["x_norm_patchtokens"].detach().cpu().numpy().astype("float32")
        token_vec = token_vec.reshape(-1, token_vec.shape[-1])

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
    return feature_path, score_path


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
        raise ValueError(f"k={args.k} is larger than the number of vectors: {features.shape[0]}")
    kmeans = faiss.Kmeans(
        d=features.shape[1],
        k=args.k,
        niter=args.niter,
        verbose=True,
        gpu=use_gpu,
        seed=args.seed,
    )
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
    repo_root = Path(__file__).resolve().parent
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    pairs = []
    if not args.no_cod10k:
        pairs.extend(collect_cod10k(resolve(repo_root, args.cod10k_root)))
    if not args.no_camo:
        pairs.extend(collect_camo(resolve(repo_root, args.camo_root)))
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise RuntimeError("No image/mask pairs found.")

    log(f"Images: {len(pairs)}")
    log(f"Tokens per image: {(args.image_size // args.patch_size) ** 2}")
    log(f"Device: {device}")
    feature_path, score_path = extract_feature_cache(args, repo_root, pairs, device)
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
        num_images=np.array([len(pairs)], dtype=np.int64),
    )
    log(f"Saved index: {output_index}")
    log(f"Saved scores: {output_scores}")
    log(f"Non-empty clusters: {int((counts > 0).sum())}/{len(counts)}")


if __name__ == "__main__":
    main()
