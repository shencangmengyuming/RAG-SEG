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


def rag_coarse_mask(img: Image.Image, model, index, scores, device: torch.device, image_size: int):
    inputs = preprocess_dinov2_image(img, image_size, device)
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model.forward_features(inputs)
        else:
            out = model.forward_features(inputs)
    token_vec = out["x_norm_patchtokens"].squeeze(0).detach().cpu().numpy()
    flat_tokens = token_vec.reshape(-1, token_vec.shape[-1]).astype("float32")
    distances, idxs = index.search(flat_tokens, 1)

    mask_vals = np.zeros(len(idxs), dtype=np.float32)
    valid = distances[:, 0] >= 0
    mask_vals[valid] = scores[idxs[valid, 0]]

    side = int(np.sqrt(flat_tokens.shape[0]))
    return mask_vals[: side * side].reshape(side, side).astype(np.float32)


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
    init_mask = rag_coarse_mask(img, model, index, scores, device, args.image_size)
    init_mask = cv2.resize(
        init_mask, (args.sam_size, args.sam_size), interpolation=cv2.INTER_LINEAR
    )

    pos_indices = np.argwhere(init_mask > args.pos_thr)
    neg_indices = np.argwhere(init_mask < args.neg_thr)
    pos_points = pos_indices[:, ::-1]
    neg_points = neg_indices[:, ::-1]

    if len(pos_points) > args.n_points:
        pos_points = pos_points[np.random.choice(len(pos_points), args.n_points, replace=False)]
    if len(neg_points) > args.n_points:
        neg_points = neg_points[np.random.choice(len(neg_points), args.n_points, replace=False)]

    if len(pos_points) == 0 and len(neg_points) == 0:
        return np.zeros((original_size[1], original_size[0]), dtype=np.uint8)

    input_points = np.concatenate([pos_points, neg_points], axis=0).astype(np.float32)
    input_labels = np.concatenate(
        [np.ones(len(pos_points)), np.zeros(len(neg_points))], axis=0
    ).astype(np.int32)

    predictor.set_image(np.array(img.resize((args.sam_size, args.sam_size))))
    mask_input = (init_mask > args.mask_thr).astype(np.float32)
    mask_input = cv2.resize(mask_input, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask_input = mask_input[None, :, :]
    masks, _, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        mask_input=mask_input,
        multimask_output=False,
    )
    pred = (masks[0] * 255).astype(np.uint8)
    return cv2.resize(pred, original_size, interpolation=cv2.INTER_LINEAR)


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
        for stem in tqdm(names, desc="infer", ncols=100):
            pred_path = pred_dir / f"{stem}.png"
            if pred_path.exists() and not args.force:
                continue
            img_path = image_dir / f"{stem}{args.image_ext}"
            pred = predict_mask(
                img_path=img_path,
                predictor=predictor,
                model=model,
                index=index,
                scores=scores,
                device=device,
                args=args,
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
