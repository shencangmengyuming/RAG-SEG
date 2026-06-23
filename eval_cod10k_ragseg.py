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
from transformers import AutoImageProcessor, Dinov2Model


PAPER_COD10K = {
    "S_alpha": 0.854,
    "maxE": 0.902,
    "F_w_beta": 0.783,
    "MAE": 0.027,
}


def prepare_dinov2(image_size: int, device: str):
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    processor.size = {"height": image_size, "width": image_size}
    processor.do_center_crop = False
    processor.do_resize = True
    model = Dinov2Model.from_pretrained("facebook/dinov2-small").to(device).eval()
    return processor, model


def load_index(root: Path):
    index = faiss.read_index(str(root / "sod_cod.index"))
    scores = np.load(root / "sod_cod_score.index.npz")["scores"].astype("float32")
    return index, scores


def load_sam2_predictor(repo_root: Path, device: str):
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


def rag_coarse_mask(img: Image.Image, processor, model, index, scores, device: str):
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**inputs)
    token_vec = out.last_hidden_state[:, 1:, :].squeeze(0).detach().cpu().numpy()
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
    processor,
    model,
    index,
    scores,
    device: str,
    seed: int,
    sam_size: int,
    pos_thr: float,
    neg_thr: float,
    mask_thr: float,
    n_points: int,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    img = Image.open(img_path).convert("RGB")
    original_size = img.size
    init_mask = rag_coarse_mask(img, processor, model, index, scores, device)
    init_mask = cv2.resize(init_mask, (sam_size, sam_size), interpolation=cv2.INTER_LINEAR)

    pos_indices = np.argwhere(init_mask > pos_thr)
    neg_indices = np.argwhere(init_mask < neg_thr)
    pos_points = pos_indices[:, ::-1]
    neg_points = neg_indices[:, ::-1]

    if len(pos_points) > n_points:
        pos_points = pos_points[np.random.choice(len(pos_points), n_points, replace=False)]
    if len(neg_points) > n_points:
        neg_points = neg_points[np.random.choice(len(neg_points), n_points, replace=False)]

    if len(pos_points) == 0 and len(neg_points) == 0:
        return np.zeros((original_size[1], original_size[0]), dtype=np.uint8)

    input_points = np.concatenate([pos_points, neg_points], axis=0).astype(np.float32)
    input_labels = np.concatenate(
        [np.ones(len(pos_points)), np.zeros(len(neg_points))], axis=0
    ).astype(np.int32)

    predictor.set_image(np.array(img.resize((sam_size, sam_size))))
    mask_input = (init_mask > mask_thr).astype(np.float32)
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


def load_gray(path: Path):
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(path)
    return arr


def load_split_names(split_path: Path):
    return [
        Path(line.split()[0]).with_suffix(".png").name
        for line in split_path.read_text().splitlines()
        if line.strip()
    ]


def compute_metrics(pred_dir: Path, gt_dir: Path, names):
    repo_root = Path(__file__).resolve().parent
    if not (repo_root / "py_sod_metrics").is_dir():
        raise FileNotFoundError(repo_root / "py_sod_metrics")
    sys.path.insert(0, str(repo_root))

    import py_sod_metrics

    sm = py_sod_metrics.Smeasure()
    em = py_sod_metrics.Emeasure()
    wfm = py_sod_metrics.WeightedFmeasure()
    mae = py_sod_metrics.MAE()

    for name in tqdm(names, desc="metrics"):
        pred = load_gray(pred_dir / name)
        gt = load_gray(gt_dir / name)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="COD10K-v3")
    parser.add_argument("--split", default="COD10K-v3/Info/CAM_test.txt")
    parser.add_argument("--out", default="outputs/ragseg_cod10k_cam_test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=784)
    parser.add_argument("--sam-size", type=int, default=1024)
    parser.add_argument("--pos-thr", type=float, default=0.99)
    parser.add_argument("--neg-thr", type=float, default=0.05)
    parser.add_argument("--mask-thr", type=float, default=0.3)
    parser.add_argument("--n-points", type=int, default=10)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    dataset = root / args.dataset
    image_dir = dataset / "Test" / "Image"
    gt_dir = dataset / "Test" / "GT_Object"
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    names = load_split_names(root / args.split)
    if args.limit:
        names = names[: args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = prepare_dinov2(args.image_size, device)
    predictor = load_sam2_predictor(root, device)
    index, scores = load_index(root)

    start = time.time()
    for name in tqdm(names, desc="infer"):
        pred_path = out_dir / name
        if pred_path.exists():
            continue
        img_path = image_dir / Path(name).with_suffix(".jpg").name
        pred = predict_mask(
            img_path=img_path,
            predictor=predictor,
            processor=processor,
            model=model,
            index=index,
            scores=scores,
            device=device,
            seed=args.seed,
            sam_size=args.sam_size,
            pos_thr=args.pos_thr,
            neg_thr=args.neg_thr,
            mask_thr=args.mask_thr,
            n_points=args.n_points,
        )
        cv2.imwrite(str(pred_path), pred)
    infer_seconds = time.time() - start

    missing = [name for name in names if not (out_dir / name).exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} predictions are missing, first: {missing[0]}")

    metrics = compute_metrics(pred_dir=out_dir, gt_dir=gt_dir, names=names)
    diff = {k: metrics[k] - PAPER_COD10K[k] for k in PAPER_COD10K}
    result = {
        "split": str(root / args.split),
        "num_images": len(names),
        "prediction_dir": str(out_dir),
        "seed": args.seed,
        "infer_seconds": infer_seconds,
        "metrics": metrics,
        "paper_cod10k": PAPER_COD10K,
        "diff_local_minus_paper": diff,
    }

    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
