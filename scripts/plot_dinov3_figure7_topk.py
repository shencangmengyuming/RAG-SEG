from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/home/lsx/workplace/dino-sod/segdino-main/SOD/DUTS")
OUT = ROOT / "outputs/dinov3_figure7_topk_20261110/figure7_topk_initial_masks.png"
OUT_PDF = OUT.with_suffix(".pdf")
SAMPLE_IDS = [
    "ILSVRC2012_test_00000003",
    "ILSVRC2012_test_00000023",
    "ILSVRC2012_test_00000025",
    "ILSVRC2012_test_00000026",
]
TOPK_DIRS = {
    1: ROOT / "outputs/dinov3_topk_kcurve_20261025/k1024_top1/DUTS-TE_prior",
    3: ROOT / "outputs/dinov3_figure7_topk_20261110/top3/DUTS-TE_prior",
    10: ROOT / "outputs/dinov3_figure7_topk_20261110/top10/DUTS-TE_prior",
    50: ROOT / "outputs/dinov3_figure7_topk_20261110/top50/DUTS-TE_prior",
    100: ROOT / "outputs/dinov3_figure7_topk_20261110/top100/DUTS-TE_prior",
}


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def read_mask(
    path: Path,
    size: tuple[int, int],
    resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> np.ndarray:
    image = Image.open(path).convert("L")
    image = image.resize(size, resample)
    return np.asarray(image)


def main() -> None:
    fig, axes = plt.subplots(len(SAMPLE_IDS), 7, figsize=(7.2, 4.2))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.90, bottom=0.02, hspace=0.08, wspace=0.04)
    headings = ["Image", "GT", "Top-1", "Top-3", "Top-10", "Top-50", "Top-100"]

    for col, title in enumerate(headings):
        axes[0, col].set_title(title, fontsize=9, pad=5)

    for row, stem in enumerate(SAMPLE_IDS):
        image = read_rgb(DATA / "test/image" / f"{stem}.jpg")
        gt = read_mask(
            DATA / "test/mask" / f"{stem}.png",
            (image.shape[1], image.shape[0]),
            Image.Resampling.NEAREST,
        )
        axes[row, 0].imshow(image)
        axes[row, 1].imshow(gt, cmap="gray", vmin=0, vmax=255)

        for col, topk in enumerate((1, 3, 10, 50, 100), start=2):
            prior_path = TOPK_DIRS[topk] / f"{stem}.png"
            prior = read_mask(prior_path, (image.shape[1], image.shape[0]))
            axes[row, col].imshow(prior, cmap="gray", vmin=0, vmax=255)

        for ax in axes[row]:
            ax.set_axis_off()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
