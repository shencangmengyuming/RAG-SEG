#!/usr/bin/env python3
"""Render the DINOv3 reproduction of RAG-SEG Appendix Figure 10."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter


K_VALUES = (1024, 2048, 4096, 8192, 16384, 32768, 65536)
BIN_EDGES = np.linspace(0.0, 1.0, 11)
COLORS = (
    "#7A8FA6",
    "#C98078",
    "#D4B483",
    "#9A9288",
    "#A58AA8",
    "#7F9A88",
    "#526A82",
)
HATCHES = ("", "//", "..", "\\\\", "xx", "--", "++")
ACADEMIC_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "savefig.dpi": 220,
    "axes.edgecolor": "#D8D1C7",
    "axes.labelcolor": "#4B5563",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 10.5,
    "grid.color": "#E7E5E4",
    "grid.linewidth": 0.65,
    "grid.alpha": 0.65,
    "font.size": 10,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "xtick.color": "#6B7280",
    "ytick.color": "#6B7280",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("indexes/dinov3/sod_k_ablation_20261019"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/dinov3_sod_k_ablation_20261019/figure10_dinov3"),
    )
    return parser.parse_args()


def load_histograms(index_dir: Path) -> tuple[dict[int, np.ndarray], dict[int, dict[str, float]]]:
    histograms: dict[int, np.ndarray] = {}
    summaries: dict[int, dict[str, float]] = {}

    for k in K_VALUES:
        score_path = index_dir / f"duts_k{k}_centroid_l2.index.npz"
        if not score_path.is_file():
            raise FileNotFoundError(f"Missing score file: {score_path}")

        archive = np.load(score_path)
        scores = archive["scores"]
        counts = archive["counts"]
        if scores.shape != (k,) or counts.shape != (k,):
            raise ValueError(
                f"K={k}: expected score/count shape {(k,)}, got {scores.shape}/{counts.shape}"
            )
        if not np.isfinite(scores).all() or scores.min() < 0.0 or scores.max() > 1.0:
            raise ValueError(f"K={k}: scores must be finite and bounded by [0, 1]")

        histogram, _ = np.histogram(scores, bins=BIN_EDGES)
        if int(histogram.sum()) != k:
            raise ValueError(f"K={k}: histogram accounts for {histogram.sum()} centers")

        histograms[k] = histogram
        summaries[k] = {
            "score_min": float(scores.min()),
            "score_max": float(scores.max()),
            "score_mean": float(scores.mean()),
            "empty_clusters": int((counts == 0).sum()),
            "assigned_tokens": int(counts.sum()),
        }

    assigned_token_counts = {summary["assigned_tokens"] for summary in summaries.values()}
    if assigned_token_counts != {2_701_568}:
        raise ValueError(f"Unexpected assigned-token totals: {sorted(assigned_token_counts)}")
    return histograms, summaries


def write_counts_csv(path: Path, histograms: dict[int, np.ndarray]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["k", "bin_index", "score_lower", "score_upper", "right_closed", "count"])
        for k in K_VALUES:
            for index, count in enumerate(histograms[k]):
                writer.writerow(
                    [
                        k,
                        index,
                        f"{BIN_EDGES[index]:.1f}",
                        f"{BIN_EDGES[index + 1]:.1f}",
                        index == len(BIN_EDGES) - 2,
                        int(count),
                    ]
                )


def render_figure(output_base: Path, histograms: dict[int, np.ndarray]) -> None:
    labels = [
        f"[{BIN_EDGES[index]:.1f}, {BIN_EDGES[index + 1]:.1f}{']' if index == 9 else ')'}"
        for index in range(10)
    ]
    x = np.arange(len(labels), dtype=np.float64)
    group_width = 0.84
    bar_width = group_width / len(K_VALUES)

    fig, ax = plt.subplots(figsize=(7.2, 3.85))
    for series_index, (k, color, hatch) in enumerate(zip(K_VALUES, COLORS, HATCHES)):
        offset = (series_index - (len(K_VALUES) - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            histograms[k],
            width=bar_width * 0.92,
            color=color,
            edgecolor="#FFFFFF",
            linewidth=0.35,
            hatch=hatch,
            label=f"K={k:,}",
            zorder=3,
        )

    ax.set_xlabel("Cluster mask-score interval")
    ax.set_ylabel("Number of cluster centers")
    ax.set_xticks(x, labels, rotation=32, ha="right", rotation_mode="anchor")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.55, len(labels) - 0.45)
    ax.set_ylim(bottom=0)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=4,
        columnspacing=1.2,
        handlelength=1.2,
        handletextpad=0.45,
    )
    ax.text(
        0.99,
        0.965,
        "DINOv3 ViT-S/16 · DUTS-TR",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="#6B7280",
    )
    fig.subplots_adjust(top=0.80, bottom=0.25, left=0.105, right=0.985)

    for extension in ("png", "pdf", "svg"):
        kwargs = {"dpi": 320} if extension == "png" else {}
        fig.savefig(output_base.with_suffix(f".{extension}"), **kwargs)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plt.rcParams.update(ACADEMIC_STYLE)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    histograms, summaries = load_histograms(args.index_dir)
    output_base = args.output_dir / "figure10_dinov3_sod_score_distribution"
    counts_path = args.output_dir / "figure10_dinov3_counts.csv"
    manifest_path = args.output_dir / "figure10_dinov3_manifest.json"

    write_counts_csv(counts_path, histograms)
    render_figure(output_base, histograms)

    manifest = {
        "surface_class": "appendix",
        "description": "DINOv3 reproduction of RAG-SEG Appendix Figure 10.",
        "source_data": [
            str(args.index_dir / f"duts_k{k}_centroid_l2.index.npz") for k in K_VALUES
        ],
        "generating_script": "scripts/plot_dinov3_figure10.py",
        "k_values": list(K_VALUES),
        "bin_edges": BIN_EDGES.tolist(),
        "histogram_counts": {str(k): histograms[k].tolist() for k in K_VALUES},
        "input_summaries": {str(k): summaries[k] for k in K_VALUES},
        "outputs": {
            "png": str(output_base.with_suffix(".png")),
            "pdf": str(output_base.with_suffix(".pdf")),
            "svg": str(output_base.with_suffix(".svg")),
            "counts_csv": str(counts_path),
        },
        "claim": (
            "With increasing K, the absolute number of high-score centers grows, while the "
            "largest share remains concentrated in the [0.0, 0.1) interval."
        ),
        "self_review_revision": (
            "Added an in-figure DINOv3/DUTS-TR provenance label and applied the academic "
            "style parameters directly to avoid Matplotlib parsing warnings from hex colors."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved figure and data to {args.output_dir}")


if __name__ == "__main__":
    main()
