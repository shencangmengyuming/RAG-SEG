import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class DinoV3FeatureOutput:
    patch_tokens: torch.Tensor
    grid_size: tuple[int, int]
    cls_token: torch.Tensor | None


def _resolve(repo_root: Path, path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else repo_root / path


def parse_layers(layers: str, zero_based: bool = False) -> list[int]:
    if not layers:
        return []
    parsed = [int(item.strip()) for item in layers.split(",") if item.strip()]
    if zero_based:
        return parsed
    return [item - 1 for item in parsed]


def parse_layer_weights(weights: str, n_layers: int) -> list[float]:
    if not weights:
        return [1.0 / n_layers] * n_layers
    parsed = [float(item.strip()) for item in weights.split(",") if item.strip()]
    if len(parsed) != n_layers:
        raise ValueError(f"Expected {n_layers} layer weights, got {len(parsed)}")
    total = sum(parsed)
    if abs(total) < 1e-12:
        raise ValueError("Layer weights must not sum to zero.")
    return [item / total for item in parsed]


class DinoV3Extractor:
    """Small wrapper that exposes DINOv3 patch tokens for RAG retrieval."""

    def __init__(
        self,
        repo_root: Path,
        dinov3_repo: str | Path,
        weights: str | Path,
        model_name: str = "dinov3_vits16",
        layers: str = "",
        fusion: str = "last",
        layer_weights: str = "",
        zero_based_layers: bool = False,
        l2_normalize: bool = False,
        device: torch.device | str = "cpu",
    ):
        self.repo_root = Path(repo_root)
        self.dinov3_repo = _resolve(self.repo_root, dinov3_repo)
        self.weights = _resolve(self.repo_root, weights)
        self.model_name = model_name
        self.layer_ids = parse_layers(layers, zero_based_layers)
        self.fusion = fusion
        self.l2_normalize = l2_normalize
        self.device = torch.device(device)

        if not self.dinov3_repo.is_dir():
            raise FileNotFoundError(f"DINOv3 repo not found: {self.dinov3_repo}")
        if not self.weights.is_file():
            raise FileNotFoundError(f"DINOv3 weights not found: {self.weights}")
        if self.fusion not in {"last", "mean", "weighted", "concat"}:
            raise ValueError(f"Unsupported fusion: {self.fusion}")
        if self.fusion != "last" and not self.layer_ids:
            raise ValueError("--layers is required when fusion is not 'last'.")

        self.layer_weights = parse_layer_weights(layer_weights, len(self.layer_ids)) if self.layer_ids else []
        self.model = self._load_model().to(self.device).eval()
        self.patch_size = int(getattr(self.model, "patch_size", 16))
        self.base_dim = int(getattr(self.model, "embed_dim", 384))
        self.feature_dim = self.base_dim * len(self.layer_ids) if self.fusion == "concat" else self.base_dim

    def _load_model(self):
        repo_path = str(self.dinov3_repo)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        from dinov3.hub import backbones

        if not hasattr(backbones, self.model_name):
            raise ValueError(f"Unknown DINOv3 backbone: {self.model_name}")
        loader = getattr(backbones, self.model_name)
        return loader(weights=str(self.weights))

    def preprocess_image(self, img: Image.Image, image_size: int) -> torch.Tensor:
        if image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size={image_size} must be divisible by DINOv3 patch_size={self.patch_size}"
            )
        img = img.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def preprocess_paths(self, paths: Iterable[Path], image_size: int) -> torch.Tensor:
        batch = [self.preprocess_image(Image.open(path), image_size).squeeze(0).cpu() for path in paths]
        return torch.stack(batch, dim=0).to(self.device)

    def extract(self, inputs: torch.Tensor) -> DinoV3FeatureOutput:
        _, _, height, width = inputs.shape
        grid_size = (height // self.patch_size, width // self.patch_size)
        with torch.inference_mode():
            if self.fusion == "last":
                out = self.model.forward_features(inputs)
                patch_tokens = out["x_norm_patchtokens"]
                cls_token = out.get("x_norm_clstoken")
            else:
                feats = self.model.get_intermediate_layers(
                    inputs,
                    n=self.layer_ids,
                    reshape=False,
                    return_class_token=True,
                    norm=True,
                )
                patch_list = [item[0] for item in feats]
                cls_list = [item[1] for item in feats]
                if self.fusion == "concat":
                    patch_tokens = torch.cat(patch_list, dim=-1)
                    cls_token = torch.cat(cls_list, dim=-1)
                elif self.fusion == "weighted":
                    weights = torch.tensor(self.layer_weights, device=inputs.device, dtype=patch_list[0].dtype)
                    stacked = torch.stack(patch_list, dim=0)
                    patch_tokens = (stacked * weights[:, None, None, None]).sum(dim=0)
                    cls_stacked = torch.stack(cls_list, dim=0)
                    cls_token = (cls_stacked * weights[:, None, None]).sum(dim=0)
                else:
                    patch_tokens = torch.stack(patch_list, dim=0).mean(dim=0)
                    cls_token = torch.stack(cls_list, dim=0).mean(dim=0)

        if self.l2_normalize:
            patch_tokens = F.normalize(patch_tokens.float(), dim=-1, p=2)
            if cls_token is not None:
                cls_token = F.normalize(cls_token.float(), dim=-1, p=2)
        return DinoV3FeatureOutput(patch_tokens=patch_tokens, grid_size=grid_size, cls_token=cls_token)

    def signature(self) -> dict:
        return {
            "model_name": self.model_name,
            "weights": str(self.weights),
            "patch_size": self.patch_size,
            "feature_dim": self.feature_dim,
            "layers": self.layer_ids,
            "fusion": self.fusion,
            "layer_weights": self.layer_weights,
            "l2_normalize": self.l2_normalize,
        }

