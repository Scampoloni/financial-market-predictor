"""
chart_classifier.py — Pre-trained CNN feature extractor for candlestick charts.

Supports two modes:
  1. Frozen EfficientNet-B0 (ImageNet weights) — default, no training required.
  2. Fine-tuned EfficientNet-B0 — loaded from models/cnn_finetuned.pth if it
     exists (run scripts/finetune_cnn.py first).

Usage:
    from src.cv.chart_classifier import ChartCNN
    cnn = ChartCNN()                          # auto-selects fine-tuned if available
    cnn = ChartCNN(use_finetuned=True)        # force fine-tuned (raises if missing)
    cnn = ChartCNN(use_finetuned=False)       # force frozen ImageNet
    embedding = cnn.embed_image("data/raw/charts/AAPL/2024-01-15.png")
    # → np.ndarray shape (1280,)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)

# EfficientNet-B0 penultimate layer output dim
EMBED_DIM = 1280

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Default path for the fine-tuned model (set in config.py or resolved here)
_DEFAULT_FINETUNED_PATH = Path(__file__).resolve().parents[2] / "models" / "cnn_finetuned.pth"


class ChartCNN:
    """EfficientNet-B0 feature extractor for chart images.

    Supports both frozen (ImageNet) and fine-tuned (domain-adapted) modes.
    Automatically uses the fine-tuned model if ``models/cnn_finetuned.pth``
    exists, unless explicitly overridden.

    Args:
        device: 'cpu', 'cuda', or None (auto-detect).
        use_finetuned: True = always use fine-tuned; False = always use frozen;
                       None (default) = auto-select (fine-tuned if file exists).
        finetuned_path: Override path to the fine-tuned weights file.
    """

    def __init__(
        self,
        device: str | None = None,
        use_finetuned: bool | None = None,
        finetuned_path: str | Path | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: nn.Module | None = None
        self._finetuned_path = Path(finetuned_path) if finetuned_path else _DEFAULT_FINETUNED_PATH

        # Resolve mode
        if use_finetuned is True:
            if not self._finetuned_path.exists():
                raise FileNotFoundError(
                    f"Fine-tuned model not found at {self._finetuned_path}. "
                    "Run: python scripts/finetune_cnn.py"
                )
            self._use_finetuned = True
        elif use_finetuned is False:
            self._use_finetuned = False
        else:
            # Auto-detect
            self._use_finetuned = self._finetuned_path.exists()

        mode = "fine-tuned" if self._use_finetuned else "frozen ImageNet"
        logger.info("ChartCNN initialized (device=%s, mode=%s)", self.device, mode)

    @property
    def mode(self) -> str:
        """Returns 'finetuned' or 'frozen'."""
        return "finetuned" if self._use_finetuned else "frozen"

    # ── Loading ──────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load the backbone (called lazily on first inference)."""
        if self._model is not None:
            return

        if self._use_finetuned:
            self._model = self._load_finetuned()
        else:
            self._model = self._load_frozen()

    def _load_frozen(self) -> nn.Module:
        """Load vanilla frozen EfficientNet-B0 (ImageNet weights)."""
        logger.info("Loading frozen EfficientNet-B0 (ImageNet weights) ...")
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        # Remove classifier → keep features + avgpool only (outputs 1280-dim)
        backbone.classifier = nn.Identity()
        backbone.eval()
        backbone.to(self.device)
        logger.info("Frozen EfficientNet-B0 loaded on %s (embed_dim=%d)", self.device, EMBED_DIM)
        return backbone

    def _load_finetuned(self) -> nn.Module:
        """Load the fine-tuned EfficientNet-B0 as a feature extractor.

        The classifier head (Linear(1280, 2)) is stripped — the 1280-dim
        avgpool output is used as the embedding, matching frozen mode.
        """
        logger.info("Loading fine-tuned EfficientNet-B0 from %s ...", self._finetuned_path)
        checkpoint = torch.load(self._finetuned_path, map_location=self.device)

        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        # Rebuild classifier head as in finetune_cnn.py (Dropout + Linear(1280, 2))
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 2),
        )
        backbone.load_state_dict(checkpoint["model_state_dict"])

        # Strip classifier → expose 1280-dim avgpool output for embedding
        backbone.classifier = nn.Identity()
        backbone.eval()
        backbone.to(self.device)

        val_f1 = checkpoint.get("val_f1_macro", 0.0)
        n_train = checkpoint.get("n_train_samples", "?")
        logger.info(
            "Fine-tuned EfficientNet-B0 loaded (val F1=%.4f, trained on %s samples)",
            val_f1, n_train,
        )
        return backbone

    # ── Inference ────────────────────────────────────────────────────────────

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        """Extract a 1280-dim embedding from a single chart image.

        Args:
            image_path: Path to a PNG chart image.

        Returns:
            1-D numpy array of shape (1280,).

        Raises:
            FileNotFoundError: If the image does not exist.
        """
        self._load()
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Chart image not found: {path}")

        img = Image.open(path).convert("RGB")
        tensor = _TRANSFORM(img).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)

        with torch.no_grad():
            embedding = self._model(tensor)  # (1, 1280)

        return embedding.squeeze(0).cpu().numpy()

    def embed_batch(
        self, image_paths: list[str | Path], batch_size: int = 16,
    ) -> np.ndarray:
        """Extract embeddings for a batch of chart images in mini-batches.

        Args:
            image_paths: List of paths to PNG chart images.
            batch_size: Number of images per mini-batch (lower = less RAM/CPU).

        Returns:
            2-D numpy array of shape (N, 1280). Rows with failed images
            are filled with zeros and a warning is logged.
        """
        self._load()
        results = np.zeros((len(image_paths), EMBED_DIM), dtype=np.float32)

        tensors, valid_indices = [], []
        for i, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert("RGB")
                tensors.append(_TRANSFORM(img))
                valid_indices.append(i)
            except Exception as exc:
                logger.warning("Failed to load chart %s: %s", path, exc)

        if not tensors:
            return results

        for start in range(0, len(tensors), batch_size):
            end = min(start + batch_size, len(tensors))
            batch = torch.stack(tensors[start:end]).to(self.device)
            with torch.no_grad():
                embeddings = self._model(batch).cpu().numpy()
            for j, emb in enumerate(embeddings):
                results[valid_indices[start + j]] = emb

        return results
