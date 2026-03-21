"""
chart_classifier.py — Pre-trained CNN feature extractor for candlestick charts.

Uses EfficientNet-B0 (ImageNet-pretrained) as a frozen feature extractor.
The final classification head is removed; the penultimate layer (1280-dim
global average pool output) is used as the embedding vector.

No fine-tuning is performed — we rely on transfer learning from ImageNet
visual features (edges, textures, shapes) which generalise to chart patterns.

Usage:
    from src.cv.chart_classifier import ChartCNN
    cnn = ChartCNN()
    embedding = cnn.embed_image("data/raw/charts/AAPL/2024-01-15.png")
    # → np.ndarray shape (1280,)
"""

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


class ChartCNN:
    """Frozen EfficientNet-B0 feature extractor for chart images.

    Lazy-loads the model on first use. Runs on CPU or CUDA automatically.
    """

    def __init__(self, device: str | None = None) -> None:
        """Initialize the feature extractor (model not loaded until first call).

        Args:
            device: 'cpu', 'cuda', or None (auto-detect).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: nn.Module | None = None
        logger.info("ChartCNN initialized (device=%s)", self.device)

    def _load(self) -> None:
        """Load EfficientNet-B0 and strip the classification head."""
        if self._model is not None:
            return
        logger.info("Loading EfficientNet-B0 (ImageNet weights) ...")
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Remove classifier — keep features + avgpool only
        backbone.classifier = nn.Identity()
        backbone.eval()
        backbone.to(self.device)
        self._model = backbone
        logger.info("EfficientNet-B0 loaded on %s (embed_dim=%d)", self.device, EMBED_DIM)

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

    def embed_batch(self, image_paths: list[str | Path]) -> np.ndarray:
        """Extract embeddings for a batch of chart images.

        Args:
            image_paths: List of paths to PNG chart images.

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

        batch = torch.stack(tensors).to(self.device)  # (N, 3, 224, 224)
        with torch.no_grad():
            embeddings = self._model(batch).cpu().numpy()  # (N, 1280)

        for out_idx, emb in zip(valid_indices, embeddings):
            results[out_idx] = emb

        return results
