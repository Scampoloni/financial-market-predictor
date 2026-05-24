"""
eval_cv_held_out.py — Held-out evaluation of the fine-tuned EfficientNet-B0.

Evaluates the chart classifier on 2025 chart images — a window that is entirely
outside both the training period (≤ 2023-12-31) and the validation period
(2024-01-01 – 2024-06-30) used during fine-tuning.  This avoids the
selection-on-validation problem: val_f1=0.538 was the metric used to select
the best checkpoint, so it cannot also serve as a clean generalization estimate.

Usage:
    python scripts/eval_cv_held_out.py

Output:
    - Prints a classification report to stdout.
    - Writes data/processed/cv_held_out_eval.json with all metrics.

Requires:
    - models/cnn_finetuned.pth   (run: python scripts/finetune_cnn.py)
    - data/processed/features_market.parquet
    - data/raw/charts/{TICKER}/{YYYY-MM-DD}.png
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torchvision import models, transforms

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    FEATURES_MARKET_PATH,
    MODELS_DIR,
    PROCESSED_DIR,
    RAW_CHARTS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
FINETUNED_PATH = MODELS_DIR / "cnn_finetuned.pth"
OUTPUT_PATH = PROCESSED_DIR / "cv_held_out_eval.json"
DATE_CUTOFF = "2025-01-01"
MIN_SAMPLES = 20
LABEL_MAP = {"DOWN": 0, "UP": 1}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Model loading (WITH classifier head — needed for UP/DOWN predictions) ─────

def _load_classifier(path: Path, device: str) -> nn.Module:
    """Load fine-tuned EfficientNet-B0 with its 2-class head intact."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    backbone = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    )
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 2),
    )
    backbone.load_state_dict(checkpoint["model_state_dict"])
    backbone.eval()
    backbone.to(device)

    val_f1 = checkpoint.get("val_f1_macro", float("nan"))
    n_train = checkpoint.get("n_train_samples", "?")
    logger.info(
        "Fine-tuned EfficientNet-B0 loaded (val F1=%.4f, trained on %s samples)",
        val_f1, n_train,
    )
    return backbone


# ── Label lookup ──────────────────────────────────────────────────────────────

def _build_label_lookup(parquet_path: Path) -> dict[tuple[str, str], str]:
    """Return {(ticker, 'YYYY-MM-DD') -> 'UP'/'DOWN'} for dates >= DATE_CUTOFF."""
    df = pd.read_parquet(parquet_path)
    df.index = pd.to_datetime(df.index)
    df = df[df.index >= DATE_CUTOFF]
    df = df[df["target"].notna()]

    lookup: dict[tuple[str, str], str] = {}
    for date, row in df.iterrows():
        key = (str(row["ticker"]), date.strftime("%Y-%m-%d"))
        lookup[key] = str(row["target"])

    logger.info("Label lookup: %d entries for dates >= %s", len(lookup), DATE_CUTOFF)
    return lookup


# ── Chart discovery ───────────────────────────────────────────────────────────

def _collect_chart_paths(
    charts_dir: Path,
    label_lookup: dict[tuple[str, str], str],
) -> list[tuple[Path, str]]:
    """
    Scan charts_dir/{TICKER}/{YYYY-MM-DD}.png.
    Return [(path, label)] for 2025 files that have a matching label.
    """
    items: list[tuple[Path, str]] = []

    if not charts_dir.exists():
        logger.warning("Charts directory not found: %s", charts_dir)
        return items

    for ticker_dir in sorted(charts_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name

        for png in sorted(ticker_dir.glob("*.png")):
            date_str = png.stem  # filename is YYYY-MM-DD
            if date_str < DATE_CUTOFF:
                continue
            label = label_lookup.get((ticker, date_str))
            if label is None:
                continue
            items.append((png, label))

    logger.info("Found %d chart files for dates >= %s with known labels", len(items), DATE_CUTOFF)
    return items


# ── Inference ─────────────────────────────────────────────────────────────────

def _predict_batch(
    model: nn.Module,
    items: list[tuple[Path, str]],
    device: str,
    batch_size: int = 32,
) -> tuple[list[str], list[str]]:
    """Run inference; return (y_true, y_pred) as label strings."""
    y_true: list[str] = []
    y_pred: list[str] = []

    paths = [p for p, _ in items]
    labels = [lbl for _, lbl in items]

    # Load all images that can be opened
    tensors: list[torch.Tensor] = []
    valid_idx: list[int] = []
    for i, path in enumerate(paths):
        try:
            img = Image.open(path).convert("RGB")
            tensors.append(_TRANSFORM(img))
            valid_idx.append(i)
        except Exception as exc:
            logger.warning("Skipping %s: %s", path, exc)

    if not tensors:
        return y_true, y_pred

    for start in range(0, len(tensors), batch_size):
        end = min(start + batch_size, len(tensors))
        batch = torch.stack(tensors[start:end]).to(device)
        with torch.no_grad():
            logits = model(batch)  # (B, 2)
        preds = logits.argmax(dim=1).cpu().numpy()
        for j, pred_idx in enumerate(preds):
            orig_i = valid_idx[start + j]
            y_true.append(labels[orig_i])
            y_pred.append(IDX_TO_LABEL[int(pred_idx)])

    return y_true, y_pred


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Check model exists
    if not FINETUNED_PATH.exists():
        print("No fine-tuned chart model found — run chart fine-tuning first.")
        print(f"  Expected: {FINETUNED_PATH}")
        sys.exit(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # 2. Load label lookup from market features
    label_lookup = _build_label_lookup(FEATURES_MARKET_PATH)

    # 3. Collect 2025 chart paths with labels
    items = _collect_chart_paths(RAW_CHARTS_DIR, label_lookup)

    # 4. Check minimum sample count
    if len(items) < MIN_SAMPLES:
        print(
            f"Only {len(items)} chart files found for {DATE_CUTOFF}+ "
            f"— held-out set too small, skipping."
        )
        sys.exit(0)

    logger.info("Running inference on %d charts ...", len(items))

    # 5. Load model and run inference
    model = _load_classifier(FINETUNED_PATH, device)
    y_true, y_pred = _predict_batch(model, items, device)

    n = len(y_true)
    if n == 0:
        print("No predictions produced — check chart files.")
        sys.exit(0)

    # 6. Compute metrics
    accuracy   = accuracy_score(y_true, y_pred)
    f1_macro   = f1_score(y_true, y_pred, average="macro", labels=["DOWN", "UP"], zero_division=0)
    f1_down    = f1_score(y_true, y_pred, average=None, labels=["DOWN", "UP"], zero_division=0)[0]
    f1_up      = f1_score(y_true, y_pred, average=None, labels=["DOWN", "UP"], zero_division=0)[1]
    cm         = confusion_matrix(y_true, y_pred, labels=["DOWN", "UP"])
    # cm layout with labels=["DOWN","UP"]: [[TN, FP], [FN, TP]] (positive = UP)

    print("\n" + "=" * 60)
    print(f"  CV Held-Out Evaluation  (2025 charts, n={n})")
    print("=" * 60)
    print(classification_report(y_true, y_pred, labels=["DOWN", "UP"], zero_division=0))
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  F1-macro  : {f1_macro:.4f}")
    print(f"  F1 DOWN   : {f1_down:.4f}")
    print(f"  F1 UP     : {f1_up:.4f}")
    print()
    print("  Confusion matrix (rows=actual, cols=predicted):")
    print("                 Pred DOWN   Pred UP")
    print(f"  Actual DOWN  :   {cm[0,0]:5d}       {cm[0,1]:5d}")
    print(f"  Actual UP    :   {cm[1,0]:5d}       {cm[1,1]:5d}")
    print("=" * 60)

    # 7. Save to JSON
    result = {
        "n_samples":        n,
        "accuracy":         round(float(accuracy), 4),
        "f1_macro":         round(float(f1_macro), 4),
        "f1_down":          round(float(f1_down), 4),
        "f1_up":            round(float(f1_up), 4),
        "confusion_matrix": cm.tolist(),   # [[TN, FP], [FN, TP]]
        "date_cutoff":      DATE_CUTOFF,
        "note": (
            "Evaluated on charts unseen during training and epoch selection. "
            "Training: dates <= 2023-12-31. Val (epoch selection): 2024-01-01 – 2024-06-30. "
            "This set: >= 2025-01-01."
        ),
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2))
    logger.info("Results saved to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
