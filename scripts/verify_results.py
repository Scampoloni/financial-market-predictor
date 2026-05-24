"""
verify_results.py — Reproducibility verification script.

Loads the committed model artifacts and ablation results from the repository
and verifies that the headline metrics match the values reported in the
documentation. No data download or GPU required.

Usage:
    python scripts/verify_results.py

Expected output (values must match docs/documentation.md Section 2A.5):
    Config A: Test F1=0.4970  Test Acc=0.4971
    Config B: Test F1=0.4826  Test Acc=0.4842
    Config C: Test F1=0.4861  Test Acc=0.4863
    Config D: Test F1=0.4850  Test Acc=0.4850
    21-day model: LightGBM val F1=0.4482, test F1=0.4723
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Expected values from documentation
EXPECTED = {
    "A": {"test_f1_macro": 0.4970, "test_accuracy": 0.4971, "n_features": 28},
    "B": {"test_f1_macro": 0.4826, "test_accuracy": 0.4842, "n_features": 56},
    "C": {"test_f1_macro": 0.4861, "test_accuracy": 0.4863, "n_features": 66},
    "D": {"test_f1_macro": 0.4850, "test_accuracy": 0.4850, "n_features": 66},
}
TOLERANCE = 0.001  # ±0.001 F1 allowed for float rounding


def _check(label: str, got: float, expected: float, tol: float = TOLERANCE) -> bool:
    ok = abs(got - expected) <= tol
    status = "OK" if ok else "FAIL"
    print(f"  {status}  {label}: got {got:.4f}  expected {expected:.4f}")
    return ok


def verify_ablation() -> bool:
    path = ROOT / "data" / "processed" / "ablation_results.json"
    if not path.exists():
        print(f"ERROR: {path} not found")
        return False

    results = json.loads(path.read_text())
    print("\n=== Ablation Study (5-day horizon) ===")
    all_ok = True
    for cfg in ("A", "B", "C", "D"):
        if cfg not in results:
            print(f"  FAIL  Config {cfg}: missing from ablation_results.json")
            all_ok = False
            continue
        r = results[cfg]
        exp = EXPECTED[cfg]
        print(f"\n  Config {cfg} ({r.get('n_features', '?')} features):")
        for key, exp_val in exp.items():
            got_val = r.get(key, float("nan"))
            ok = _check(key, got_val, exp_val)
            all_ok = all_ok and ok
    return all_ok


def verify_model_artifact() -> bool:
    path = ROOT / "models" / "stacking_final_D.pkl"
    if not path.exists():
        # Try legacy filename
        path = ROOT / "models" / "stacking_final.pkl"
    if not path.exists():
        print("\nERROR: No model artifact found in models/")
        return False

    print(f"\n=== Model Artifact ({path.name}) ===")
    with open(path, "rb") as f:
        bundle = pickle.load(f)

    n_feats = len(bundle.get("feature_cols", []))
    model_type = bundle.get("best_model_type", "unknown")
    print(f"  Model type   : {model_type}")
    print(f"  Feature count: {n_feats}  (expected 66)")
    ok = n_feats == 66
    print(f"  {'OK' if ok else 'FAIL'}  feature count")
    return ok


def verify_21d_model() -> bool:
    path = ROOT / "models" / "model_21d.pkl"
    if not path.exists():
        print(f"\nERROR: {path} not found")
        return False

    print("\n=== 21-Day Model ===")
    with open(path, "rb") as f:
        bundle = pickle.load(f)

    val_f1 = bundle.get("val_f1_macro", float("nan"))
    test_f1 = bundle.get("test_f1_macro", float("nan"))
    best_model = bundle.get("best_model", "unknown")
    n_feats = len(bundle.get("feature_cols", []))

    print(f"  Best model   : {best_model}")
    print(f"  Feature count: {n_feats}  (expected 66)")
    ok1 = _check("val_f1_macro", val_f1, 0.4482)
    ok2 = _check("test_f1_macro", test_f1, 0.4723)
    ok3 = n_feats == 66
    print(f"  {'OK' if ok3 else 'FAIL'}  feature count")
    return ok1 and ok2 and ok3


def main() -> None:
    print("Financial Market Predictor — Results Verification")
    print("=" * 52)
    print(f"Repo root: {ROOT}")

    ok_ablation = verify_ablation()
    ok_model    = verify_model_artifact()
    ok_21d      = verify_21d_model()

    print("\n" + "=" * 52)
    if ok_ablation and ok_model and ok_21d:
        print("ALL CHECKS PASSED — results match documentation.")
    else:
        print("SOME CHECKS FAILED — see above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
