"""woody-classifier/evaluate.py — AUC, precision/recall curve, feature importances.

Usage
-----
    python woody-classifier/evaluate.py
    python woody-classifier/evaluate.py --model-dir outputs/woody-classifier
    python woody-classifier/evaluate.py --plot        # save PR curve and importance chart

Reads val_summaries.parquet and val_labels.parquet from --model-dir.
Prints a summary table; optionally writes PNG plots.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util as _ilu
def _import_sibling(name: str):
    spec = _ilu.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_score_mod = _import_sibling("score")
_train_mod = _import_sibling("train")
load_model              = _score_mod.load_model
THRESHOLD_HIGH_PRECISION = _train_mod.THRESHOLD_HIGH_PRECISION

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("woody.evaluate")

_DEFAULT_MODEL_DIR = PROJECT_ROOT / "outputs" / "woody-classifier"

# Verification targets from docs/WOODY-CLASSIFIER.md
_TARGET_AUC          = 0.92
_TARGET_RECALL_BARE  = 0.10   # <10% bare soil passing at 0.85
_TARGET_RECALL_WOODY = 0.80   # >80% of Parkinsonia pixels passing at 0.85


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _threshold_metrics(y: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-6)
    fpr       = fp / max(fp + tn, 1)
    return dict(threshold=threshold, tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall, f1=f1, fpr=fpr)


def _feature_importances(model, feat_names: list[str], backend: str,
                          X_val: np.ndarray, y_val: np.ndarray) -> pd.Series:
    if backend == "xgboost":
        imp = pd.Series(model.feature_importances_, index=feat_names)
    else:
        from sklearn.inspection import permutation_importance
        log.info("Computing permutation importances (n_repeats=10)...")
        pi = permutation_importance(
            model, X_val, y_val, n_repeats=10, scoring="roc_auc",
            random_state=42, n_jobs=-1,
        )
        imp = pd.Series(pi.importances_mean, index=feat_names)
    return imp.sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_pr_curve(y: np.ndarray, probs: np.ndarray, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, average_precision_score
    except ImportError:
        log.warning("matplotlib not available — skipping PR curve plot")
        return

    precision, recall, thresholds = precision_recall_curve(y, probs)
    ap = average_precision_score(y, probs)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, lw=1.5, label=f"AP={ap:.3f}")

    # Mark the two key thresholds
    for thr in [0.5, THRESHOLD_HIGH_PRECISION]:
        m = _threshold_metrics(y, probs, thr)
        ax.scatter(m["recall"], m["precision"], zorder=5,
                   label=f"thr={thr}  P={m['precision']:.2f} R={m['recall']:.2f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Woody Classifier — Precision-Recall Curve (val)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("PR curve saved to %s", out_path)


def _plot_importances(importances: pd.Series, out_path: Path, top_n: int = 20) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — skipping importance plot")
        return

    top = importances.head(top_n)
    fig, ax = plt.subplots(figsize=(7, 0.4 * top_n + 1))
    y_pos = np.arange(len(top))
    ax.barh(y_pos, top.values[::-1])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top.index[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Woody Classifier — Top {top_n} Features")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Importance chart saved to %s", out_path)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(model_dir: Path, plot: bool = False) -> dict:
    from sklearn.metrics import roc_auc_score

    model, feat_names, backend = load_model(model_dir)

    va_sum_path = model_dir / "val_summaries.parquet"
    va_lbl_path = model_dir / "val_labels.parquet"
    if not va_sum_path.exists() or not va_lbl_path.exists():
        raise FileNotFoundError(
            f"Val data not found in {model_dir}. Run woody-classifier/train.py first."
        )

    va_sum = pd.read_parquet(va_sum_path)
    va_lbl = pd.read_parquet(va_lbl_path)["label"]
    va_lbl.index = va_sum.index

    X_val = np.zeros((len(va_sum), len(feat_names)), dtype=np.float32)
    for j, name in enumerate(feat_names):
        if name in va_sum.columns:
            col = va_sum[name].values.astype(np.float32)
            X_val[:, j] = np.where(np.isnan(col), 0.0, col)

    y_val  = va_lbl.values.astype(np.float32)
    probs  = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, probs)

    m50 = _threshold_metrics(y_val, probs, 0.50)
    m85 = _threshold_metrics(y_val, probs, THRESHOLD_HIGH_PRECISION)

    # Verification targets
    auc_ok    = val_auc >= _TARGET_AUC
    recall_ok = m85["recall"] >= _TARGET_RECALL_WOODY
    fpr_ok    = m85["fpr"]    <= _TARGET_RECALL_BARE

    print()
    print(f"  Val AUC : {val_auc:.4f}  {'OK' if auc_ok else 'FAIL (target >= 0.92)'}")
    print()
    print(f"  Threshold 0.50 — precision={m50['precision']:.3f}  recall={m50['recall']:.3f}  "
          f"f1={m50['f1']:.3f}  fpr={m50['fpr']:.3f}")
    print(f"  Threshold {THRESHOLD_HIGH_PRECISION:.2f} — precision={m85['precision']:.3f}  "
          f"recall={m85['recall']:.3f}  f1={m85['f1']:.3f}  fpr={m85['fpr']:.3f}")
    print(f"    Recall on woody  {'OK' if recall_ok else 'FAIL'} (target ≥ {_TARGET_RECALL_WOODY})")
    print(f"    FPR on non-woody {'OK' if fpr_ok else 'FAIL'} (target ≤ {_TARGET_RECALL_BARE})")

    # Feature importances
    importances = _feature_importances(model, feat_names, backend, X_val, y_val)
    (model_dir / "feature_importances.csv").write_text(
        importances.to_csv(header=["importance"])
    )
    print()
    print(f"  Top-10 features:")
    for name, val in importances.head(10).items():
        print(f"    {name:<22} {val:.6f}")

    # Key feature check: B11_p5 and s1_mean_vh_dry in top 5
    top5 = set(importances.head(5).index)
    for key in ["B11_p5", "s1_mean_vh_dry"]:
        status = "OK" if key in top5 else "NOTE (not in top 5)"
        print(f"  {key} in top 5: {status}")

    if plot:
        _plot_pr_curve(y_val, probs, model_dir / "pr_curve.png")
        _plot_importances(importances, model_dir / "feature_importances.png")

    return {
        "val_auc":         val_auc,
        "precision_0.85":  m85["precision"],
        "recall_0.85":     m85["recall"],
        "fpr_0.85":        m85["fpr"],
        "auc_ok":          auc_ok,
        "recall_ok":       recall_ok,
        "fpr_ok":          fpr_ok,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the woody-classifier model on the val set.")
    ap.add_argument("--model-dir", type=Path, default=_DEFAULT_MODEL_DIR,
                    help=f"Directory with model and val parquets (default: {_DEFAULT_MODEL_DIR})")
    ap.add_argument("--plot", action="store_true",
                    help="Save PR curve and feature importance chart as PNGs")
    args = ap.parse_args()

    model_dir = PROJECT_ROOT / args.model_dir if not args.model_dir.is_absolute() else args.model_dir
    results = evaluate(model_dir, plot=args.plot)

    passed = sum(1 for k in ["auc_ok", "recall_ok", "fpr_ok"] if results[k])
    print()
    print(f"  {passed}/3 verification targets passed.")
    sys.exit(0 if passed == 3 else 1)


if __name__ == "__main__":
    main()
