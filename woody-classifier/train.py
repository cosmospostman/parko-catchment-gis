"""woody-classifier/train.py — load data → fit XGBoost → save model.

Usage
-----
    python woody-classifier/train.py
    python woody-classifier/train.py --out outputs/woody-classifier
    python woody-classifier/train.py --no-cache   # ignore cached summaries

Reads training regions from data/locations/woody-classifier.yaml.
Val regions are those with tags containing "val"; the rest are train.

Output written to --out (default: outputs/woody-classifier/):
    model.json          XGBoost model (or model.joblib for sklearn fallback)
    feature_names.json  ordered list of feature names for inference
    train_summaries.parquet   cached pixel summaries
    train_labels.parquet      cached labels
    val_summaries.parquet
    val_labels.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.regions import load_regions, TrainingRegion
from utils.training_collector import _region_parquet_path

# woody-classifier has a hyphen so it can't be imported as a normal package;
# import siblings via importlib using the directory path.
import importlib.util as _ilu
def _import_sibling(name: str):
    spec = _ilu.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_features_mod = _import_sibling("features")
compute_woody_features = _features_mod.compute_woody_features
WOODY_FEATURE_NAMES    = _features_mod.WOODY_FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("woody.train")

_YAML = PROJECT_ROOT / "data" / "locations" / "woody-classifier.yaml"
_DEFAULT_OUT = PROJECT_ROOT / "outputs" / "woody-classifier"

# Threshold for the high-precision mask
THRESHOLD_HIGH_PRECISION = 0.85


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_region_pixels(region: TrainingRegion) -> pd.DataFrame | None:
    path = _region_parquet_path(region.id)
    if not path.exists():
        log.warning("Missing parquet for region %s — skipping (%s)", region.id, path)
        return None
    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    want = ["point_id", "date", "source", "scl_purity", "lon", "lat",
            "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
            "vh", "vv"]
    read_cols = [c for c in want if c in available]
    chunks = [pf.read_row_group(rg, columns=read_cols).to_pandas()
              for rg in range(pf.metadata.num_row_groups)]
    return pd.concat(chunks, ignore_index=True) if chunks else None


def _build_summaries(
    regions: list[TrainingRegion],
) -> tuple[pd.DataFrame, pd.Series]:
    """Load pixel parquets for regions and compute woody features per pixel."""
    feat_chunks: list[pd.DataFrame] = []
    label_chunks: list[pd.Series] = []

    for region in regions:
        df = _load_region_pixels(region)
        if df is None or df.empty:
            continue
        feats = compute_woody_features(df)
        if feats.empty:
            continue
        feat_chunks.append(feats)
        labels = pd.Series(
            1.0 if region.label == "presence" else 0.0,
            index=feats.index,
            name="label",
        )
        label_chunks.append(labels)

    if not feat_chunks:
        raise RuntimeError("No training data loaded — run training_collector first.")

    summaries = pd.concat(feat_chunks)
    summaries = summaries[~summaries.index.duplicated(keep="first")]
    labels = pd.concat(label_chunks)
    labels = labels[~labels.index.duplicated(keep="first")]
    labels = labels.loc[summaries.index]
    return summaries, labels


def load_splits(
    out_dir: Path,
    no_cache: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Return (train_X, train_y, val_X, val_y), using cache when available."""
    tr_sum_path = out_dir / "train_summaries.parquet"
    tr_lbl_path = out_dir / "train_labels.parquet"
    va_sum_path = out_dir / "val_summaries.parquet"
    va_lbl_path = out_dir / "val_labels.parquet"

    if (not no_cache and
            tr_sum_path.exists() and tr_lbl_path.exists() and
            va_sum_path.exists() and va_lbl_path.exists()):
        log.info("Loading cached summaries from %s", out_dir)
        tr_sum = pd.read_parquet(tr_sum_path)
        tr_lbl = pd.read_parquet(tr_lbl_path)["label"]
        tr_lbl.index = tr_sum.index
        va_sum = pd.read_parquet(va_sum_path)
        va_lbl = pd.read_parquet(va_lbl_path)["label"]
        va_lbl.index = va_sum.index
        log.info(
            "Train: %d px (%d presence, %d absence)  Val: %d px (%d presence, %d absence)",
            len(tr_lbl), int(tr_lbl.sum()), int((tr_lbl == 0).sum()),
            len(va_lbl), int(va_lbl.sum()), int((va_lbl == 0).sum()),
        )
        return tr_sum, tr_lbl, va_sum, va_lbl

    all_regions = load_regions(_YAML)
    train_regions = [r for r in all_regions if "val" not in r.tags]
    val_regions   = [r for r in all_regions if "val" in r.tags]

    log.info("Building training summaries (%d regions)...", len(train_regions))
    tr_sum, tr_lbl = _build_summaries(train_regions)
    log.info("Building val summaries (%d regions)...", len(val_regions))
    va_sum, va_lbl = _build_summaries(val_regions)

    out_dir.mkdir(parents=True, exist_ok=True)
    tr_sum.to_parquet(tr_sum_path)
    pd.DataFrame({"label": tr_lbl.values}, index=tr_sum.index).to_parquet(tr_lbl_path)
    va_sum.to_parquet(va_sum_path)
    pd.DataFrame({"label": va_lbl.values}, index=va_sum.index).to_parquet(va_lbl_path)

    log.info(
        "Train: %d px (%d presence, %d absence)  Val: %d px (%d presence, %d absence)",
        len(tr_lbl), int(tr_lbl.sum()), int((tr_lbl == 0).sum()),
        len(va_lbl), int(va_lbl.sum()), int((va_lbl == 0).sum()),
    )
    return tr_sum, tr_lbl, va_sum, va_lbl


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _build_model(scale_pos_weight: float):
    try:
        from xgboost import XGBClassifier
        log.info("Using XGBoost")
        return XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            early_stopping_rounds=30,
            n_jobs=-1,
            random_state=42,
        ), "xgboost"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        log.info("XGBoost not found — using HistGradientBoostingClassifier")
        return HistGradientBoostingClassifier(
            max_iter=500,
            max_depth=5,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42,
        ), "hgb"


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(
    tr_sum: pd.DataFrame,
    tr_lbl: pd.Series,
    va_sum: pd.DataFrame,
    va_lbl: pd.Series,
    out_dir: Path,
) -> tuple:
    from sklearn.metrics import roc_auc_score

    feat_names = WOODY_FEATURE_NAMES
    X_train = tr_sum[feat_names].values.astype(np.float32)
    y_train = tr_lbl.loc[tr_sum.index].values.astype(np.float32)
    X_val   = va_sum[feat_names].values.astype(np.float32)
    y_val   = va_lbl.loc[va_sum.index].values.astype(np.float32)

    n_absence  = int((y_train == 0).sum())
    n_presence = int((y_train == 1).sum())
    scale_pos_weight = n_absence / max(n_presence, 1)

    model, backend = _build_model(scale_pos_weight)

    if backend == "xgboost":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    else:
        model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    val_auc   = roc_auc_score(y_val, val_probs)
    log.info("Val AUC: %.4f", val_auc)

    # High-precision threshold metrics
    preds_85 = (val_probs >= THRESHOLD_HIGH_PRECISION).astype(int)
    tp = int(((preds_85 == 1) & (y_val == 1)).sum())
    fp = int(((preds_85 == 1) & (y_val == 0)).sum())
    fn = int(((preds_85 == 0) & (y_val == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    log.info(
        "Threshold=%.2f  precision=%.3f  recall=%.3f  (tp=%d fp=%d fn=%d)",
        THRESHOLD_HIGH_PRECISION, precision, recall, tp, fp, fn,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    if backend == "xgboost":
        model_path = out_dir / "model.json"
        model.save_model(model_path)
    else:
        import joblib
        model_path = out_dir / "model.joblib"
        joblib.dump(model, model_path)

    (out_dir / "feature_names.json").write_text(json.dumps(feat_names, indent=2))
    log.info("Model saved to %s", model_path)

    return model, feat_names, val_auc, backend


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Train woody-classifier stage-1 XGBoost mask.")
    ap.add_argument("--out", default=str(_DEFAULT_OUT), type=Path,
                    help="Output directory (default: outputs/woody-classifier)")
    ap.add_argument("--no-cache", action="store_true",
                    help="Ignore cached summaries and recompute from parquets")
    args = ap.parse_args()

    out_dir = PROJECT_ROOT / args.out if not args.out.is_absolute() else args.out
    tr_sum, tr_lbl, va_sum, va_lbl = load_splits(out_dir, no_cache=args.no_cache)
    train(tr_sum, tr_lbl, va_sum, va_lbl, out_dir)


if __name__ == "__main__":
    main()
