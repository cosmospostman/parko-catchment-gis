"""swir_structure_xgb.py — Band-summary-only gradient-boosted tree ablation.

Trains on the same V9 region set using only per-pixel [p5, p95, std] summaries
(33 features: 11 bands × 3 stats). No time-series sequence.

Purpose: test whether SWIR structural signal alone is discriminating, without
the confounds of the Transformer architecture or phenological curve learning.

Uses XGBoost if available, otherwise falls back to sklearn HistGradientBoosting.

Usage
-----
    python analysis/swir_structure_xgb.py
    python analysis/swir_structure_xgb.py --out outputs/swir-xgb
    python analysis/swir_structure_xgb.py --infer-dir data/pixels/longreach-8x8km
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tam.core.dataset import V9_FEATURE_COLS
from tam.experiments.v9_spectral import EXPERIMENT
from tam.utils import label_pixels
from utils.regions import select_regions
from utils.training_collector import tile_ids_for_regions, tile_parquet_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("swir_xgb")

FEATURE_COLS = V9_FEATURE_COLS  # 11 bands
SCL_PURITY_MIN = 0.5


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _add_spectral_indices(df: pd.DataFrame) -> pd.DataFrame:
    if "NDVI" not in df.columns and {"B08", "B04"}.issubset(df.columns):
        denom = df["B08"] + df["B04"]
        df = df.copy()
        df["NDVI"] = np.where(denom != 0, (df["B08"] - df["B04"]) / denom, 0.0)
    if "NDWI" not in df.columns and {"B03", "B08"}.issubset(df.columns):
        denom = df["B03"] + df["B08"]
        df = df.copy()
        df["NDWI"] = np.where(denom != 0, (df["B03"] - df["B08"]) / denom, 0.0)
    return df


def _band_summaries(pixel_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per-pixel [p5, p95, std] for each col. Returns DataFrame indexed by point_id.

    Uses a single numpy percentile pass per group rather than three separate
    pandas groupby aggregations — roughly 3× faster on large observation frames.
    """
    src = pixel_df
    if "source" in pixel_df.columns:
        src = pixel_df[pixel_df["source"] == "S2"]
    if "scl_purity" in src.columns:
        src = src[src["scl_purity"] >= SCL_PURITY_MIN]
    present = [c for c in cols if c in src.columns]

    src = src[["point_id"] + present].sort_values("point_id")
    pids = src["point_id"].to_numpy()
    vals = src[present].to_numpy(dtype=np.float32)

    _, first = np.unique(pids, return_index=True)
    splits = np.split(vals, first[1:])  # list of (n_obs, n_bands) arrays per pixel
    unique_pids = pids[first]

    out_cols = []
    for c in present:
        out_cols += [f"{c}_p5", f"{c}_p95", f"{c}_std"]

    # One percentile call per group (covers all bands at once)
    rows = np.empty((len(unique_pids), len(out_cols)), dtype=np.float32)
    for row_idx, group in enumerate(splits):
        p5_95 = np.percentile(group, [5, 95], axis=0)  # (2, n_bands)
        std    = group.std(axis=0, ddof=1) if len(group) > 1 else np.zeros(len(present))
        for bi in range(len(present)):
            ci = bi * 3
            rows[row_idx, ci]     = p5_95[0, bi]
            rows[row_idx, ci + 1] = p5_95[1, bi]
            rows[row_idx, ci + 2] = std[bi]

    return pd.DataFrame(rows, index=unique_pids, columns=out_cols)


def load_training_summaries(cache_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load V9 training pixels → band summaries + labels.

    Band summaries are cached to cache_dir/train_summaries.parquet on first run
    and reloaded on subsequent runs, skipping the expensive parquet read + groupby.
    """
    cache_path  = cache_dir / "train_summaries.parquet"
    labels_path = cache_dir / "train_labels.parquet"

    if cache_path.exists() and labels_path.exists():
        log.info("Loading cached summaries from %s", cache_path)
        summaries = pd.read_parquet(cache_path)
        labels    = pd.read_parquet(labels_path)["label"]
        labels.index = summaries.index
        log.info(
            "Training (cached): %d pixels, %d presence, %d absence, %d features",
            len(labels), int(labels.sum()), int((labels == 0).sum()), summaries.shape[1],
        )
        return summaries, labels

    regions  = select_regions(EXPERIMENT.region_ids)
    tile_ids = tile_ids_for_regions(EXPERIMENT.region_ids)

    band_chunks:  list[pd.DataFrame] = []
    coord_chunks: list[pd.DataFrame] = []

    for tid in tile_ids:
        path = tile_parquet_path(tid)
        if not path.exists():
            log.error("Missing tile parquet: %s — run training_collector first", path)
            sys.exit(1)
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        # date not needed — summaries are over the full multi-year stack
        extra     = [c for c in ("source", "scl_purity") if c in available]
        band_cols = [c for c in ("B02","B03","B04","B05","B07","B08","B8A","B11","B12",
                                  "NDVI","NDWI") if c in available]
        read_cols = list(dict.fromkeys(["point_id"] + band_cols + extra))

        tile_chunks = []
        for rg in range(pf.metadata.num_row_groups):
            tile_chunks.append(pf.read_row_group(rg, columns=read_cols).to_pandas())
        tile_df = pd.concat(tile_chunks, ignore_index=True)
        tile_df = _add_spectral_indices(tile_df)
        band_chunks.append(tile_df)

        # Read coords (point_id, lon, lat) from row group 0 only — enough for dedup
        coord_cols = [c for c in ("point_id", "lon", "lat") if c in available]
        coord_chunks.append(
            pf.read_row_group(0, columns=coord_cols).to_pandas().drop_duplicates("point_id")
        )
        log.info("Loaded tile %s: %d rows", tid, len(tile_df))

    pixel_df = pd.concat(band_chunks, ignore_index=True)
    coord_df = pd.concat(coord_chunks, ignore_index=True).drop_duplicates("point_id")

    labelled = label_pixels(coord_df, regions).dropna(subset=["is_presence"])
    labels   = labelled.set_index("point_id")["is_presence"].map({True: 1.0, False: 0.0})

    log.info("Computing band summaries for %d unique pixels...", pixel_df["point_id"].nunique())
    summaries = _band_summaries(pixel_df, FEATURE_COLS)
    summaries = summaries.loc[summaries.index.isin(labels.index)]
    labels    = labels.loc[labels.index.isin(summaries.index)]

    log.info(
        "Training: %d pixels, %d presence, %d absence, %d features",
        len(labels), int(labels.sum()), int((labels == 0).sum()), summaries.shape[1],
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    summaries.to_parquet(cache_path)
    pd.DataFrame({"label": labels.values}, index=summaries.index).to_parquet(labels_path)
    log.info("Cached summaries to %s", cache_path)

    return summaries, labels


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def _build_model():
    try:
        from xgboost import XGBClassifier
        log.info("Using XGBoost")
        return XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            early_stopping_rounds=30,
            n_jobs=-1,
            random_state=42,
        ), "xgboost"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        log.info("XGBoost not found — using sklearn HistGradientBoostingClassifier")
        return HistGradientBoostingClassifier(
            max_iter=500,
            max_depth=5,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42,
        ), "hgb"


def train(summaries: pd.DataFrame, labels: pd.Series, out_dir: Path):
    import joblib
    from sklearn.metrics import roc_auc_score
    from sklearn.inspection import permutation_importance

    X = summaries.values.astype(np.float32)
    y = labels.loc[summaries.index].values.astype(np.float32)
    feat_names = summaries.columns.tolist()

    # Etna pixels as val set (match V9 protocol)
    etna_mask = summaries.index.str.startswith("etna_")
    X_val, y_val = X[etna_mask], y[etna_mask]
    X_train, y_train = X[~etna_mask], y[~etna_mask]
    log.info("Train: %d  Val (Etna): %d", len(y_train), len(y_val))

    model, backend = _build_model()

    if backend == "xgboost":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    else:
        model.fit(X_train, y_train)

    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    log.info("Val AUC (Etna): %.4f", val_auc)

    # Feature importances — native for XGBoost, permutation for HGB
    if backend == "xgboost":
        importances = pd.Series(model.feature_importances_, index=feat_names)
    else:
        log.info("Computing permutation importances on val set (n_repeats=10)...")
        pi = permutation_importance(
            model, X_val, y_val, n_repeats=10, scoring="roc_auc",
            random_state=42, n_jobs=-1,
        )
        importances = pd.Series(pi.importances_mean, index=feat_names)

    importances = importances.sort_values(ascending=False)
    log.info("Top-15 features:\n%s", importances.head(15).to_string())

    out_dir.mkdir(parents=True, exist_ok=True)
    importances.to_csv(out_dir / "feature_importances.csv", header=["importance"])

    model_path = out_dir / ("xgb_swir.json" if backend == "xgboost" else "hgb_swir.joblib")
    if backend == "xgboost":
        model.save_model(model_path)
    else:
        joblib.dump(model, model_path)
    log.info("Saved model and importances to %s", out_dir)

    return model, feat_names, val_auc


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer(model, feat_names: list[str], infer_dir: Path, out_dir: Path):
    """Score all parquets in infer_dir and write a probability CSV."""
    parquets = sorted(infer_dir.rglob("*.parquet"))
    parquets = [p for p in parquets if "coords" not in p.name]
    if not parquets:
        log.warning("No parquets found under %s", infer_dir)
        return

    all_rows = []
    for path in parquets:
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        band_cols = [c for c in ("B02","B03","B04","B05","B07","B08","B8A","B11","B12") if c in available]
        read_cols = list(dict.fromkeys(["point_id"] + band_cols + [c for c in ("scl_purity","source") if c in available]))
        chunks = [pf.read_row_group(rg, columns=read_cols).to_pandas()
                  for rg in range(pf.metadata.num_row_groups)]
        df = pd.concat(chunks, ignore_index=True)
        df = _add_spectral_indices(df)
        summaries = _band_summaries(df, FEATURE_COLS)

        # Align columns to training order
        missing = [c for c in feat_names if c not in summaries.columns]
        for c in missing:
            summaries[c] = 0.0
        X = summaries[feat_names].values.astype(np.float32)
        probs = model.predict_proba(X)[:, 1]
        chunk_df = pd.DataFrame({"point_id": summaries.index, "prob": probs})
        all_rows.append(chunk_df)
        log.info("Scored %s: %d pixels", path.name, len(chunk_df))

    result = pd.concat(all_rows, ignore_index=True)
    result = result.groupby("point_id")["prob"].mean().reset_index()
    out_path = out_dir / "longreach_swir_probs.csv"
    result.to_csv(out_path, index=False)
    log.info(
        "Inference complete: %d pixels, prob mean=%.3f, p95=%.3f — written to %s",
        len(result), result["prob"].mean(), result["prob"].quantile(0.95), out_path,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/swir-xgb", type=Path)
    ap.add_argument("--infer-dir", type=Path, default=None,
                    help="Directory of year parquets to score (e.g. data/pixels/longreach-8x8km)")
    ap.add_argument("--skip-train", action="store_true",
                    help="Load saved model instead of retraining")
    args = ap.parse_args()
    out_dir = PROJECT_ROOT / args.out

    if args.skip_train:
        import joblib
        xgb_path = out_dir / "xgb_swir.json"
        hgb_path = out_dir / "hgb_swir.joblib"
        if xgb_path.exists():
            from xgboost import XGBClassifier
            model = XGBClassifier()
            model.load_model(xgb_path)
        else:
            model = joblib.load(hgb_path)
        importances = pd.read_csv(out_dir / "feature_importances.csv", index_col=0)
        feat_names = importances.index.tolist()
        log.info("Loaded model from %s", out_dir)
    else:
        summaries, labels = load_training_summaries(out_dir)
        model, feat_names, _ = train(summaries, labels, out_dir)

    if args.infer_dir:
        infer_dir = PROJECT_ROOT / args.infer_dir if not args.infer_dir.is_absolute() else args.infer_dir
        infer(model, feat_names, infer_dir, out_dir)


if __name__ == "__main__":
    main()
