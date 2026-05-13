"""woody-classifier/score.py — load model → score pixel parquets → write prob parquet.

Usage
-----
    python woody-classifier/score.py <parquet_or_dir> [<parquet_or_dir> ...]
    python woody-classifier/score.py data/pixels/longreach-8x8km
    python woody-classifier/score.py data/pixels/longreach-8x8km --out outputs/woody-classifier/scores
    python woody-classifier/score.py data/pixels/longreach-8x8km --threshold 0.5

Output
------
One parquet per input parquet written to --out, named <stem>_woody_probs.parquet,
with columns: point_id, prob_woody (float32).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util as _ilu
def _import_sibling(name: str):
    spec = _ilu.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_features_mod = _import_sibling("features")
compute_woody_features = _features_mod.compute_woody_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("woody.score")

_DEFAULT_MODEL_DIR = PROJECT_ROOT / "outputs" / "woody-classifier"
_DEFAULT_OUT       = _DEFAULT_MODEL_DIR / "scores"
_THRESHOLD_HIGH    = 0.85


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_dir: Path):
    """Load XGBoost or sklearn model from model_dir. Returns (model, feat_names, backend)."""
    feat_path = model_dir / "feature_names.json"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"feature_names.json not found in {model_dir}. "
            "Run woody-classifier/train.py first."
        )
    feat_names = json.loads(feat_path.read_text())

    xgb_path  = model_dir / "model.json"
    hgb_path  = model_dir / "model.joblib"

    if xgb_path.exists():
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model(xgb_path)
        return model, feat_names, "xgboost"
    elif hgb_path.exists():
        import joblib
        model = joblib.load(hgb_path)
        return model, feat_names, "hgb"
    else:
        raise FileNotFoundError(
            f"No model file found in {model_dir} (expected model.json or model.joblib)."
        )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_parquet(
    parquet_path: Path,
    model,
    feat_names: list[str],
    out_dir: Path,
    threshold: float = _THRESHOLD_HIGH,
) -> Path:
    """Score one pixel parquet and write prob_woody parquet to out_dir."""
    pf = pq.ParquetFile(parquet_path)
    available = set(pf.schema_arrow.names)

    want = ["point_id", "date", "source", "scl_purity",
            "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
            "vh", "vv"]
    read_cols = [c for c in want if c in available]

    chunks = [pf.read_row_group(rg, columns=read_cols).to_pandas()
              for rg in range(pf.metadata.num_row_groups)]
    if not chunks:
        log.warning("Empty parquet: %s", parquet_path)
        return None

    df = pd.concat(chunks, ignore_index=True)
    if df.empty:
        log.warning("No rows in %s", parquet_path)
        return None

    feats = compute_woody_features(df)
    if feats.empty:
        log.warning("No features computed for %s", parquet_path)
        return None

    # Align columns to training order; fill missing with 0 (feature mean in normalised space)
    X = np.zeros((len(feats), len(feat_names)), dtype=np.float32)
    for j, name in enumerate(feat_names):
        if name in feats.columns:
            col = feats[name].values.astype(np.float32)
            X[:, j] = np.where(np.isnan(col), 0.0, col)

    probs = model.predict_proba(X)[:, 1].astype(np.float32)

    result = pd.DataFrame({
        "point_id":  feats.index.to_numpy(),
        "prob_woody": probs,
    })

    above = int((probs >= threshold).sum())
    log.info(
        "Scored %s: %d pixels  prob mean=%.3f p95=%.3f  above_%.2f=%d (%.1f%%)",
        parquet_path.name, len(result),
        probs.mean(), np.percentile(probs, 95),
        threshold, above, 100 * above / max(len(result), 1),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{parquet_path.stem}_woody_probs.parquet"
    table = pa.Table.from_pandas(result, preserve_index=False)
    pq.write_table(table, out_path)
    return out_path


def score_dir_or_file(
    path: Path,
    model,
    feat_names: list[str],
    out_dir: Path,
    threshold: float = _THRESHOLD_HIGH,
) -> list[Path]:
    if path.is_file():
        parquets = [path]
    else:
        parquets = sorted(path.rglob("*.parquet"))
        parquets = [p for p in parquets if "coords" not in p.name and "woody_probs" not in p.name]

    if not parquets:
        log.warning("No parquets found under %s", path)
        return []

    written = []
    for p in parquets:
        out = score_parquet(p, model, feat_names, out_dir, threshold)
        if out:
            written.append(out)
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Score pixel parquets with the woody-classifier model.")
    ap.add_argument("inputs", nargs="+", type=Path,
                    help="Parquet files or directories containing parquets to score")
    ap.add_argument("--model-dir", type=Path, default=_DEFAULT_MODEL_DIR,
                    help=f"Directory with model.json and feature_names.json (default: {_DEFAULT_MODEL_DIR})")
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT,
                    help=f"Output directory for prob parquets (default: {_DEFAULT_OUT})")
    ap.add_argument("--threshold", type=float, default=_THRESHOLD_HIGH,
                    help=f"Log threshold for reporting (default: {_THRESHOLD_HIGH})")
    args = ap.parse_args()

    model_dir = PROJECT_ROOT / args.model_dir if not args.model_dir.is_absolute() else args.model_dir
    out_dir   = PROJECT_ROOT / args.out       if not args.out.is_absolute()       else args.out

    model, feat_names, backend = load_model(model_dir)
    log.info("Loaded %s model from %s  (%d features)", backend, model_dir, len(feat_names))

    all_written = []
    for inp in args.inputs:
        path = PROJECT_ROOT / inp if not inp.is_absolute() else inp
        written = score_dir_or_file(path, model, feat_names, out_dir, args.threshold)
        all_written.extend(written)

    log.info("Done. %d output parquet(s) written.", len(all_written))
    for p in all_written:
        print(p)


if __name__ == "__main__":
    main()
