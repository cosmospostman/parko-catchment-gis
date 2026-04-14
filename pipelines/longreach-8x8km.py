"""pipelines/longreach-8x8km.py — Longreach 8×8 km subscene analysis pipeline.

Trains the Parkinsonia classifier on the original Longreach training labels
(presence/absence sub-bboxes from longreach.yaml), then applies it to all
pixels in the 8×8 km subscene (longreach-8x8km.yaml).

Signal params are loaded from longreach-8x8km.yaml (tuned for Longreach).

Usage
-----
    python -m pipelines."longreach-8x8km"
    python -m pipelines."longreach-8x8km" --no-plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils.location import get
from utils.heatmap import plot_prob_heatmaps
from signals import extract_parko_features, NdviIntegralSignal
from analysis.classifier import ParkoClassifier
from pipelines.common import label_pixels, summarise, save_pixel_ranking

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "longreach-8x8"
FEATURES = ["nir_cv", "rec_p", "re_p10", "ndvi_integral"]

# Pre-computed NDVI curve — shared with recession/greenup investigations.
# Build once via research/recession-and-greenup/longreach-recession-greenup.py
# if not present.
CURVE_CACHE = (
    PROJECT_ROOT / "research" / "recession-and-greenup"
    / "longreach-recession-greenup" / "_cache_ndvi_curve.parquet"
)

TRAIN_LOC_ID = "longreach"        # source of sub-bbox training labels
SCENE_LOC_ID = "longreach-8x8km"  # full scene to score



def run(plots: bool = True) -> None:
    train_loc = get(TRAIN_LOC_ID)
    scene_loc = get(SCENE_LOC_ID)
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Train on the original Longreach labels
    # ------------------------------------------------------------------
    print(f"Loading training pixels from {train_loc.parquet_path()} ...")
    train_raw = pd.read_parquet(train_loc.parquet_path())

    print("Extracting features (training set)...")
    train_features = extract_parko_features(train_raw, train_loc)

    if CURVE_CACHE.exists():
        integral_train = NdviIntegralSignal().compute(None, train_loc, _curve=CURVE_CACHE)
        train_features = train_features.merge(
            integral_train[["point_id", "ndvi_integral"]], on="point_id", how="left"
        )
    else:
        print(f"  WARNING: CURVE_CACHE not found — ndvi_integral will be NaN for training set.\n  {CURVE_CACHE}")

    print("Labelling training pixels...")
    train_labelled = label_pixels(train_features, train_loc)
    n_pres = (train_labelled["is_presence"] == True).sum()   # noqa: E712
    n_abs  = (train_labelled["is_presence"] == False).sum()  # noqa: E712
    print(f"  Presence: {n_pres:,}  Absence: {n_abs:,}  Unlabelled: {train_labelled['is_presence'].isna().sum():,}")

    print("Fitting classifier...")
    train_set = train_labelled[train_labelled["is_presence"].notna()].copy()
    train_set["is_presence"] = train_set["is_presence"].astype(bool)
    clf = ParkoClassifier(features=FEATURES)
    clf.fit(train_set, label_col="is_presence")
    print(clf.summary())

    # ------------------------------------------------------------------
    # Score the 8×8 km scene (tuned params auto-loaded from YAML)
    # ------------------------------------------------------------------
    scene_path = scene_loc.parquet_path()
    sorted_path = scene_path.with_name(scene_path.stem + "-by-pixel.parquet")
    if not sorted_path.exists():
        from signals._shared import sort_parquet_by_pixel
        print(f"\nSorting {scene_path.name} by pixel (one-time, ~2 min)...")
        sort_parquet_by_pixel(scene_path, sorted_path)
        print(f"Saved: {sorted_path}")

    print(f"\nExtracting features from {sorted_path.name} (chunked, one row-group at a time)...")
    scene_features = extract_parko_features(sorted_path, scene_loc)

    if CURVE_CACHE.exists():
        print("Computing ndvi_integral from curve cache...")
        integral_scene = NdviIntegralSignal().compute(None, scene_loc, _curve=CURVE_CACHE)
        scene_features = scene_features.merge(
            integral_scene[["point_id", "ndvi_integral"]], on="point_id", how="left"
        )
    else:
        print(f"  WARNING: CURVE_CACHE not found — ndvi_integral will be NaN.\n  {CURVE_CACHE}")

    # Carry over training labels where pixels overlap the sub-bboxes
    scene_labelled = label_pixels(scene_features, train_loc)

    print("Scoring scene...")
    scored = clf.score(scene_labelled)

    summarise(scored, scene_loc)

    save_pixel_ranking(scored, out_dir / "longreach_8x8km_pixel_ranking.csv", FEATURES)

    if plots:
        plot_prob_heatmaps(scored, scene_loc, out_dir, stem="longreach_8x8km")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    run(plots=not args.no_plots)
