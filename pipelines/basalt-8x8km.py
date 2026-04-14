"""pipelines/basalt-8x8km.py — Basalt 8×8 km scene analysis pipeline.

Trains the Parkinsonia classifier on the original Longreach training labels
(presence/absence sub-bboxes from longreach.yaml), then applies it to all
pixels in the 8×8 km Basalt scene (basalt-8x8km.yaml).

Signal params are loaded from basalt-8x8km.yaml (tune as needed).

Usage
-----
    python -m pipelines."basalt-8x8km"
    python -m pipelines."basalt-8x8km" --no-plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils.location import get
from utils.heatmap import plot_prob_heatmaps
from signals import extract_parko_features
from analysis.classifier import ParkoClassifier
from pipelines.common import label_pixels, summarise, save_pixel_ranking

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "basalt-8x8"
FEATURES = ["nir_cv", "rec_p", "re_p10"]

TRAIN_LOC_ID = "longreach"       # source of sub-bbox training labels
SCENE_LOC_ID = "basalt-8x8km"   # full scene to score



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
    print(f"\nLoading scene pixels from {scene_loc.parquet_path()} ...")
    scene_raw = pd.read_parquet(scene_loc.parquet_path())

    print("Extracting features (scene, tuned params)...")
    scene_features = extract_parko_features(scene_raw, scene_loc)

    # Carry over training labels where pixels overlap the sub-bboxes
    scene_labelled = label_pixels(scene_features, train_loc)

    print("Scoring scene...")
    scored = clf.score(scene_labelled)

    summarise(scored, scene_loc)

    save_pixel_ranking(scored, out_dir / "basalt_8x8km_pixel_ranking.csv", FEATURES)

    if plots:
        plot_prob_heatmaps(scored, scene_loc, out_dir, stem="basalt_8x8km", wms_width=1024)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    run(plots=not args.no_plots)
