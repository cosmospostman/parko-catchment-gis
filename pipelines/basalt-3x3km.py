"""pipelines/basalt-3x3km.py — Basalt 3×3 km scene analysis pipeline.

Trains the Parkinsonia classifier on the original Longreach training labels
(presence/absence sub-bboxes from longreach.yaml), then applies it to all
pixels in the 3×3 km Basalt scene (basalt-3x3km.yaml).

Signal params are loaded from basalt-3x3km.yaml (tune as needed).

Usage
-----
    python -m pipelines."basalt-3x3km"
    python -m pipelines."basalt-3x3km" --no-plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils.location import get
from utils.heatmap import plot_prob_heatmaps
from signals import extract_parko_features
from analysis.classifier import ParkoClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "basalt-3x3"
FEATURES = ["nir_cv", "rec_p", "re_p10"]

TRAIN_LOC_ID = "longreach"       # source of sub-bbox training labels
SCENE_LOC_ID = "basalt-3x3km"   # full scene to score


def label_pixels(features_df: pd.DataFrame, train_loc) -> pd.DataFrame:
    """Assign is_presence from the training location's presence/absence sub_bboxes.

    Pixels outside any labelled sub-bbox get NaN — scored but not trained on.
    """
    df = features_df.copy()
    df["is_presence"] = pd.NA

    for sub in train_loc.sub_bboxes.values():
        lon_min, lat_min, lon_max, lat_max = sub.bbox
        mask = (
            df["lon"].between(lon_min, lon_max) &
            df["lat"].between(lat_min, lat_max)
        )
        if sub.role == "presence":
            df.loc[mask, "is_presence"] = True
        elif sub.role == "absence":
            df.loc[mask, "is_presence"] = False

    return df


def summarise(scored_df: pd.DataFrame, scene_loc) -> None:
    """Print per-class probability statistics."""
    print(f"\n{'='*60}")
    print(f"Site: {scene_loc.name}  ({len(scored_df):,} pixels)")
    print(f"{'='*60}")

    labelled = scored_df[scored_df["is_presence"].notna()]
    if not labelled.empty:
        print("\nProbability by class (mean / median / std):")
        for val, label in [(True, "Presence"), (False, "Absence")]:
            sub = labelled[labelled["is_presence"] == val]["prob_lr"]
            if not sub.empty:
                print(f"  {label:10s}  mean={sub.mean():.3f}  median={sub.median():.3f}  std={sub.std():.3f}")

    all_scored = scored_df["prob_lr"].dropna()
    print(f"\nFull scene  ({len(all_scored):,} scored pixels):")
    print(f"  mean={all_scored.mean():.3f}  median={all_scored.median():.3f}  std={all_scored.std():.3f}")
    for pct in (75, 90, 95):
        print(f"  p{pct}={all_scored.quantile(pct/100):.3f}")


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
    # Score the 3×3 km scene (tuned params auto-loaded from YAML)
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

    ranked_path = out_dir / "basalt_3x3km_pixel_ranking.csv"
    cols = ["point_id", "lon", "lat", "is_presence", "prob_lr", "rank"] + FEATURES
    scored[[c for c in cols if c in scored.columns]].sort_values("rank").to_csv(
        ranked_path, index=False, float_format="%.4f"
    )
    print(f"Saved: {ranked_path}")

    if plots:
        plot_prob_heatmaps(scored, scene_loc, out_dir, stem="basalt_3x3km", wms_width=1024)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    run(plots=not args.no_plots)
