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

from utils.location import get
from utils.heatmap import plot_prob_heatmaps
from signals import extract_parko_features
from analysis.classifier import ParkoClassifier
from pipelines.common import label_pixels, summarise, save_pixel_ranking

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "basalt-3x3"
FEATURES = ["nir_cv", "rec_p", "re_p10", "ndvi_integral"]

TRAIN_LOC_ID = "longreach"          # source of sub-bbox training labels
TRAIN_DATA_ID = "longreach-8x8km"  # pixel data for training (wider scene, more years)
SCENE_LOC_ID = "basalt-3x3km"   # full scene to score



def run(plots: bool = True, year_from: int | None = None, year_to: int | None = None) -> None:
    train_loc = get(TRAIN_LOC_ID)
    train_data_loc = get(TRAIN_DATA_ID)
    scene_loc = get(SCENE_LOC_ID)
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if year_from is not None or year_to is not None:
        label = f"{year_from or '?'}–{year_to or '?'}"
        print(f"Filtering to years {label}")

    # ------------------------------------------------------------------
    # Train on the longreach-8x8km pixels, labelled by longreach sub-bboxes
    # ------------------------------------------------------------------
    # Pre-filter to the union of training sub-bboxes so we only load the ~700
    # labelled pixels rather than the full 8×8 km scene (~380,000 pixels).
    sub_bboxes = list(train_loc.sub_bboxes.values())
    train_bbox = (
        min(s.bbox[0] for s in sub_bboxes),  # lon_min
        min(s.bbox[1] for s in sub_bboxes),  # lat_min
        max(s.bbox[2] for s in sub_bboxes),  # lon_max
        max(s.bbox[3] for s in sub_bboxes),  # lat_max
    )
    print(f"Loading training pixels from {train_data_loc.parquet_path()} ...")
    print(f"Extracting features (training set, bbox={train_bbox})...")
    train_features = extract_parko_features(
        train_data_loc.parquet_path(), train_data_loc,
        year_from=year_from, year_to=year_to,
        bbox=train_bbox,
    )

    print("Labelling training pixels...")
    train_labelled = label_pixels(train_features, train_loc)
    n_pres = (train_labelled["is_presence"] == True).sum()   # noqa: E712
    n_abs  = (train_labelled["is_presence"] == False).sum()  # noqa: E712
    print(f"  Presence: {n_pres:,}  Absence: {n_abs:,}  Unlabelled: {train_labelled['is_presence'].isna().sum():,}")

    print("Fitting classifier...")
    train_set = train_labelled[train_labelled["is_presence"].notna()].copy()
    train_set["is_presence"] = train_set["is_presence"].astype(bool)
    active_features = [f for f in FEATURES if train_set[f].notna().any()]
    if len(active_features) < len(FEATURES):
        dropped = [f for f in FEATURES if f not in active_features]
        print(f"  Warning: dropping all-null features from classifier: {dropped}")
    clf = ParkoClassifier(features=active_features)
    clf.fit(train_set, label_col="is_presence")
    print(clf.summary())

    # ------------------------------------------------------------------
    # Score the 3×3 km scene (tuned params auto-loaded from YAML)
    # ------------------------------------------------------------------
    print(f"\nLoading scene pixels from {scene_loc.parquet_path()} ...")
    print("Extracting features (scene, tuned params)...")
    scene_features = extract_parko_features(
        scene_loc.parquet_path(), scene_loc,
        year_from=year_from, year_to=year_to,
    )

    # Carry over training labels where pixels overlap the sub-bboxes
    scene_labelled = label_pixels(scene_features, train_loc)

    print("Scoring scene...")
    scored = clf.score(scene_labelled)

    summarise(scored, scene_loc)

    save_pixel_ranking(scored, out_dir / "basalt_3x3km_pixel_ranking.csv", FEATURES)

    if plots:
        plot_prob_heatmaps(scored, scene_loc, out_dir, stem="basalt_3x3km")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--year-from", type=int, default=None, metavar="YEAR",
                        help="Ignore observations before this year (inclusive)")
    parser.add_argument("--year-to", type=int, default=None, metavar="YEAR",
                        help="Ignore observations after this year (inclusive)")
    args = parser.parse_args()
    run(plots=not args.no_plots, year_from=args.year_from, year_to=args.year_to)
