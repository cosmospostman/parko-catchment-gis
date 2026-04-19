"""pipelines/kowanyama_pormpuraaw.py — Kowanyama scoring with Pormpuraaw-trained classifier.

Trains the Parkinsonia classifier on the Pormpuraaw training labels
(presence/absence sub-bboxes from pormpuraaw.yaml), then applies it to all
pixels in the Kowanyama scene.

Usage
-----
    python -m pipelines.kowanyama_pormpuraaw
    python -m pipelines.kowanyama_pormpuraaw --no-plots
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
OUT_DIR = PROJECT_ROOT / "outputs" / "kowanyama-pormpuraaw"
FEATURES = ["re_p10", "rec_p", "nir_cv", "ndvi_integral", "swir_p10"]

TRAIN_LOC_ID = "pormpuraaw"   # source of sub-bbox training labels
SCENE_LOC_ID = "kowanyama"    # scene to score


def run(plots: bool = True, tile_id: str | None = None) -> None:
    train_loc = get(TRAIN_LOC_ID)
    scene_loc = get(SCENE_LOC_ID)
    out_dir = OUT_DIR
    if tile_id:
        out_dir = out_dir / f"tile-{tile_id.lower()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Train on the Pormpuraaw labels
    # ------------------------------------------------------------------
    print(f"Loading training pixels from {train_loc.parquet_path()} ...")
    train_raw = pd.read_parquet(train_loc.parquet_path())

    print("Extracting features (training set)...")
    train_features = extract_parko_features(
        train_raw, train_loc,
        calibration_path=train_loc.calibration_path(),
    )

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
    # Score the Kowanyama scene
    # ------------------------------------------------------------------
    scene_path = scene_loc.parquet_path()

    tile_msg = f" (tile {tile_id} only)" if tile_id else ""
    print(f"\nExtracting features from {scene_path.name} (chunked, one row-group at a time){tile_msg}...")
    # year_to=2025: the parquet extends into 2026 (partial wet-season data from a later
    # STAC fetch).  That partial year straddles the min_obs_per_year=10 threshold —
    # the 54LWJ tile adds just enough observations north of ~-15.465 to tip all those
    # pixels over 10, while the 54LWH-only south sits at ~9.25.  The resulting patchy
    # 2026 inclusion creates visible north-south stripes in the output.  Capping at
    # 2025 removes the partial year uniformly across the scene.
    scene_features = extract_parko_features(
        scene_path, scene_loc,
        year_to=2025,
        calibration_path=scene_loc.calibration_path(),
        tile_id=tile_id,
    )

    # No training labels exist for Kowanyama — all pixels will be unlabelled
    scene_labelled = label_pixels(scene_features, train_loc)

    # Water mask: pixels with ndvi_integral < 0 are open water (Mitchell River system).
    # The Pormpuraaw classifier was trained on dryland savanna and has never seen water;
    # strongly negative ndvi_integral pushes prob_lr to 1.0, creating a false-positive
    # block across the western floodplain.  Drop them before scoring.
    n_water = (scene_labelled["ndvi_integral"] < 0).sum()
    if n_water:
        print(f"  Masking {n_water:,} water pixels (ndvi_integral < 0) before scoring.")
    scene_labelled = scene_labelled[scene_labelled["ndvi_integral"] >= 0].copy()

    print("Scoring scene...")
    scored = clf.score(scene_labelled)

    summarise(scored, scene_loc, show_scene_percentiles=True)
    stem = f"kowanyama_pormpuraaw_tile_{tile_id.lower()}" if tile_id else "kowanyama_pormpuraaw"
    save_pixel_ranking(scored, out_dir / f"{stem}_pixel_ranking.csv", FEATURES)

    if plots:
        plot_prob_heatmaps(scored, scene_loc, out_dir, stem=stem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--tile", metavar="TILE_ID",
                        help="Restrict scene observations to a single S2 tile (e.g. 54LWH)")
    args = parser.parse_args()
    run(plots=not args.no_plots, tile_id=args.tile)
