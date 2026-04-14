"""pipelines/longreach-expanded-tuned.py — Longreach expansion bbox, tuned signals.

Same as longreach-expanded.py but uses tuned signal params (red_edge floor_percentile=0.84)
for both training and scoring, via longreach-tuned.yaml and longreach-expansion-tuned.yaml.

Usage
-----
    python -m pipelines.longreach-expanded-tuned
    python -m pipelines.longreach-expanded-tuned --no-plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils.location import get
from utils.heatmap import plot_prob_heatmaps
from signals import extract_parko_features
from signals._shared import load_signal_params
from analysis.classifier import ParkoClassifier
from pipelines.common import label_pixels, summarise, save_pixel_ranking

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "longreach-expanded-tuned"
FEATURES = ["nir_cv", "rec_p", "re_p10"]

TRAIN_LOC_ID  = "longreach"            # source of parquet + sub-bbox labels
SCENE_LOC_ID  = "longreach-expansion"  # source of scene parquet + loc metadata
TUNED_LOC_ID  = "longreach-expansion-tuned"  # carries tuned signal params



def run(plots: bool = True) -> None:
    train_loc = get(TRAIN_LOC_ID)
    scene_loc = get(SCENE_LOC_ID)
    tuned_loc = get(TUNED_LOC_ID)
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tuned signal params — applied consistently to both training and scene
    # feature extraction so the feature space matches.
    tuned_re_params = load_signal_params(tuned_loc, "red_edge")

    # ------------------------------------------------------------------
    # Train on the original Longreach labels
    # ------------------------------------------------------------------
    print(f"Loading training pixels from {train_loc.parquet_path()} ...")
    train_raw = pd.read_parquet(train_loc.parquet_path())

    print("Extracting features (training set)...")
    train_features = extract_parko_features(train_raw, train_loc,
                                            red_edge_params=tuned_re_params)

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
    # Score the full expansion scene
    # ------------------------------------------------------------------
    print(f"\nLoading expansion scene pixels from {scene_loc.parquet_path()} ...")
    scene_raw = pd.read_parquet(scene_loc.parquet_path())

    print("Extracting features (expansion scene)...")
    scene_features = extract_parko_features(scene_raw, scene_loc,
                                            red_edge_params=tuned_re_params)

    scene_labelled = label_pixels(scene_features, train_loc)

    print("Scoring expansion scene...")
    scored = clf.score(scene_labelled)

    summarise(scored, scene_loc)

    save_pixel_ranking(scored, out_dir / "longreach_expanded_tuned_pixel_ranking.csv", FEATURES)

    if plots:
        # Build annotation: white dashed rect marking the training infestation patch
        presence_bbox = next(
            sub.bbox for sub in train_loc.sub_bboxes.values()
            if sub.role == "presence"
        )
        annotations = [dict(
            xy=(presence_bbox[0], presence_bbox[1]),
            width=presence_bbox[2] - presence_bbox[0],
            height=presence_bbox[3] - presence_bbox[1],
            edgecolor="white", linewidth=1.2, linestyle="--",
            label="Training: infestation patch",
        )]
        plot_prob_heatmaps(
            scored, tuned_loc, out_dir,
            stem="longreach_expanded_tuned",
            wms_width=2048,
            annotations=annotations,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    run(plots=not args.no_plots)
