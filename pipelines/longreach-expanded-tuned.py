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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "longreach-expanded-tuned"
FEATURES = ["nir_cv", "rec_p", "re_p10"]

TRAIN_LOC_ID  = "longreach"            # source of parquet + sub-bbox labels
SCENE_LOC_ID  = "longreach-expansion"  # source of scene parquet + loc metadata
TUNED_LOC_ID  = "longreach-expansion-tuned"  # carries tuned signal params


def label_pixels(features_df: pd.DataFrame, train_loc) -> pd.DataFrame:
    """Assign is_presence from the training location's presence/absence sub_bboxes."""
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
    labelled = scored_df[scored_df["is_presence"].notna()]
    print(f"\n{'='*60}")
    print(f"Site: {scene_loc.name}  ({len(scored_df):,} pixels)")
    print(f"{'='*60}")

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

    ranked_path = out_dir / "longreach_expanded_tuned_pixel_ranking.csv"
    cols = ["point_id", "lon", "lat", "is_presence", "prob_lr", "rank"] + FEATURES
    scored[[c for c in cols if c in scored.columns]].sort_values("rank").to_csv(
        ranked_path, index=False, float_format="%.4f"
    )
    print(f"Saved: {ranked_path}")

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
