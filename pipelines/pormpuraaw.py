"""pipelines/pormpuraaw.py — Pormpuraaw Parkinsonia analysis pipeline.

Trains the Parkinsonia classifier on the Pormpuraaw training labels
(presence/absence sub-bboxes from pormpuraaw.yaml), then applies it to all
pixels in the scene (pormpuraaw-scene.yaml).

Feature selection based on explore.py diagnostic run (2016–2021):

  Signal        | Separability | Notes
  --------------|-------------|----------------------------------------
  re_p10        |   -4.51     | Strongest — red-edge floor lower on presence
  rec_p         |   +3.09     | NDVI wet/dry amplitude higher on presence
  nir_cv        |   +2.78     | Dry-season NIR variability higher on presence
  ndvi_integral |   -1.66     | Mean annual NDVI lower on presence
  swir_p10      |   -1.51     | SWIR floor lower on presence (borderline)
  peak_doy      |   +0.16     | No discriminating power — excluded

Usage
-----
    python -m pipelines.pormpuraaw
    python -m pipelines.pormpuraaw --no-plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.location import get
from utils.heatmap import plot_prob_heatmaps
from signals import extract_parko_features
from analysis.classifier import ParkoClassifier
from pipelines.common import label_pixels, summarise, save_pixel_ranking

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "pormpuraaw-analysis"

# Feature order: strongest discriminators first.
# re_p10 sign is negative (presence lower) — classifier handles this internally.
FEATURES = ["re_p10", "rec_p", "nir_cv", "ndvi_integral", "swir_p10"]

TRAIN_LOC_ID = "pormpuraaw"        # source of sub-bbox training labels
SCENE_LOC_ID = "pormpuraaw-south"  # full scene to score


def plot_feature_space(scored_df: pd.DataFrame, loc, out_dir: Path) -> list[Path]:
    """Pairwise 2D feature-space scatters for the top-3 features with class ellipses."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    LABELS = {True: "Presence", False: "Absence"}
    top3 = FEATURES[:3]
    pairs = [(top3[i], top3[j]) for i in range(len(top3)) for j in range(i + 1, len(top3))]

    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5))
    if len(pairs) == 1:
        axes = [axes]
    fig.suptitle(f"{loc.name} — pairwise feature projections", fontsize=13, y=1.01)

    for ax, (fx, fy) in zip(axes, pairs):
        unlabelled = scored_df[scored_df["is_presence"].isna()]
        if not unlabelled.empty:
            ax.scatter(unlabelled[fx], unlabelled[fy], s=8, alpha=0.15,
                       color="#aaaaaa", label="Unlabelled")

        for is_pres, colour in [(True, "#2ca02c"), (False, "#ff7f0e")]:
            sub = scored_df[scored_df["is_presence"] == is_pres]
            if sub.empty:
                continue
            ax.scatter(sub[fx], sub[fy], s=8, alpha=0.4, color=colour,
                       label=LABELS[is_pres])
            if len(sub) > 2:
                cov = np.cov(sub[fx].values, sub[fy].values)
                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                w, h = 2 * 1.5 * np.sqrt(vals)
                ax.add_patch(Ellipse(
                    xy=(sub[fx].mean(), sub[fy].mean()), width=w, height=h,
                    angle=theta, edgecolor=colour, facecolor="none",
                    linewidth=1.5, zorder=3,
                ))
                ax.scatter(sub[fx].mean(), sub[fy].mean(), s=80, color=colour,
                           marker="D", edgecolors="black", linewidth=0.8, zorder=4)

        ax.set_xlabel(fx, fontsize=10)
        ax.set_ylabel(fy, fontsize=10)
        ax.tick_params(labelsize=9)

    axes[0].legend(fontsize=9)
    plt.tight_layout()
    p = out_dir / "pormpuraaw_pairwise_2d.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")
    return [p]


def run(plots: bool = True) -> None:
    train_loc = get(TRAIN_LOC_ID)
    scene_loc = get(SCENE_LOC_ID)
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Train on the Pormpuraaw labels
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
    # Score the scene (separate parquet, no overlap with training pixels)
    # ------------------------------------------------------------------
    print(f"\nExtracting features from {scene_loc.parquet_path().name} ...")
    scene_features = extract_parko_features(scene_loc.parquet_path(), scene_loc)

    # Carry over training labels where scene pixels overlap the sub-bboxes
    scene_labelled = label_pixels(scene_features, train_loc)

    print("Scoring scene...")
    scored = clf.score(scene_labelled)

    summarise(scored, scene_loc, show_scene_percentiles=True)
    save_pixel_ranking(scored, out_dir / "pormpuraaw_pixel_ranking.csv", FEATURES)

    if plots:
        plot_prob_heatmaps(scored, scene_loc, out_dir, stem="pormpuraaw")
        plot_feature_space(scored, loc=scene_loc, out_dir=out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    run(plots=not args.no_plots)
