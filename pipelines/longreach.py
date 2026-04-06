"""pipelines/longreach.py — Longreach Parkinsonia analysis pipeline.

Computes features, labels end-member pixels, trains and applies the classifier,
summarises results, and writes diagnostic plots.

Usage
-----
    python -m pipelines.longreach
    python -m pipelines.longreach --no-plots
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "longreach-analysis"
FEATURES = ["nir_cv", "rec_p", "re_p10"]


def label_pixels(features_df: pd.DataFrame, loc) -> pd.DataFrame:
    """Assign is_presence from the location's presence/absence sub_bboxes.

    Returns copy of features_df with is_presence (True / False / NaN).
    Pixels outside any labelled sub-bbox get NaN — scored but not trained on.
    """
    df = features_df.copy()
    df["is_presence"] = pd.NA

    for sub in loc.sub_bboxes.values():
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


def summarise(scored_df: pd.DataFrame, loc) -> None:
    """Print per-class probability statistics and top/bottom 20 pixels."""
    print(f"\n{'='*60}")
    print(f"Site: {loc.name}  ({len(scored_df):,} pixels)")
    print(f"{'='*60}")

    labelled = scored_df[scored_df["is_presence"].notna()]
    if not labelled.empty:
        print("\nProbability by class (mean / median / std):")
        for val, label in [(True, "Presence"), (False, "Absence")]:
            sub = labelled[labelled["is_presence"] == val]["prob_lr"]
            if not sub.empty:
                print(f"  {label:10s}  mean={sub.mean():.3f}  median={sub.median():.3f}  std={sub.std():.3f}")

    rank_cols = ["point_id", "lon", "lat", "is_presence", "prob_lr", "rank"]
    print("\nTop 20 by Parkinsonia probability:")
    print(scored_df.nsmallest(20, "rank")[rank_cols].to_string(index=False))
    print("\nBottom 20 by Parkinsonia probability:")
    print(scored_df.nlargest(20, "rank")[rank_cols].to_string(index=False))


def plot_feature_space(scored_df: pd.DataFrame, loc, out_dir: Path) -> list[Path]:
    """Pairwise 2D feature-space scatters with class ellipses.

    Produces:
      longreach_pairwise_2d.png
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Ellipse

    CLASS_COLOURS = {True: "#2ca02c", False: "#ff7f0e", pd.NA: "#aaaaaa"}
    LABELS = {True: "Presence", False: "Absence", pd.NA: "Unlabelled"}

    pairs = [(FEATURES[i], FEATURES[j]) for i in range(len(FEATURES)) for j in range(i + 1, len(FEATURES))]

    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5))
    fig.suptitle(f"{loc.name} — pairwise feature projections", fontsize=13, y=1.01)

    for ax, (fx, fy) in zip(axes, pairs):
        for is_pres, colour in [(True, "#2ca02c"), (False, "#ff7f0e")]:
            sub = scored_df[scored_df["is_presence"] == is_pres]
            if sub.empty:
                continue
            ax.scatter(sub[fx], sub[fy], s=8, alpha=0.4, color=colour, label=LABELS[is_pres])
            if len(sub) > 2:
                cov = np.cov(sub[fx].values, sub[fy].values)
                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                w, h = 2 * 1.5 * np.sqrt(vals)
                ax.add_patch(Ellipse(
                    xy=(sub[fx].mean(), sub[fy].mean()), width=w, height=h, angle=theta,
                    edgecolor=colour, facecolor="none", linewidth=1.5, zorder=3,
                ))
                ax.scatter(sub[fx].mean(), sub[fy].mean(), s=80, color=colour,
                           marker="D", edgecolors="black", linewidth=0.8, zorder=4)

        unlabelled = scored_df[scored_df["is_presence"].isna()]
        if not unlabelled.empty:
            ax.scatter(unlabelled[fx], unlabelled[fy], s=8, alpha=0.15, color="#aaaaaa", label="Unlabelled")

        ax.set_xlabel(fx, fontsize=10)
        ax.set_ylabel(fy, fontsize=10)
        ax.tick_params(labelsize=9)

    axes[0].legend(fontsize=9)
    plt.tight_layout()
    p = out_dir / "longreach_pairwise_2d.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")
    return [p]


def run(plots: bool = True) -> None:
    loc = get("longreach")
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pixels...")
    raw = pd.read_parquet(loc.parquet_path())

    print("Extracting features...")
    features = extract_parko_features(raw, loc)

    print("Labelling pixels...")
    labelled = label_pixels(features, loc)
    n_pres = (labelled["is_presence"] == True).sum()   # noqa: E712
    n_abs  = (labelled["is_presence"] == False).sum()  # noqa: E712
    print(f"  Presence: {n_pres:,}  Absence: {n_abs:,}  Unlabelled: {(labelled['is_presence'].isna()).sum():,}")

    print("Fitting classifier...")
    train = labelled[labelled["is_presence"].notna()].copy()
    train["is_presence"] = train["is_presence"].astype(bool)
    clf = ParkoClassifier(features=FEATURES)
    clf.fit(train, label_col="is_presence")
    print(clf.summary())

    print("Scoring pixels...")
    scored = clf.score(labelled)

    summarise(scored, loc)

    ranked_path = out_dir / "longreach_pixel_ranking.csv"
    cols = ["point_id", "lon", "lat", "is_presence", "prob_lr", "rank"] + FEATURES
    scored[[c for c in cols if c in scored.columns]].sort_values("rank").to_csv(
        ranked_path, index=False, float_format="%.4f"
    )
    print(f"Saved: {ranked_path}")

    if plots:
        plot_prob_heatmaps(scored, loc, out_dir, stem="longreach")
        plot_feature_space(scored, loc, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    run(plots=not args.no_plots)
