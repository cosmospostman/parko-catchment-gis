"""pipelines/frenchs_stage1.py — Frenchs Stage 1: woody vs non-woody classifier.

Trains a binary classifier on end-member pixels:
  Positive (woody)     — sub_bboxes with role "woody"
  Negative (non-woody) — sub_bboxes with role "absence"

Features: re_p10 and ndvi_integral (best separability from explore.py: +42.3 / +46.5).

Outputs
-------
  outputs/frenchs-stage1/frenchs_stage1_prob_vs_imagery.png
  outputs/frenchs-stage1/frenchs_stage1_prob_black.png
  outputs/frenchs-stage1/frenchs_stage1_pairwise_2d.png
  outputs/frenchs-stage1/frenchs_stage1_pixel_ranking.csv

Usage
-----
    python -m pipelines.frenchs_stage1
    python -m pipelines.frenchs_stage1 --no-plots
    python -m pipelines.frenchs_stage1 --year-from 2020 --year-to 2023
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.location import get
from utils.heatmap import plot_prob_heatmaps
from signals import extract_parko_features, SwirSignal, NdviIntegralSignal, QualityParams
from signals._shared import ensure_pixel_sorted
from analysis.classifier import ParkoClassifier
from pipelines.common import summarise, save_pixel_ranking

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "frenchs-stage1"

# Stage 1 features — dry-season only signals, flood-robust
FEATURES = ["re_p10", "swir_p10", "nir_cv"]

# Same tuned params as the main frenchs pipeline
SWIR_PARAMS = SwirSignal.Params(quality=QualityParams(), floor_percentile=0.20)
INTEGRAL_PARAMS = NdviIntegralSignal.Params(quality=QualityParams(), smooth_days=15)


def label_woody(features_df: pd.DataFrame, loc) -> pd.DataFrame:
    """Label pixels using woody/absence sub-bbox roles.

    woody   → is_woody = True
    absence → is_woody = False
    presence (Parkinsonia) and unlabelled → NaN (scored but not trained on)
    """
    df = features_df.copy()
    df["is_woody"] = pd.NA

    for sub in loc.sub_bboxes.values():
        lon_min, lat_min, lon_max, lat_max = sub.bbox
        mask = (
            df["lon"].between(lon_min, lon_max) &
            df["lat"].between(lat_min, lat_max)
        )
        if sub.role == "woody":
            df.loc[mask, "is_woody"] = True
        elif sub.role == "absence":
            df.loc[mask, "is_woody"] = False

    return df


def summarise_stage1(scored_df: pd.DataFrame, loc) -> None:
    """Print per-class woody probability statistics."""
    print(f"\n{'='*60}")
    print(f"Site: {loc.name} — Stage 1 woody mask  ({len(scored_df):,} pixels)")
    print(f"{'='*60}")

    labelled = scored_df[scored_df["is_woody"].notna()]
    if not labelled.empty:
        print("\nP(woody) by class (mean / median / std):")
        for val, label in [(True, "Woody"), (False, "Non-woody")]:
            sub = labelled[labelled["is_woody"] == val]["prob_lr"]
            if not sub.empty:
                print(f"  {label:12s}  mean={sub.mean():.3f}  median={sub.median():.3f}  std={sub.std():.3f}")

    all_scored = scored_df["prob_lr"].dropna()
    print(f"\nFull scene  ({len(all_scored):,} scored pixels):")
    print(f"  mean={all_scored.mean():.3f}  median={all_scored.median():.3f}  std={all_scored.std():.3f}")
    for pct in (75, 90, 95):
        print(f"  p{pct}={all_scored.quantile(pct/100):.3f}")

    # Scene-level woody fraction at candidate thresholds
    print("\nWoody pixel counts at probability thresholds:")
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        n = (all_scored >= thr).sum()
        pct_scene = 100 * n / len(all_scored)
        print(f"  p >= {thr:.1f}:  {n:,} pixels  ({pct_scene:.1f}% of scene)")


def plot_feature_space(scored_df: pd.DataFrame, loc, out_dir: Path) -> list[Path]:
    """Pairwise 2D feature-space scatters for all FEATURES pairs."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    LABELS = {True: "Woody", False: "Non-woody"}
    COLOURS = {True: "#2ca02c", False: "#ff7f0e"}

    pairs = [(FEATURES[i], FEATURES[j]) for i in range(len(FEATURES)) for j in range(i + 1, len(FEATURES))]
    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5))
    if len(pairs) == 1:
        axes = [axes]
    fig.suptitle(f"{loc.name} — Stage 1 feature projections", fontsize=13, y=1.01)

    for ax, (fx, fy) in zip(axes, pairs):
        for is_woody, colour in [(True, COLOURS[True]), (False, COLOURS[False])]:
            sub = scored_df[scored_df["is_woody"] == is_woody]
            if sub.empty:
                continue
            ax.scatter(sub[fx], sub[fy], s=8, alpha=0.4, color=colour, label=LABELS[is_woody])
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

        unlabelled = scored_df[scored_df["is_woody"].isna()]
        if not unlabelled.empty:
            ax.scatter(unlabelled[fx], unlabelled[fy], s=6, alpha=0.12,
                       color="#aaaaaa", label="Unlabelled")

        ax.set_xlabel(fx, fontsize=10)
        ax.set_ylabel(fy, fontsize=10)
        ax.tick_params(labelsize=9)

    axes[0].legend(fontsize=9)
    plt.tight_layout()

    p = out_dir / "frenchs_stage1_pairwise_2d.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")
    return [p]


def run(plots: bool = True, year_from: int | None = None, year_to: int | None = None) -> None:
    loc = get("frenchs")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sorted_path = ensure_pixel_sorted(loc.parquet_path())

    print("Extracting features (re_p10, ndvi_integral)...")
    features = extract_parko_features(
        sorted_path, loc,
        swir_params=SWIR_PARAMS,
        ndvi_integral_params=INTEGRAL_PARAMS,
        year_from=year_from,
        year_to=year_to,
    )

    print("Labelling pixels (woody vs non-woody)...")
    labelled = label_woody(features, loc)
    n_woody  = (labelled["is_woody"] == True).sum()   # noqa: E712
    n_nonw   = (labelled["is_woody"] == False).sum()  # noqa: E712
    print(f"  Woody: {n_woody:,}  Non-woody: {n_nonw:,}  Unlabelled: {labelled['is_woody'].isna().sum():,}")

    print("Fitting Stage 1 classifier...")
    train = labelled[labelled["is_woody"].notna()].copy()
    train["is_woody"] = train["is_woody"].astype(bool)
    clf = ParkoClassifier(features=FEATURES)
    clf.fit(train, label_col="is_woody")
    print(clf.summary())

    print("Scoring pixels...")
    # Score on a temporary copy with is_presence alias so heatmap/summarise helpers work
    scored = clf.score(labelled)
    # Rename is_woody → is_presence so plot_prob_heatmaps labels work cleanly
    scored["is_presence"] = scored["is_woody"]

    summarise_stage1(scored, loc)
    save_pixel_ranking(scored, OUT_DIR / "frenchs_stage1_pixel_ranking.csv", FEATURES)

    if plots:
        plot_prob_heatmaps(scored, loc, OUT_DIR, stem="frenchs_stage1")
        plot_feature_space(scored, loc, OUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--year-from", type=int, default=None, metavar="YYYY")
    parser.add_argument("--year-to",   type=int, default=None, metavar="YYYY")
    args = parser.parse_args()
    run(plots=not args.no_plots, year_from=args.year_from, year_to=args.year_to)
