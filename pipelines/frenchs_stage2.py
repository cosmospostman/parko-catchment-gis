"""pipelines/frenchs_stage2.py — Frenchs Stage 2: Parkinsonia ranking within woody mask.

Takes the stage 1 pixel ranking CSV, filters to the woody mask (prob_lr >= WOODY_THRESHOLD),
then trains a supervised classifier using:
  Positive (Parkinsonia) — sub_bboxes with role "presence"
  Negative (woody)       — sub_bboxes with role "woody"

Features: re_p10 and swir_p10 — both dry-season signals that show clear separation
between Parkinsonia and native woody pixels within the woody mask.

Outputs
-------
  outputs/frenchs-stage2/frenchs_stage2_prob_vs_imagery.png
  outputs/frenchs-stage2/frenchs_stage2_prob_black.png
  outputs/frenchs-stage2/frenchs_stage2_pairwise_2d.png
  outputs/frenchs-stage2/frenchs_stage2_pixel_ranking.csv

Usage
-----
    python -m pipelines.frenchs_stage2
    python -m pipelines.frenchs_stage2 --no-plots
    python -m pipelines.frenchs_stage2 --woody-threshold 0.3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.location import get
from utils.heatmap import plot_prob_heatmaps
from analysis.classifier import ParkoClassifier
from pipelines.common import save_pixel_ranking

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR      = PROJECT_ROOT / "outputs" / "frenchs-stage2"
STAGE1_CSV   = PROJECT_ROOT / "outputs" / "frenchs-stage1" / "frenchs_stage1_pixel_ranking.csv"

# Stage 2 features — dry-season signals that separate Parkinsonia from native woody
FEATURES = ["re_p10", "swir_p10"]

# Pixels with stage 1 P(woody) >= this threshold enter stage 2
WOODY_THRESHOLD = 0.1


def label_parkinsonia(df: pd.DataFrame, loc) -> pd.DataFrame:
    """Label pixels within the woody mask by Parkinsonia/woody sub-bbox roles.

    presence → is_parkinsonia = True
    woody    → is_parkinsonia = False
    absence and unlabelled → NaN (not used for training)
    """
    out = df.copy()
    out["is_parkinsonia"] = pd.NA

    for sub in loc.sub_bboxes.values():
        lon_min, lat_min, lon_max, lat_max = sub.bbox
        mask = (
            out["lon"].between(lon_min, lon_max) &
            out["lat"].between(lat_min, lat_max)
        )
        if sub.role == "presence":
            out.loc[mask, "is_parkinsonia"] = True
        elif sub.role == "woody":
            out.loc[mask, "is_parkinsonia"] = False

    return out


def summarise_stage2(scored_df: pd.DataFrame, loc, woody_threshold: float) -> None:
    print(f"\n{'='*60}")
    print(f"Site: {loc.name} — Stage 2 Parkinsonia ranking")
    print(f"  Woody mask threshold: p >= {woody_threshold}")
    print(f"  Pixels in woody mask: {len(scored_df):,}")
    print(f"{'='*60}")

    labelled = scored_df[scored_df["is_parkinsonia"].notna()]
    if not labelled.empty:
        print("\nP(Parkinsonia) by class (mean / median / std):")
        for val, label in [(True, "Parkinsonia"), (False, "Woody")]:
            sub = labelled[labelled["is_parkinsonia"] == val]["prob_lr"]
            if not sub.empty:
                print(f"  {label:14s}  mean={sub.mean():.3f}  median={sub.median():.3f}  std={sub.std():.3f}")

    all_scored = scored_df["prob_lr"].dropna()
    print(f"\nFull woody mask  ({len(all_scored):,} scored pixels):")
    print(f"  mean={all_scored.mean():.3f}  median={all_scored.median():.3f}  std={all_scored.std():.3f}")
    for pct in (75, 90, 95):
        print(f"  p{pct}={all_scored.quantile(pct/100):.3f}")

    print("\nParkinsonia pixel counts at probability thresholds:")
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        n = (all_scored >= thr).sum()
        pct_mask = 100 * n / len(all_scored)
        print(f"  p >= {thr:.1f}:  {n:,} pixels  ({pct_mask:.1f}% of woody mask)")


def plot_feature_space(scored_df: pd.DataFrame, loc, out_dir: Path) -> list[Path]:
    """re_p10 vs swir_p10 scatter coloured by Parkinsonia/woody."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    LABELS  = {True: "Parkinsonia", False: "Woody"}
    COLOURS = {True: "#d62728", False: "#2ca02c"}

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(f"{loc.name} — Stage 2: re_p10 vs swir_p10", fontsize=13)

    for is_parko, colour in [(True, COLOURS[True]), (False, COLOURS[False])]:
        sub = scored_df[scored_df["is_parkinsonia"] == is_parko]
        if sub.empty:
            continue
        ax.scatter(sub["re_p10"], sub["swir_p10"], s=8, alpha=0.4,
                   color=colour, label=LABELS[is_parko])
        if len(sub) > 2:
            cov = np.cov(sub["re_p10"].values, sub["swir_p10"].values)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            w, h = 2 * 1.5 * np.sqrt(vals)
            ax.add_patch(Ellipse(
                xy=(sub["re_p10"].mean(), sub["swir_p10"].mean()),
                width=w, height=h, angle=theta,
                edgecolor=colour, facecolor="none", linewidth=1.5, zorder=3,
            ))
            ax.scatter(sub["re_p10"].mean(), sub["swir_p10"].mean(),
                       s=80, color=colour, marker="D",
                       edgecolors="black", linewidth=0.8, zorder=4)

    unlabelled = scored_df[scored_df["is_parkinsonia"].isna()]
    if not unlabelled.empty:
        ax.scatter(unlabelled["re_p10"], unlabelled["swir_p10"],
                   s=6, alpha=0.12, color="#aaaaaa", label="Unlabelled")

    ax.set_xlabel("re_p10", fontsize=11)
    ax.set_ylabel("swir_p10", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9)
    plt.tight_layout()

    p = out_dir / "frenchs_stage2_pairwise_2d.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")
    return [p]


def run(plots: bool = True, woody_threshold: float = WOODY_THRESHOLD) -> None:
    loc = get("frenchs")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not STAGE1_CSV.exists():
        raise FileNotFoundError(
            f"Stage 1 pixel ranking not found: {STAGE1_CSV}\n"
            "Run pipelines.frenchs_stage1 first."
        )

    print(f"Loading stage 1 pixel ranking: {STAGE1_CSV.name}")
    stage1 = pd.read_csv(STAGE1_CSV)
    print(f"  {len(stage1):,} total pixels")

    print(f"Applying woody mask (prob_lr >= {woody_threshold})...")
    woody = stage1[stage1["prob_lr"] >= woody_threshold].copy()
    print(f"  {len(woody):,} pixels in woody mask ({100*len(woody)/len(stage1):.1f}% of scene)")

    print("Labelling pixels (Parkinsonia vs woody)...")
    labelled = label_parkinsonia(woody, loc)
    n_pres = (labelled["is_parkinsonia"] == True).sum()   # noqa: E712
    n_wood = (labelled["is_parkinsonia"] == False).sum()  # noqa: E712
    print(f"  Parkinsonia: {n_pres:,}  Woody: {n_wood:,}  Unlabelled: {labelled['is_parkinsonia'].isna().sum():,}")

    print("Fitting Stage 2 classifier...")
    train = labelled[labelled["is_parkinsonia"].notna()].copy()
    train["is_parkinsonia"] = train["is_parkinsonia"].astype(bool)
    clf = ParkoClassifier(features=FEATURES)
    clf.fit(train, label_col="is_parkinsonia")
    print(clf.summary())

    print("Scoring woody mask pixels...")
    scored = clf.score(labelled)
    scored["is_presence"] = scored["is_parkinsonia"]

    summarise_stage2(scored, loc, woody_threshold)
    save_pixel_ranking(scored, OUT_DIR / "frenchs_stage2_pixel_ranking.csv", FEATURES)

    if plots:
        plot_prob_heatmaps(scored, loc, OUT_DIR, stem="frenchs_stage2")
        plot_feature_space(scored, loc, OUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--woody-threshold", type=float, default=WOODY_THRESHOLD, metavar="P",
                        help=f"Stage 1 probability threshold for woody mask (default {WOODY_THRESHOLD})")
    args = parser.parse_args()
    run(plots=not args.no_plots, woody_threshold=args.woody_threshold)
