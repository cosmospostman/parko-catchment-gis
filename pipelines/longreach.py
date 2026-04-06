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


def plot_prob_vs_imagery(scored_df: pd.DataFrame, loc, out_dir: Path) -> list[Path]:
    """Probability heatmap and top/bottom decile overlay on WMS imagery.

    Produces:
      longreach_prob_vs_imagery.png
      longreach_prob_deciles.png
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    import importlib.util

    spec = importlib.util.spec_from_file_location("qglobe_plot", PROJECT_ROOT / "utils" / "qglobe-plot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    margin = 0.00005
    lon_min = scored_df["lon"].min() - margin
    lon_max = scored_df["lon"].max() + margin
    lat_min = scored_df["lat"].min() - margin
    lat_max = scored_df["lat"].max() + margin

    try:
        img = mod.fetch_wms_image([lon_min, lat_min, lon_max, lat_max], width_px=512)
    except Exception as exc:
        print(f"  WARNING: WMS fetch failed ({exc}) — plots will render without imagery")
        img = None

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0, vmax=1)
    half = 0.000045

    # Figure 1: imagery + heatmap
    fig, (ax_img, ax_score) = plt.subplots(1, 2, figsize=(14, 18))
    fig.suptitle(
        f"Parkinsonia probability vs Queensland Globe 20cm imagery\n"
        f"{loc.name}  ({len(scored_df):,} pixels)",
        fontsize=12,
    )

    if img is not None:
        ax_img.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max], origin="upper", aspect="auto")
    else:
        ax_img.set_facecolor("#222222")
    ax_img.set_title("Queensland Globe 20cm + probability overlay", fontsize=10)

    for _, row in scored_df.iterrows():
        if pd.isna(row["prob_lr"]):
            continue
        ax_img.add_patch(mpatches.Rectangle(
            (row["lon"] - half, row["lat"] - half),
            width=half * 2, height=half * 2,
            linewidth=0, facecolor=cmap(norm(row["prob_lr"])), alpha=0.55,
        ))

    ax_score.set_facecolor("#222222")
    ax_score.set_title("Parkinsonia probability score (logistic regression)", fontsize=10)
    sc = ax_score.scatter(
        scored_df["lon"], scored_df["lat"],
        c=scored_df["prob_lr"], cmap=cmap, norm=norm,
        s=60, marker="s", linewidths=0,
    )
    cb = plt.colorbar(sc, ax=ax_score, fraction=0.03, pad=0.04)
    cb.set_label("Probability (Parkinsonia)", fontsize=9)
    for thresh in np.percentile(scored_df["prob_lr"].dropna(), np.arange(10, 100, 10)):
        cb.ax.axhline(thresh, color="white", linewidth=0.8, linestyle="--")

    for ax in (ax_img, ax_score):
        ax.set_xlabel("Longitude", fontsize=9)
        ax.set_ylabel("Latitude", fontsize=9)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    p1 = out_dir / "longreach_prob_vs_imagery.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p1}")

    # Figure 2: top / bottom decile
    p90 = scored_df["prob_lr"].quantile(0.90)
    p10 = scored_df["prob_lr"].quantile(0.10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 18))
    fig.suptitle(f"Top vs bottom decile Parkinsonia probability\n{loc.name}", fontsize=12)

    for ax, subset, colour, title in [
        (axes[0], scored_df[scored_df["prob_lr"] >= p90], "#e74c3c",
         f"Top decile (prob ≥ {p90:.2f}, n={(scored_df['prob_lr'] >= p90).sum():,})"),
        (axes[1], scored_df[scored_df["prob_lr"] <= p10], "#3498db",
         f"Bottom decile (prob ≤ {p10:.2f}, n={(scored_df['prob_lr'] <= p10).sum():,})"),
    ]:
        if img is not None:
            ax.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max], origin="upper", aspect="auto")
        else:
            ax.set_facecolor("#222222")
        for _, row in subset.iterrows():
            ax.add_patch(mpatches.Rectangle(
                (row["lon"] - half, row["lat"] - half),
                width=half * 2, height=half * 2,
                linewidth=0.5, edgecolor="white", facecolor=colour, alpha=0.65,
            ))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Longitude", fontsize=9)
        ax.set_ylabel("Latitude", fontsize=9)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    p2 = out_dir / "longreach_prob_deciles.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p2}")

    return [p1, p2]


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
        plot_prob_vs_imagery(scored, loc, out_dir)
        plot_feature_space(scored, loc, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    run(plots=not args.no_plots)
