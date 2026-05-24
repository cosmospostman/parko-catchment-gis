"""
Diagnostic: VH dry-season distributions for training presence, training absence,
and any scored false-positive pixels.

Usage:
    python utils/diag_s1_vh.py                     # distributions only
    python utils/diag_s1_vh.py --fp-csv PATH        # include false-positive pixel CSVs

The script loads all training pixel parquets, computes per-pixel dry-season mean VH (dB),
and plots overlapping histograms + prints overlap statistics.  If --fp-csv is given
(columns: point_id or lon/lat, score) it also shows those pixels' VH distributions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tam.core.dataset import lin_to_db

DRY_DOY_MIN = 121   # ~May 1
DRY_DOY_MAX = 304   # ~Oct 31

# ── helpers ───────────────────────────────────────────────────────────────────

def load_all_training_pixels(regions: list[str] | None = None) -> pl.DataFrame:
    """Load all (or specified) training pixels from tile parquets, return combined frame.

    point_id has the form  {region_id}_{row}_{col}, so we extract region_id from it.
    """
    tiles_dir = ROOT / "data/training/tiles"
    tile_files = sorted(tiles_dir.glob("*.parquet"))
    if not tile_files:
        raise RuntimeError(f"No tile parquets found in {tiles_dir}")

    frames = []
    for tp in tile_files:
        df = pl.read_parquet(tp)
        # Extract region_id: everything before the last two underscore-separated tokens
        # point_id = "{region_id}_{row:04d}_{col:04d}"
        df = df.with_columns(
            pl.col("point_id")
            .str.extract(r"^(.+)_\d{4}_\d{4}$", 1)
            .alias("region_id")
        )
        frames.append(df)

    combined = pl.concat(frames, how="diagonal_relaxed")

    if regions:
        combined = combined.filter(pl.col("region_id").is_in(regions))

    if len(combined) == 0:
        raise RuntimeError("No rows after filtering.")
    return combined


def compute_vh_dry_means(df: pl.DataFrame) -> pl.DataFrame:
    """Return per-pixel dry-season mean VH (dB), with label and region_id."""
    s1 = df.filter(pl.col("source") == "S1")
    if len(s1) == 0:
        raise RuntimeError("No S1 rows found.")

    # Compute doy if absent
    if "doy" not in s1.columns:
        s1 = s1.with_columns(
            pl.col("date").dt.ordinal_day().alias("doy")
        )

    dry = s1.filter(
        (pl.col("doy") >= DRY_DOY_MIN) & (pl.col("doy") <= DRY_DOY_MAX)
        & pl.col("vh").is_not_null()
    )

    # Convert linear vh → dB
    dry = dry.with_columns(
        (10.0 * (pl.col("vh").log(base=10.0))).alias("vh_db")
    ).filter(pl.col("vh_db").is_finite())

    agg = (
        dry.group_by(["point_id", "region_id"])
        .agg(pl.col("vh_db").mean().alias("mean_vh_db"))
    )
    return agg


def add_labels(vh_df: pl.DataFrame) -> pl.DataFrame:
    """Attach presence/absence label from region_id."""
    vh_df = vh_df.with_columns(
        pl.when(pl.col("region_id").str.contains("presence"))
        .then(pl.lit("presence"))
        .when(pl.col("region_id").str.contains("absence"))
        .then(pl.lit("absence"))
        .otherwise(pl.lit("unknown"))
        .alias("label")
    )
    return vh_df


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp-csv", type=Path, default=None,
                        help="CSV of false-positive pixels (needs point_id or lon+lat cols)")
    parser.add_argument("--sites", nargs="+", default=None,
                        help="Filter to these site prefixes (e.g. frenchs burdekin)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Save figure to this path instead of showing")
    args = parser.parse_args()

    # ── load training data ────────────────────────────────────────────────────
    print("Loading training pixels ...")
    idx = pl.read_parquet(ROOT / "data/training/index.parquet")
    regions = idx["region_id"].to_list()
    if args.sites:
        regions = [r for r in regions if any(r.startswith(s) for s in args.sites)]
        print(f"  Filtered to {len(regions)} regions matching {args.sites}")

    df = load_all_training_pixels(regions)
    print(f"  Loaded {len(df):,} rows from {df['region_id'].n_unique()} regions")

    vh_df = compute_vh_dry_means(df)
    vh_df = add_labels(vh_df)

    presence = vh_df.filter(pl.col("label") == "presence")["mean_vh_db"].to_numpy()
    absence  = vh_df.filter(pl.col("label") == "absence")["mean_vh_db"].to_numpy()

    print(f"\nPresence pixels : {len(presence):,}")
    print(f"  mean={presence.mean():.2f}  p5={np.percentile(presence,5):.2f}  "
          f"p25={np.percentile(presence,25):.2f}  p75={np.percentile(presence,75):.2f}  "
          f"p95={np.percentile(presence,95):.2f}")
    print(f"Absence pixels  : {len(absence):,}")
    print(f"  mean={absence.mean():.2f}  p5={np.percentile(absence,5):.2f}  "
          f"p25={np.percentile(absence,25):.2f}  p75={np.percentile(absence,75):.2f}  "
          f"p95={np.percentile(absence,95):.2f}")

    # Overlap: fraction of presence pixels below absence median
    absence_median = np.median(absence)
    presence_median = np.median(presence)
    overlap_pct = 100 * (presence < absence_median).mean()
    print(f"\nAbsence median VH: {absence_median:.2f} dB")
    print(f"Presence median VH: {presence_median:.2f} dB")
    print(f"Overlap: {overlap_pct:.1f}% of presence pixels are BELOW absence median")

    # Per-region breakdown
    print("\nPer-region mean VH (dB):")
    region_agg = (
        vh_df.group_by(["region_id", "label"])
        .agg(
            pl.col("mean_vh_db").mean().alias("mean"),
            pl.col("mean_vh_db").count().alias("n_pixels"),
        )
        .sort("label", "region_id")
    )
    for row in region_agg.iter_rows(named=True):
        print(f"  {row['label']:8s}  {row['region_id']:45s}  mean={row['mean']:6.2f} dB  n={row['n_pixels']}")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.linspace(-26, -8, 60)

    ax = axes[0]
    ax.hist(absence,  bins=bins, alpha=0.55, color="steelblue",  density=True, label=f"absence (n={len(absence):,})")
    ax.hist(presence, bins=bins, alpha=0.55, color="darkorange", density=True, label=f"presence (n={len(presence):,})")
    ax.axvline(absence_median,  color="steelblue",  linestyle="--", lw=1.5, label=f"absence median {absence_median:.1f}")
    ax.axvline(presence_median, color="darkorange", linestyle="--", lw=1.5, label=f"presence median {presence_median:.1f}")
    ax.set_xlabel("Dry-season mean VH (dB)")
    ax.set_ylabel("Density")
    ax.set_title("VH distribution: presence vs absence (training data)")
    ax.legend(fontsize=9)

    # per-site box plot
    ax2 = axes[1]
    site_labels = []
    site_data = []
    colors = []
    for row in region_agg.sort("label", "region_id").iter_rows(named=True):
        rid = row["region_id"]
        lbl = row["label"]
        vals = vh_df.filter(pl.col("region_id") == rid)["mean_vh_db"].to_numpy()
        site_labels.append(rid.replace("_presence", "\n(pres)").replace("_absence", "\n(abs)"))
        site_data.append(vals)
        colors.append("darkorange" if lbl == "presence" else "steelblue")

    bp = ax2.boxplot(site_data, patch_artist=True, vert=True, showfliers=False,
                     medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_xticks(range(1, len(site_labels)+1))
    ax2.set_xticklabels(site_labels, rotation=90, fontsize=6)
    ax2.set_ylabel("Mean VH (dB)")
    ax2.set_title("Per-region VH box plots")
    pres_patch = mpatches.Patch(color="darkorange", alpha=0.6, label="presence")
    abs_patch  = mpatches.Patch(color="steelblue",  alpha=0.6, label="absence")
    ax2.legend(handles=[pres_patch, abs_patch], fontsize=9)

    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=150)
        print(f"\nFigure saved to {args.out}")
    else:
        out_path = ROOT / "outputs/diag_vh_distributions.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"\nFigure saved to {out_path}")


if __name__ == "__main__":
    main()
