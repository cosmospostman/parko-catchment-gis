"""tam/diag_sequence.py — Inspect model input sequences at a score-jump boundary.

Selects two adjacent lat bands (low-score and high-score side of a visible streak),
samples N pixels from each, reconstructs the exact sequence the model would see
(same SCL filter, same feature columns, same band normalisation), and plots:

  - DOY distribution of observations (histogram)
  - Per-observation band values along the sequence (sorted by DOY)
  - Sequence length distribution

Usage:
    python -m tam.diag_sequence \
        --location quaids \
        --year 2025 \
        --score outputs/scores/quaids/tam-v9_spectral_d256_l3.csv \
        --checkpoint outputs/models/tam-v9_spectral_d256_l3 \
        --lat-lo -16.775 \
        --lat-hi -16.774 \
        --n-pixels 200 \
        --out outputs/diag_sequence_quaids.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _read_pixels_for_band(
    lat_lo: float,
    lat_hi: float,
    tile_paths: list[Path],
    feature_cols: list[str],
    scl_purity_min: float,
    year: int,
) -> pd.DataFrame:
    """Load all observations for pixels whose lat rounds to lat_lo or lat_hi."""
    target_lats = {lat_lo, lat_hi}
    raw_band_cols = [c for c in feature_cols if c not in ("NDVI", "NDWI", "EVI")]
    read_cols = ["point_id", "lat", "date", "tile_id", "item_id", "scl_purity"] + raw_band_cols

    chunks = []
    for tp in tile_paths:
        pf = pq.ParquetFile(tp)
        available = set(pf.schema_arrow.names)
        cols = [c for c in read_cols if c in available]
        for i in range(pf.metadata.num_row_groups):
            chunk = pf.read_row_group(i, columns=cols).to_pandas()
            chunk["lat_f64"] = chunk["lat"].astype("float64")
            chunk["lat_r"]   = chunk["lat_f64"].round(3)
            chunk = chunk[chunk["lat_r"].isin(target_lats)]
            if chunk.empty:
                continue
            chunk = chunk[chunk["scl_purity"] >= scl_purity_min]
            chunk["date"] = pd.to_datetime(chunk["date"])
            chunk["yr"]   = chunk["date"].dt.year
            chunk = chunk[chunk["yr"] == year]
            if chunk.empty:
                continue
            chunk["doy"]  = chunk["date"].dt.dayofyear
            chunk["month"] = chunk["date"].dt.month
            chunks.append(chunk)

    if not chunks:
        raise ValueError(f"No data found for lat bands {lat_lo}/{lat_hi}")

    df = pd.concat(chunks, ignore_index=True)

    # derive tile_id from item_id where blank
    if "item_id" in df.columns:
        import re
        _re = re.compile(r"^S2[ABC]_(\d{2}[A-Z]{3})_")
        mask = df["tile_id"].isna() | (df["tile_id"] == "")
        if mask.any():
            df.loc[mask, "tile_id"] = (df.loc[mask, "item_id"]
                                        .str.extract(_re, expand=False))
    return df


def _sequence_stats(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Per-pixel sequence statistics."""
    raw_cols = [c for c in feature_cols if c in df.columns]
    stats = []
    for pid, grp in df.groupby("point_id"):
        grp = grp.sort_values("doy")
        n = len(grp)
        doys = grp["doy"].values
        tile_counts = grp["tile_id"].value_counts().to_dict()
        row = {
            "point_id": pid,
            "lat_r": grp["lat_r"].iloc[0],
            "n_obs": n,
            "doy_mean": doys.mean(),
            "doy_std": doys.std(),
            "doy_min": doys.min(),
            "doy_max": doys.max(),
            "doy_range": doys.max() - doys.min(),
            "tile_counts": tile_counts,
        }
        # DOY coverage in wet vs dry
        row["n_wet"] = int(((doys >= 305) | (doys <= 90)).sum())   # Nov–Mar
        row["n_dry"] = int(((doys >= 121) & (doys <= 273)).sum())  # May–Sep
        for col in raw_cols:
            row[f"{col}_mean"] = grp[col].mean()
            row[f"{col}_std"]  = grp[col].std()
        stats.append(row)
    return pd.DataFrame(stats)


def run(
    location: str,
    year: int,
    score_csv: Path,
    checkpoint_dir: Path,
    lat_lo: float,
    lat_hi: float,
    n_pixels: int,
    out_path: Path,
    scl_purity_min: float = 0.5,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── load checkpoint config ─────────────────────────────────────────────
    cfg = json.load(open(checkpoint_dir / "tam_config.json"))
    feature_cols: list[str] = cfg.get("feature_cols") or [
        "B02","B03","B04","B05","B07","B08","B8A","B11","B12","NDVI","NDWI"
    ]
    print(f"feature_cols: {feature_cols}")

    # ── locate tile parquets ───────────────────────────────────────────────
    pixel_dir = PROJECT_ROOT / "data" / "pixels" / location / str(year)
    tile_paths = sorted(
        p for p in pixel_dir.iterdir()
        if p.suffix == ".parquet"
        and not p.stem.endswith("-by-pixel")
        and "coords" not in p.stem
    )
    print(f"Tile parquets: {[p.name for p in tile_paths]}")

    # ── load observations for the two lat bands ────────────────────────────
    print(f"Loading observations for lat {lat_lo} and {lat_hi} ...")
    df = _read_pixels_for_band(lat_lo, lat_hi, tile_paths, feature_cols,
                                scl_purity_min, year)
    print(f"  {len(df):,} observations, {df['point_id'].nunique():,} unique pixels")

    # ── join scores ────────────────────────────────────────────────────────
    scores = pd.read_csv(score_csv)[["point_id", "prob_tam"]]
    df = df.merge(scores, on="point_id", how="left")

    # ── sub-sample pixels ─────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    sampled_pids: dict[float, list] = {}
    for lat_r, grp in df.groupby("lat_r"):
        pids = grp["point_id"].unique()
        chosen = rng.choice(pids, size=min(n_pixels, len(pids)), replace=False)
        sampled_pids[lat_r] = list(chosen)

    dfs: dict[float, pd.DataFrame] = {}
    for lat_r, pids in sampled_pids.items():
        dfs[lat_r] = df[df["point_id"].isin(pids)].copy()

    # ── per-pixel sequence stats ───────────────────────────────────────────
    pid_scores = scores.set_index("point_id")["prob_tam"]
    stats: dict[float, pd.DataFrame] = {}
    for lat_r, sub in dfs.items():
        st = _sequence_stats(sub, feature_cols)
        st["prob_tam"] = st["point_id"].map(pid_scores)
        stats[lat_r] = st

    lat_labels = {lat_lo: f"lat {lat_lo}  (median prob={scores.merge(pd.DataFrame({'point_id': sampled_pids[lat_lo]}))['prob_tam'].median():.3f})",
                  lat_hi: f"lat {lat_hi}  (median prob={scores.merge(pd.DataFrame({'point_id': sampled_pids[lat_hi]}))['prob_tam'].median():.3f})"}
    colors = {lat_lo: "steelblue", lat_hi: "darkorange"}

    # ── plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    fig.suptitle(
        f"{location} {year}  —  sequence diagnostic\n"
        f"lat {lat_lo} vs {lat_hi}  |  {n_pixels} pixels sampled per band",
        fontsize=11,
    )

    def _hist(ax, col, title, xlabel, bins=20):
        for lat_r, st in stats.items():
            vals = st[col].dropna()
            ax.hist(vals, bins=bins, alpha=0.6, color=colors[lat_r],
                    label=lat_labels[lat_r], density=True)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _scatter(ax, xcol, ycol, title):
        for lat_r, st in stats.items():
            ax.scatter(st[xcol].dropna(), st[ycol].dropna(),
                       alpha=0.3, s=8, color=colors[lat_r],
                       label=lat_labels[lat_r])
        ax.set_title(title)
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Row 0: sequence length, DOY spread, wet/dry count
    _hist(axes[0, 0], "n_obs",     "Sequence length (n observations)", "n obs")
    _hist(axes[0, 1], "doy_range", "DOY range (last − first obs)",     "days")
    _hist(axes[0, 2], "doy_mean",  "Mean DOY of observations",         "DOY")

    # Row 1: wet vs dry obs count, doy_std, prob_tam
    _hist(axes[1, 0], "n_wet", "Obs in wet season (Nov–Mar)", "count")
    _hist(axes[1, 1], "n_dry", "Obs in dry season (May–Sep)", "count")
    _hist(axes[1, 2], "doy_std", "DOY std dev", "std (days)")

    # Row 2: key band means, n_obs vs prob_tam scatter
    plot_bands = [f"{b}_mean" for b in ["B11", "B08", "B04"]
                  if f"{b}_mean" in next(iter(stats.values())).columns]
    for ax, band in zip(axes[2, :2], plot_bands[:2]):
        _hist(ax, band, f"{band} distribution", band)

    _scatter(axes[2, 2], "n_obs", "prob_tam", "n_obs vs prob_tam")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # ── text summary ──────────────────────────────────────────────────────
    print("\n── Per-band summary (mean ± std across sampled pixels) ──")
    stat_cols = ["n_obs", "doy_range", "doy_mean", "doy_std",
                 "n_wet", "n_dry"] + [f"{c}_mean" for c in feature_cols if c in df.columns]
    for lat_r, st in stats.items():
        print(f"\n  lat {lat_r}  (n={len(st)} pixels):")
        for col in stat_cols:
            if col not in st.columns:
                continue
            print(f"    {col:20s}  {st[col].mean():.3f}  ±  {st[col].std():.3f}")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--location",   required=True)
    parser.add_argument("--year",       type=int, default=2025)
    parser.add_argument("--score",      required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lat-lo",     type=float, default=-16.775)
    parser.add_argument("--lat-hi",     type=float, default=-16.774)
    parser.add_argument("--n-pixels",   type=int, default=200)
    parser.add_argument("--out",        default=None)
    args = parser.parse_args()

    out_path = (Path(args.out) if args.out
                else PROJECT_ROOT / "outputs" / f"diag_sequence_{args.location}.png")

    run(
        location=args.location,
        year=args.year,
        score_csv=Path(args.score),
        checkpoint_dir=Path(args.checkpoint),
        lat_lo=args.lat_lo,
        lat_hi=args.lat_hi,
        n_pixels=args.n_pixels,
        out_path=out_path,
    )


if __name__ == "__main__":
    _main()
