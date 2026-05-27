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
import polars as pl
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.constants import add_spectral_indices, ensure_float32_bands


def _read_pixels_for_band(
    lat_lo: float,
    lat_hi: float,
    tile_paths: list[Path],
    feature_cols: list[str],
    scl_purity_min: float,
    year: int,
) -> pl.DataFrame:
    """Load all observations for pixels whose lat rounds to lat_lo or lat_hi."""
    import re
    target_lats = {lat_lo, lat_hi}
    raw_band_cols = [c for c in feature_cols if c not in ("NDVI", "NDWI", "EVI")]
    read_cols = ["point_id", "lat", "date", "tile_id", "item_id", "scl_purity"] + raw_band_cols
    _re = re.compile(r"^S2[ABC]_(\d{2}[A-Z]{3})_")

    chunks = []
    for tp in tile_paths:
        pf = pq.ParquetFile(tp)
        available = set(pf.schema_arrow.names)
        cols = [c for c in read_cols if c in available]
        for i in range(pf.metadata.num_row_groups):
            chunk = pl.from_arrow(pf.read_row_group(i, columns=cols)).with_columns(
                pl.col("lat").cast(pl.Float64).round(3).alias("lat_r")
            ).filter(pl.col("lat_r").is_in(list(target_lats))).filter(
                pl.col("scl_purity") >= scl_purity_min
            ).with_columns([
                pl.col("date").cast(pl.Date).dt.year().alias("yr"),
                pl.col("date").cast(pl.Date).dt.ordinal_day().alias("doy"),
                pl.col("date").cast(pl.Date).dt.month().alias("month"),
            ]).filter(pl.col("yr") == year)
            if chunk.is_empty():
                continue
            chunks.append(chunk)

    if not chunks:
        raise ValueError(f"No data found for lat bands {lat_lo}/{lat_hi}")

    df = add_spectral_indices(ensure_float32_bands(pl.concat(chunks)))

    # derive tile_id from item_id where blank
    if "item_id" in df.columns and "tile_id" in df.columns:
        tile_ids = df["tile_id"].to_list()
        item_ids = df["item_id"].to_list()
        fixed = [
            tid if tid else (m.group(1) if (m := _re.match(iid or "")) else None)
            for tid, iid in zip(tile_ids, item_ids)
        ]
        df = df.with_columns(pl.Series("tile_id", fixed))
    return df


def _sequence_stats(df: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    """Per-pixel sequence statistics."""
    raw_cols = [c for c in feature_cols if c in df.columns]
    stats = []
    for (pid,), grp in df.sort("doy").group_by(["point_id"], maintain_order=True):
        doys = grp["doy"].to_numpy()
        n = len(doys)
        tile_arr = grp["tile_id"].to_list() if "tile_id" in grp.columns else []
        from collections import Counter
        tile_counts = dict(Counter(tile_arr))
        row: dict = {
            "point_id": pid,
            "lat_r":    float(grp["lat_r"][0]),
            "n_obs":    n,
            "doy_mean": float(doys.mean()),
            "doy_std":  float(doys.std()),
            "doy_min":  float(doys.min()),
            "doy_max":  float(doys.max()),
            "doy_range": float(doys.max() - doys.min()),
            "n_wet": int(((doys >= 305) | (doys <= 90)).sum()),
            "n_dry": int(((doys >= 121) & (doys <= 273)).sum()),
        }
        for col in raw_cols:
            vals = grp[col].drop_nulls().to_numpy()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) > 0 else float("nan")
            row[f"{col}_std"]  = float(vals.std())  if len(vals) > 1 else float("nan")
        stats.append(row)
    return pl.DataFrame(stats) if stats else pl.DataFrame()


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
    print(f"  {len(df):,} observations, {df['point_id'].n_unique():,} unique pixels")

    # ── join scores ────────────────────────────────────────────────────────
    scores = pl.read_csv(score_csv).select(["point_id", "prob_tam"])
    pid_score_map = {r["point_id"]: r["prob_tam"] for r in scores.iter_rows(named=True)}
    df = df.join(scores, on="point_id", how="left")

    # ── sub-sample pixels ─────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    sampled_pids: dict[float, list] = {}
    for (lat_r,), grp in df.group_by(["lat_r"], maintain_order=True):
        pids = grp["point_id"].unique().to_list()
        chosen = rng.choice(pids, size=min(n_pixels, len(pids)), replace=False)
        sampled_pids[float(lat_r)] = list(chosen)

    dfs: dict[float, pl.DataFrame] = {}
    for lat_r, pids in sampled_pids.items():
        dfs[lat_r] = df.filter(pl.col("point_id").is_in(pids))

    # ── per-pixel sequence stats ───────────────────────────────────────────
    stats: dict[float, pl.DataFrame] = {}
    for lat_r, sub in dfs.items():
        st = _sequence_stats(sub, feature_cols)
        if not st.is_empty():
            st = st.with_columns(
                pl.col("point_id").map_elements(
                    lambda p: pid_score_map.get(p, float("nan")), return_dtype=pl.Float64
                ).alias("prob_tam")
            )
        stats[lat_r] = st

    def _median_prob(pids: list) -> float:
        vals = [pid_score_map.get(p, float("nan")) for p in pids]
        valid = [v for v in vals if not np.isnan(v)]
        return float(np.median(valid)) if valid else float("nan")

    lat_labels = {
        lat_lo: f"lat {lat_lo}  (median prob={_median_prob(sampled_pids[lat_lo]):.3f})",
        lat_hi: f"lat {lat_hi}  (median prob={_median_prob(sampled_pids[lat_hi]):.3f})",
    }
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
            if st.is_empty() or col not in st.columns:
                continue
            vals = st[col].drop_nulls().to_numpy()
            ax.hist(vals, bins=bins, alpha=0.6, color=colors[lat_r],
                    label=lat_labels[lat_r], density=True)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _scatter(ax, xcol, ycol, title):
        for lat_r, st in stats.items():
            if st.is_empty() or xcol not in st.columns or ycol not in st.columns:
                continue
            sub = st.select([xcol, ycol]).drop_nulls()
            ax.scatter(sub[xcol].to_numpy(), sub[ycol].to_numpy(),
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
    first_st = next(iter(stats.values())) if stats else pl.DataFrame()
    plot_bands = [f"{b}_mean" for b in ["B11", "B08", "B04"]
                  if f"{b}_mean" in first_st.columns]
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
            if st.is_empty() or col not in st.columns:
                continue
            vals = st[col].drop_nulls().to_numpy()
            if len(vals) > 0:
                print(f"    {col:20s}  {vals.mean():.3f}  ±  {vals.std():.3f}")


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
