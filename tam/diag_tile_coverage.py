"""tam/diag_tile_coverage.py — Diagnose horizontal streaking from tile coverage imbalance.

For each 0.001-deg lat band, computes:
  - Mean observations per pixel from each tile
  - Mean band values by season (wet / dry)
  - Median prob_tam from scored CSV

Outputs a multi-panel PNG showing whether feature-space differences track score differences.

Usage:
    python -m tam.diag_tile_coverage \
        --location quaids \
        --year 2025 \
        --score outputs/scores/quaids/tam-v9_spectral_d256_l3.csv \
        --out outputs/diag_tile_coverage_quaids.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── helpers ──────────────────────────────────────────────────────────────────

WET_MONTHS   = {11, 12, 1, 2, 3}   # Nov–Mar
DRY_MONTHS   = {5, 6, 7, 8, 9}     # May–Sep
FEATURE_BANDS = ["B04", "B08", "B11", "NDVI"]


def _read_tile(path: Path, bands: list[str]) -> pl.DataFrame:
    """Read a tile parquet, return point_id / lat / date / tile_id / bands."""
    import re
    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    cols = [c for c in ["point_id", "lat", "date", "item_id", "tile_id", "scl_purity"] + bands
            if c in available]
    chunks = [pl.from_arrow(pf.read_row_group(i, columns=cols))
              for i in range(pf.metadata.num_row_groups)]
    df = pl.concat(chunks).filter(pl.col("scl_purity") >= 0.5).with_columns([
        pl.col("date").cast(pl.Date).dt.month().alias("month"),
    ])
    # season: months 1-3 and 10-12 → wet, 4-9 → dry
    df = df.with_columns(
        pl.when(pl.col("month").is_between(4, 9)).then(pl.lit("dry")).otherwise(pl.lit("wet")).alias("season")
    )
    # derive tile_id from item_id if tile_id column is blank
    if "item_id" in df.columns and "tile_id" in df.columns:
        _re = re.compile(r"^S2[ABC]_(\d{2}[A-Z]{3})_")
        def _extract_tile(s: str) -> str | None:
            m = _re.match(s or "")
            return m.group(1) if m else None
        tile_ids = df["tile_id"].to_list()
        item_ids = df["item_id"].to_list()
        fixed = [
            tid if tid else _extract_tile(iid)
            for tid, iid in zip(tile_ids, item_ids)
        ]
        df = df.with_columns(pl.Series("tile_id", fixed))
    return df


def _lat_band_stats(df: pl.DataFrame, bands: list[str]) -> pl.DataFrame:
    """Per lat-band (0.001 deg): obs count, wet/dry mean per band."""
    df = df.with_columns(
        pl.col("lat").cast(pl.Float64).round(3).alias("lat_r")
    )
    agg_exprs = [
        pl.len().alias("n_obs"),
        pl.col("point_id").n_unique().alias("n_pts"),
    ]
    for band in bands:
        if band not in df.columns:
            continue
        agg_exprs += [
            pl.col(band).filter(pl.col("season") == "wet").mean().alias(f"{band}_wet"),
            pl.col(band).filter(pl.col("season") == "dry").mean().alias(f"{band}_dry"),
            pl.col(band).mean().alias(f"{band}_all"),
        ]
    result = df.group_by("lat_r").agg(agg_exprs).sort("lat_r")
    return result.with_columns(
        (pl.col("n_obs") / pl.col("n_pts").clip(lower_bound=1)).alias("obs_per_pt")
    )


# ── main ─────────────────────────────────────────────────────────────────────

def run(location: str, year: int, score_csv: Path, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pixel_dir = PROJECT_ROOT / "data" / "pixels" / location / str(year)
    if not pixel_dir.exists():
        raise FileNotFoundError(pixel_dir)

    tile_paths = sorted(
        p for p in pixel_dir.iterdir()
        if p.suffix == ".parquet"
        and not p.stem.endswith("-by-pixel")
        and "coords" not in p.stem
    )
    if not tile_paths:
        raise FileNotFoundError(f"No tile parquets in {pixel_dir}")

    print(f"Tiles: {[p.name for p in tile_paths]}")

    # ── read each tile ────────────────────────────────────────────────────────
    tile_dfs: dict[str, pl.DataFrame] = {}
    for tp in tile_paths:
        print(f"  Reading {tp.name} ...")
        df = _read_tile(tp, FEATURE_BANDS)
        tile_id = df["tile_id"].drop_nulls().mode()[0] if len(df) > 0 else tp.stem
        tile_dfs[tile_id] = df

    # ── per-tile lat-band stats ───────────────────────────────────────────────
    tile_stats: dict[str, pl.DataFrame] = {}
    for tile_id, df in tile_dfs.items():
        tile_stats[tile_id] = _lat_band_stats(df, FEATURE_BANDS)

    # ── combined lat-band stats (all tiles together) ──────────────────────────
    all_df = pl.concat(list(tile_dfs.values()))
    combined_stats = _lat_band_stats(all_df, FEATURE_BANDS)

    # ── score CSV ─────────────────────────────────────────────────────────────
    scores = pl.read_csv(score_csv).with_columns(
        pl.col("lat").cast(pl.Float64).round(3).alias("lat_r")
    )
    score_lat = scores.group_by("lat_r").agg([
        pl.col("prob_tam").median().alias("prob_median"),
        pl.col("prob_tam").std().alias("prob_std"),
    ])

    merged = combined_stats.join(score_lat, on="lat_r", how="left")

    # ── plot ──────────────────────────────────────────────────────────────────
    tiles = list(tile_stats.keys())
    n_bands = len(FEATURE_BANDS)
    n_rows = 2 + n_bands   # obs counts, prob_tam, one row per band
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3 * n_rows),
                             sharex=True, constrained_layout=True)
    fig.suptitle(f"{location} {year} — tile coverage diagnostic\n({score_csv.name})",
                 fontsize=11)

    lats = merged["lat_r"].to_numpy()

    # Row 0: observations per pixel per tile
    ax = axes[0]
    for tile_id, st in tile_stats.items():
        ax.plot(st["lat_r"].to_numpy(), st["obs_per_pt"].to_numpy(), label=tile_id, linewidth=1)
    ax.set_ylabel("obs / pixel")
    ax.set_title("Observations per pixel by tile")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 1: prob_tam with std band
    ax = axes[1]
    prob_med = merged["prob_median"].to_numpy()
    prob_std = merged["prob_std"].fill_null(0).to_numpy()
    ax.plot(lats, prob_med, color="steelblue", linewidth=1.2, label="median")
    ax.fill_between(lats, prob_med - prob_std, prob_med + prob_std,
                    alpha=0.2, color="steelblue", label="±1σ")
    ax.set_ylabel("prob_tam")
    ax.set_ylim(0, 1)
    ax.set_title("TAM probability (0.001° lat bins)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Rows 2+: per-band wet vs dry mean
    colors = {"wet": "royalblue", "dry": "darkorange"}
    for i, band in enumerate(FEATURE_BANDS):
        ax = axes[2 + i]
        for season, color in colors.items():
            col = f"{band}_{season}"
            if col not in merged.columns:
                continue
            ax.plot(lats, merged[col].to_numpy(), color=color, linewidth=1,
                    label=f"{season} season", alpha=0.85)
        ax.set_ylabel(band)
        ax.set_title(f"{band} — wet vs dry season mean")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("latitude")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # ── text summary ──────────────────────────────────────────────────────────
    print("\n── Correlation: obs_per_pt vs prob_tam ──")
    for tile_id, st in tile_stats.items():
        m = st.join(score_lat, on="lat_r", how="inner")
        if m.is_empty():
            continue
        r = float(np.corrcoef(m["obs_per_pt"].to_numpy(), m["prob_median"].to_numpy())[0, 1])
        print(f"  {tile_id}: r = {r:.3f}")

    print("\n── Correlation: band means vs prob_tam ──")
    for band in FEATURE_BANDS:
        for season in ("wet", "dry", "all"):
            col = f"{band}_{season}"
            if col not in merged.columns:
                continue
            x = merged[col].drop_nulls().to_numpy()
            y = merged.filter(pl.col(col).is_not_null())["prob_median"].to_numpy()
            if len(x) > 1:
                r = float(np.corrcoef(x, y)[0, 1])
                print(f"  {band} ({season}): r = {r:.3f}")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--location", required=True)
    parser.add_argument("--year",     type=int, default=2025)
    parser.add_argument("--score",    required=True, help="Scored CSV path")
    parser.add_argument("--out",      default=None)
    args = parser.parse_args()

    score_csv = Path(args.score)
    out_path = (Path(args.out) if args.out
                else PROJECT_ROOT / "outputs" / f"diag_tile_coverage_{args.location}.png")

    run(args.location, args.year, score_csv, out_path)


if __name__ == "__main__":
    _main()
