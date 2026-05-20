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
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── helpers ──────────────────────────────────────────────────────────────────

WET_MONTHS   = {11, 12, 1, 2, 3}   # Nov–Mar
DRY_MONTHS   = {5, 6, 7, 8, 9}     # May–Sep
FEATURE_BANDS = ["B04", "B08", "B11", "NDVI"]


def _read_tile(path: Path, bands: list[str]) -> pd.DataFrame:
    """Read a tile parquet, return point_id / lat / date / tile_id / bands."""
    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    cols = [c for c in ["point_id", "lat", "date", "item_id", "tile_id", "scl_purity"] + bands
            if c in available]
    chunks = [pf.read_row_group(i, columns=cols).to_pandas()
              for i in range(pf.metadata.num_row_groups)]
    df = pd.concat(chunks, ignore_index=True)
    df = df[df["scl_purity"] >= 0.5]
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["season"] = pd.cut(df["month"],
                          bins=[0, 3, 9, 12],
                          labels=["wet", "dry", "wet2"]).astype(str)
    # merge wet/wet2
    df["season"] = df["season"].replace("wet2", "wet")
    # derive tile_id from item_id if tile_id column is blank
    if "item_id" in df.columns:
        mask = df["tile_id"].isna() | (df["tile_id"] == "")
        if mask.any():
            import re
            _re = re.compile(r"^S2[ABC]_(\d{2}[A-Z]{3})_")
            derived = df.loc[mask, "item_id"].str.extract(_re, expand=False)
            df.loc[mask, "tile_id"] = derived
    return df


def _lat_band_stats(df: pd.DataFrame, bands: list[str]) -> pd.DataFrame:
    """Per lat-band (0.001 deg): obs count, wet/dry mean per band."""
    df = df.copy()
    df["lat_r"] = df["lat"].astype("float64").round(3)

    rows = []
    for lat_r, grp in df.groupby("lat_r"):
        row: dict = {"lat_r": lat_r}
        row["n_obs"] = len(grp)
        row["n_pts"] = grp["point_id"].nunique()
        row["obs_per_pt"] = row["n_obs"] / max(row["n_pts"], 1)
        for band in bands:
            if band not in grp.columns:
                continue
            row[f"{band}_wet"] = grp.loc[grp["season"] == "wet",  band].mean()
            row[f"{band}_dry"] = grp.loc[grp["season"] == "dry",  band].mean()
            row[f"{band}_all"] = grp[band].mean()
        rows.append(row)

    return pd.DataFrame(rows).sort_values("lat_r").reset_index(drop=True)


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
    tile_dfs: dict[str, pd.DataFrame] = {}
    for tp in tile_paths:
        print(f"  Reading {tp.name} ...")
        df = _read_tile(tp, FEATURE_BANDS)
        tile_id = df["tile_id"].mode().iloc[0] if len(df) > 0 else tp.stem
        tile_dfs[tile_id] = df

    # ── per-tile lat-band stats ───────────────────────────────────────────────
    tile_stats: dict[str, pd.DataFrame] = {}
    for tile_id, df in tile_dfs.items():
        tile_stats[tile_id] = _lat_band_stats(df, FEATURE_BANDS)

    # ── combined lat-band stats (all tiles together) ──────────────────────────
    all_df = pd.concat(tile_dfs.values(), ignore_index=True)
    combined_stats = _lat_band_stats(all_df, FEATURE_BANDS)

    # ── score CSV ─────────────────────────────────────────────────────────────
    scores = pd.read_csv(score_csv)
    scores["lat_r"] = scores["lat"].round(3)
    score_lat = (scores.groupby("lat_r")["prob_tam"]
                 .agg(["median", "std"])
                 .rename(columns={"median": "prob_median", "std": "prob_std"})
                 .reset_index())

    merged = combined_stats.merge(score_lat, on="lat_r", how="left")

    # ── plot ──────────────────────────────────────────────────────────────────
    tiles = list(tile_stats.keys())
    n_bands = len(FEATURE_BANDS)
    n_rows = 2 + n_bands   # obs counts, prob_tam, one row per band
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3 * n_rows),
                             sharex=True, constrained_layout=True)
    fig.suptitle(f"{location} {year} — tile coverage diagnostic\n({score_csv.name})",
                 fontsize=11)

    lats = merged["lat_r"]

    # Row 0: observations per pixel per tile
    ax = axes[0]
    for tile_id, st in tile_stats.items():
        ax.plot(st["lat_r"], st["obs_per_pt"], label=tile_id, linewidth=1)
    ax.set_ylabel("obs / pixel")
    ax.set_title("Observations per pixel by tile")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 1: prob_tam with std band
    ax = axes[1]
    ax.plot(lats, merged["prob_median"], color="steelblue", linewidth=1.2, label="median")
    ax.fill_between(lats,
                    merged["prob_median"] - merged["prob_std"],
                    merged["prob_median"] + merged["prob_std"],
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
            ax.plot(lats, merged[col], color=color, linewidth=1,
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
        m = st.merge(score_lat, on="lat_r", how="inner")
        r = m["obs_per_pt"].corr(m["prob_median"])
        print(f"  {tile_id}: r = {r:.3f}")

    print("\n── Correlation: band means vs prob_tam ──")
    for band in FEATURE_BANDS:
        for season in ("wet", "dry", "all"):
            col = f"{band}_{season}"
            if col not in merged.columns:
                continue
            r = merged[col].corr(merged["prob_median"])
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
