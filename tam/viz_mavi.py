"""Plot MAVI and ΔMAVI/Δt temporal profiles grouped by region tag class.

MAVI = (B08 - B04) / (B08 + B04 + B11)

Computes per-pixel MAVI time series, then derives ΔMAVI/Δt (rate of change
per day between consecutive observations). Both are binned by DOY and plotted
as mean ± std for each tag class.

Usage:
    python -m tam.viz_mavi --regions landsend_sparse_presence_1 \
        landsend_presence_1 landsend_absence_1 frenchs_absence_bare_soil_2

    python -m tam.viz_mavi --tags sparse bare_soil riparian \
        --out outputs/mavi_halo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import re
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.regions import load_regions
from utils.training_collector import tile_ids_for_regions, tile_parquet_path

LOAD_COLS = ["point_id", "lon", "lat", "date", "scl_purity", "B04", "B08", "B11", "NDVI", "vh", "source"]

# Woody filter thresholds (matches TAMConfig defaults)
PRESENCE_MIN_VH_DRY_DB     = -21.0
PRESENCE_NDVI_RESCUE_VH_DB = -23.0
PRESENCE_NDVI_RESCUE_MIN   = 0.50
DRY_DOY_MIN = 121   # May 1
DRY_DOY_MAX = 304   # Oct 31

DOY_BINS   = np.arange(1, 366, 10)
BIN_CENTRES = (DOY_BINS[:-1] + DOY_BINS[1:]) / 2
MONTH_DOYS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
MONTH_LBLS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# Colour cycle — up to 12 tag classes
COLOURS = ["steelblue", "coral", "seagreen", "mediumpurple",
           "darkorange", "crimson", "teal", "saddlebrown",
           "gold", "navy", "hotpink", "olivedrab"]

SCL_PURITY_MIN = 0.5


def apply_woody_filter(pixel_df: pl.DataFrame) -> pl.DataFrame:
    """Drop presence pixel-years that fail the S1 VH woody filter."""
    if "source" not in pixel_df.columns or "vh" not in pixel_df.columns:
        print("Woody filter: skipped — no source/vh columns")
        return pixel_df

    presence_pids = set(
        pixel_df.filter(pl.col("tag_class").str.contains("presence"))["point_id"].to_list()
    )
    if not presence_pids:
        return pixel_df

    s1_pres = pixel_df.filter(
        (pl.col("source") == "S1") &
        pl.col("point_id").is_in(presence_pids) &
        pl.col("doy").is_between(DRY_DOY_MIN, DRY_DOY_MAX)
    )
    if s1_pres.is_empty():
        print("Woody filter: skipped — no S1 rows found for presence pixels")
        return pixel_df

    vh_lin = s1_pres["vh"].to_numpy().astype(np.float64)
    vh_db  = np.where(vh_lin > 0, 10.0 * np.log10(vh_lin), np.nan)
    s1_pres = s1_pres.with_columns(pl.Series("_vh_db", vh_db)).filter(
        pl.col("_vh_db").is_not_null() & pl.col("_vh_db").is_not_nan()
    )
    mean_vh_df = s1_pres.group_by(["point_id", "year"]).agg(
        pl.col("_vh_db").mean().alias("mean_vh")
    )
    mean_vh_map = {(r["point_id"], r["year"]): r["mean_vh"]
                   for r in mean_vh_df.iter_rows(named=True)}

    s2_pres = pixel_df.filter(
        (pl.col("source") == "S2") &
        pl.col("point_id").is_in(presence_pids) &
        pl.col("doy").is_between(DRY_DOY_MIN, DRY_DOY_MAX) &
        pl.col("NDVI").is_not_null()
    )
    mean_ndvi_map = {(r["point_id"], r["year"]): r["mean_ndvi"]
                     for r in s2_pres.group_by(["point_id", "year"])
                                     .agg(pl.col("NDVI").mean().alias("mean_ndvi"))
                                     .iter_rows(named=True)}

    drop_py: set[tuple] = set()
    for (pid, yr), vh in mean_vh_map.items():
        if vh < PRESENCE_MIN_VH_DRY_DB:
            ndvi = mean_ndvi_map.get((pid, yr), float("nan"))
            rescued = (vh >= PRESENCE_NDVI_RESCUE_VH_DB and
                       not np.isnan(ndvi) and ndvi >= PRESENCE_NDVI_RESCUE_MIN)
            if not rescued:
                drop_py.add((pid, yr))

    if drop_py:
        is_presence = pixel_df["tag_class"].str.contains("presence").to_numpy()
        pids = pixel_df["point_id"].to_numpy()
        years = pixel_df["year"].to_numpy()
        in_drop = np.array([
            bool(is_presence[i]) and (pids[i], int(years[i])) in drop_py
            for i in range(len(pixel_df))
        ])
        n_dropped = int(in_drop.sum())
        print(f"Woody filter: dropped {n_dropped} presence obs across {len(drop_py)} pixel-years")
        return pixel_df.filter(~pl.Series(in_drop))

    print("Woody filter: no pixel-years dropped")
    return pixel_df


def compute_mavi(df: pl.DataFrame) -> pl.Series:
    b04 = df["B04"].to_numpy().astype("float32")
    b08 = df["B08"].to_numpy().astype("float32")
    b11 = df["B11"].to_numpy().astype("float32")
    denom = b08 + b04 + b11
    mavi = np.where(denom != 0, (b08 - b04) / denom, np.nan)
    return pl.Series("MAVI", mavi)


def compute_dmavi(pixel_df: pl.DataFrame) -> pl.DataFrame:
    """Compute ΔMAVI/Δt (per day) for each pixel's consecutive observations."""
    pixel_df = pixel_df.sort(["point_id", "date"])
    rows = []
    for (pid,), grp in pixel_df.group_by(["point_id"], maintain_order=True):
        mavi = grp["MAVI"].to_numpy().astype(float)
        days = grp["date"].cast(pl.Date).to_numpy().astype("datetime64[D]").astype(float)
        doy  = grp["doy"].to_numpy()
        tag  = grp["tag_class"].to_numpy()
        dt = np.diff(days)
        dm = np.diff(mavi)
        rate = np.where(dt > 0, dm / dt, np.nan)
        for i in range(len(rate)):
            rows.append({"point_id": pid, "doy": int(doy[i + 1]),
                         "DMAVI": float(rate[i]), "tag_class": str(tag[i + 1])})
    return pl.DataFrame(rows) if rows else pl.DataFrame({"point_id": [], "doy": [], "DMAVI": [], "tag_class": []})


# Seasonal windows for histogram: (label, doy_min, doy_max)
SEASON_WINDOWS = [
    ("Wet (Jan–Mar)",    1,   90),
    ("Wet→Dry (Apr)",   91,  120),
    ("Early dry (May)", 121, 151),
    ("Mid dry (Jun–Aug)", 152, 243),
    ("Late dry (Sep–Oct)", 244, 304),
]


def plot_histograms(pixel_df: pl.DataFrame, regions, tag_class_for,
                    out_dir: Path, colours: list[str]) -> None:
    """One PNG per region: KDE of per-pixel dry-season MAVI across seasonal windows."""
    from scipy.stats import gaussian_kde

    n_windows = len(SEASON_WINDOWS)

    for region in regions:
        lon_min, lat_min, lon_max, lat_max = region.bbox
        rdf = pixel_df.filter(
            (pl.col("lon") >= lon_min) & (pl.col("lon") <= lon_max) &
            (pl.col("lat") >= lat_min) & (pl.col("lat") <= lat_max)
        ).select(["point_id", "doy", "MAVI"]).drop_nulls()
        if rdf.is_empty():
            continue

        tc = tag_class_for(region)
        n_pix = rdf["point_id"].n_unique()

        fig, axes = plt.subplots(1, n_windows, figsize=(4 * n_windows, 4), sharey=False)
        fig.suptitle(f"{region.id}  [{tc}]  —  {n_pix} pixels", fontsize=10)

        colour = colours[0] if "presence" in tc else colours[min(3, len(colours) - 1)]

        for ax, (win_label, doy_min, doy_max) in zip(axes, SEASON_WINDOWS):
            vals = rdf.filter(pl.col("doy").is_between(doy_min, doy_max))["MAVI"].to_numpy()
            vals = vals[np.isfinite(vals)]
            ax.set_title(win_label, fontsize=8)
            ax.set_xlabel("MAVI", fontsize=8)
            ax.axvline(0, color="grey", linewidth=0.7, linestyle="--")

            if len(vals) < 10:
                ax.text(0.5, 0.5, "insufficient\ndata", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color="grey")
                continue

            # Histogram bars
            ax.hist(vals, bins=30, density=True, alpha=0.3, color=colour)

            # KDE overlay
            try:
                kde = gaussian_kde(vals, bw_method="scott")
                x = np.linspace(vals.min(), vals.max(), 200)
                ax.plot(x, kde(x), color=colour, linewidth=1.5)
            except Exception:
                pass

            ax.axvline(np.median(vals), color=colour, linewidth=1.0,
                       linestyle=":", label=f"median={np.median(vals):.3f}")
            ax.legend(fontsize=7, loc="upper right")
            ax.tick_params(labelsize=7)

        plt.tight_layout()
        out_path = out_dir / f"hist_{region.id}.png"
        plt.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


def bin_mean_std(series_doy: np.ndarray, series_val: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    means = np.full(len(BIN_CENTRES), np.nan)
    stds  = np.full(len(BIN_CENTRES), np.nan)
    bin_idx = np.searchsorted(DOY_BINS, series_doy, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, len(BIN_CENTRES) - 1)
    for b in range(len(BIN_CENTRES)):
        vals = series_val[bin_idx == b]
        vals = vals[~np.isnan(vals)]
        if len(vals) >= 3:
            means[b] = np.mean(vals)
            stds[b]  = np.std(vals)
    return means, stds


def plot_signal(ax: plt.Axes, df: pl.DataFrame, col: str,
                tag_classes: list[str], colours: list[str]) -> None:
    for tag, colour in zip(tag_classes, colours):
        sub = df.filter(pl.col("tag_class") == tag).select([col, "doy"]).drop_nulls()
        if sub.is_empty():
            continue
        means, stds = bin_mean_std(sub["doy"].to_numpy(), sub[col].to_numpy())
        valid = ~np.isnan(means)
        if not valid.any():
            continue
        ax.plot(BIN_CENTRES[valid], means[valid], color=colour,
                label=tag, linewidth=1.5)
        ax.fill_between(BIN_CENTRES[valid],
                        (means - stds)[valid],
                        (means + stds)[valid],
                        color=colour, alpha=0.15)
    ax.set_xticks(MONTH_DOYS)
    ax.set_xticklabels(MONTH_LBLS, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--regions", nargs="+",
                       help="Explicit region IDs to load")
    group.add_argument("--tags", nargs="+",
                       help="Load all regions whose tags include ANY of these")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: outputs/mavi_halo)")
    parser.add_argument("--min-purity", type=float, default=SCL_PURITY_MIN,
                        help="Minimum scl_purity to include (default 0.5)")
    parser.add_argument("--group-by-site", action="store_true",
                        help="Group by site+label (e.g. corfield/presence) instead of tags")
    parser.add_argument("--pixel-zscore", action="store_true",
                        help="Apply per-pixel z-score to MAVI (same normalisation as v9 training)")
    parser.add_argument("--min-obs-per-year", type=int, default=8,
                        help="Drop pixel-years with fewer observations (default 8, matches training)")
    parser.add_argument("--woody-filter", action="store_true",
                        help="Apply S1 VH woody filter to presence pixels (matches training label cleaning)")
    parser.add_argument("--hist", action="store_true",
                        help="Also produce per-bbox MAVI histograms across seasonal windows")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs/mavi_halo")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_regions = load_regions()

    if args.regions:
        region_ids = args.regions
        all_by_id  = {r.id: r for r in all_regions}
        missing    = [rid for rid in region_ids if rid not in all_by_id]
        if missing:
            print(f"Unknown region IDs: {missing}", file=sys.stderr)
            sys.exit(1)
        regions = [all_by_id[rid] for rid in region_ids]
    else:
        wanted_tags = set(args.tags)
        regions = [r for r in all_regions if wanted_tags & set(r.tags)]
        if not regions:
            print(f"No regions matched tags: {args.tags}", file=sys.stderr)
            sys.exit(1)
        region_ids = [r.id for r in regions]

    print(f"Loading {len(regions)} regions: {[r.id for r in regions]}")

    # Build tag_class label per region
    def tag_class_for(region) -> str:
        if args.group_by_site:
            # e.g. "corfield/presence", "landsend/sparse_presence"
            site = re.match(r"^(.+?)_(presence|absence|sparse)", region.id)
            site_name = site.group(1) if site else region.id
            return f"{site_name}/{region.label}"
        distinguish = [t for t in region.tags
                       if t not in ("arid", "semi-arid", "monsoonal", "riparian",
                                    "savanna", "gps_surveyed")]
        suffix = f"/{distinguish[0]}" if distinguish else ""
        return f"{region.label}{suffix}"

    # Load parquet data
    tile_ids = tile_ids_for_regions(region_ids)
    chunks: list[pl.DataFrame] = []
    for tid in tile_ids:
        path = tile_parquet_path(tid)
        if not path.exists():
            continue
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        cols = [c for c in LOAD_COLS if c in available]
        for rg in range(pf.metadata.num_row_groups):
            chunks.append(pl.from_arrow(pf.read_row_group(rg, columns=cols)))

    if not chunks:
        print("No parquet data found for the selected regions.", file=sys.stderr)
        sys.exit(1)

    pixel_df = pl.concat(chunks).with_columns(
        pl.col("date").cast(pl.Date).dt.ordinal_day().alias("doy")
    )
    # SCL filter applies to S2 rows only; S1 rows have NaN scl_purity — keep them for woody filter
    pixel_df = pixel_df.filter(
        (pl.col("scl_purity") >= args.min_purity) | (pl.col("source") == "S1")
    )

    # Assign tag_class by matching point_id to region bboxes
    tag_class_arr = [None] * len(pixel_df)
    lons = pixel_df["lon"].to_numpy()
    lats = pixel_df["lat"].to_numpy()
    for region in regions:
        lon_min, lat_min, lon_max, lat_max = region.bbox
        tc = tag_class_for(region)
        mask = (lons >= lon_min) & (lons <= lon_max) & (lats >= lat_min) & (lats <= lat_max)
        for i in np.where(mask)[0]:
            tag_class_arr[i] = tc

    pixel_df = pixel_df.with_columns(
        pl.Series("tag_class", tag_class_arr)
    ).filter(pl.col("tag_class").is_not_null()).with_columns(
        pl.col("date").cast(pl.Date).dt.year().alias("year")
    )

    # Apply S1 VH woody filter before dropna (needs vh column)
    if args.woody_filter:
        pixel_df = apply_woody_filter(pixel_df)

    # Drop rows with NaN in any required band (matches training dropna)
    pixel_df = pixel_df.drop_nulls(subset=["B04", "B08", "B11"])
    obs_counts = pixel_df.group_by(["point_id", "year"]).agg(pl.len().alias("n"))
    keep_py = set(
        (r["point_id"], r["year"])
        for r in obs_counts.filter(pl.col("n") >= args.min_obs_per_year).iter_rows(named=True)
    )
    py_arr = list(zip(pixel_df["point_id"].to_list(), pixel_df["year"].to_list()))
    pixel_df = pixel_df.filter(pl.Series([py in keep_py for py in py_arr]))

    # Compute MAVI on raw reflectance
    pixel_df = pixel_df.with_columns(compute_mavi(pixel_df))

    # Per-pixel z-score of MAVI — mirrors training pixel_zscore=True
    if args.pixel_zscore:
        stats_df = pixel_df.group_by("point_id").agg([
            pl.col("MAVI").mean().alias("_m"),
            pl.col("MAVI").std().alias("_s"),
        ])
        pixel_df = pixel_df.join(stats_df, on="point_id", how="left").with_columns(
            ((pl.col("MAVI") - pl.col("_m")) /
             pl.col("_s").fill_null(1e-6).clip(lower_bound=1e-6)).alias("MAVI")
        ).drop(["_m", "_s"])

    tag_classes = sorted(pixel_df["tag_class"].drop_nulls().unique().to_list())
    colours = COLOURS[:len(tag_classes)]

    print(f"Tag classes found: {tag_classes}")
    for tc in tag_classes:
        n = pixel_df.filter(pl.col("tag_class") == tc)["point_id"].n_unique()
        print(f"  {tc}: {n} pixels")

    # Per-bbox histograms
    if args.hist:
        hist_dir = out_dir / "histograms"
        hist_dir.mkdir(exist_ok=True)
        plot_histograms(pixel_df, regions, tag_class_for, hist_dir, colours)

    # Compute ΔMAVI/Δt
    dmavi_df = compute_dmavi(pixel_df)

    # --- Plot -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    ax_mavi, ax_dm = axes

    plot_signal(ax_mavi, pixel_df, "MAVI",  tag_classes, colours)
    ax_mavi.set_ylabel("MAVI", fontsize=10)
    ax_mavi.set_title("MAVI by DOY — mean ± 1 std", fontsize=11)
    ax_mavi.axhline(0, color="grey", linewidth=0.7, linestyle="--")

    plot_signal(ax_dm, dmavi_df, "DMAVI", tag_classes, colours)
    ax_dm.set_ylabel("ΔMAVI / day", fontsize=10)
    ax_dm.set_title("ΔMAVI/Δt by DOY — rate of change per day", fontsize=11)
    ax_dm.axhline(0, color="grey", linewidth=0.7, linestyle="--")
    ax_dm.set_xlabel("Day of year", fontsize=9)

    notes = []
    if args.pixel_zscore:  notes.append("pixel z-scored")
    if args.woody_filter:  notes.append("woody filter")
    note_str = f" [{', '.join(notes)}]" if notes else ""
    fig.suptitle(f"MAVI soil moisture halo analysis — Parkinsonia{note_str}", fontsize=12)
    plt.tight_layout()

    out_path = out_dir / "mavi_temporal.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Per-class CSV summary of MAVI and ΔMAVI/Δt dry-season minima
    summary_rows = []
    for tc in tag_classes:
        dry_m  = pixel_df.filter(
            (pl.col("tag_class") == tc) & pl.col("doy").is_between(150, 270)
        )["MAVI"].drop_nulls().to_numpy()
        dry_dm = dmavi_df.filter(
            (pl.col("tag_class") == tc) & pl.col("doy").is_between(150, 270)
        )["DMAVI"].drop_nulls().to_numpy()
        summary_rows.append({
            "tag_class":      tc,
            "mavi_dry_mean":  float(np.mean(dry_m))  if len(dry_m)  > 0 else float("nan"),
            "mavi_dry_std":   float(np.std(dry_m))   if len(dry_m)  > 0 else float("nan"),
            "dmavi_dry_mean": float(np.mean(dry_dm)) if len(dry_dm) > 0 else float("nan"),
            "dmavi_dry_min":  float(np.min(dry_dm))  if len(dry_dm) > 0 else float("nan"),
        })
    summary_df = pl.DataFrame(summary_rows)
    csv_path   = out_dir / "mavi_summary.csv"
    summary_df.write_csv(csv_path, float_precision=4)
    print(f"Saved: {csv_path}")
    print(summary_df)


if __name__ == "__main__":
    main()
