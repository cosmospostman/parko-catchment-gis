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
import pandas as pd
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


def apply_woody_filter(pixel_df: pd.DataFrame) -> pd.DataFrame:
    """Drop presence pixel-years that fail the S1 VH woody filter.

    Mirrors _apply_presence_filter in train.py:
      drop if mean_vh_dry < PRESENCE_MIN_VH_DRY_DB
      AND NOT (mean_vh_dry >= PRESENCE_NDVI_RESCUE_VH_DB AND mean_ndvi_dry >= PRESENCE_NDVI_RESCUE_MIN)
    Only applied to presence pixels; absence pixels are unchanged.

    VH is computed from S1-sourced rows only (source == "S1").
    NDVI rescue uses S2-sourced dry-season rows.
    """
    if "source" not in pixel_df.columns or "vh" not in pixel_df.columns:
        print("Woody filter: skipped — no source/vh columns")
        return pixel_df

    presence_pids = set(pixel_df[pixel_df["tag_class"].str.contains("presence")]["point_id"])
    if not presence_pids:
        return pixel_df

    # S1 rows only — VH in linear, convert to dB
    s1_pres = pixel_df[
        (pixel_df["source"] == "S1") &
        pixel_df["point_id"].isin(presence_pids) &
        pixel_df["doy"].between(DRY_DOY_MIN, DRY_DOY_MAX)
    ].copy()

    if s1_pres.empty:
        print("Woody filter: skipped — no S1 rows found for presence pixels")
        return pixel_df

    s1_pres["vh_db"] = np.where(s1_pres["vh"] > 0, 10.0 * np.log10(s1_pres["vh"]), np.nan)
    s1_pres = s1_pres[np.isfinite(s1_pres["vh_db"])]
    mean_vh = s1_pres.groupby(["point_id", "year"])["vh_db"].mean()

    # S2 NDVI rescue — clear-sky dry-season obs
    s2_pres = pixel_df[
        (pixel_df["source"] == "S2") &
        pixel_df["point_id"].isin(presence_pids) &
        pixel_df["doy"].between(DRY_DOY_MIN, DRY_DOY_MAX) &
        pixel_df["NDVI"].notna()
    ]
    mean_ndvi = s2_pres.groupby(["point_id", "year"])["NDVI"].mean()

    fails_strict = mean_vh < PRESENCE_MIN_VH_DRY_DB
    rescued = (
        (mean_vh >= PRESENCE_NDVI_RESCUE_VH_DB) &
        (mean_ndvi.reindex(mean_vh.index) >= PRESENCE_NDVI_RESCUE_MIN)
    )
    drop_py = set(map(tuple, mean_vh.index[fails_strict & ~rescued.fillna(False)].tolist()))

    if drop_py:
        presence_mask = pixel_df["tag_class"].str.contains("presence")
        py_index = list(zip(pixel_df["point_id"], pixel_df["year"]))
        in_drop = pd.Series([tuple(x) in drop_py for x in py_index], index=pixel_df.index)
        keep = ~(presence_mask & in_drop)
        n_dropped = (~keep).sum()
        n_py = len(drop_py)
        print(f"Woody filter: dropped {n_dropped} presence obs across {n_py} pixel-years")
        return pixel_df[keep].copy()

    print("Woody filter: no pixel-years dropped")
    return pixel_df


def compute_mavi(df: pd.DataFrame) -> pd.Series:
    b04 = df["B04"].values.astype("float32")
    b08 = df["B08"].values.astype("float32")
    b11 = df["B11"].values.astype("float32")
    denom = b08 + b04 + b11
    mavi = np.where(denom != 0, (b08 - b04) / denom, np.nan)
    return pd.Series(mavi, index=df.index, name="MAVI")


def compute_dmavi(pixel_df: pd.DataFrame) -> pd.DataFrame:
    """Compute ΔMAVI/Δt (per day) for each pixel's consecutive observations."""
    pixel_df = pixel_df.sort_values(["point_id", "date"])
    pixel_df["date_dt"] = pd.to_datetime(pixel_df["date"])

    rows = []
    for pid, grp in pixel_df.groupby("point_id", sort=False):
        grp = grp.sort_values("date_dt")
        mavi = grp["MAVI"].values
        days = grp["date_dt"].values.astype("datetime64[D]").astype(float)
        dt = np.diff(days)
        dm = np.diff(mavi)
        # rate = dm/dt; assign to the later observation's DOY
        rate = np.where(dt > 0, dm / dt, np.nan)
        doy  = grp["doy"].values[1:]
        tag  = grp["tag_class"].values[1:]
        for i in range(len(rate)):
            rows.append({"point_id": pid, "doy": doy[i],
                         "DMAVI": rate[i], "tag_class": tag[i]})
    return pd.DataFrame(rows)


# Seasonal windows for histogram: (label, doy_min, doy_max)
SEASON_WINDOWS = [
    ("Wet (Jan–Mar)",    1,   90),
    ("Wet→Dry (Apr)",   91,  120),
    ("Early dry (May)", 121, 151),
    ("Mid dry (Jun–Aug)", 152, 243),
    ("Late dry (Sep–Oct)", 244, 304),
]


def plot_histograms(pixel_df: pd.DataFrame, regions, tag_class_for,
                    out_dir: Path, colours: list[str]) -> None:
    """One PNG per region: KDE of per-pixel dry-season MAVI across seasonal windows."""
    from scipy.stats import gaussian_kde

    # Build region_id -> tag_class map for titles
    n_windows = len(SEASON_WINDOWS)

    for region in regions:
        lon_min, lat_min, lon_max, lat_max = region.bbox
        mask = (
            (pixel_df["lon"] >= lon_min) & (pixel_df["lon"] <= lon_max) &
            (pixel_df["lat"] >= lat_min) & (pixel_df["lat"] <= lat_max)
        )
        rdf = pixel_df[mask][["point_id", "doy", "MAVI"]].dropna()
        if rdf.empty:
            continue

        tc = tag_class_for(region)
        n_pix = rdf["point_id"].nunique()

        fig, axes = plt.subplots(1, n_windows, figsize=(4 * n_windows, 4), sharey=False)
        fig.suptitle(f"{region.id}  [{tc}]  —  {n_pix} pixels", fontsize=10)

        colour = colours[0] if "presence" in tc else colours[min(3, len(colours) - 1)]

        for ax, (win_label, doy_min, doy_max) in zip(axes, SEASON_WINDOWS):
            vals = rdf[rdf["doy"].between(doy_min, doy_max)]["MAVI"].values
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


def plot_signal(ax: plt.Axes, df: pd.DataFrame, col: str,
                tag_classes: list[str], colours: list[str]) -> None:
    for tag, colour in zip(tag_classes, colours):
        sub = df[df["tag_class"] == tag][[col, "doy"]].dropna()
        if sub.empty:
            continue
        means, stds = bin_mean_std(sub["doy"].values, sub[col].values)
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
    chunks: list[pd.DataFrame] = []
    for tid in tile_ids:
        path = tile_parquet_path(tid)
        if not path.exists():
            continue
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(rg, columns=LOAD_COLS)
            chunks.append(tbl.to_pandas())

    if not chunks:
        print("No parquet data found for the selected regions.", file=sys.stderr)
        sys.exit(1)

    pixel_df = pd.concat(chunks, ignore_index=True)
    # SCL filter applies to S2 rows only; S1 rows have NaN scl_purity — keep them for woody filter
    pixel_df = pixel_df[(pixel_df["scl_purity"] >= args.min_purity) | (pixel_df["source"] == "S1")].copy()
    pixel_df["date"] = pd.to_datetime(pixel_df["date"])
    pixel_df["doy"]  = pixel_df["date"].dt.day_of_year

    # Assign tag_class by matching point_id to region bboxes
    pixel_df["tag_class"] = pd.NA
    for region in regions:
        lon_min, lat_min, lon_max, lat_max = region.bbox
        mask = (
            (pixel_df["lon"] >= lon_min) & (pixel_df["lon"] <= lon_max) &
            (pixel_df["lat"] >= lat_min) & (pixel_df["lat"] <= lat_max)
        )
        tc = tag_class_for(region)
        pixel_df.loc[mask, "tag_class"] = tc

    pixel_df = pixel_df[pixel_df["tag_class"].notna()].copy()
    pixel_df["year"] = pixel_df["date"].dt.year

    # Apply S1 VH woody filter before dropna (needs vh column)
    if args.woody_filter:
        pixel_df = apply_woody_filter(pixel_df)

    # Drop rows with NaN in any required band (matches training dropna)
    pixel_df = pixel_df.dropna(subset=["B04", "B08", "B11"])
    obs_counts = pixel_df.groupby(["point_id", "year"]).size()
    keep = obs_counts[obs_counts >= args.min_obs_per_year].index
    pixel_df = pixel_df.set_index(["point_id", "year"]).loc[
        pixel_df.set_index(["point_id", "year"]).index.isin(keep)
    ].reset_index()

    # Compute MAVI on raw reflectance
    pixel_df["MAVI"] = compute_mavi(pixel_df)

    # Per-pixel z-score of MAVI — mirrors training pixel_zscore=True
    # Applied to MAVI (not raw bands) so the index formulation is preserved
    if args.pixel_zscore:
        stats = pixel_df.groupby("point_id")["MAVI"].agg(["mean", "std"])
        pid_arr = pixel_df["point_id"].values
        m = stats["mean"].reindex(pid_arr).values
        s = stats["std"].reindex(pid_arr).clip(lower=1e-6).fillna(1e-6).values
        pixel_df["MAVI"] = (pixel_df["MAVI"].values - m) / s

    tag_classes = sorted(pixel_df["tag_class"].unique())
    colours = COLOURS[:len(tag_classes)]

    print(f"Tag classes found: {tag_classes}")
    for tc in tag_classes:
        n = pixel_df[pixel_df["tag_class"] == tc]["point_id"].nunique()
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
        sub_m  = pixel_df[pixel_df["tag_class"] == tc][["doy", "MAVI"]].dropna()
        sub_dm = dmavi_df[dmavi_df["tag_class"] == tc][["doy", "DMAVI"]].dropna()
        # Dry season = DOY 150-270 (roughly May-Sep in northern Australia)
        dry_m  = sub_m[(sub_m["doy"] >= 150) & (sub_m["doy"] <= 270)]["MAVI"]
        dry_dm = sub_dm[(sub_dm["doy"] >= 150) & (sub_dm["doy"] <= 270)]["DMAVI"]
        summary_rows.append({
            "tag_class":          tc,
            "mavi_dry_mean":      dry_m.mean()  if len(dry_m)  > 0 else np.nan,
            "mavi_dry_std":       dry_m.std()   if len(dry_m)  > 0 else np.nan,
            "dmavi_dry_mean":     dry_dm.mean() if len(dry_dm) > 0 else np.nan,
            "dmavi_dry_min":      dry_dm.min()  if len(dry_dm) > 0 else np.nan,
        })
    summary_df = pd.DataFrame(summary_rows)
    csv_path   = out_dir / "mavi_summary.csv"
    summary_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved: {csv_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
