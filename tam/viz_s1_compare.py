"""Compare Sentinel-1 VH/VV backscatter between two bboxes or region IDs.

Fetches S1 GRD data directly from Element84 STAC, computes mean backscatter
and VH/VV ratio by DOY, and plots presence vs absence side by side.

Usage:
    # Compare two region IDs from training.yaml
    python -m tam.viz_s1_compare \
        --region-a norman_road_presence_1 \
        --region-b norman_road_absence_7 \
        --out outputs/s1_compare_nr_p1_a7.png

    # Compare arbitrary bboxes
    python -m tam.viz_s1_compare \
        --bbox-a 141.647945 -20.397868 141.649492 -20.396426 \
        --bbox-b 141.727942 -20.187984 141.730785 -20.185069 \
        --year 2025 \
        --label-a "NR presence 1" \
        --label-b "NR absence 7" \
        --out outputs/s1_compare_candidate.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.stac import search_sentinel1, load_stackstac, filter_items_by_bbox
from utils.regions import select_regions
from utils.pipeline import setup_gdal_env
setup_gdal_env()

STAC_ENDPOINT_ELEMENT84 = "https://earth-search.aws.element84.com/v1"
S1_COLLECTION           = "sentinel-1-grd"

DOY_BINS   = np.arange(1, 366, 14)  # fortnightly bins
MONTH_DOYS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
MONTH_LBLS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
S1_BANDS   = ["vv", "vh"]
S1_CACHE_DIR = PROJECT_ROOT / "data" / "s1_cache"


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Append VH−VV ratio and RVI rows to a VH/VV DataFrame."""
    vv = df[df["band"] == "vv"].set_index("date")["mean"]
    vh = df[df["band"] == "vh"].set_index("date")["mean"]
    common = vv.index.intersection(vh.index)
    if len(common) == 0:
        return df

    extras = []

    # VH−VV ratio (dB difference = log ratio)
    extras.append(pd.DataFrame({
        "date": common,
        "band": "vh_vv_ratio",
        "mean": (vh.loc[common] - vv.loc[common]).values,
        "std":  0.0,
        "n":    vv.loc[common].values,
        "doy":  pd.to_datetime(common).day_of_year,
    }))

    # Radar Vegetation Index: 4*VH_lin / (VV_lin + VH_lin)
    # Convert dB back to linear for RVI, then store as linear (0-1)
    vv_lin = 10 ** (vv.loc[common] / 10)
    vh_lin = 10 ** (vh.loc[common] / 10)
    rvi = 4 * vh_lin / (vv_lin + vh_lin)
    extras.append(pd.DataFrame({
        "date": common,
        "band": "rvi",
        "mean": rvi.values,
        "std":  0.0,
        "n":    vv.loc[common].values,
        "doy":  pd.to_datetime(common).day_of_year,
    }))

    return pd.concat([df] + extras, ignore_index=True)


def _cache_path(bbox: list[float], year: int) -> Path:
    key = "_".join(f"{v:.6f}" for v in bbox) + f"_y{year}"
    return S1_CACHE_DIR / f"{key}.parquet"


def fetch_s1(bbox: list[float], year: int) -> pd.DataFrame:
    """Fetch S1 VH/VV observations for a bbox and year window, return as DataFrame.

    Results are cached to data/s1_cache/ as parquet — subsequent calls for the
    same bbox and year are served from disk without hitting S3.
    """
    cache = _cache_path(bbox, year)
    if cache.exists():
        print(f"  Loading from cache: {cache.name}")
        df = pd.read_parquet(cache)
        return _add_derived(df)

    start = f"{year - 2}-01-01"
    end   = f"{year}-12-31"

    items = search_sentinel1(
        bbox=bbox,
        start=start,
        end=end,
        endpoint=STAC_ENDPOINT_ELEMENT84,
        collection=S1_COLLECTION,
    )
    items = filter_items_by_bbox(items, bbox)

    if not items:
        print(f"  No S1 items found for bbox {bbox} in {start}/{end}")
        return pd.DataFrame()

    print(f"  Found {len(items)} S1 items")

    # Read S1 COGs directly via rasterio.
    # S1 GRD COGs have no embedded CRS or transform — reconstruct from STAC item
    # properties (proj:transform and proj:bbox are in EPSG:4326).
    import rasterio
    import rasterio.windows
    from rasterio.transform import Affine

    records = []
    for item in items:
        date = pd.to_datetime(item.datetime).date()

        # Reconstruct affine transform from item properties
        # proj:transform is [xres, 0, x_origin, 0, yres, y_origin]
        pt = item.properties.get("proj:transform")
        if pt is None:
            continue
        # stackstac convention: [xres, 0, x_origin, 0, yres, y_origin, 0, 0, 1]
        # rasterio Affine: (xres, xskew, x_origin, yskew, yres, y_origin)
        # proj:transform: [xres, xskew, x_origin, yskew, yres, y_origin, ...]
        # Affine(a, b, c, d, e, f) = (xres, xskew, x_origin, yskew, yres, y_origin)
        affine = Affine(pt[0], pt[1], pt[2], pt[3], pt[4], pt[5]) if len(pt) >= 6 else None
        if affine is None:
            continue

        for band in S1_BANDS:
            if band not in item.assets:
                continue
            href = item.assets[band].href
            try:
                with rasterio.open(href) as src:
                    lon_min, lat_min, lon_max, lat_max = bbox
                    win = rasterio.windows.from_bounds(
                        lon_min, lat_min, lon_max, lat_max,
                        transform=affine,
                    )
                    win = win.intersection(
                        rasterio.windows.Window(0, 0, src.width, src.height)
                    )
                    if win.width <= 0 or win.height <= 0:
                        continue
                    arr = src.read(1, window=win).astype(np.float32)
                    arr[arr == 0] = np.nan
                    arr = 10 * np.log10(arr)  # linear → dB
                    arr = arr[np.isfinite(arr)]
                    if len(arr) == 0:
                        continue
                    records.append({
                        "date": date,
                        "band": band,
                        "mean": float(np.mean(arr)),
                        "std":  float(np.std(arr)),
                        "n":    len(arr),
                    })
            except Exception as e:
                print(f"  Warning: could not read {band} for {item.id}: {e}")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["doy"] = pd.to_datetime(df["date"]).dt.day_of_year

    S1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    print(f"  Cached to {cache.name}")
    return _add_derived(df)


def bbox_for_region(region_id: str) -> tuple[list[float], int]:
    """Return (bbox, year) for a region ID from training.yaml."""
    regions = select_regions([region_id])
    r = regions[0]
    year = r.year if r.year is not None else 2024
    return list(r.bbox), year


def bin_by_doy(df: pd.DataFrame, band: str) -> tuple[np.ndarray, np.ndarray]:
    sub = df[df["band"] == band].dropna(subset=["mean"])
    if sub.empty:
        return np.full(len(DOY_BINS) - 1, np.nan), np.full(len(DOY_BINS) - 1, np.nan)
    bin_idx = np.clip(np.searchsorted(DOY_BINS, sub["doy"].values, side="right") - 1,
                      0, len(DOY_BINS) - 2)
    n_bins = len(DOY_BINS) - 1
    means  = np.full(n_bins, np.nan)
    stds   = np.full(n_bins, np.nan)
    for b in range(n_bins):
        vals = sub.loc[bin_idx == b, "mean"].values
        if len(vals) >= 2:
            means[b] = np.mean(vals)
            stds[b]  = np.std(vals)
    return means, stds


def fetch_regions(region_ids: list[str]) -> pd.DataFrame:
    """Fetch and concatenate S1 data for multiple region IDs."""
    chunks = []
    for rid in region_ids:
        bbox, year = bbox_for_region(rid)
        print(f"  Fetching {rid}...")
        df = fetch_s1(bbox, year)
        if not df.empty:
            df["region"] = rid
            chunks.append(df)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--presence", nargs="+", required=True,
                        help="Region IDs for presence group")
    parser.add_argument("--absence",  nargs="+", required=True,
                        help="Region IDs for absence group")
    parser.add_argument("--label-presence", default="presence")
    parser.add_argument("--label-absence",  default="absence")
    parser.add_argument("--out",  default="outputs/s1_compare.png")
    args = parser.parse_args()

    print("Fetching S1 for presence regions...")
    df_pres = fetch_regions(args.presence)
    print("Fetching S1 for absence regions...")
    # Fetch each absence region separately so we can plot them individually
    absence_dfs: dict[str, pd.DataFrame] = {}
    for rid in args.absence:
        bbox, year = bbox_for_region(rid)
        print(f"  Fetching {rid}...")
        df = fetch_s1(bbox, year)
        if not df.empty:
            df["region"] = rid
            absence_dfs[rid] = df

    if df_pres.empty or not absence_dfs:
        print("Error: no S1 data returned for one or both groups.")
        sys.exit(1)

    bands_to_plot = ["vh", "vv", "vh_vv_ratio", "rvi"]
    band_labels   = {"vh": "VH (dB)", "vv": "VV (dB)", "vh_vv_ratio": "VH−VV ratio (dB)", "rvi": "RVI (0–1)"}
    bin_centres   = (DOY_BINS[:-1] + DOY_BINS[1:]) / 2

    # Colour palette: presence always steelblue, each absence region gets its own colour
    absence_colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(absence_dfs)))
    absence_items  = list(absence_dfs.items())

    fig, axes = plt.subplots(len(bands_to_plot), 1,
                             figsize=(12, 4 * len(bands_to_plot)), sharex=True)
    fig.suptitle(
        f"S1 backscatter — presence: {', '.join(args.presence)}",
        fontsize=9,
    )

    summary_lines = [
        f"S1 backscatter — presence: {', '.join(args.presence)}",
        f"n_obs  presence: {len(df_pres)}",
        "",
        f"{'region':<35}  {'band':<16}  {'peak_sep_month':<14}  {'max_sep':>8}  direction",
        "-" * 85,
    ]

    for ax, band in zip(axes, bands_to_plot):
        # Plot pooled presence
        mp, sp = bin_by_doy(df_pres, band)
        valid = ~np.isnan(mp)
        if valid.any():
            ax.plot(bin_centres[valid], mp[valid], color="steelblue",
                    label=args.label_presence, linewidth=2.0)
            ax.fill_between(bin_centres[valid],
                            (mp - sp)[valid], (mp + sp)[valid],
                            color="steelblue", alpha=0.15)

        # Plot each absence region separately
        for (rid, df_a), color in zip(absence_items, absence_colors):
            ma, sa = bin_by_doy(df_a, band)
            valid_a = ~np.isnan(ma)
            if not valid_a.any():
                continue
            short = rid.replace("norman_road_", "").replace("frenchs_", "")
            ax.plot(bin_centres[valid_a], ma[valid_a], color=color,
                    label=short, linewidth=1.5, linestyle="--")
            ax.fill_between(bin_centres[valid_a],
                            (ma - sa)[valid_a], (ma + sa)[valid_a],
                            color=color, alpha=0.1)

            # Summary per absence region
            sep = mp - ma
            valid_sep = ~(np.isnan(mp) | np.isnan(ma))
            if valid_sep.any():
                max_b     = np.nanargmax(np.abs(sep))
                max_sep   = sep[max_b]
                doy_peak  = bin_centres[max_b]
                month_idx = max(0, np.searchsorted(MONTH_DOYS, doy_peak, side="right") - 1)
                month     = MONTH_LBLS[month_idx]
                direction = "pres>abs" if max_sep > 0 else "abs>pres"
                summary_lines.append(
                    f"{rid:<35}  {band_labels.get(band,band):<16}  {month:<14}  "
                    f"{abs(max_sep):>8.3f}  {direction}"
                )

        ax.set_ylabel(band_labels.get(band, band), fontsize=9)
        ax.set_xticks(MONTH_DOYS)
        ax.set_xticklabels(MONTH_LBLS, fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        summary_lines.append("")

    axes[-1].set_xlabel("Day of year", fontsize=9)
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary_text = "\n".join(summary_lines)
    out_path.with_suffix(".txt").write_text(summary_text)
    print(summary_text)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
