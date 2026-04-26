"""cli/location.py — Location registry introspection and fetch triggering.

Usage
-----
  python cli/location.py list
  python cli/location.py info <id>
  python cli/location.py bbox <id>
  python cli/location.py fetch <id> --years YYYY [YYYY ...]
                                     [--cloud-max N] [--no-nbar]
  python cli/location.py training list
  python cli/location.py training fetch [--regions ID ...] [--all]
                                         [--cloud-max N] [--no-nbar]
  python cli/location.py training verify

Examples
--------
  python cli/location.py list
  python cli/location.py info longreach
  python cli/location.py bbox muttaburra
  python cli/location.py fetch longreach --years 2020 2021 2022
  python cli/location.py fetch longreach --years 2024
  python cli/location.py training list
  python cli/location.py training fetch --all
  python cli/location.py training fetch --regions lake_mueller_presence barcoorah_presence
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.location import all_locations, get  # noqa: E402


def _fmt_size(n_bytes: int) -> str:
    if n_bytes >= 1e12:
        return f"{n_bytes/1e12:.1f} TB"
    if n_bytes >= 1e9:
        return f"{n_bytes/1e9:.1f} GB"
    if n_bytes >= 1e6:
        return f"{n_bytes/1e6:.0f} MB"
    return f"{n_bytes/1e3:.0f} KB"


def _dir_size(p: "Path") -> int:
    import os
    total = 0
    for entry in os.scandir(p):
        if entry.is_file(follow_symlinks=False):
            total += entry.stat().st_size
        elif entry.is_dir(follow_symlinks=False):
            total += _dir_size(Path(entry.path))
    return total


def cmd_list(args: argparse.Namespace) -> None:
    locs = sorted(all_locations(), key=lambda l: l.id)

    rows = []
    for loc in locs:
        chips = loc.chips_path()
        years = loc.parquet_years()
        if years:
            years_str = f"{years[0]}–{years[-1]}" if len(years) > 1 else str(years[0])
            tile_paths = loc.parquet_tile_paths()
            total_bytes = sum(p.stat().st_size for ps in tile_paths.values() for p in ps)
            parquet_str = _fmt_size(total_bytes)
        else:
            years_str   = "—"
            parquet_str = "—"
        chips_str  = _fmt_size(_dir_size(chips)) if chips.exists() else "—"
        area_str   = f"{loc.area_km2:.1f}"
        pixels_str = f"{loc.pixel_count:,}"
        rows.append((loc.id, years_str, area_str, pixels_str, chips_str, parquet_str))

    headers = ("ID", "YEARS", "AREA km²", "PIXELS", "CHIPS", "PARQUET")
    # columns 0,1 are left-aligned; 2-5 are right-aligned
    widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt_row(cols):
        id_col, yr, area, px, chips, parq = cols
        w = widths
        return (f"  {id_col:<{w[0]}} {yr:<{w[1]}} {area:>{w[2]}}  {px:>{w[3]}}  "
                f"{chips:>{w[4]}}  {parq:>{w[5]}}")

    print(fmt_row(headers))
    print("  " + "-" * (sum(widths) + 12))
    for row in rows:
        print(fmt_row(row))


def cmd_info(args: argparse.Namespace) -> None:
    try:
        loc = get(args.id)
    except KeyError:
        print(f"Unknown location: {args.id!r}", file=sys.stderr)
        sys.exit(1)

    print(loc.summary())

    years = loc.parquet_years()
    if not years:
        return

    import pandas as pd

    print()
    print(f"  Fetched years: {', '.join(str(y) for y in years)}")

    tile_paths_by_year = loc.parquet_tile_paths()
    for year in years:
        tile_paths = tile_paths_by_year.get(year, [])
        total_size = sum(p.stat().st_size for p in tile_paths)
        size_str = _fmt_size(total_size)
        dfs = [pd.read_parquet(p, columns=["date"]) for p in tile_paths]
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["date"])
        df["date"] = pd.to_datetime(df["date"])
        counts = (
            df.groupby(df["date"].dt.to_period("M"))["date"]
            .nunique()
            .rename("n")
        )
        print()
        print(f"  {year}  ({size_str})  — scene count per month")
        for period, n in counts.items():
            label = period.strftime("%b %Y").upper()
            bar = "#" * n
            print(f"    {label:<12} {n:>4}  {bar}")


def cmd_bbox(args: argparse.Namespace) -> None:
    try:
        print(get(args.id).bbox_cli)
    except KeyError:
        print(f"Unknown location: {args.id!r}", file=sys.stderr)
        sys.exit(1)


def cmd_fetch(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("rasterio").setLevel(logging.WARNING)

    try:
        loc = get(args.id)
    except KeyError:
        print(f"Unknown location: {args.id!r}", file=sys.stderr)
        sys.exit(1)

    written = loc.fetch(
        years=args.years,
        cloud_max=args.cloud_max,
        apply_nbar=not args.no_nbar,
    )
    for path in written:
        print(f"Written: {path}")


def cmd_training_list(args: argparse.Namespace) -> None:
    from utils.regions import load_regions
    from utils.location import _bbox_pixel_count

    regions = load_regions()
    totals: dict[str, int] = {}

    print(f"  {'ID':<40} {'LABEL':<10} {'YEAR':<6} {'PIXELS':>8}")
    print("  " + "-" * 68)
    for r in regions:
        n = _bbox_pixel_count(r.bbox)
        totals[r.label] = totals.get(r.label, 0) + n
        year_str = str(r.year) if r.year else "—"
        print(f"  {r.id:<40} {r.label:<10} {year_str:<6} {n:>8,}")

    print("  " + "-" * 68)
    for label, total in sorted(totals.items()):
        print(f"  {'Total ' + label:<40} {'':10} {'':6} {total:>8,}")


def cmd_training_fetch(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("rasterio").setLevel(logging.WARNING)
    from utils.regions import load_regions, select_regions
    from utils.training_collector import ensure_training_pixels

    regions = load_regions() if args.all else select_regions(args.regions)
    ensure_training_pixels(
        regions=regions,
        cloud_max=args.cloud_max,
        apply_nbar=not args.no_nbar,
        max_concurrent=args.max_concurrent,
    )


def cmd_training_verify(args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd
    from utils.regions import load_regions
    from utils.training_collector import _region_parquet_path

    BAND_COLS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

    regions = load_regions()
    issues: list[str] = []
    rows = []

    for r in regions:
        path = _region_parquet_path(r.id)
        if not path.exists():
            issues.append(f"MISSING  {r.id} — parquet not found at {path}")
            continue

        df = pd.read_parquet(path)

        n_pixels = df["point_id"].nunique()
        n_obs    = len(df)
        dupes    = df.duplicated(subset=["point_id", "date"]).sum()

        # Observations per pixel
        obs_per_pixel = df.groupby("point_id").size()
        obs_min, obs_med, obs_max = obs_per_pixel.min(), obs_per_pixel.median(), obs_per_pixel.max()

        # Band stats — mean and std across all obs
        band_means = df[BAND_COLS].mean()
        band_stds  = df[BAND_COLS].std()
        nan_counts = df[BAND_COLS].isna().sum().sum()

        # Flag suspicious values
        if dupes > 0:
            issues.append(f"DUPES    {r.id} — {dupes} duplicate (point_id, date) rows")
        if n_pixels < 5:
            issues.append(f"SPARSE   {r.id} — only {n_pixels} pixels")
        if obs_min < 4:
            issues.append(f"LOW_OBS  {r.id} — some pixels have <4 observations (min={obs_min})")
        if nan_counts > 0:
            issues.append(f"NAN      {r.id} — {nan_counts} NaN band values")
        if band_means.max() > 1.5 or band_means.min() < -0.5:
            issues.append(f"RANGE    {r.id} — band means outside expected range [{band_means.min():.2f}, {band_means.max():.2f}]")

        rows.append((r.id, r.label, n_pixels, n_obs, int(obs_min), int(obs_med), int(obs_max),
                     f"{band_means.mean():.3f}", f"{band_stds.mean():.3f}"))

    # Print table
    headers = ("REGION", "LABEL", "PIXELS", "OBS", "MIN_OBS", "MED_OBS", "MAX_OBS", "MEAN_BAND", "STD_BAND")
    widths  = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]

    def fmt(cols):
        return (f"  {str(cols[0]):<{widths[0]}}  {str(cols[1]):<{widths[1]}}"
                f"  {str(cols[2]):>{widths[2]}}  {str(cols[3]):>{widths[3]}}"
                f"  {str(cols[4]):>{widths[4]}}  {str(cols[5]):>{widths[5]}}"
                f"  {str(cols[6]):>{widths[6]}}  {str(cols[7]):>{widths[7]}}"
                f"  {str(cols[8]):>{widths[8]}}")

    print(fmt(headers))
    print("  " + "-" * (sum(widths) + 18))
    for row in rows:
        print(fmt(row))

    # Summary
    if rows:
        total_pixels = sum(r[2] for r in rows)
        presence_px  = sum(r[2] for r in rows if r[1] == "presence")
        absence_px   = sum(r[2] for r in rows if r[1] == "absence")
        print()
        print(f"  Total pixels: {total_pixels:,}  (presence: {presence_px:,}  absence: {absence_px:,}  ratio: 1:{absence_px//max(presence_px,1)})")

    if issues:
        print()
        print(f"  {len(issues)} issue(s) found:")
        for iss in issues:
            print(f"    ! {iss}")
    else:
        print()
        print("  No issues found.")


def cmd_training(args: argparse.Namespace) -> None:
    {
        "list":   cmd_training_list,
        "fetch":  cmd_training_fetch,
        "verify": cmd_training_verify,
    }[args.training_cmd](args)


def main() -> None:
    p = argparse.ArgumentParser(
        prog="python cli/location.py",
        description="Inspect and fetch Parkinsonia analysis locations.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List all known locations")

    pi = sub.add_parser("info", help="Print full summary for a location")
    pi.add_argument("id", help="Location id (e.g. longreach, muttaburra)")

    pb = sub.add_parser("bbox", help="Print bbox as 'lon_min,lat_min,lon_max,lat_max'")
    pb.add_argument("id", help="Location id")

    pf = sub.add_parser("fetch", help="Fetch Sentinel-2 pixel observations")
    pf.add_argument("id", help="Location id")
    pf.add_argument("--years", nargs="+", type=int, required=True, metavar="YYYY",
                    help="Calendar years to fetch (e.g. --years 2020 2021 2022)")
    pf.add_argument("--cloud-max", type=int, default=30, metavar="N",
                    help="Max cloud cover %% (default: 30)")
    pf.add_argument("--no-nbar", action="store_true",
                    help="Disable BRDF NBAR c-factor correction")

    pt = sub.add_parser("training", help="Manage training regions and pixel collection")
    tsub = pt.add_subparsers(dest="training_cmd", required=True)

    tsub.add_parser("list", help="List all training regions with estimated pixel counts")

    tsub.add_parser("verify", help="Verify training data quality and flag issues")

    tf = tsub.add_parser("fetch", help="Fetch pixels for training regions")
    grp = tf.add_mutually_exclusive_group(required=True)
    grp.add_argument("--regions", nargs="+", metavar="ID",
                     help="Region IDs to fetch")
    grp.add_argument("--all", action="store_true",
                     help="Fetch all regions in training.yaml")
    tf.add_argument("--cloud-max", type=int, default=80, metavar="N")
    tf.add_argument("--no-nbar", action="store_true")
    tf.add_argument("--max-concurrent", type=int, default=32, metavar="N",
                    help="Max concurrent HTTP patch fetches per tile (default: 32)")

    args = p.parse_args()
    {
        "list":     cmd_list,
        "info":     cmd_info,
        "bbox":     cmd_bbox,
        "fetch":    cmd_fetch,
        "training": cmd_training,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
