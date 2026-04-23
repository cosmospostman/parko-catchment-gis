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


def cmd_training(args: argparse.Namespace) -> None:
    {
        "list":  cmd_training_list,
        "fetch": cmd_training_fetch,
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
