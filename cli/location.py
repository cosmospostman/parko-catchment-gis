"""cli/location.py — Location registry introspection and fetch triggering.

Usage
-----
  python cli/location.py list
  python cli/location.py info <id>
  python cli/location.py bbox <id>
  python cli/location.py fetch <id> [--start YYYY-MM-DD] [--end YYYY-MM-DD]
                                     [--cloud-max N] [--no-nbar]
  python cli/location.py training list
  python cli/location.py training fetch [--regions ID ...] [--all]
                                         [--cloud-max N] [--no-nbar]

Examples
--------
  python cli/location.py list
  python cli/location.py info longreach
  python cli/location.py bbox muttaburra
  python cli/location.py fetch barcaldine --start 2022-01-01 --end 2024-12-31
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
    print(f"  {'ID':<26} {'STATUS':<9} {'CHIPS':>8}  {'PARQUET':>8}")
    print("  " + "-" * 56)
    for loc in locs:
        parquet = loc.parquet_path()
        chips   = parquet.parent / (loc.id + ".chips")
        fetched      = parquet.exists()
        status       = "fetched" if fetched else ""
        parquet_str  = _fmt_size(parquet.stat().st_size) if fetched else "—"
        chips_str    = _fmt_size(_dir_size(chips)) if chips.exists() else "—"
        print(f"  {loc.id:<26} {status:<9} {chips_str:>8}  {parquet_str:>8}")


def cmd_info(args: argparse.Namespace) -> None:
    try:
        loc = get(args.id)
    except KeyError:
        print(f"Unknown location: {args.id!r}", file=sys.stderr)
        sys.exit(1)

    print(loc.summary())

    parquet = loc.parquet_path()
    if not parquet.exists():
        return

    import pandas as pd

    df = pd.read_parquet(parquet, columns=["date"])
    df["date"] = pd.to_datetime(df["date"])
    counts = (
        df.groupby(df["date"].dt.to_period("M"))["date"]
        .nunique()
        .rename("n")
    )
    if counts.empty:
        return

    print()
    print("  Scene count per month")
    for period, n in counts.items():
        label = period.strftime("%b %Y").upper()
        bar = "#" * n
        print(f"  {label:<12} {n:>4}  {bar}")


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

    out = loc.fetch(
        start=args.start,
        end=args.end,
        cloud_max=args.cloud_max,
        apply_nbar=not args.no_nbar,
    )
    print(f"Written: {out}")


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
    pf.add_argument("--start", default="2020-01-01", help="Start date (default: 2020-01-01)")
    pf.add_argument("--end", default=None, help="End date (default: today)")
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
