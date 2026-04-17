"""cli/location.py — Location registry introspection and fetch triggering.

Usage
-----
  python cli/location.py list
  python cli/location.py info <id>
  python cli/location.py bbox <id>
  python cli/location.py fetch <id> [--start YYYY-MM-DD] [--end YYYY-MM-DD]
                                     [--cloud-max N] [--stride N] [--no-nbar]

Examples
--------
  python cli/location.py list
  python cli/location.py info longreach
  python cli/location.py bbox muttaburra
  python cli/location.py fetch barcaldine --start 2022-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.location import all_locations, get  # noqa: E402


def cmd_list(args: argparse.Namespace) -> None:
    locs = sorted(all_locations(), key=lambda l: l.id)
    for loc in locs:
        print(f"  {loc.id:<22} {loc.name}")


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
        stride=args.stride,
        apply_nbar=not args.no_nbar,
    )
    print(f"Written: {out}")


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
    pf.add_argument("--stride", type=int, default=1,
                    help="Pixel grid stride, 1=every pixel (default: 1)")
    pf.add_argument("--no-nbar", action="store_true",
                    help="Disable BRDF NBAR c-factor correction")

    args = p.parse_args()
    {"list": cmd_list, "info": cmd_info, "bbox": cmd_bbox, "fetch": cmd_fetch}[args.cmd](args)


if __name__ == "__main__":
    main()
