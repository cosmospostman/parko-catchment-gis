"""cli/describe.py — Single-bbox signal description / fingerprint report.

Usage
-----
  python cli/describe.py --location frenchs --bbox woody_1 --out outputs/describe-frenchs-woody1

  python cli/describe.py --location frenchs \
      --bbox "[141.537, -15.805, 141.539, -15.803]" \
      --out outputs/describe-frenchs-custom

  python cli/describe.py --location frenchs --bbox woody_1 \
      --year-from 2021 --year-to 2024 \
      --out outputs/describe-frenchs-woody1-2021-2024
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.location import get as get_location                         # noqa: E402
from signals import (                                                  # noqa: E402
    QualityParams,
    RecessionSensitivitySignal,
    GreenupTimingSignal,
    extract_parko_features,
)
from signals.scl_composition import SclCompositionSignal               # noqa: E402
from signals.diagnostics import (                                      # noqa: E402
    plot_signal_map,
    plot_distributions,
    plot_scl_timeseries,
)
from signals._shared import annual_ndvi_curve_chunked, ensure_pixel_sorted  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python cli/describe.py",
        description="Compute a signal fingerprint for a single bbox.",
    )
    p.add_argument("--location", required=True, help="Location id.")
    p.add_argument(
        "--bbox", required=True,
        help='sub_bbox key from location YAML, or inline "[lon_min, lat_min, lon_max, lat_max]".',
    )
    p.add_argument("--out", required=True, type=Path, help="Output directory.")
    p.add_argument("--year-from", type=int, default=None, metavar="YYYY")
    p.add_argument("--year-to",   type=int, default=None, metavar="YYYY")
    return p.parse_args()


def resolve_bbox(
    bbox_arg: str, loc
) -> tuple[tuple[float, float, float, float], str]:
    """Return (bbox_tuple, label) from a named sub_bbox key or inline JSON list."""
    if bbox_arg in loc.sub_bboxes:
        sub = loc.sub_bboxes[bbox_arg]
        return tuple(sub.bbox), sub.label
    parsed = json.loads(bbox_arg)
    assert len(parsed) == 4, "Inline bbox must have exactly 4 values."
    return tuple(parsed), f"[{bbox_arg}]"


# ---------------------------------------------------------------------------
# Signal descriptive stats
# ---------------------------------------------------------------------------

TABULAR_SIGNALS = [
    ("nir_cv",        "NIR CV (dry-season inter-annual)",  "viridis_r"),
    ("rec_p",         "NDVI amplitude (wet/dry swing)",    "YlGn"),
    ("re_p10",        "Red-edge ratio floor (p10)",        "YlGn"),
    ("swir_p10",      "SWIR moisture floor (p10)",         "YlGn"),
    ("ndvi_integral", "NDVI integral (mean annual NDVI)",  "YlGn"),
]

CURVE_SIGNALS = [
    ("recession_sensitivity", RecessionSensitivitySignal,
     "Recession sensitivity (r: slope vs wet-NDWI)", "RdYlGn"),
    ("peak_doy", GreenupTimingSignal,
     "Green-up peak DOY (mean annual)", "RdYlBu_r"),
]


def _signal_stats(series: pd.Series) -> dict:
    vals = series.dropna()
    if len(vals) == 0:
        return {"n_pixels": 0, "pct_valid": 0.0,
                "median": None, "iqr": None, "p10": None, "p90": None}
    return {
        "n_pixels":  len(vals),
        "pct_valid": round(len(vals) / len(series) * 100, 1),
        "median":    round(float(np.median(vals)), 4),
        "iqr":       round(float(np.percentile(vals, 75) - np.percentile(vals, 25)), 4),
        "p10":       round(float(np.percentile(vals, 10)), 4),
        "p90":       round(float(np.percentile(vals, 90)), 4),
    }


# ---------------------------------------------------------------------------
# Core steps
# ---------------------------------------------------------------------------

def step_tabular_features(
    loc, bbox_tuple, out_dir, year_from, year_to
) -> pd.DataFrame:
    log("\n[1/3] Extracting tabular features ...")
    features = extract_parko_features(
        loc.parquet_path(), loc,
        bbox=bbox_tuple,
        year_from=year_from,
        year_to=year_to,
    )
    features.to_csv(out_dir / "features.csv", index=False, float_format="%.5f")
    log(f"      {len(features):,} pixels  →  features.csv")
    return features


def step_curve_signals(
    loc, bbox_tuple, out_dir, features
) -> pd.DataFrame:
    """Compute recession and green-up signals, post-filter to bbox."""
    log("\n[2/3] Curve-based signals (recession, greenup) ...")

    parquet_path = loc.parquet_path()
    if not parquet_path.exists():
        log("  WARNING: parquet not found — skipping curve-based signals")
        return features

    sorted_path = ensure_pixel_sorted(parquet_path)

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        curve_path = Path(f.name)

    try:
        log("  Building NDVI curve (smooth_days=30) ...")
        annual_ndvi_curve_chunked(
            sorted_path, out_path=curve_path,
            smooth_days=30,
            min_obs_per_year=QualityParams().min_obs_per_year,
            scl_purity_min=0.0,
        )
        log(f"  Curve written ({curve_path.stat().st_size / 1e6:.1f} MB)")

        log("  Loading raw pixels for curve-based signals ...")
        raw_df = pd.read_parquet(sorted_path)
        log(f"    {len(raw_df):,} observations")

        lon_min, lat_min, lon_max, lat_max = bbox_tuple

        for col, signal_cls, _desc, _cmap in CURVE_SIGNALS:
            log(f"  {col} ...")
            stats = signal_cls().compute(raw_df, loc, _curve=curve_path)

            if col not in stats.columns or stats[col].isna().all():
                log(f"    {col}: no data — skipping")
                continue

            # Post-filter to bbox pixels
            stats = stats[
                stats["lon"].between(lon_min, lon_max) &
                stats["lat"].between(lat_min, lat_max)
            ]
            log(f"    {col}: {len(stats):,} bbox pixels")

            features = features.merge(
                stats[["point_id", col]], on="point_id", how="left"
            )

    finally:
        curve_path.unlink(missing_ok=True)

    return features


def step_scl_timeseries(loc, bbox_tuple, out_dir) -> pd.DataFrame | None:
    """Compute SCL class fractions as an actual time series for the bbox."""
    log("\n[3/3] SCL class composition time series ...")

    parquet_path = loc.parquet_path()
    if not parquet_path.exists():
        log("  WARNING: parquet not found — skipping SCL")
        return None

    raw_df = pd.read_parquet(parquet_path)
    lon_min, lat_min, lon_max, lat_max = bbox_tuple
    raw_bbox = raw_df[
        raw_df["lon"].between(lon_min, lon_max) &
        raw_df["lat"].between(lat_min, lat_max)
    ]

    if raw_bbox.empty:
        log("  WARNING: no pixels in bbox — skipping SCL")
        return None

    ts = SclCompositionSignal().compute_timeseries(raw_bbox, loc)
    if ts.empty:
        return None

    log(f"  {ts['point_id'].nunique():,} pixels, "
        f"{ts[['year','month']].drop_duplicates().__len__()} time steps")
    return ts


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt_stat(v) -> str:
    if v is None:
        return "—"
    return f"{v:.4f}"


def _high_variance_flag(iqr) -> str:
    if iqr is None:
        return ""
    return "⚠ high variance" if iqr > 0.15 else ""


def write_report(
    loc,
    bbox_tuple,
    bbox_label,
    out_dir,
    features,
    ts_df,
    year_from,
    year_to,
) -> None:
    log("\nWriting report.md ...")

    year_label = "all years"
    if year_from and year_to:
        year_label = f"{year_from}–{year_to}"
    elif year_from:
        year_label = f"{year_from}–present"
    elif year_to:
        year_label = f"up to {year_to}"

    all_signals = [
        (col, desc, cmap) for col, desc, cmap in TABULAR_SIGNALS
    ] + [
        (col, desc, cmap) for col, cls, desc, cmap in CURVE_SIGNALS
    ]

    # Compute stats and produce per-signal plots
    stat_rows = []
    for col, desc, cmap in all_signals:
        if col not in features.columns or features[col].isna().all():
            continue
        s = _signal_stats(features[col])
        stat_rows.append({
            "signal": col,
            "description": desc,
            "n_valid": s["n_pixels"],
            "pct_valid": s["pct_valid"],
            "median": s["median"],
            "iqr": s["iqr"],
            "p10": s["p10"],
            "p90": s["p90"],
            "flag": _high_variance_flag(s["iqr"]),
        })

        plot_signal_map(
            features, col, loc,
            title=f"{loc.name} — {desc}",
            out_path=out_dir / f"map_{col}.png",
            colormap=cmap,
        )
        plot_distributions(
            features, col, loc,
            out_path=out_dir / f"dist_{col}.png",
        )

    # SCL timeseries plot
    scl_plot_path = out_dir / "scl_timeseries.png"
    if ts_df is not None and not ts_df.empty:
        plot_scl_timeseries(
            ts_df,
            title=f"{loc.name} — SCL class composition ({bbox_label})",
            out_path=scl_plot_path,
        )

    def _fmt_bbox(b) -> str:
        return f"`{b[0]:.6f}, {b[1]:.6f}, {b[2]:.6f}, {b[3]:.6f}`"

    lines = [
        f"# Signal Description — {loc.name}",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Location:** `{loc.id}`  ",
        f"**Bbox:** {bbox_label} — {_fmt_bbox(bbox_tuple)}  ",
        f"**Pixels:** {len(features):,}  ",
        f"**Years:** {year_label}  ",
        f"**Parquet:** `{loc.parquet_path()}`",
        "",
        "## Signal Summary",
        "",
        "| signal | description | n_valid | pct_valid | median | IQR | p10 | p90 | flag |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for r in stat_rows:
        lines.append(
            f"| {r['signal']} | {r['description']} | {r['n_valid']} | {r['pct_valid']}% "
            f"| {_fmt_stat(r['median'])} | {_fmt_stat(r['iqr'])} "
            f"| {_fmt_stat(r['p10'])} | {_fmt_stat(r['p90'])} | {r['flag']} |"
        )

    lines += [""]

    # SCL timeseries section
    if ts_df is not None and not ts_df.empty:
        lines += [
            "## SCL Class Composition",
            "",
            f"![SCL timeseries](scl_timeseries.png)",
            "",
        ]

    # Per-signal sections
    for r in stat_rows:
        col  = r["signal"]
        desc = r["description"]
        lines += [
            f"## Signal: `{col}`",
            "",
            f"**{desc}**",
            "",
            f"| n_valid | pct_valid | median | IQR | p10 | p90 |",
            f"| --- | --- | --- | --- | --- | --- |",
            f"| {r['n_valid']} | {r['pct_valid']}% | {_fmt_stat(r['median'])} "
            f"| {_fmt_stat(r['iqr'])} | {_fmt_stat(r['p10'])} | {_fmt_stat(r['p90'])} |",
            "",
            f"| Map | Distribution |",
            f"| --- | --- |",
            f"| ![map](map_{col}.png) | ![dist](dist_{col}.png) |",
            "",
        ]

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    log(f"      → {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    loc = get_location(args.location)
    bbox_tuple, bbox_label = resolve_bbox(args.bbox, loc)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Location : {loc.name}  ({loc.id})")
    log(f"Bbox     : {bbox_label}  {bbox_tuple}")
    log(f"Parquet  : {loc.parquet_path()}")
    log(f"Output   : {out_dir}")

    features = step_tabular_features(
        loc, bbox_tuple, out_dir, args.year_from, args.year_to
    )
    features = step_curve_signals(loc, bbox_tuple, out_dir, features)
    ts_df    = step_scl_timeseries(loc, bbox_tuple, out_dir)

    write_report(
        loc, bbox_tuple, bbox_label,
        out_dir, features, ts_df,
        args.year_from, args.year_to,
    )
    log(f"\nDone.  Open {out_dir}/report.md to view the report.")


if __name__ == "__main__":
    main()
