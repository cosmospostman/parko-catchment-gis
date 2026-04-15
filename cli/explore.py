"""cli/explore.py — Signal diagnostic report generator.

Reads presence/absence bboxes from a Location YAML, computes all signals,
runs a parameter sweep on each, and writes a comprehensive Markdown report.

Usage
-----
  python cli/explore.py --location longreach-8x8km --out outputs/explore-longreach

  python cli/explore.py --location longreach-8x8km \
      --year-from 2021 --year-to 2024 \
      --out outputs/explore-longreach

The location must have at least one sub_bbox with role 'presence' and one
with role 'absence'.  These define the infestation and background classes.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.location import get as get_location                         # noqa: E402
from signals import (                                                  # noqa: E402
    QualityParams,
    NirCvSignal,
    RecPSignal,
    RedEdgeSignal,
    SwirSignal,
    NdviIntegralSignal,
    RecessionSensitivitySignal,
    GreenupTimingSignal,
    extract_parko_features,
    sweep_signal,
)
from signals.diagnostics import (                                      # noqa: E402
    _resolve_classes,
    separability_score,
    plot_signal_map,
    plot_distributions,
)
from signals._shared import annual_ndvi_curve_chunked, ensure_pixel_sorted  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def _fmt_sep(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.3f}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python cli/explore.py",
        description="Generate a signal diagnostic report for a location.",
    )
    p.add_argument(
        "--location", required=True,
        help="Location id (e.g. longreach-8x8km).  Must have presence + absence sub_bboxes.",
    )
    p.add_argument(
        "--out", required=True, type=Path,
        help="Output directory for report and figures.",
    )
    p.add_argument("--year-from", type=int, default=None, metavar="YYYY")
    p.add_argument("--year-to",   type=int, default=None, metavar="YYYY")
    return p.parse_args()


def _validate_location(loc) -> None:
    has_pres = any(s.role == "presence" for s in loc.sub_bboxes.values())
    has_abs  = any(s.role == "absence"  for s in loc.sub_bboxes.values())
    if not has_pres or not has_abs:
        sys.exit(
            f"ERROR: Location '{loc.id}' must have at least one sub_bbox with "
            f"role='presence' and one with role='absence'.\n"
            f"Found roles: {[s.role for s in loc.sub_bboxes.values()]}"
        )


# ---------------------------------------------------------------------------
# Tabular feature signals
# ---------------------------------------------------------------------------

TABULAR_SIGNALS = [
    ("nir_cv",        "NIR CV (dry-season inter-annual)",  "viridis_r"),
    ("rec_p",         "NDVI amplitude (wet/dry swing)",    "YlGn"),
    ("re_p10",        "Red-edge ratio floor (p10)",        "YlGn"),
    ("swir_p10",      "SWIR moisture floor (p10)",         "YlGn"),
    ("ndvi_integral", "NDVI integral (mean annual NDVI)",  "YlGn"),
]

SWEEP_GRIDS: dict[str, dict] = {
    "nir_cv": {
        "scl_purity_min": [0.3, 0.5, 0.7],
        "min_obs_dry":    [3, 5, 8],
    },
    "rec_p": {
        "scl_purity_min":   [0.3, 0.5, 0.7],
        "min_obs_per_year": [5, 10, 15],
    },
    "re_p10": {
        "floor_percentile": [0.05, 0.10, 0.20],
        "scl_purity_min":   [0.3, 0.5, 0.7],
    },
    "swir_p10": {
        "floor_percentile": [0.05, 0.10, 0.20],
        "scl_purity_min":   [0.3, 0.5, 0.7],
    },
    "ndvi_integral": {
        "smooth_days": [15, 30, 45],
        "min_years":   [2, 3, 4],
    },
}

SIGNAL_CLASS_MAP = {
    "nir_cv":        NirCvSignal,
    "rec_p":         RecPSignal,
    "re_p10":        RedEdgeSignal,
    "swir_p10":      SwirSignal,
    "ndvi_integral": NdviIntegralSignal,
}


# ---------------------------------------------------------------------------
# Curve-based signals
# ---------------------------------------------------------------------------

CURVE_SIGNALS = [
    ("recession_sensitivity", RecessionSensitivitySignal,
     "Recession sensitivity (r: slope vs wet-NDWI)", "RdYlGn"),
    ("peak_doy", GreenupTimingSignal,
     "Green-up peak DOY (mean annual)", "RdYlBu_r"),
]


# ---------------------------------------------------------------------------
# Core steps
# ---------------------------------------------------------------------------

def step_tabular_features(
    loc,
    out_dir: Path,
    year_from: int | None,
    year_to: int | None,
) -> pd.DataFrame:
    log("\n[1/4] Extracting tabular features (nir_cv, rec_p, re_p10, swir_p10, ndvi_integral) ...")
    features = extract_parko_features(
        loc.parquet_path(), loc,
        year_from=year_from,
        year_to=year_to,
    )
    features.to_csv(out_dir / "features.csv", index=False, float_format="%.5f")
    log(f"      {len(features):,} pixels  →  features.csv")
    return features


def step_tabular_diagnostics(
    features: pd.DataFrame,
    loc,
    out_dir: Path,
) -> list[dict]:
    """Map + distribution PNG for each tabular signal; returns separability rows.

    Maps and distributions are produced from the pre-computed features table.
    Parameter sweeps re-load raw pixel observations because each signal
    recomputes from scratch with different quality/algorithm params.
    """
    log("\n[2/4] Per-signal maps, distributions, and parameter sweeps ...")
    presence_ids, absence_ids = _resolve_classes(features, loc)

    # Load raw observations once for the sweeps (all tabular signals need them).
    raw_df: pd.DataFrame | None = None
    parquet_path = loc.parquet_path()
    if parquet_path.exists():
        log(f"  Loading raw pixels for parameter sweeps ...")
        raw_df = pd.read_parquet(parquet_path)
        log(f"    {len(raw_df):,} observations, {raw_df['point_id'].nunique():,} pixels")
    else:
        log("  WARNING: parquet not found — parameter sweeps will be skipped")

    rows = []

    for col, description, cmap in TABULAR_SIGNALS:
        if col not in features.columns or features[col].isna().all():
            log(f"  {col}: no data — skipping")
            continue

        sep = separability_score(features, col, presence_ids, absence_ids)
        log(f"  {col}: separability = {_fmt_sep(sep)}")

        plot_signal_map(
            features, col, loc,
            title=f"{loc.name} — {description}",
            out_path=out_dir / f"map_{col}.png",
            colormap=cmap,
        )
        plot_distributions(
            features, col, loc,
            presence_ids=presence_ids,
            absence_ids=absence_ids,
            out_path=out_dir / f"dist_{col}.png",
        )

        # Parameter sweep — needs raw observations
        best_sweep_sep = None
        if raw_df is not None:
            signal_cls = SIGNAL_CLASS_MAP[col]
            sweep_grid = SWEEP_GRIDS[col]
            log(f"    sweeping {list(sweep_grid.keys())} ...")
            sweep_df = sweep_signal(signal_cls, sweep_grid, raw_df, loc)
            sweep_df.to_csv(out_dir / f"sweep_{col}.csv", index=False, float_format="%.4f")
            best_sweep_sep = sweep_df["separability"].abs().max() if len(sweep_df) else None

        rows.append({
            "signal": col,
            "description": description,
            "separability": sep,
            "n_presence": len(presence_ids) if presence_ids is not None else None,
            "n_absence":  len(absence_ids)  if absence_ids  is not None else None,
            "best_sweep_sep": best_sweep_sep,
        })

    return rows


def step_curve_signals(
    loc,
    out_dir: Path,
    features: pd.DataFrame,
) -> list[dict]:
    """Recession sensitivity and green-up timing signals (need NDVI curve)."""
    log("\n[3/4] Curve-based signals (recession, greenup) ...")

    parquet_path = loc.parquet_path()
    if not parquet_path.exists():
        log("  WARNING: parquet not found — skipping curve-based signals")
        return []

    sorted_path = ensure_pixel_sorted(parquet_path)
    presence_ids, absence_ids = _resolve_classes(features, loc)

    rows = []

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

        # Load raw observations once — recession signal needs NDWI columns
        # which are not in the NDVI curve parquet.
        log("  Loading raw pixels for curve-based signals ...")
        raw_df = pd.read_parquet(sorted_path)
        log(f"    {len(raw_df):,} observations")

        for col, signal_cls, description, cmap in CURVE_SIGNALS:
            log(f"  {col} ...")
            sig = signal_cls()
            stats = sig.compute(raw_df, loc, _curve=curve_path)

            if col not in stats.columns or stats[col].isna().all():
                log(f"    {col}: no data — skipping")
                continue

            sep = separability_score(stats, col, presence_ids, absence_ids)
            log(f"    {col}: separability = {_fmt_sep(sep)}")

            plot_signal_map(
                stats, col, loc,
                title=f"{loc.name} — {description}",
                out_path=out_dir / f"map_{col}.png",
                colormap=cmap,
            )
            plot_distributions(
                stats, col, loc,
                presence_ids=presence_ids,
                absence_ids=absence_ids,
                out_path=out_dir / f"dist_{col}.png",
            )

            rows.append({
                "signal": col,
                "description": description,
                "separability": sep,
                "n_presence": len(presence_ids) if presence_ids is not None else None,
                "n_absence":  len(absence_ids)  if absence_ids  is not None else None,
                "best_sweep_sep": None,  # sweep not implemented for curve-based signals
            })

    finally:
        curve_path.unlink(missing_ok=True)

    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _md_table(rows: list[dict], cols: list[str]) -> str:
    """Format a list of dicts as a Markdown table."""
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines  = [header, sep]
    for row in rows:
        cells = []
        for c in cols:
            v = row.get(c)
            if v is None:
                cells.append("—")
            elif isinstance(v, float):
                cells.append(f"{v:+.3f}" if c in ("separability", "best_sweep_sep") else f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _sweep_table_md(csv_path: Path) -> str:
    if not csv_path.exists():
        return "_Sweep results not available._"
    df = pd.read_csv(csv_path)
    if df.empty:
        return "_No sweep results._"
    # Sort by |separability| descending
    df = df.copy()
    if "separability" in df.columns:
        df["_abs"] = df["separability"].abs()
        df = df.sort_values("_abs", ascending=False).drop(columns=["_abs"])
    cols = [c for c in df.columns if not c.startswith("_")]
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines  = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                cells.append("—")
            elif isinstance(v, float):
                cells.append(f"{v:+.4f}" if c == "separability" else f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    loc,
    out_dir: Path,
    all_rows: list[dict],
    year_from: int | None,
    year_to: int | None,
) -> None:
    log("\n[4/4] Writing report.md ...")

    year_label = "all years"
    if year_from and year_to:
        year_label = f"{year_from}–{year_to}"
    elif year_from:
        year_label = f"{year_from}–present"
    elif year_to:
        year_label = f"up to {year_to}"

    # Presence / absence bbox info
    presence_sub = [s for s in loc.sub_bboxes.values() if s.role == "presence"]
    absence_sub  = [s for s in loc.sub_bboxes.values() if s.role == "absence"]

    def _fmt_bbox(b: list) -> str:
        return f"`{b[0]:.5f}, {b[1]:.5f}, {b[2]:.5f}, {b[3]:.5f}`"

    lines = [
        f"# Signal Exploration Report — {loc.name}",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Location:** `{loc.id}`  ",
        f"**Parquet:** `{loc.parquet_path()}`  ",
        f"**Years:** {year_label}",
        "",
        "## Classes",
        "",
    ]

    if presence_sub:
        for s in presence_sub:
            lines.append(f"- **Presence** ({s.label}): {_fmt_bbox(s.bbox)}")
    if absence_sub:
        for s in absence_sub:
            lines.append(f"- **Absence** ({s.label}): {_fmt_bbox(s.bbox)}")
    lines.append("")

    # Overview separability table
    lines += [
        "## Separability Overview",
        "",
        "> Separability = (presence median − absence median) / pooled SD.  ",
        "> Positive → presence higher; negative → absence higher.  ",
        "> |separability| > 1 is a useful rule of thumb for a discriminating signal.",
        "",
        _md_table(
            all_rows,
            ["signal", "description", "n_presence", "n_absence", "separability", "best_sweep_sep"],
        ),
        "",
    ]

    # Per-signal sections
    for row in all_rows:
        col   = row["signal"]
        desc  = row["description"]
        sep   = row.get("separability")
        sweep_csv = out_dir / f"sweep_{col}.csv"

        lines += [
            f"## Signal: `{col}`",
            "",
            f"**{desc}**  ",
            f"**Separability (default params):** {_fmt_sep(sep)}",
            "",
            f"| Map | Distribution |",
            f"|-----|-------------|",
            f"| ![map](map_{col}.png) | ![dist](dist_{col}.png) |",
            "",
        ]

        if sweep_csv.exists():
            lines += [
                "### Parameter Sweep",
                "",
                _sweep_table_md(sweep_csv),
                "",
            ]
        else:
            lines += [
                "### Parameter Sweep",
                "",
                "_Not available for this signal (requires NDVI curve — run separately)._",
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
    _validate_location(loc)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Location : {loc.name}  ({loc.id})")
    log(f"Parquet  : {loc.parquet_path()}")
    log(f"Output   : {out_dir}")
    presence_subs = [s for s in loc.sub_bboxes.values() if s.role == "presence"]
    absence_subs  = [s for s in loc.sub_bboxes.values() if s.role == "absence"]
    for s in presence_subs:
        log(f"  Presence  [{s.label}]: {s.bbox}")
    for s in absence_subs:
        log(f"  Absence   [{s.label}]: {s.bbox}")

    features  = step_tabular_features(loc, out_dir, args.year_from, args.year_to)
    tab_rows  = step_tabular_diagnostics(features, loc, out_dir)
    curv_rows = step_curve_signals(loc, out_dir, features)
    all_rows  = tab_rows + curv_rows

    # Separability summary CSV
    sep_df = pd.DataFrame(all_rows)
    sep_df.to_csv(out_dir / "separability.csv", index=False, float_format="%.4f")

    write_report(loc, out_dir, all_rows, args.year_from, args.year_to)
    log(f"\nDone.  Open {out_dir}/report.md to view the report.")


if __name__ == "__main__":
    main()
