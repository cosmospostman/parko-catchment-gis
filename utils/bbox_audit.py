"""utils/bbox_audit.py — Multi-year quality audit for Mitchell training bboxes.

For each region in training.yaml matching --prefix, runs pixel_timeseries for
every year in the region's years list, reduces each year to five summary scalars,
applies label-appropriate pass/flag/fail thresholds, and writes a markdown report.

Usage::

    python utils/bbox_audit.py \\
        --root /mnt/external/chunkstore \\
        --training data/locations/training.yaml \\
        --prefix mitchell \\
        --out docs/MITCHELL-BBOX-AUDIT.md

    # Single region for a quick check:
    python utils/bbox_audit.py --bbox-id mitchell_presence_1 ...
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.pixel_reader import ChunkIndex
from utils.pixel_timeseries import _mean, _percentile, _safe_div
from signals.base import Signal

# ---------------------------------------------------------------------------
# Cover-type thresholds (from docs/MITCHELL-TRAIN.md)
# ---------------------------------------------------------------------------

@dataclass
class Thresholds:
    wet_ndvi_lo: float
    wet_ndvi_hi: float
    dry_ndvi_lo: float
    dry_ndvi_hi: float
    delta_lo: float
    delta_hi: float
    vhvv_lo: float   # most negative (e.g. −9)
    vhvv_hi: float   # least negative (e.g. −3)
    check_water: bool = False


THRESHOLDS: dict[str, Thresholds] = {
    "presence": Thresholds(
        wet_ndvi_lo=0.75, wet_ndvi_hi=0.90,
        dry_ndvi_lo=0.20, dry_ndvi_hi=0.35,
        delta_lo=0.50,    delta_hi=9.9,
        vhvv_lo=-6.0,     vhvv_hi=-3.0,
    ),
    "mangrove": Thresholds(
        wet_ndvi_lo=0.85, wet_ndvi_hi=0.97,
        dry_ndvi_lo=0.60, dry_ndvi_hi=0.97,
        delta_lo=0.0,     delta_hi=0.30,
        vhvv_lo=-6.0,     vhvv_hi=-3.0,
    ),
    "bare": Thresholds(
        wet_ndvi_lo=0.45, wet_ndvi_hi=0.80,
        dry_ndvi_lo=0.08, dry_ndvi_hi=0.20,
        delta_lo=0.40,    delta_hi=9.9,
        vhvv_lo=-9.9,     vhvv_hi=-6.5,
    ),
    "riparian": Thresholds(
        wet_ndvi_lo=0.65, wet_ndvi_hi=0.90,
        dry_ndvi_lo=0.25, dry_ndvi_hi=0.55,
        delta_lo=0.25,    delta_hi=0.55,
        vhvv_lo=-6.0,     vhvv_hi=-3.0,
    ),
    "water": Thresholds(
        wet_ndvi_lo=-1.0, wet_ndvi_hi=0.0,
        dry_ndvi_lo=-1.0, dry_ndvi_hi=0.0,
        delta_lo=-9.9,    delta_hi=9.9,
        vhvv_lo=-9.9,     vhvv_hi=9.9,
        check_water=True,
    ),
}

_MARGIN = 0.10  # relative boundary margin for ⚠ flag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cover_type(region_id: str) -> str:
    for key in ("mangrove", "bare", "riparian", "water"):
        if f"_{key}" in region_id:
            return key
    return "presence"


def _fmt(v: Optional[float], decimals: int = 2) -> str:
    return "—" if v is None else f"{v:.{decimals}f}"


def _in_range(val: Optional[float], lo: float, hi: float) -> Optional[bool]:
    if val is None:
        return None
    return lo <= val <= hi


def _near_boundary(val: float, lo: float, hi: float) -> bool:
    span = hi - lo
    if span <= 0:
        return False
    return (val - lo) / span < _MARGIN or (hi - val) / span < _MARGIN


# ---------------------------------------------------------------------------
# Per-year reduction
# ---------------------------------------------------------------------------

@dataclass
class YearSummary:
    year: int
    wet_ndvi_peak: Optional[float]
    dry_ndvi_floor: Optional[float]
    wet_dry_delta: Optional[float]
    vhvv_dry: Optional[float]
    iqr_max: Optional[float]
    tile_used: str
    verdict: str
    notes: list[str] = field(default_factory=list)


_WET_MONTHS  = {1, 2, 3, 4, 5}
_DRY_MONTHS  = {7, 8, 9, 10, 11}


def _month(date_str: str) -> int:
    return int(str(date_str)[5:7])


def reduce_series(series: list[dict]) -> tuple[
    Optional[float], Optional[float], Optional[float], Optional[float], Optional[float],
    list[str]
]:
    notes: list[str] = []
    if not series:
        return None, None, None, None, None, notes

    wet_ndvi, dry_ndvi, dry_vhvv, all_iqr = [], [], [], []

    for row in series:
        m = _month(row["date"])
        ndvi = row.get("ndvi")
        p25  = row.get("ndvi_p25")
        p75  = row.get("ndvi_p75")
        vhvv = row.get("vh_vv")

        if ndvi is not None:
            if m in _WET_MONTHS:
                wet_ndvi.append(ndvi)
            if m in _DRY_MONTHS:
                dry_ndvi.append(ndvi)

        if vhvv is not None and m in _DRY_MONTHS:
            dry_vhvv.append(vhvv)

        if p25 is not None and p75 is not None:
            all_iqr.append(p75 - p25)
            # Flag single-date NDVI near-zero mid wet season (cloud/shadow)
            if m in _WET_MONTHS and ndvi is not None and ndvi < 0.05:
                notes.append(f"near-zero NDVI {row['date']} ({ndvi:.2f}) — likely cloud/shadow")

    wet_peak  = float(max(wet_ndvi))  if wet_ndvi  else None
    dry_floor = float(min(dry_ndvi))  if dry_ndvi  else None
    delta     = (wet_peak - dry_floor) if (wet_peak is not None and dry_floor is not None) else None
    vhvv_mean = float(np.mean(dry_vhvv)) if dry_vhvv else None
    iqr_max   = float(max(all_iqr))   if all_iqr   else None

    if iqr_max is not None and iqr_max > 0.12:
        notes.append(f"IQR max {iqr_max:.2f} > 0.12 — possible spatial mixing")

    return wet_peak, dry_floor, delta, vhvv_mean, iqr_max, notes


def verdict(
    ctype: str,
    wet: Optional[float],
    dry: Optional[float],
    delta: Optional[float],
    vhvv: Optional[float],
    iqr_max: Optional[float],
) -> str:
    th = THRESHOLDS[ctype]

    if ctype == "water":
        if wet is None:
            return "— no data"
        fail = (wet is not None and wet >= 0.10) or (dry is not None and dry >= 0.10)
        flag = (wet is not None and wet >= 0.0)
        return "✗ fail" if fail else ("⚠ flag" if flag else "✓ pass")

    if wet is None and dry is None:
        return "— no data"

    fails = []
    flags = []

    for val, lo, hi, name in [
        (wet,   th.wet_ndvi_lo, th.wet_ndvi_hi,  "wet NDVI"),
        (dry,   th.dry_ndvi_lo, th.dry_ndvi_hi,  "dry NDVI"),
        (delta, th.delta_lo,    th.delta_hi,      "Δ"),
        (vhvv,  th.vhvv_lo,     th.vhvv_hi,      "VH/VV"),
    ]:
        if val is None:
            continue
        in_r = _in_range(val, lo, hi)
        if not in_r:
            fails.append(name)
        elif _near_boundary(val, lo, hi):
            flags.append(name)

    if iqr_max is not None and iqr_max > 0.12:
        flags.append("IQR")

    if fails:
        return "✗ fail"
    if flags:
        return "⚠ flag"
    return "✓ pass"


# ---------------------------------------------------------------------------
# Tile selection
# ---------------------------------------------------------------------------

_TILE_PRIMARY   = "54LWH"
_TILE_SECONDARY = "54KWG"


def _compute_timeseries_quality_filtered(
    root: Path,
    year: int,
    tile_id: str,
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
) -> list[dict]:
    """Like compute_timeseries, but applies Signal.quality_mask (the exact
    gate the training pipeline uses: source == "S2" and scl_purity >= 0.5)
    before computing optical indices — so the audit measures the same pixels
    the model actually trains on. VH/VV (Sentinel-1) is unaffected by the
    optical quality gate and is computed from all rows carrying radar bands.
    """
    chunk_idx = ChunkIndex(root, year, tile_id)
    tbl = chunk_idx.query_bbox(lon_min, lat_min, lon_max, lat_max)
    if tbl.num_rows == 0:
        return []

    df = pl.from_arrow(tbl)
    # Raw chunkstore rows store S2 with source=null (only S1 is explicitly
    # tagged); the training pipeline backfills null -> "S2" before gating
    # (see utils/training_collector.py), so mirror that here.
    if "source" in df.columns:
        df = df.with_columns(pl.col("source").fill_null("S2"))
    good = Signal.quality_mask(df)

    def col(frame: pl.DataFrame, name: str) -> np.ndarray:
        return frame[name].to_numpy().astype("float32")

    # --- Optical indices: only from rows passing the training quality gate.
    opt = df.filter(good)
    if len(opt) > 0:
        b04 = col(opt, "B04")
        b08 = col(opt, "B08")
        b11 = col(opt, "B11")
        ndvi = _safe_div(b08 - b04, b08 + b04)
        mavi = _safe_div(b08 - b04, b08 + b04 + b11)
        opt_dates = opt["date"].to_numpy()
    else:
        ndvi = mavi = np.array([], dtype="float32")
        opt_dates = np.array([])

    # --- Radar (VH/VV): SCL purity does not apply to S1; use all rows with
    # valid radar bands (mirrors how downstream signals treat S1 separately).
    rad_mask = df["vh"].is_not_null() & df["vv"].is_not_null()
    rad = df.filter(rad_mask)
    if len(rad) > 0:
        vh = col(rad, "vh")
        vv = col(rad, "vv")
        with np.errstate(divide="ignore", invalid="ignore"):
            vh_db = np.where(vh > 0, 10.0 * np.log10(vh.astype("float64")), np.nan).astype("float32")
            vv_db = np.where(vv > 0, 10.0 * np.log10(vv.astype("float64")), np.nan).astype("float32")
        vh_vv = vh_db - vv_db
        rad_dates = rad["date"].to_numpy()
    else:
        vh_vv = np.array([], dtype="float32")
        rad_dates = np.array([])

    unique_dates = sorted(set(opt_dates.tolist()) | set(rad_dates.tolist()))

    series = []
    for d in unique_dates:
        row: dict[str, object] = {"date": str(d)}
        opt_g = ndvi[opt_dates == d]
        mavi_g = mavi[opt_dates == d]
        rad_g = vh_vv[rad_dates == d]
        for name, g in [("ndvi", opt_g), ("mavi", mavi_g), ("vh_vv", rad_g)]:
            row[name]          = _mean(g)
            row[f"{name}_p25"] = _percentile(g, 25)
            row[f"{name}_p75"] = _percentile(g, 75)
        series.append(row)

    return series


def fetch_year(root: Path, year: int, bbox: list[float]) -> tuple[list[dict], str]:
    lon_min, lat_min, lon_max, lat_max = bbox
    for tile in (_TILE_PRIMARY, _TILE_SECONDARY):
        try:
            series = _compute_timeseries_quality_filtered(root, year, tile, lon_min, lat_min, lon_max, lat_max)
        except Exception:
            series = []
        if series:
            return series, tile
    return [], _TILE_PRIMARY  # empty — record primary tile


# ---------------------------------------------------------------------------
# Region loading
# ---------------------------------------------------------------------------

@dataclass
class Region:
    id: str
    label: str
    bbox: list[float]
    years: list[int]
    role: str  # "train" or "val"


def load_regions(training_yaml: Path, prefix: str, bbox_id: Optional[str]) -> list[Region]:
    with open(training_yaml) as f:
        doc = yaml.safe_load(f)
    raw = doc["regions"] if isinstance(doc, dict) and "regions" in doc else doc

    results = []
    for entry in raw:
        rid = entry.get("id", "")
        if bbox_id:
            if rid != bbox_id:
                continue
        elif not rid.startswith(prefix):
            continue

        role = "val" if "_val_" in rid else "train"
        results.append(Region(
            id=rid,
            label=entry.get("label", "absence"),
            bbox=entry["bbox"],
            years=sorted(entry.get("years", [])),
            role=role,
        ))

    return sorted(results, key=lambda r: r.id)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

_COVER_ORDER = ["presence", "mangrove", "bare", "riparian", "water"]


def render_report(summaries: dict[str, list[YearSummary]], regions: list[Region]) -> str:
    lines: list[str] = []

    # --- Header counts ---
    all_verdicts: list[str] = [ys.verdict for rl in summaries.values() for ys in rl]
    n_pass  = sum(1 for v in all_verdicts if v.startswith("✓"))
    n_flag  = sum(1 for v in all_verdicts if v.startswith("⚠"))
    n_fail  = sum(1 for v in all_verdicts if v.startswith("✗"))
    n_nodat = sum(1 for v in all_verdicts if v.startswith("—"))

    # Per-category counts
    cat_counts: dict[str, dict[str, int]] = {}
    for r in regions:
        ct = cover_type(r.id)
        if ct not in cat_counts:
            cat_counts[ct] = {"pass": 0, "flag": 0, "fail": 0, "nodata": 0}
        for ys in summaries.get(r.id, []):
            if ys.verdict.startswith("✓"):
                cat_counts[ct]["pass"] += 1
            elif ys.verdict.startswith("⚠"):
                cat_counts[ct]["flag"] += 1
            elif ys.verdict.startswith("✗"):
                cat_counts[ct]["fail"] += 1
            else:
                cat_counts[ct]["nodata"] += 1

    lines += [
        "# Mitchell Training Bbox — Multi-Year Quality Audit",
        "",
        f"**Total year-rows:** {len(all_verdicts)}  "
        f"| ✓ pass: {n_pass}  | ⚠ flag: {n_flag}  | ✗ fail: {n_fail}  | — no data: {n_nodat}",
        "",
        "## Summary by cover type",
        "",
        "| Cover type | ✓ pass | ⚠ flag | ✗ fail | — no data |",
        "|------------|--------|--------|--------|-----------|",
    ]
    for ct in _COVER_ORDER:
        if ct in cat_counts:
            c = cat_counts[ct]
            lines.append(
                f"| {ct:<10} | {c['pass']:6} | {c['flag']:6} | {c['fail']:6} | {c['nodata']:9} |"
            )
    lines += ["", "---", ""]

    # --- Per-region sections ---
    region_map = {r.id: r for r in regions}
    for ct in _COVER_ORDER:
        ct_regions = [r for r in regions if cover_type(r.id) == ct]
        if not ct_regions:
            continue
        lines += [f"## {ct.capitalize()} regions", ""]
        for r in ct_regions:
            ys_list = summaries.get(r.id, [])
            tile_used = ys_list[0].tile_used if ys_list else "?"
            lines += [
                f"### {r.id}  ({r.label} · {r.role})",
                f"Bbox: {r.bbox}  Tile: {tile_used}",
                "",
                "| Year | Wet NDVI (peak) | Dry NDVI (floor) | Wet–dry Δ | VH/VV dry (dB) | IQR max | Verdict |",
                "|------|----------------|-----------------|-----------|---------------|---------|---------|",
            ]
            year_notes: list[str] = []
            for ys in ys_list:
                lines.append(
                    f"| {ys.year} "
                    f"| {_fmt(ys.wet_ndvi_peak)} "
                    f"| {_fmt(ys.dry_ndvi_floor)} "
                    f"| {_fmt(ys.wet_dry_delta)} "
                    f"| {_fmt(ys.vhvv_dry)} "
                    f"| {_fmt(ys.iqr_max)} "
                    f"| {ys.verdict} |"
                )
                for note in ys.notes:
                    year_notes.append(f"- {ys.year}: {note}")

            # All-year summary row
            valid_ys = [ys for ys in ys_list if ys.wet_ndvi_peak is not None]
            if valid_ys:
                def _rng(vals: list[Optional[float]]) -> str:
                    fv = [v for v in vals if v is not None]
                    if not fv:
                        return "—"
                    lo, hi = min(fv), max(fv)
                    return f"{lo:.2f}–{hi:.2f}" if lo != hi else f"{lo:.2f}"

                wet_vals  = [ys.wet_ndvi_peak  for ys in valid_ys]
                dry_vals  = [ys.dry_ndvi_floor for ys in valid_ys]
                dlt_vals  = [ys.wet_dry_delta  for ys in valid_ys]
                vhv_vals  = [ys.vhvv_dry       for ys in valid_ys]
                overall_v = (
                    "**✗ fail**" if any(ys.verdict.startswith("✗") for ys in ys_list) else
                    "**⚠ flag**" if any(ys.verdict.startswith("⚠") for ys in ys_list) else
                    "**✓ consistent**"
                )
                lines.append(
                    f"| **All-year** "
                    f"| {_rng(wet_vals)} "
                    f"| {_rng(dry_vals)} "
                    f"| {_rng(dlt_vals)} "
                    f"| {_rng(vhv_vals)} "
                    f"| — "
                    f"| {overall_v} |"
                )

            lines.append("")
            if year_notes:
                lines.append("**Notes:**")
                lines.extend(year_notes)
                lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-year bbox quality audit for training regions.")
    ap.add_argument("--root",     default="/mnt/external/chunkstore")
    ap.add_argument("--training", default="data/locations/training.yaml")
    ap.add_argument("--prefix",   default="mitchell")
    ap.add_argument("--out",      default="docs/MITCHELL-BBOX-AUDIT.md")
    ap.add_argument("--years",    default="2017-2025",
                    help="Year range as YYYY-YYYY (default: 2017-2025)")
    ap.add_argument("--bbox-id",  default=None,
                    help="Audit a single region by id")
    args = ap.parse_args()

    root = Path(args.root)
    training_yaml = Path(args.training)
    out_path = Path(args.out)

    y_lo, y_hi = (int(y) for y in args.years.split("-"))
    year_override = list(range(y_lo, y_hi + 1))

    regions = load_regions(training_yaml, args.prefix, args.bbox_id)
    if not regions:
        print(f"No regions found matching prefix={args.prefix!r} / bbox_id={args.bbox_id!r}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Auditing {len(regions)} region(s) × up to {len(year_override)} years each…",
          file=sys.stderr)

    summaries: dict[str, list[YearSummary]] = {}

    for r in regions:
        years_to_run = [y for y in r.years if y in year_override]
        print(f"  {r.id}  ({len(years_to_run)} years)", file=sys.stderr)
        ctype = cover_type(r.id)
        ys_list: list[YearSummary] = []

        for yr in years_to_run:
            series, tile_used = fetch_year(root, yr, r.bbox)
            wet, dry, delta, vhvv, iqr_max, notes = reduce_series(series)
            v = verdict(ctype, wet, dry, delta, vhvv, iqr_max)
            ys_list.append(YearSummary(
                year=yr,
                wet_ndvi_peak=wet,
                dry_ndvi_floor=dry,
                wet_dry_delta=delta,
                vhvv_dry=vhvv,
                iqr_max=iqr_max,
                tile_used=tile_used,
                verdict=v,
                notes=notes,
            ))

        summaries[r.id] = ys_list

    md = render_report(summaries, regions)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"\nReport written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
