"""utils/contamination_check.py — does cloud/shadow contamination move v10's
per-pixel global features (p5 / p95 / std of NDVI & MAVI)?

v10 trains on `_compute_band_summaries` (tam/core/train.py): for every pixel,
across its *entire* multi-year S2 timeseries (after the scl_purity>=0.5 gate),
it computes [p5, p95, std] per band/index and feeds those to the model as
global features, then z-scores them across all pixels.

The Mitchell bbox audit (docs/MITCHELL-BBOX-AUDIT.md) flagged specific dates
where bbox-mean wet-season NDVI dropped near zero or negative ("likely
cloud/shadow") — values that pass the scl_purity gate untouched (it's a no-op
on this chunkstore; scl_purity is uniformly 1.0).

This script asks the concrete question: for a sample of bboxes, if we drop
those flagged dates before computing [p5, p95, std] per pixel, how much do
the resulting global features move? p5/p95 are percentile-based and should be
fairly resistant; std has no such protection and is the feature we expect to
be sensitive.

Usage::

    python utils/contamination_check.py --root /mnt/external/chunkstore \\
        --training data/locations/training.yaml \\
        --bbox-id mitchell_presence_4 mitchell_absence_mangrove_3 ...
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.pixel_reader import ChunkIndex

_TILE_PRIMARY   = "54LWH"
_TILE_SECONDARY = "54KWG"

_WET_MONTHS = {1, 2, 3, 4, 5}
# Same heuristic the audit uses to flag suspected cloud/shadow contamination:
# a wet-season date whose bbox-mean NDVI drops near zero or negative.
_CONTAM_NDVI_THRESHOLD = 0.05


def _month(date_str: str) -> int:
    return int(str(date_str)[5:7])


def _safe_ratio(num: pl.Expr, denom: pl.Expr) -> pl.Expr:
    return pl.when(denom != 0).then(num / denom).otherwise(pl.lit(None))


def _load_pixel_df(root: Path, years: list[int], tile_id: str, bbox: list[float]) -> pl.DataFrame:
    lon_min, lat_min, lon_max, lat_max = bbox
    parts = []
    for year in years:
        try:
            chunk_idx = ChunkIndex(root, year, tile_id)
            tbl = chunk_idx.query_bbox(lon_min, lat_min, lon_max, lat_max)
        except Exception:
            continue
        if tbl.num_rows == 0:
            continue
        df = pl.from_arrow(tbl)
        if "source" in df.columns:
            df = df.with_columns(pl.col("source").fill_null("S2"))
        parts.append(df)
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="diagonal_relaxed")


def _band_summaries(df: pl.DataFrame, flagged_dates: set[str] | None = None) -> pl.DataFrame:
    """Mirror tam.core.train._compute_band_summaries for NDVI & MAVI only.

    If flagged_dates is given, those dates are dropped before aggregation —
    simulating "what if we filtered out suspected cloud/shadow contamination".
    """
    lf = df.lazy().filter(pl.col("source") == "S2")
    if "scl_purity" in df.columns:
        lf = lf.filter(pl.col("scl_purity") >= 0.5)
    if flagged_dates:
        lf = lf.filter(~pl.col("date").cast(pl.Utf8).is_in(list(flagged_dates)))

    lf = lf.with_columns([
        _safe_ratio(pl.col("B08") - pl.col("B04"), pl.col("B08") + pl.col("B04")).alias("NDVI"),
        _safe_ratio(pl.col("B08") - pl.col("B04"),
                    pl.col("B08") + pl.col("B04") + pl.col("B11")).alias("MAVI"),
    ])

    aggs = []
    for c in ("NDVI", "MAVI"):
        aggs += [
            pl.col(c).quantile(0.05).alias(f"{c}_p5"),
            pl.col(c).quantile(0.95).alias(f"{c}_p95"),
            pl.col(c).std().alias(f"{c}_std"),
            pl.col(c).count().alias(f"{c}_n"),
        ]
    return lf.group_by("point_id").agg(aggs).collect()


def _find_contaminated_dates(df: pl.DataFrame) -> set[str]:
    """Reproduce the audit's per-date "near-zero wet-season NDVI" flag at
    bbox level: mean NDVI across all pixels for that date < threshold.
    """
    lf = df.lazy().filter(pl.col("source") == "S2")
    if "scl_purity" in df.columns:
        lf = lf.filter(pl.col("scl_purity") >= 0.5)
    lf = lf.with_columns(
        _safe_ratio(pl.col("B08") - pl.col("B04"), pl.col("B08") + pl.col("B04")).alias("NDVI")
    )
    per_date = (
        lf.group_by(pl.col("date").cast(pl.Utf8))
        .agg(pl.col("NDVI").mean().alias("ndvi_mean"))
        .collect()
    )
    flagged = set()
    for row in per_date.iter_rows(named=True):
        d = row["date"]
        if _month(d) in _WET_MONTHS and row["ndvi_mean"] is not None and row["ndvi_mean"] < _CONTAM_NDVI_THRESHOLD:
            flagged.add(d)
    return flagged


def _compare(bbox_id: str, before: pl.DataFrame, after: pl.DataFrame, flagged: set[str]) -> str:
    lines = [f"### {bbox_id}", f"Flagged (suspected contaminated) dates dropped: {len(flagged)}"]
    if flagged:
        lines.append("  " + ", ".join(sorted(flagged)))

    if before.is_empty():
        lines.append("(no pixel data)")
        return "\n".join(lines)

    joined = before.join(after, on="point_id", suffix="_clean", how="inner")
    n_pixels = joined.height
    lines.append(f"Pixels compared: {n_pixels}")

    for c in ("NDVI", "MAVI"):
        for stat in ("p5", "p95", "std"):
            a = joined[f"{c}_{stat}"].to_numpy().astype("float64")
            b = joined[f"{c}_{stat}_clean"].to_numpy().astype("float64")
            valid = np.isfinite(a) & np.isfinite(b)
            if valid.sum() == 0:
                continue
            d = b[valid] - a[valid]
            lines.append(
                f"  {c}_{stat}: mean Δ={np.mean(d):+.4f}  "
                f"max|Δ|={np.max(np.abs(d)):.4f}  "
                f"pixels with |Δ|>0.02: {(np.abs(d) > 0.02).sum()}/{valid.sum()}"
            )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--training", type=Path, required=True)
    ap.add_argument("--bbox-id", nargs="+", required=True)
    args = ap.parse_args()

    with open(args.training) as f:
        doc = yaml.safe_load(f)
    raw = doc["regions"] if isinstance(doc, dict) and "regions" in doc else doc
    by_id = {e["id"]: e for e in raw}

    out = []
    for bbox_id in args.bbox_id:
        entry = by_id.get(bbox_id)
        if entry is None:
            print(f"[skip] {bbox_id}: not found in training.yaml", file=sys.stderr)
            continue
        bbox = entry["bbox"]
        years = sorted(entry.get("years", []))
        print(f"[run] {bbox_id}: {len(years)} years", file=sys.stderr)

        df = pl.DataFrame()
        used_tile = None
        for tile in (_TILE_PRIMARY, _TILE_SECONDARY):
            df = _load_pixel_df(args.root, years, tile, bbox)
            if not df.is_empty():
                used_tile = tile
                break
        if df.is_empty():
            print(f"[skip] {bbox_id}: no data", file=sys.stderr)
            continue

        flagged = _find_contaminated_dates(df)
        before = _band_summaries(df)
        after = _band_summaries(df, flagged_dates=flagged)
        out.append(_compare(bbox_id, before, after, flagged))

    print("\n\n".join(out))


if __name__ == "__main__":
    main()
