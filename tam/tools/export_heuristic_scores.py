"""tam/tools/export_heuristic_scores.py — Score training-region pixels with the VH heuristic.

Usage:
    python3 tam/tools/export_heuristic_scores.py <region_id>

Writes outputs/heuristic_scores/{region_id}.json with per-pixel heuristic score, lon, lat.

Heuristic logic (Sentinel-1 VH backscatter, dry-season mean):
    mean_vh_dry ≥ -18 dB  →  1.0  (woody)
    mean_vh_dry < -18 dB  →  0.0  (non-woody / bare)
    No S1 data            →  None

Score is written as prob_woody so the UI can be shared with the classifier pipeline.
Cache strategy: outputs/heuristic_scores/{region_id}.json is returned as-is if it exists.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_DATA_DIR = _REPO / "data" / "training"
_OUT_DIR  = _REPO / "outputs" / "heuristic_scores"

# Physical threshold (dB) — pixels at or above this are woody, below are rejected
_VH_WOODY_FLOOR_DB = -18.0

from tam.core.constants import DRY_DOY_MIN as _DRY_DOY_MIN, DRY_DOY_MAX as _DRY_DOY_MAX


def _region_label(region_id: str) -> str:
    try:
        import yaml
        with open(_REPO / "data" / "locations" / "training.yaml") as f:
            data = yaml.safe_load(f)
        for r in data.get("regions", []):
            if r["id"] == region_id:
                return r.get("label", "unknown")
    except Exception:
        pass
    if "_presence" in region_id:
        return "presence"
    if "_absence" in region_id:
        return "absence"
    return "unknown"


def _apply_heuristic(s1: pl.DataFrame) -> dict[str, float]:
    """Compute per-pixel heuristic score from S1 observations.

    Returns dict mapping point_id → score ∈ {0.0, 1.0, nan}.
    """
    if "doy" not in s1.columns:
        s1 = s1.with_columns(
            pl.col("date").cast(pl.Date).dt.ordinal_day().alias("doy")
        )

    vh_lin = s1["vh"].to_numpy().astype(np.float64)
    vh_db  = np.where(vh_lin > 0, 10.0 * np.log10(vh_lin), np.nan)
    s1 = s1.with_columns(pl.Series("_vh_db", vh_db))

    dry = s1.filter(
        pl.col("_vh_db").is_not_null() & pl.col("_vh_db").is_not_nan() &
        (pl.col("doy") >= _DRY_DOY_MIN) & (pl.col("doy") <= _DRY_DOY_MAX)
    )

    mean_vh = {
        row["point_id"]: row["mean_vh"]
        for row in dry.group_by("point_id")
                      .agg(pl.col("_vh_db").mean().alias("mean_vh"))
                      .iter_rows(named=True)
    }

    all_pids = s1["point_id"].unique().to_list()
    return {
        pid: (1.0 if mean_vh[pid] >= _VH_WOODY_FLOOR_DB else 0.0)
             if pid in mean_vh else float("nan")
        for pid in all_pids
    }


def export_region(region_id: str) -> str:
    """Apply VH heuristic to pixels for region_id and write JSON. Returns output path."""
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / f"{region_id}.json"

    import pyarrow.parquet as pq

    idx = pl.read_parquet(_DATA_DIR / "index.parquet")
    rows = idx.filter(pl.col("region_id") == region_id)
    if rows.is_empty():
        raise ValueError(f"Region '{region_id}' not found in training index.")
    tile_id = rows["tile_id"][0]

    tile_path = _DATA_DIR / "tiles" / f"{tile_id}.parquet"
    print(f"Loading tile {tile_id} ...", file=sys.stderr)

    pf = pq.ParquetFile(tile_path)
    available = set(pf.schema_arrow.names)
    want = ["point_id", "lon", "lat", "date", "source", "vh", "vv"]
    read_cols = [c for c in want if c in available]

    chunks = [pl.from_arrow(pf.read_row_group(rg, columns=read_cols))
              for rg in range(pf.metadata.num_row_groups)]
    tile_df = pl.concat(chunks)

    prefix = region_id + "_"
    pixel_df = tile_df.filter(pl.col("point_id").str.starts_with(prefix))

    label = _region_label(region_id)

    if pixel_df.is_empty():
        print(f"No pixels found for region '{region_id}' — writing empty result.", file=sys.stderr)
        payload = {"region_id": region_id, "label": label, "pixels": [], "missing_data": True}
        out_path.write_text(json.dumps(payload, separators=(",", ":")))
        return str(out_path)

    print(f"  {len(pixel_df):,} observations, "
          f"{pixel_df['point_id'].n_unique():,} unique pixels", file=sys.stderr)

    coords = {
        row["point_id"]: (row["lon"], row["lat"])
        for row in pixel_df.select(["point_id", "lon", "lat"])
                            .unique("point_id")
                            .iter_rows(named=True)
    }

    # S1 only
    if "source" in pixel_df.columns:
        s1_df = pixel_df.filter(pl.col("source") == "S1")
    else:
        s1_df = pl.DataFrame()

    if not s1_df.is_empty() and "vh" in s1_df.columns:
        scores = _apply_heuristic(s1_df)
    else:
        print("  No S1 data found — all pixels will have score=None", file=sys.stderr)
        scores = {pid: float("nan") for pid in coords}

    pixels = []
    for pid, (lon, lat) in coords.items():
        score = scores.get(pid, float("nan"))
        pixels.append({
            "id":         pid,
            "lon":        round(float(lon), 7),
            "lat":        round(float(lat), 7),
            "prob_woody": None if np.isnan(score) else round(float(score), 1),
        })

    payload = {"region_id": region_id, "label": label, "pixels": pixels}
    out_path.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"Wrote {len(pixels):,} pixels → {out_path}", file=sys.stderr)
    return str(out_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <region_id>", file=sys.stderr)
        sys.exit(1)
    export_region(sys.argv[1])
