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
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_DATA_DIR = _REPO / "data" / "training"
_OUT_DIR  = _REPO / "outputs" / "heuristic_scores"

# Physical threshold (dB) — pixels at or above this are woody, below are rejected
_VH_WOODY_FLOOR_DB = -18.0

# Dry season: May 1 – Oct 31 (DOY 121–304)
_DRY_DOY_MIN = 121
_DRY_DOY_MAX = 304


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


def _apply_heuristic(s1: pd.DataFrame) -> pd.Series:
    """Compute per-pixel heuristic score from S1 observations.

    Parameters
    ----------
    s1 : DataFrame with columns point_id, vh (linear power), date or doy.

    Returns
    -------
    Series indexed by point_id with float score ∈ {0.0, 0.5, 1.0}.
    """
    df = s1.copy()
    if "doy" not in df.columns:
        df["doy"] = pd.to_datetime(df["date"]).dt.day_of_year

    vh_lin = df["vh"].values.astype(np.float64)
    valid  = vh_lin > 0
    df["_vh_db"] = np.where(valid, 10.0 * np.log10(vh_lin), np.nan)

    has_db = df["_vh_db"].notna()
    dry_mask = (df["doy"] >= _DRY_DOY_MIN) & (df["doy"] <= _DRY_DOY_MAX)

    dry_df = df[has_db & dry_mask]

    mean_vh_dry = dry_df.groupby("point_id")["_vh_db"].mean()

    all_pids = pd.Index(df["point_id"].unique(), name="point_id")
    mean_vh_dry = mean_vh_dry.reindex(all_pids)

    scores = pd.Series(np.nan, index=all_pids, name="prob_woody", dtype=float)

    scores[mean_vh_dry >= _VH_WOODY_FLOOR_DB] = 1.0
    scores[mean_vh_dry <  _VH_WOODY_FLOOR_DB] = 0.0

    return scores


def export_region(region_id: str) -> str:
    """Apply VH heuristic to pixels for region_id and write JSON. Returns output path."""
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / f"{region_id}.json"

    idx = pd.read_parquet(_DATA_DIR / "index.parquet")
    rows = idx[idx["region_id"] == region_id]
    if rows.empty:
        raise ValueError(f"Region '{region_id}' not found in training index.")
    tile_id = rows["tile_id"].iloc[0]

    tile_path = _DATA_DIR / "tiles" / f"{tile_id}.parquet"
    print(f"Loading tile {tile_id} ...", file=sys.stderr)

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(tile_path)
    available = set(pf.schema_arrow.names)
    want = ["point_id", "lon", "lat", "date", "source", "vh", "vv"]
    read_cols = [c for c in want if c in available]

    chunks = [pf.read_row_group(rg, columns=read_cols).to_pandas()
              for rg in range(pf.metadata.num_row_groups)]
    tile_df = pd.concat(chunks, ignore_index=True)

    prefix = region_id + "_"
    pixel_df = tile_df[tile_df["point_id"].str.startswith(prefix)].copy()

    label = _region_label(region_id)

    if pixel_df.empty:
        print(f"No pixels found for region '{region_id}' — writing empty result.", file=sys.stderr)
        payload = {"region_id": region_id, "label": label, "pixels": [], "missing_data": True}
        out_path.write_text(json.dumps(payload, separators=(",", ":")))
        return str(out_path)

    print(f"  {len(pixel_df):,} observations, "
          f"{pixel_df['point_id'].nunique():,} unique pixels", file=sys.stderr)

    coords = (
        pixel_df[["point_id", "lon", "lat"]]
        .drop_duplicates("point_id")
        .set_index("point_id")
    )

    # S1 only
    if "source" in pixel_df.columns:
        s1_df = pixel_df[pixel_df["source"] == "S1"].copy()
    else:
        s1_df = pd.DataFrame()

    if not s1_df.empty and "vh" in s1_df.columns:
        scores = _apply_heuristic(s1_df)
    else:
        print("  No S1 data found — all pixels will have score=None", file=sys.stderr)
        scores = pd.Series(np.nan, index=pd.Index(pixel_df["point_id"].unique(), name="point_id"))

    joined = coords.join(scores.rename("prob_woody"), how="left")

    pixels = []
    for pid, row in joined.iterrows():
        score = row["prob_woody"]
        pixels.append({
            "id":         pid,
            "lon":        round(float(row["lon"]), 7),
            "lat":        round(float(row["lat"]), 7),
            "prob_woody": None if (score is None or (isinstance(score, float) and np.isnan(score)))
                          else round(float(score), 1),
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
