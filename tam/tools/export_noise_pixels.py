"""tam/tools/export_noise_pixels.py — Export per-pixel noise-filter features for a training region.

Usage:
    python3 -m tam.tools.export_noise_pixels <region_id>
    python3 tam/tools/export_noise_pixels.py <region_id>

Writes outputs/noise_pixels/{region_id}.json with per-pixel dry_ndvi, rec_p, nir_cv
and lon/lat. The browser uses this to render a keep/drop overlay with interactive
threshold sliders.

Cache strategy: if any outputs/*/global_features_cache.parquet already covers the
region's pixels, reuse it (fast). Otherwise compute from the tile parquet (slow,
~1–2 min for large regions).
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_DATA_DIR    = os.path.join(_REPO, "data", "training")
_OUT_DIR     = os.path.join(_REPO, "outputs", "noise_pixels")
_OUTPUTS_DIR = os.path.join(_REPO, "outputs")

NOISE_FEATURES = ["dry_ndvi", "rec_p", "nir_cv", "s1_mean_vh_dry"]


def _find_cached_features(point_ids: set[str]) -> pd.DataFrame | None:
    """Return a DataFrame with NOISE_FEATURES if any existing cache covers all pixels."""
    sample = list(point_ids)[:20]
    for cache_path in sorted(
        glob.glob(os.path.join(_OUTPUTS_DIR, "**", "global_features_cache.parquet"),
                  recursive=True)
    ):
        try:
            gf = pd.read_parquet(cache_path, columns=NOISE_FEATURES)
        except Exception:
            continue
        if all(pid in gf.index for pid in sample):
            full = gf.reindex(list(point_ids))
            if full[NOISE_FEATURES].notna().any(axis=None):
                print(f"Reusing cache: {cache_path}", file=sys.stderr)
                return full
    return None


def export_region(region_id: str) -> str:
    """Compute and write noise pixel JSON for region_id. Returns the output path."""
    os.makedirs(_OUT_DIR, exist_ok=True)
    out_path = os.path.join(_OUT_DIR, f"{region_id}.json")

    # --- find tile_id ---
    idx = pd.read_parquet(os.path.join(_DATA_DIR, "index.parquet"))
    rows = idx[idx["region_id"] == region_id]
    if rows.empty:
        raise ValueError(f"Region '{region_id}' not found in training index.")
    tile_id = rows["tile_id"].iloc[0]

    # --- load pixel time series for this region ---
    tile_path = os.path.join(_DATA_DIR, "tiles", f"{tile_id}.parquet")
    print(f"Loading tile {tile_id} ...", file=sys.stderr)
    prefix = region_id + "_"
    tile_df = pd.read_parquet(tile_path)
    pixel_df = tile_df[tile_df["point_id"].str.startswith(prefix)].copy()
    if pixel_df.empty:
        print(f"No pixels found for region '{region_id}' in tile {tile_id} — writing empty result.", file=sys.stderr)
        label = _region_label(region_id)
        payload = {"region_id": region_id, "label": label, "pixels": [], "missing_data": True}
        with open(out_path, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        return out_path

    print(f"  {len(pixel_df):,} observations, "
          f"{pixel_df['point_id'].nunique():,} unique pixels", file=sys.stderr)

    # --- lon/lat per pixel (one row each) ---
    coords = (
        pixel_df[["point_id", "lon", "lat"]]
        .drop_duplicates("point_id")
        .set_index("point_id")
    )

    # --- global features: reuse cache or compute ---
    point_ids = set(coords.index)
    gf = _find_cached_features(point_ids)
    if gf is None:
        print("Computing global features (this may take a while) ...", file=sys.stderr)
        # add year column if missing
        if "year" not in pixel_df.columns:
            pixel_df["year"] = pd.to_datetime(pixel_df["date"]).dt.year
        from tam.core.global_features import compute_global_features
        gf_full = compute_global_features(pixel_df)
        gf = gf_full[NOISE_FEATURES]

    gf = gf.reindex(list(point_ids))

    # --- build output ---
    joined = coords.join(gf, how="left")

    def _val(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return None if np.isnan(f) else round(f, 6)

    pixels = [
        {
            "id":             pid,
            "lon":            round(float(row["lon"]), 7),
            "lat":            round(float(row["lat"]), 7),
            "dry_ndvi":       _val(row.get("dry_ndvi")),
            "rec_p":          _val(row.get("rec_p")),
            "nir_cv":         _val(row.get("nir_cv")),
            "s1_mean_vh_dry": _val(row.get("s1_mean_vh_dry")),
        }
        for pid, row in joined.iterrows()
    ]

    # load label from training.yaml
    label = _region_label(region_id)

    payload = {"region_id": region_id, "label": label, "pixels": pixels}
    with open(out_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    print(f"Wrote {len(pixels):,} pixels → {out_path}", file=sys.stderr)
    return out_path


def _region_label(region_id: str) -> str:
    try:
        import yaml
        yaml_path = os.path.join(_REPO, "data", "locations", "training.yaml")
        with open(yaml_path) as f:
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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <region_id>", file=sys.stderr)
        sys.exit(1)
    export_region(sys.argv[1])
