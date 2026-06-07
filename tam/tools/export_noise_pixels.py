"""tam/tools/export_noise_pixels.py — Export per-pixel noise-filter features for a training region.

Usage:
    python3 -m tam.tools.export_noise_pixels <region_id>
    python3 tam/tools/export_noise_pixels.py <region_id>

Writes outputs/noise_pixels/{region_id}.json with per-pixel dry_ndvi, rec_p, nir_cv
and lon/lat. The browser uses this to render a keep/drop overlay with interactive
threshold sliders.

Cache strategy: if any outputs/*/annual_features_cache.parquet already covers the
region's pixels, reuse it (fast). Otherwise compute from the tile parquet (slow,
~1–2 min for large regions).
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import polars as pl

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_DATA_DIR    = os.path.join(_REPO, "data", "training")
_OUT_DIR     = os.path.join(_REPO, "outputs", "noise_pixels")
_OUTPUTS_DIR = os.path.join(_REPO, "outputs")

NOISE_FEATURES = ["dry_ndvi", "rec_p", "nir_cv", "s1_mean_vh_dry"]


def _find_cached_features(point_ids: set[str]) -> pl.DataFrame | None:
    """Return a DataFrame with NOISE_FEATURES if any existing cache covers all pixels."""
    sample = list(point_ids)[:20]
    for cache_path in sorted(
        glob.glob(os.path.join(_OUTPUTS_DIR, "**", "annual_features_cache.parquet"),
                  recursive=True)
    ):
        try:
            gf = pl.read_parquet(cache_path, columns=["point_id"] + NOISE_FEATURES)
        except Exception:
            continue
        cached_pids = set(gf["point_id"].to_list())
        if all(pid in cached_pids for pid in sample):
            full = gf.filter(pl.col("point_id").is_in(point_ids))
            if any(full[c].drop_nulls().len() > 0 for c in NOISE_FEATURES):
                print(f"Reusing cache: {cache_path}", file=sys.stderr)
                return full
    return None


def export_region(region_id: str) -> str:
    """Compute and write noise pixel JSON for region_id. Returns the output path."""
    os.makedirs(_OUT_DIR, exist_ok=True)
    out_path = os.path.join(_OUT_DIR, f"{region_id}.json")

    # --- find tile_id ---
    idx = pl.read_parquet(os.path.join(_DATA_DIR, "index.parquet"))
    rows = idx.filter(pl.col("region_id") == region_id)
    if rows.is_empty():
        raise ValueError(f"Region '{region_id}' not found in training index.")
    tile_id = rows["tile_id"][0]

    # --- load pixel time series for this region ---
    tile_path = os.path.join(_DATA_DIR, "tiles", f"{tile_id}.parquet")
    print(f"Loading tile {tile_id} ...", file=sys.stderr)
    prefix = region_id + "_"
    tile_df = pl.read_parquet(tile_path)
    pixel_df = tile_df.filter(pl.col("point_id").str.starts_with(prefix))
    if pixel_df.is_empty():
        print(f"No pixels found for region '{region_id}' in tile {tile_id} — writing empty result.", file=sys.stderr)
        label = _region_label(region_id)
        payload = {"region_id": region_id, "label": label, "pixels": [], "missing_data": True}
        with open(out_path, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        return out_path

    print(f"  {len(pixel_df):,} observations, "
          f"{pixel_df['point_id'].n_unique():,} unique pixels", file=sys.stderr)

    # --- lon/lat per pixel (one row each) ---
    coords = {
        row["point_id"]: (row["lon"], row["lat"])
        for row in pixel_df.select(["point_id", "lon", "lat"])
                            .unique("point_id")
                            .iter_rows(named=True)
    }

    # --- annual features: reuse cache or compute ---
    point_ids = set(coords)
    gf_df = _find_cached_features(point_ids)
    if gf_df is None:
        print("Computing annual features (this may take a while) ...", file=sys.stderr)
        from tam.core.annual_features import compute_annual_features
        gf_full = compute_annual_features(pixel_df)
        gf_df = gf_full.select(["point_id"] + NOISE_FEATURES)

    gf_lookup: dict[str, dict] = {
        row["point_id"]: row
        for row in gf_df.iter_rows(named=True)
    }

    def _val(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return None if np.isnan(f) else round(f, 6)

    pixels = [
        {
            "id":             pid,
            "lon":            round(float(lon), 7),
            "lat":            round(float(lat), 7),
            "dry_ndvi":       _val(gf_lookup.get(pid, {}).get("dry_ndvi")),
            "rec_p":          _val(gf_lookup.get(pid, {}).get("rec_p")),
            "nir_cv":         _val(gf_lookup.get(pid, {}).get("nir_cv")),
            "s1_mean_vh_dry": _val(gf_lookup.get(pid, {}).get("s1_mean_vh_dry")),
        }
        for pid, (lon, lat) in coords.items()
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
