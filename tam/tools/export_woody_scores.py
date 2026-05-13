"""tam/tools/export_woody_scores.py — Score training-region pixels with the woody classifier.

Usage:
    python3 tam/tools/export_woody_scores.py <region_id>

Writes outputs/woody_scores/{region_id}.json with per-pixel prob_woody, lon, lat.
The browser uses this to render a kept/dropped overlay with a single threshold slider.

Cache strategy: if outputs/woody_scores/{region_id}.json already exists it is returned
immediately (delete it to force a rebuild).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_DATA_DIR = _REPO / "data" / "training"
_OUT_DIR  = _REPO / "outputs" / "woody_scores"


def _load_scorer():
    """Import load_model and score functions from woody-classifier/score.py."""
    spec = importlib.util.spec_from_file_location(
        "woody_score", _REPO / "woody-classifier" / "score.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_model, mod.score_parquet


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


def export_region(region_id: str) -> str:
    """Score pixels for region_id and write JSON. Returns the output path."""
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
    want = ["point_id", "lon", "lat", "date", "source", "scl_purity",
            "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
            "vh", "vv"]
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

    # Score via woody-classifier
    load_model, _score_parquet = _load_scorer()
    model_dir = _REPO / "outputs" / "woody-classifier"
    model, feat_names, backend = load_model(model_dir)
    print(f"Loaded {backend} model ({len(feat_names)} features)", file=sys.stderr)

    # score_parquet expects a file path; write a temp parquet then score it
    import tempfile, pyarrow as pa, pyarrow.parquet as pq2
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "region.parquet"
        pq2.write_table(pa.Table.from_pandas(pixel_df, preserve_index=False), tmp_path)
        tmp_out = Path(tmp) / "scores"
        out_file = _score_parquet(tmp_path, model, feat_names, tmp_out)
        if out_file is None:
            raise RuntimeError(f"Scoring returned no output for region {region_id}")
        scores_df = pd.read_parquet(out_file).set_index("point_id")

    joined = coords.join(scores_df, how="left")

    pixels = [
        {
            "id":         pid,
            "lon":        round(float(row["lon"]), 7),
            "lat":        round(float(row["lat"]), 7),
            "prob_woody": None if np.isnan(float(row["prob_woody"])) else round(float(row["prob_woody"]), 4),
        }
        for pid, row in joined.iterrows()
    ]

    payload = {"region_id": region_id, "label": label, "pixels": pixels}
    out_path.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"Wrote {len(pixels):,} pixels → {out_path}", file=sys.stderr)
    return str(out_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <region_id>", file=sys.stderr)
        sys.exit(1)
    export_region(sys.argv[1])
