# Plan: Inter-tile radiometric harmonisation

## Problem

The Kowanyama scene mixes observations from two overlapping S2 MGRS tiles (54LWH and 54LWJ).
After NBAR correction, same-pixel same-day pairs from the two tiles still show a systematic
band offset:

| Band | H/J median ratio | 
|------|-----------------|
| B04  | 1.018           |
| B05  | 1.019           |
| B07  | 1.013           |
| B08  | 1.013           |

The offset is **uncorrelated with VZA difference** (r = 0.007), confirming it is radiometric
rather than geometric. It also drifts over time (B07 ratio: 1.019 in 2019 → 1.080 in 2025),
consistent with inter-sensor calibration drift between S2A and S2B, which predominantly
illuminate different tiles at Kowanyama.

NBAR cannot fix this — it only corrects for BRDF/viewing-geometry effects.

## Approach

Two-pass approach that leaves the raw parquet untouched:

**Pass 1 — calibrate:** scan the parquet, find all same-pixel same-day multi-tile
observations (the overlap zone), and compute per-(tile, band, year) scale factors
relative to a reference tile. Write to a small correction table parquet.

**Pass 2 — apply:** in `signals/_shared.py:compute_features_chunked()`, load the
correction table and multiply each observation's band values by the appropriate
scale factor before computing derived quantities (ndvi, re_ratio, swir_mi).

This generalises automatically — any number of tiles, as long as at least one day
of overlap exists with the reference tile.

---

## Architecture

```
utils/tile_harmonisation.py        — new: calibrate() and load_corrections()
signals/_shared.py                 — modify: apply corrections in compute_features_chunked()
data/calibration/<loc>.parquet     — new: correction table (small, ~KB)
```

---

## Step 1 — New module: `utils/tile_harmonisation.py`

### 1a. `calibrate(parquet_path, out_path, bands)`

Scans the parquet to find same-pixel same-day observations from different tiles,
then computes per-(tile, band, year) scale factors.

```python
def calibrate(
    parquet_path: Path,
    out_path: Path,
    bands: list[str] = ["B04", "B05", "B07", "B08", "B11"],
) -> pd.DataFrame:
    """Compute per-(tile, band, year) scale factors from overlap observations.

    Reads the parquet row-group by row-group. For each row group, finds all
    (point_id, date_only) pairs that have observations from more than one tile.
    Computes the median ratio of each non-reference tile to the reference tile
    for each band and year.

    Reference tile: the tile with the most total observations across all years.

    Returns and writes a DataFrame with columns:
        tile, band, year, scale_factor
    where scale_factor multiplies the tile's band value to bring it onto the
    reference tile's radiometry.
    """
```

**Implementation notes:**

- Read only `["point_id", "date", "tile_id", "B04", "B05", "B07", "B08", "B11"]`.
- Extract `date_only = date.dt.date` to match observations from the same calendar day
  (tiles on the same swath are ~14 seconds apart — same illumination, same atmosphere).
- **Tail-buffer (same pattern as `compute_features_chunked`):** the pixel-sorted
  parquet may split a pixel across row-group boundaries — the last pixel in rg N may
  continue into rg N+1.  Peel off the last pixel before processing each row group and
  prepend it to the next:

  ```python
  tail_buf: pl.DataFrame | None = None
  for rg_idx in range(n_rg):
      chunk = pl.from_arrow(pf.read_row_groups([rg_idx], columns=LOAD_COLS))
      if tail_buf is not None:
          if chunk["point_id"][0] == tail_buf["point_id"][0]:
              chunk = pl.concat([tail_buf, chunk])
          tail_buf = None
      if rg_idx < n_rg - 1:
          last_pid = chunk["point_id"][-1]
          tail_mask = chunk["point_id"] == last_pid
          tail_buf = chunk.filter(tail_mask)
          chunk = chunk.filter(~tail_mask)
      # ... process chunk ...
  ```

  Without this, any overlap pixel that straddles a row-group boundary silently
  contributes zero pairs — the calibration is still computed but from fewer samples
  than expected.

- For each row group, pivot to wide form on `tile_id`, drop rows where any tile is
  missing, compute `ratio = other_tile_band / ref_tile_band`.
- Accumulate all ratios across row groups into a list; at the end take the
  **weighted median** (by observation count) per (tile, band, year).
- Reference tile tie-breaker: if two tiles have equal observation counts, pick the
  alphabetically first tile id (`sorted(counts.items())[0]`) so the result is
  deterministic.
- Clamp scale factors to [0.85, 1.15] to guard against degenerate overlap samples.
- Print a summary table on completion so the caller can inspect the corrections.

### 1b. `load_corrections(calibration_path)`

```python
def load_corrections(calibration_path: Path) -> dict[tuple[str, str, int], float] | None:
    """Load correction table as a lookup dict keyed by (tile_id, band, year).

    Returns None if the file does not exist (correction disabled — graceful
    fallback so existing pipelines continue to work without re-calibrating).
    """
```

### 1c. CLI entry point

```bash
python -m utils.tile_harmonisation --location kowanyama
```

Writes to `data/calibration/kowanyama.parquet`. Prints the correction table.

---

## Step 2 — Apply corrections in `signals/_shared.py`

### 2a. Load correction table at the start of `compute_features_chunked()`

Add an optional parameter:

```python
def compute_features_chunked(
    path: Path,
    ...
    calibration_path: Path | None = None,   # <-- new
) -> ...:
```

At the top of the function body:

```python
from utils.tile_harmonisation import load_corrections
corrections = load_corrections(calibration_path) if calibration_path else None
```

### 2b. Apply per-observation scale factors inside the row-group loop

The current pipeline adds `year`/`month` and computes `ndvi`/`re_ratio`/`swir_mi`
in one `.with_columns()` call.  Split that block into three stages so that `year`
is available for the correction join before the derived bands are computed:

```python
# Stage 1 — temporal columns (needed as join key for corrections)
chunk = chunk.with_columns([
    pl.col("date").dt.year().alias("year"),
    pl.col("date").dt.month().alias("month"),
])

# Stage 2 — apply tile corrections (join on tile_id + year)
# <correction block — see below>

# Stage 3 — derived bands (use corrected B04/B05/B07/B08/B11)
chunk = chunk.with_columns([
    ((pl.col("B08") - pl.col("B04")) / (pl.col("B08") + pl.col("B04"))).alias("ndvi"),
    (pl.col("B07") / pl.col("B05")).alias("re_ratio"),
    ((pl.col("B08") - pl.col("B11")) / (pl.col("B08") + pl.col("B11"))).alias("swir_mi"),
])
```

The correction block (Stage 2):

```python
if corrections is not None and "tile_id" in chunk.columns:
    # Build a per-row scale factor for each band using the correction table.
    # Use polars map_elements or a join on (tile_id, year) for efficiency.
    for band in ["B04", "B05", "B07", "B08", "B11"]:
        chunk = chunk.with_columns(
            pl.struct(["tile_id", "year", pl.lit(band).alias("_band")])
            .map_elements(lambda r: corrections.get((r["tile_id"], band, r["year"]), 1.0),
                          return_dtype=pl.Float32)
            .alias(f"_cf_{band}")
        )
        chunk = chunk.with_columns(
            (pl.col(band) * pl.col(f"_cf_{band}")).alias(band)
        ).drop(f"_cf_{band}")
```

> **Note:** `map_elements` per row is slow for large chunks. Prefer a polars join
> approach: materialise the corrections dict as a small DataFrame, join on
> `(tile_id, year)`, then multiply. This avoids Python-level row iteration entirely.

Preferred approach — join-based:

```python
if corrections is not None:
    corr_rows = [
        {"tile_id": t, "band": b, "year": y, "scale": s}
        for (t, b, y), s in corrections.items()
    ]
    corr_df = pl.DataFrame(corr_rows)  # columns: tile_id, band, year, scale

    for band in ["B04", "B05", "B07", "B08", "B11"]:
        band_corr = (
            corr_df.filter(pl.col("band") == band)
            .select(["tile_id", "year", "scale"])
        )
        chunk = (
            chunk
            .join(band_corr, on=["tile_id", "year"], how="left")
            .with_columns(
                pl.when(pl.col("scale").is_not_null())
                  .then(pl.col(band) * pl.col("scale"))
                  .otherwise(pl.col(band))
                  .alias(band)
            )
            .drop("scale")
        )
```

Corrections are applied **before** the ndvi / re_ratio / swir_mi derived columns
are computed, so all features benefit from the correction.

### 2c. `LOAD_COLS` — add `tile_id`

`compute_features_chunked()` currently loads:

```python
LOAD_COLS = ["point_id", "lon", "lat", "date", "scl_purity",
             "B04", "B05", "B07", "B08", "B11"]
```

Add `"tile_id"` conditionally — read the parquet schema first so older parquets
(fetched before `tile_id` was added) don't raise on load:

```python
_schema_names = set(pf.schema_arrow.names)
LOAD_COLS = ["point_id", "lon", "lat", "date", "scl_purity",
             "B04", "B05", "B07", "B08", "B11"]
if "tile_id" in _schema_names:
    LOAD_COLS = ["point_id", "lon", "lat", "date", "tile_id", "scl_purity",
                 "B04", "B05", "B07", "B08", "B11"]
```

If `tile_id` is absent the correction block is a no-op (the `if "tile_id" in chunk.columns`
guard in Stage 2 handles it — and no `calibration_path` will exist for that location anyway).

### 2d. Wire calibration_path through `extract_parko_features()`

`extract_parko_features()` in `signals/__init__.py` calls `compute_features_chunked()`.
Add a `calibration_path` parameter and pass it through:

```python
def extract_parko_features(
    pixel_df,
    loc,
    ...,
    calibration_path: Path | None = None,   # <-- new
) -> pd.DataFrame:
```

Default: `None` (no correction) for backwards compatibility. Callers can pass
`loc.calibration_path()` once that method exists.

### 2e. Default calibration path on Location

Add a convenience method to `utils/location.py`:

```python
def calibration_path(self) -> Path | None:
    """Return the tile harmonisation table path if it exists, else None."""
    p = PROJECT_ROOT / "data" / "calibration" / f"{self.id}.parquet"
    return p if p.exists() else None
```

Pipelines then need no changes — just pass `calibration_path=scene_loc.calibration_path()`
to `extract_parko_features()`.

---

## Step 3 — Update `pipelines/kowanyama_pormpuraaw.py`

```python
scene_features = extract_parko_features(
    scene_path, scene_loc,
    year_to=2025,
    calibration_path=scene_loc.calibration_path(),   # <-- add
)
```

The training set (Pormpuraaw) also benefits if it has a calibration table:

```python
train_features = extract_parko_features(
    train_raw, train_loc,
    calibration_path=train_loc.calibration_path(),   # <-- add
)
```

---

## Verification

### V1 — Unit: scale factors are correct at nadir case

On a synthetic parquet where tile A is exactly 1.02× tile B for band B07 in 2022,
`calibrate()` should produce `scale_factor = 1/1.02 ≈ 0.9804` for tile A in B07/2022.

### V2 — Diagnostic: H/J ratio collapses to 1.0 after correction

Re-run the same-pixel same-day pivot analysis from the diagnosis:

```python
# Before: B07 H/J = 1.013 median
# After:  B07 H/J should be ≈ 1.000 median
```

### V3 — Feature stripe disappears

Re-run the lat-bin feature profile (rec_p, nir_cv at 0.005° resolution through
lat -15.47 to -15.46). The step-change should flatten.

### V4 — Pipeline: stripe absent in heatmap

Re-run `python -m pipelines.kowanyama_pormpuraaw`. The N-S band at lat -15.465
should be absent in `kowanyama_pormpuraaw_prob_black.png`.

---

## Re-collection not required

The raw parquet is unchanged. Correction is applied at feature-extraction time.
The pipeline re-run is fast (feature extraction only, no STAC fetch).

---

## Generalisation to new tiles

When a new location is added with a different tile overlap (e.g. three tiles),
the same `calibrate()` call handles it:

1. Reference tile is chosen automatically (most observations).
2. Each other tile gets its own scale factors derived from its overlap days with
   the reference.
3. If tile C overlaps tile B but not tile A (the reference), a two-hop correction
   is needed (C→B scale, B→A scale, compose). This case is deferred until needed
   but the architecture supports it — the calibration step would need a graph
   traversal to find the reference path.
