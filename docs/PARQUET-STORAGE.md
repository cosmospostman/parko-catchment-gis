# Parquet Storage Format

Tile parquets are written by `utils/pixel_collector.py` (S2) and `utils/s1_collector.py`
(S1), then merged by `utils/parquet_utils.py` into combined files — one per MGRS tile per
year, stored under `data/pixels/<location>/<year>/<tile_id>.parquet`.

---

## Schema

Each row is one (pixel, acquisition-date) observation. S2 and S1 rows share the same
file, distinguished by the `source` column. Columns irrelevant to a row's sensor are null.

| Column | Dtype | S2 | S1 | Notes |
|---|---|---|---|---|
| `point_id` | `Utf8` (dict) | ✓ | ✓ | Grid pixel identifier, e.g. `"px_0042_0031"` |
| `lon` | `Float32` | ✓ | ✓ | Pixel centre longitude (EPSG:4326) — ~1 mm precision |
| `lat` | `Float32` | ✓ | ✓ | Pixel centre latitude |
| `date` | `Date32` | ✓ | ✓ | Acquisition date (days since epoch) |
| `source` | `Utf8` (dict) | `"S2"` | `"S1"` | Sensor tag |
| `item_id` | `Utf8` (dict) | ✓ | null | STAC item ID |
| `tile_id` | `Utf8` (dict) | ✓ | null | MGRS tile code, e.g. `"54LWJ"` |
| `orbit` | `Utf8` (dict) | null | ✓ | S1 relative orbit identifier |
| `scl_purity` | `Int8` | ✓ | null | 0 or 1 — clear-pixel flag for the 1×1 chip |
| `scl` | `Int8` | ✓ | null | Raw SCL class (4=veg, 5=soil, 6=water, 11=snow) |
| `aot` | `UInt8` | ✓ | null | Aerosol quality × 100 (0–100) |
| `view_zenith` | `UInt8` | ✓ | null | Inverse view zenith × 100 (0–100; 100 = nadir) |
| `sun_zenith` | `UInt8` | ✓ | null | Inverse sun zenith × 100 (0–100; 100 = high sun) |
| `B02`–`B12` (10 bands) | `UInt16` (nullable) | ✓ | null | Surface reflectance × 10 000 |
| `vh` | `Float32` (nullable) | null | ✓ | VH backscatter (linear power, not dB) |
| `vv` | `Float32` (nullable) | null | ✓ | VV backscatter (linear power, not dB) |

**Not stored — computed at read time:**
- S2: NDVI, NDWI, EVI, MAVI, NDRE, CI\_RE — via `add_spectral_indices()` in `analysis/constants.py`
- S1: s1\_vh (dB), s1\_vv (dB), s1\_vh\_vv (dB ratio), s1\_rvi — via `TAMDataset` at load time

---

## Encoding decisions

### Bands: UInt16 × 10 000

Sentinel-2 L2A surface reflectance is already quantised at 1/10 000 by ESA before
distribution. Storing as `float32` adds no information — it just wastes two bytes per
value. `uint16 × 10 000` is an exact round-trip within the source data's native precision
(max round-trip error < 2×10⁻⁸, measured).

Pixels with no S2 observation carry `null` rather than `NaN` in the band columns —
`float32` cannot be stored in a uint16 column, so NaN becomes a nullable null.

### Quality scalars: UInt8 / Int8

`aot`, `view_zenith`, and `sun_zenith` are normalised to [0, 1] at extraction time and
stored as `uint8 × 100`. Precision is 0.01 (1%), which is adequate for quality gating.

`scl_purity` is binary (0 or 1) because the extraction chip is 1×1 — stored as `int8`
with no scale factor. `scl` retains its existing `int8` encoding.

### lon / lat: Float32

Sentinel-2 10 m pixels are ~10 m apart. Float32 gives ~7 decimal digits of precision
(~1 mm at equator), which is more than sufficient. Storing as Float64 wastes 4 bytes per
row with no benefit.

### date: Date32

Dates are daily granularity with no sub-day component. `date32` (days since epoch, INT32)
halves the column size compared to `timestamp[us]` (INT64) and improves RLE encoding.

All consumers use `pl.col("date").cast(pl.Date)` or pandas `pd.to_datetime()`, both of
which handle `date32` natively.

### String columns: dictionary encoding

`item_id`, `tile_id`, and `point_id` are low-cardinality repeated strings — e.g. `tile_id`
repeats millions of times per file. Dictionary encoding writes a small dictionary page and
integer indices, reducing these columns substantially.

### S1 backscatter: Float32 (unchanged)

`vh` and `vv` are stored as raw linear power (not dB), as output by the MPC S1 RTC
pipeline. Linear power is retained rather than dB because the dB conversion and derived
features (s1\_vh\_vv ratio, RVI) are computed by `TAMDataset` at load time — analogous to
the spectral index pattern for S2.

Float32 is appropriate here: S1 SAR backscatter has ~2–3 dB radiometric accuracy, which
maps to ~25–50% variance in linear power. Uint16 encoding would require choosing a fixed
scale factor across a dynamic range of ~10⁻⁴ to ~1 (40 dB), which would either clip
bright targets or waste precision on dark ones.

### No derived indices

Six derived S2 spectral indices and four S1 features were previously stored as `float32`
columns. They are fully deterministic functions of the raw bands, so storing them is pure
redundancy.

---

## Storage savings

**Measured on `54LWJ.parquet` (86.8 M rows):**

| Change | Saving |
|---|---|
| Drop 6 derived indices | ~23% (~522 MB) |
| Bands float32 → uint16 × 10 000 | ~15% (~344 MB) |
| Quality scalars float32 → uint8 | ~2% (~49 MB) |
| **Dtype changes total** | **~50%** (2 250 MB → 1 118 MB, measured) |

**Writer-level optimisations (ZSTD, dictionary, row groups, Float32 lon/lat, Date32):**

| File | Before | Estimated after |
|---|---|---|
| kowanyama (79 GB, 1.68 B rows) | 79 GB | ~55–65 GB |
| kowtown (~70 GB est.) | ~70 GB | ~48–58 GB |
| Total /data/pixels (~337 GB) | ~337 GB | ~230–270 GB |

Mitchell River catchment projection (17 tiles, 1 yr): ~1 765 GB → **~880 GB** combining
both layers of optimisation.

---

## Writer configuration

```python
WRITE_OPTS = dict(
    compression="zstd",
    compression_level=3,
    use_dictionary=["point_id", "item_id", "tile_id"],
    write_statistics=True,
)
```

ZSTD level 3 is the sweet spot — levels above 3 yield diminishing returns on float
columns. Row groups are consolidated to 5 000 000 rows to reduce footer-seek overhead and
improve predicate pushdown on date and point_id filters.

---

## Read throughput

The dtype and codec changes are primarily a **storage win, not a read-speed win**. Band
columns dominate file size and are near-incompressible; ZSTD decompresses marginally
slower than SNAPPY, so sequential full-table scans are neutral.

Where read performance does improve:

- **Row group consolidation** — the dominant throughput win. The kowanyama baseline had
  2 084 row groups (one per sort pass), meaning 2 084 footer seeks per full scan.
  Consolidating to ~340 groups (5 M rows each) reduces open/metadata overhead and makes
  min/max predicate pushdown effective.
- **Dictionary columns** (`item_id`, `tile_id`, `point_id`) — filter and group-by
  operations work on integers rather than strings; no benefit for pure scans.
- **Date32** — smaller column, better RLE; marginal benefit on date-range filters.

---

## Reading the parquet

Every consumer must call `ensure_float32_bands()` and `add_spectral_indices()` before
accessing band or index columns. Both are in `analysis/constants.py`.

```python
from analysis.constants import add_spectral_indices, ensure_float32_bands

df = pl.read_parquet("54LWJ.parquet")
df = add_spectral_indices(ensure_float32_bands(df))
# df now has float32 B02–B12 and float32 NDVI/NDWI/EVI/MAVI/NDRE/CI_RE
```

`ensure_float32_bands` is **idempotent** — if a band column is already `float32` (old
parquet written before this optimisation) it is left unchanged. `add_spectral_indices`
calls `ensure_float32_bands` first, so calling both is equivalent to calling
`add_spectral_indices` alone for Polars frames.

### Consumer sites

| Site | Transform applied |
|---|---|
| `tam/core/train.py` — `_compute_band_stats_worker` | `ensure_float32_bands` after arrow read |
| `tam/core/train.py` — `_dataset_worker` | `ensure_float32_bands` after arrow read |
| `tam/_prep_worker.py` — slim scan | `ensure_float32_bands` after collect |
| `tam/core/score.py` — `_preprocess` | `add_spectral_indices` (calls ensure_float32 first) |
| `tam/core/dataset.py` — `TAMDataset` | `add_spectral_indices` guarded by column presence |
| `signals/temporal.py` — `TemporalVarianceSignal` | NDVI computed inline from B08/B04 |
| Diagnostic scripts (`diag_sequence`, `diag_tile_coverage`, etc.) | `add_spectral_indices(ensure_float32_bands(...))` |

---

## Quality columns — physical values

Consumers that need physical quality values must divide by the scale factor
`UINT8_QUALITY_SCALE = 100.0` from `analysis/constants.py`. The TAM pipeline gates on
`scl_purity >= 1` (binary) and `aot` threshold comparisons scaled accordingly.
