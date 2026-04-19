# Plan: cli/describe.py — Single-Bbox Signal Description

## Context
`cli/explore.py` requires labeled presence/absence sub_bboxes to compute separability scores. This script provides a complementary descriptive mode: given any single bbox (named or inline), compute all signals and produce a "signal fingerprint" — spatial maps, per-signal histograms, and summary stats. Useful for characterising unlabeled patches (survey zones, woody areas, ad-hoc coordinates) and identifying confounding factors such as water frequency, fire history effects, soil moisture regime, and vegetation type.

## CLI

```
python cli/describe.py --location frenchs --bbox woody_1 --out outputs/describe-frenchs-woody1
python cli/describe.py --location frenchs --bbox "[141.537, -15.805, 141.539, -15.803]" --out outputs/describe-frenchs-custom
python cli/describe.py --location frenchs --bbox woody_1 --year-from 2021 --year-to 2024 --out outputs/...
```

Arguments:
- `--location` (required): location id
- `--bbox` (required): sub_bbox key from location YAML **or** inline `"[lon_min, lat_min, lon_max, lat_max]"`
- `--out` (required): output directory
- `--year-from`, `--year-to`: optional year filter

## Bbox resolution

```python
def resolve_bbox(bbox_arg: str, loc) -> tuple[float, float, float, float]:
    # 1. Try as named sub_bbox key
    if bbox_arg in loc.sub_bboxes:
        sub = loc.sub_bboxes[bbox_arg]
        return tuple(sub.bbox), sub.label
    # 2. Try as inline JSON list
    parsed = json.loads(bbox_arg)
    assert len(parsed) == 4
    return tuple(parsed), f"[{bbox_arg}]"
```

## Steps

### Step 1 — Extract tabular features
```python
features = extract_parko_features(
    loc.parquet_path(), loc,
    bbox=bbox_tuple, year_from=..., year_to=...,
)
```
Reuses the existing bbox row-group scan in `signals/__init__.py`. No changes needed.

### Step 2 — Curve-based signals
Same pattern as `explore.py`'s `step_curve_signals`:
1. `ensure_pixel_sorted(parquet_path)` — `signals/_shared.py`
2. `annual_ndvi_curve_chunked(sorted_path, out_path=tmp, smooth_days=30, ...)` — `signals/_shared.py`
3. Load raw pixels filtered to bbox from sorted parquet
4. `RecessionSensitivitySignal().compute(raw_df, loc, _curve=curve_path)`
5. `GreenupTimingSignal().compute(raw_df, loc, _curve=curve_path)`
6. Filter results to `bbox_tuple` pixels (curve signals operate over full loc, so post-filter by lon/lat)

### Step 3 — Descriptive stats
For each signal column over the bbox pixels:
```python
{"n_pixels": ..., "pct_valid": ..., "median": ..., "iqr": ..., "p10": ..., "p90": ...}
```
High IQR noted in report as "high internal variance" (suggests mixed cover — the most useful flag for sub-pixel interpretation).

### Step 4 — Plots
- `plot_signal_map(features, col, loc, ...)` — reused as-is, all pixels same colour scale
- `plot_distributions(features, col, loc, presence_ids=None, absence_ids=None, ...)` — already renders single histogram when class args are None

Both functions are in `signals/diagnostics.py` and require no modification.

### Step 5 — Report
`report.md` sections:
1. **Header**: location, bbox label, bbox coordinates, pixel count, year range, parquet path
2. **Signal summary table**: `signal | description | n_valid | median | IQR | p10 | p90 | variance flag`
3. **Per-signal section**: map image + distribution image + stat table

## Future signals to wire in (not in scope now, but describe.py should be easy to extend)

Time-series style signals (preserve temporal dimension rather than collapsing to a scalar):
- **SCL class composition time series** ✓ implemented (`signals/scl_composition.py`) — wire into describe.py
- **NDVI time series** — per-pixel smoothed NDVI curve (already built in step 2, just needs a plot)
- **SWIR time series** — B11/B12 per observation; captures soil/canopy moisture dynamics over time
- **NDWI time series** — (B03−B08)/(B03+B08); complements SWIR for inundation characterisation
- **NDVI long-term trend** — Sen's slope over full observation period; flags encroachment or clearing
- **Inter-annual phenological consistency** — year-to-year variance of green-up DOY

Scalar signals:
- **Fire signal** — from fire log parquet (see NAFI.md)
- **SWIR2/SWIR1 ratio** — B12/B11; soil vs canopy moisture discrimination
- **Green season duration** — days above NDVI threshold per year

## Files touched
- **New**: `cli/describe.py`
- No changes to `utils/` or `explore.py`
- `signals/scl_composition.py` — already implemented, just needs wiring in

## Key reused functions
| Function | File |
|---|---|
| `extract_parko_features(..., bbox=...)` | `signals/__init__.py` |
| `ensure_pixel_sorted` | `signals/_shared.py` |
| `annual_ndvi_curve_chunked` | `signals/_shared.py` |
| `plot_signal_map` | `signals/diagnostics.py` |
| `plot_distributions` | `signals/diagnostics.py` |
| `RecessionSensitivitySignal`, `GreenupTimingSignal` | `signals/recession.py`, `signals/greenup.py` |
| `QualityParams` | `signals/__init__.py` |

## Verification
```bash
python cli/describe.py --location frenchs --bbox woody_1 --out /tmp/desc-test
# Expect: report.md, map_*.png, dist_*.png in /tmp/desc-test

python cli/describe.py --location frenchs --bbox "[141.551545, -15.769770, 141.554368, -15.768536]" --out /tmp/desc-test2
# Same bbox as woody_1 supplied inline — output should match
```
