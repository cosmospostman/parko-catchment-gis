# signals

Spectral/temporal signal abstractions for Parkinsonia discrimination.

Each signal computes a per-pixel summary statistic from a Sentinel-2 pixel
parquet. Signals were developed at Longreach and are designed to be applied
at any site with site-specific parameter tuning.

## The five signals

| Signal | Class | Metric | Physics | Parkinsonia direction |
|--------|-------|--------|---------|----------------------|
| Dry-season NIR CV | `NirCvSignal` | `nir_cv` | Inter-annual stability of dry-season B08 | **lower** (stable evergreen) |
| Wet/dry amplitude | `RecPSignal` | `rec_p` | Annual NDVI p90 − p10 | **higher** (deep seasonal swing) |
| Red-edge floor | `RedEdgeSignal` | `re_p10` | Annual p10 of B07/B05 ratio | **higher** (retained chlorophyll) |
| SWIR moisture floor | `SwirSignal` | `swir_p10` | Annual p10 of (B08−B11)/(B08+B11) | **higher** (sustained canopy water) |
| Flowering flash | `FloweringSignal` | `fi_p90_cg` | Contrast-gated DOY anomaly of FI_by | **higher** (isolated flowering events) |

`nir_cv` and `rec_p` are the primary classifier features at Longreach (Mahalanobis
distance 1.43). `re_p10` is retained for generalisability. `swir_p10` is correlated
with `re_p10` (r ≈ 0.73) and is included as an independent canopy-moisture check.
`FloweringSignal` is the only signal that uniquely suppresses riparian false positives.

## Design goals

Signals should capture transferable behaviour patterns — things that are true of
Parkinsonia as a plant — rather than assumptions that are specific to one location
or season. A signal that fires because "it's the wet season at Longreach" is fragile;
one that fires because "this pixel has an unusually deep annual NDVI swing" travels.
Concretely: prefer statistics that are self-referential (within-pixel z-scores,
inter-annual CV) or derived directly from the data (p90 − p10 across the full year)
over ones that require hard-coded calendar windows.

That said, assumptions are unavoidable — and signals are explicitly experimental.
`dry_months` is a Location-level assumption; `doy_bin_days` and `haze_b02_anom_max`
in FloweringSignal are algorithmic assumptions. The design accepts this: every
assumption is surfaced as a named, tunable parameter rather than buried in code,
so it can be tested and adjusted as signals are applied to new sites. Use
`sweep_signal` to challenge assumptions rather than leaving them at their defaults.

## Quick start

```python
from utils.location import get
from signals import NirCvSignal

loc = get("longreach")
df  = pd.read_parquet(loc.parquet_path())

result = NirCvSignal().diagnose(df, loc, out_dir=Path("outputs/longreach-nir-cv"))
# Writes: map.png, distributions.png
# Returns: {'signal': 'nir_cv', 'separability': 1.43, 'figures': [...], ...}
```

All four tabular features in one call:

```python
from signals import extract_parko_features

features = extract_parko_features(df, loc)
# → DataFrame[point_id, lon, lat, nir_cv, rec_p, re_p10, swir_p10]
```

## Interface

Every signal class follows this pattern:

```python
sig = NirCvSignal()                         # or NirCvSignal(NirCvSignal.Params(...))
stats  = sig.compute(df, loc)               # → per-pixel DataFrame
result = sig.diagnose(df, loc, out_dir=...) # → dict + writes figures
```

`compute()` returns a lean per-pixel DataFrame. `diagnose()` calls `compute()`,
writes `map.png` and `distributions.png` to `out_dir`, and returns a summary dict:

```python
{
    "signal":          "nir_cv",
    "site":            "longreach",
    "n_pixels":        40996,
    "presence_median": 0.047,
    "absence_median":  0.110,
    "separability":    1.43,    # (presence_med - absence_med) / pooled_std
    "figures":         [Path("outputs/longreach-nir-cv/map.png"), ...],
}
```

`FloweringSignal` follows the same interface but its internal algorithm is different
(DOY-binned z-scores, haze masking, contrast gating). It also writes an additional
`anomaly_profile.png` figure. It does not accept `dry_months` from `Location` because
it is not season-bounded — the anomaly baseline is DOY-bin-relative across the full year.

## Parameters

Every signal has a nested `Params` dataclass. The shared quality fields live in
`QualityParams`, which is composed into each signal's `Params`.

```python
from signals import QualityParams
from signals import NirCvSignal, FloweringSignal

# Default params (same as calling signal with no arguments):
NirCvSignal.Params()
# NirCvSignal.Params(quality=QualityParams(scl_purity_min=0.5, min_obs_per_year=10, min_obs_dry=5))

FloweringSignal.Params()
# FloweringSignal.Params(quality=..., doy_bin_days=14, haze_b02_anom_max=0.010,
#                        peak_percentile=75, riparian_nir_percentile=90, min_pixel_obs=10)
```

Key parameters by signal:

| Signal | Parameter | Default | Effect |
|--------|-----------|---------|--------|
| All | `scl_purity_min` | 0.5 | Minimum fraction of valid SCL pixels per observation |
| All | `min_obs_per_year` | 10 | Min observations per (pixel, year) for percentile stats |
| `NirCvSignal` | `min_obs_dry` | 5 | Min dry-season obs per (pixel, year) |
| `RecPSignal` | `peak_months` | {3,4,5} | Reference wet-season window (not used for `rec_p` — kept for diagnostic `rec_mean`) |
| `RecPSignal` | `trough_months` | {7,8,9} | Reference dry-season window (same caveat) |
| `FloweringSignal` | `haze_b02_anom_max` | 0.010 | Scene-mean B02 anomaly threshold above which a date is flagged as hazy |
| `FloweringSignal` | `doy_bin_days` | 14 | DOY bin width for anomaly baseline (26 bins/year) |
| `FloweringSignal` | `min_pixel_obs` | 10 | Min total observations to include a pixel |

`dry_months` is **not** a signal parameter — it comes from the `Location` object
(`loc.dry_months`) so it stays consistent with the site's fetch and other analyses.

## Tuning at a new site

### 1. Run with defaults first

```python
loc = get("barcaldine")
df  = pd.read_parquet(loc.parquet_path())

result = NirCvSignal().diagnose(df, loc, out_dir=Path("outputs/barcaldine-nir-cv"))
print(result["separability"])
```

Look at the `separability` score and the distribution figure. A score above ~0.8
suggests the signal is discriminating. Below that, examine whether the issue is
data density (relax `min_obs_dry`) or a genuinely weak signal at this site.

### 2. Grid search over parameters

`sweep_signal` evaluates all combinations of a parameter grid and returns a
DataFrame ranked by `|separability|`. Both quality params and signal-specific
params can be swept simultaneously or independently.

```python
from signals import sweep_signal, NirCvSignal

results = sweep_signal(
    NirCvSignal,
    param_grid={
        "scl_purity_min": [0.3, 0.5, 0.7],
        "min_obs_dry":    [3, 5, 8],
    },
    df=df,
    loc=loc,
)
print(results)
# scl_purity_min  min_obs_dry  separability  n_pixels  n_presence  n_absence
#            0.3            3         -1.61     41203         312        8940
#            0.5            3         -1.55     39880         305        8712
#            ...
```

Quality keys (`scl_purity_min`, `min_obs_per_year`, `min_obs_dry`) are routed
into `QualityParams` automatically; all other keys go to the signal's `Params`.

The same call works for signal-specific params:

```python
from signals import sweep_signal, FloweringSignal

results = sweep_signal(
    FloweringSignal,
    param_grid={
        "haze_b02_anom_max": [0.005, 0.010, 0.015, 0.020],
        "doy_bin_days":      [7, 14, 21],
    },
    df=df,
    loc=loc,
)
```

**Watch `n_presence` and `n_absence` alongside `separability`.** A high score
from a small surviving population is not trustworthy. A sharp pixel drop at the
same threshold as a separability spike is a warning sign of quality-filter
artefacts.

### 3. Override specific params manually

```python
from signals import NirCvSignal, QualityParams

result = NirCvSignal(NirCvSignal.Params(
    quality=QualityParams(min_obs_dry=3)    # less data than Longreach
)).diagnose(df, loc)
```

### 4. Save tuned params to the location YAML

Once you have working params, add them to `data/locations/<id>.yaml`:

```yaml
signals:
  nir_cv:
    min_obs_dry: 3
  flowering:
    haze_b02_anom_max: 0.012
```

Quality overrides (`scl_purity_min`, `min_obs_per_year`, `min_obs_dry`) are
recognised automatically and applied to `QualityParams`. All other keys are
passed directly to the signal's `Params`.

### 5. Load params from YAML in scoring scripts

```python
from signals._shared import load_signal_params

params  = load_signal_params(loc, "nir_cv")
result  = NirCvSignal(params).compute(df, loc)
```

`load_signal_params` merges site overrides with signal defaults — absent keys
keep their default values.

## Presence / absence class assignment

`diagnose()` and `separability_score()` derive class boundaries from
`loc.sub_bboxes`. A sub-bbox with `role: presence` defines the Parkinsonia
pixels; `role: absence` defines the background. If neither is defined for a
site, class-split figures and separability score are skipped.

Add sub-bboxes to a location's YAML to enable class-aware diagnostics:

```yaml
sub_bboxes:
  infestation:
    label: "Confirmed Parkinsonia strip"
    role: presence
    bbox: [lon_min, lat_min, lon_max, lat_max]
  grassland:
    label: "Absence background"
    role: absence
    bbox: [lon_min, lat_min, lon_max, lat_max]
```

## Files

```
signals/
  __init__.py      QualityParams, all five signal classes, extract_parko_features, sweep_signal
  _shared.py       load_and_filter, annual_percentile, dry_season_cv, load_signal_params
  nir_cv.py        NirCvSignal
  wet_dry_amp.py   RecPSignal
  red_edge.py      RedEdgeSignal
  swir.py          SwirSignal
  flowering.py     FloweringSignal
  diagnostics.py   plot_signal_map, plot_distributions, separability_score, _resolve_classes
  tuning.py        sweep_signal
```
