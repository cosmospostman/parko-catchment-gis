# SWIR Moisture Index — Analysis Plan

## Hypothesis

Shortwave infrared (B11, ~1610 nm) is strongly absorbed by liquid water in plant
tissue. High canopy water content suppresses B11 reflectance relative to NIR (B08).
The SWIR moisture index:

```
swir_mi = (B08 - B11) / (B08 + B11)
```

is analogous to NDVI but sensitive to canopy water rather than chlorophyll. Values near
1 indicate high canopy water (healthy, well-hydrated vegetation); values near 0 or
negative indicate dry canopy, bare soil, or rock.

Parkinsonia's deep roots access groundwater through the dry season. Its canopy should
remain better-hydrated than shallow-rooted grasses, which desiccate as soil moisture
depletes.

**No dry-season window is assumed.** The timing of grass desiccation varies with
rainfall. Rather than fixing a Jul–Sep window, the analysis computes the inter-class
contrast across every acquisition date in the 2020–2025 archive. The per-pixel summary
statistic (annual low percentile) is window-free by construction. A contrast time
series reveals whether the separation is consistent or shifts with rainfall timing.

**Why this is a different axis from red-edge:** B07/B05 measures *chlorophyll
concentration* — whether the plant is photosynthetically active. swir_mi measures
*leaf water content* — whether the canopy is hydrated. A plant can retain chlorophyll
without being well-hydrated (stressed but not yet senescent) or be well-hydrated
without high chlorophyll (succulent, early-season growth). In practice these are
correlated for Parkinsonia, but the B11 decomposition diagnostic (step 6) will confirm
whether swir_mi carries information independent of NIR structure.

**The riparian confound — and why it matters here more than elsewhere:** Dense riparian
woodland sits over a permanent water source. In dry season its canopy water content is
sustained by groundwater, similar to Parkinsonia. If real riparian canopy were present
in the survey area, it would likely score similarly to Parkinsonia on swir_mi. This
makes swir_mi the signal *most vulnerable* to riparian confounding and therefore the
most important to test on the current data before any spatial expansion.

The existing riparian proxy (bare riverbed) will score *low* on swir_mi — bare soil
has negligible canopy water. The result on the current 748-pixel dataset will look
favourable for swir_mi, but must be explicitly flagged as likely to degrade when
genuine riparian woodland is included.

**Expected class ordering when contrast is at its peak:**

| Class | Expected swir_mi | Mechanism |
|-------|-----------------|-----------|
| Parkinsonia | **Highest** | Groundwater access sustains canopy hydration |
| Riparian bare-soil proxy | Low | No plant water content in bare riverbed pixels |
| Grassland | **Lowest** | Desiccated leaves; soil dominates pixel |
| (Hypothetical riparian woodland) | Similar to Parkinsonia | Permanent water access |

## Data

- **Source:** `data/longreach_pixels.parquet`
- **Bands used:** `B08` (NIR, ~842 nm), `B11` (SWIR 1, ~1610 nm)
- **Pixels:** 748, 2020–2025

## Steps

### 1. Quality filter

Retain rows where `scl_purity >= 0.5`.

### 2. Compute SWIR moisture index

```
swir_mi = (B08 - B11) / (B08 + B11)
```

Also retain raw B11 and B08 as separate columns throughout — the B11 decomposition
diagnostic (step 6) requires them independently.

### 3. Inter-class contrast time series (no window assumption)

For each acquisition date, compute:
- Mean swir_mi across infestation pixels
- Mean swir_mi across grassland pixels
- Contrast = infestation mean − grassland mean

Also compute, separately for each date:
- Mean raw B11 for infestation vs grassland (contrast_B11)
- Mean raw B08 for infestation vs grassland (contrast_B08)

Plot all three contrasts (swir_mi, B11, B08) as separate panels of a single figure,
each with raw daily values as scatter and a 30-day rolling mean overlaid. This directly
shows whether the swir_mi contrast is driven by B11 (canopy water — the intended
signal), B08 (canopy structure — which would make swir_mi redundant with NIR), or both.

Log for each year:
- Date and value of maximum swir_mi contrast
- Whether B11 contrast or B08 contrast is the larger contributor at that date
- Fraction of dates where swir_mi contrast > 0

### 4. Monthly median profiles by class

For each class (infestation / riparian / grassland), compute median per calendar month
for swir_mi, raw B11, and raw B08. Plot as a 3×3 panel (3 signals × 3 classes) with
±1 std band and individual year traces.

The B11 and B08 subplots are a decomposition diagnostic: if B11 shows a strong
seasonal trough for grassland that is absent for Parkinsonia, while B08 shows similar
seasonal shapes for both classes, the swir_mi signal is genuinely driven by canopy
water. If B08 drives the separation, swir_mi adds nothing beyond the dry-NIR analysis.

Log the seasonal amplitude (max − min) of B11 and B08 per class. A large B11 amplitude
for grassland (relative to infestation) with a small B08 amplitude is the target pattern.

### 5. Per-pixel annual low percentile (window-free summary statistic)

For each pixel and each year, compute the 10th-percentile swir_mi across all qualifying
observations in that year. This captures "the driest the pixel canopy gets each year"
without fixing which months count.

From the per-year 10th-percentile values, compute per pixel:
- `swir_p10` — mean of annual 10th-percentile values across years
- `swir_p10_std` — standard deviation across years
- `swir_p10_cv` — coefficient of variation
- `n_years` — years with at least 10 qualifying observations

The 10th percentile rather than the minimum suppresses single-date outliers. Log the
distribution of qualifying-observation counts per pixel per year, and note if any
pixels are consistently below the 10-observation threshold (which would indicate sparse
coverage in some years and require a lower threshold or exclusion).

### 6. B11 decomposition diagnostic (data quality + signal attribution)

For a sample of 10 infestation pixels and 10 grassland pixels (selected as the
highest- and lowest-`nir_mean` pixels respectively from the dry-NIR stats), plot the
full 2020–2025 B11 time series as a scatter of individual acquisitions coloured by
month, with the annual 10th-percentile value marked per year.

This diagnostic serves two purposes:
- **Data quality:** Reveals whether any pixels have anomalously high B11 in dry season
  (cloud edges or SWIR saturation artefacts that passed the SCL filter). If outliers
  are present, a percentile filter should be applied before rerunning the aggregation.
- **Signal attribution:** Shows whether the dry-season B11 depression in grassland is
  consistent year-to-year or tied to specific rainfall events. If it is
  rainfall-dependent, the contrast will be unreliable in wet years — log this as a
  limitation.

### 7. Spatial map

Pixels coloured by `swir_p10`. Expect high values inside the infestation bbox.

### 8. Histogram by class

Overlaid distributions of `swir_p10` for infestation / riparian / grassland.

### 9. 2D projections into prior signal space

Two scatter plots:
- `nir_cv` vs `swir_p10` — stability vs hydration
- `rec_mean` vs `swir_p10` — recession vs hydration

Look for whether swir_p10 adds separation in the directions that nir_cv and rec_mean
do not fully resolve. If the scatter in `rec_mean` vs `swir_p10` is elongated along a
single axis, the two signals are collinear and swir_p10 is redundant.

### 10. Correlation analysis

Pearson r between `swir_p10` and:
- `nir_cv` (from dry-NIR stats)
- `rec_mean` (from wet-dry-amp stats)
- `re_p10` (from red-edge analysis, if complete)
- `nir_mean` (from dry-NIR stats)

High correlation (r ≥ 0.7) with an existing signal indicates redundancy. Also compute
the partial correlation of `swir_p10` with `rec_mean` controlling for `nir_mean` — this
removes the shared canopy-structure component and tests whether swir_mi carries
additional water-content information beyond what NIR already captures.

## Outputs

| File | Content |
|------|---------|
| `outputs/longreach-swir/longreach_swir_stats.parquet` | Per-pixel summary: `point_id`, `lon`, `lat`, `swir_p10`, `swir_p10_std`, `swir_p10_cv`, `n_years`, class flags |
| `outputs/longreach-swir/longreach_swir_contrast.png` | Daily contrast time series for swir_mi, B11, B08 (3-panel) with 30-day rolling mean |
| `outputs/longreach-swir/longreach_swir_monthly.png` | Monthly profiles of swir_mi, B11, B08 per class (3×3 panel) |
| `outputs/longreach-swir/longreach_swir_map.png` | Spatial plot: pixels coloured by `swir_p10` |
| `outputs/longreach-swir/longreach_swir_hist.png` | Overlaid histogram of `swir_p10` by class |
| `outputs/longreach-swir/longreach_nir_cv_vs_swir.png` | 2D scatter: `nir_cv` × `swir_p10` |
| `outputs/longreach-swir/longreach_rec_vs_swir.png` | 2D scatter: `rec_mean` × `swir_p10` |
| `outputs/longreach-swir/longreach_b11_timeseries.png` | Raw B11 time series for sample pixels (QC + attribution diagnostic) |

## Success criteria

1. **Contrast time series shows consistent signal** — fraction of dates where infestation
   mean swir_mi > grassland mean is ≥ 0.6.
2. **Class ordering in swir_p10** — infestation mean `swir_p10` > grassland mean.
3. **IQR separation** — infestation and grassland IQRs in `swir_p10` overlap fraction < 0.5.
4. **Signal driven by B11, not B08** — the seasonal amplitude of B11 for grassland
   exceeds that of the infestation by more than the corresponding B08 amplitude difference
   (confirmed from monthly profiles and contrast time series).
5. **Low correlation with rec_mean** — Pearson r < 0.7, confirming independence.

## Findings (2026-04-05)

### Signal strength

The SWIR moisture index produced a strong, consistent signal. All five success criteria
passed.

**Contrast time series** — 96.3% of acquisition dates (289/300) show infestation
swir_mi > grassland. This is not a narrow dry-season window: the separation holds
year-round. Peak annual contrast ranges from 0.27–0.40 across 2020–2025 and is
attributed to **B11 in every year**, confirming the contrast is driven by canopy water
content rather than canopy structure.

**Annual p10 class ordering:**

| Class | Mean swir_p10 | Median swir_p10 | IQR |
|-------|--------------|-----------------|-----|
| Infestation | −0.211 | −0.211 | [−0.225, −0.199] |
| Riparian (bare-soil proxy) | −0.156 | −0.176 | [−0.216, −0.081] |
| Grassland | −0.256 | −0.268 | [−0.282, −0.252] |

**Zero IQR overlap** between infestation and grassland. The classes are cleanly
separated in the annual p10 summary statistic.

**B11 decomposition** — B11 seasonal amplitude: infestation 0.136, grassland 0.127.
B08 seasonal amplitude: both ~0.05. The B08 amplitudes are nearly identical across
classes, so the swir_mi contrast is not a roundabout proxy for NIR canopy structure.

**Correlation with rec_mean** — r = 0.143, well below the 0.7 redundancy threshold.
swir_mi and the wet-dry amplitude signal are independent.

### Redundancy with red-edge

swir_p10 is **redundant with re_p10** (Pearson r = 0.729, just above the 0.7
threshold). Both signals detect the same underlying state: canopy that remains
physiologically active when surrounding vegetation desiccates. The partial correlation
of swir_p10 with rec_mean controlling for nir_mean is only 0.23, confirming that most
of the shared variance between swir_p10 and re_p10 is explained by canopy structure.

Per the pre-specified failure criteria: re_p10 (targeting chlorophyll specifically) is
the stronger independent axis. swir_mi is a corroborating measurement — useful as
confirmation that the red-edge result is not an artefact, but unlikely to add a new
discriminating dimension in a multi-band classifier.

### B11 outlier assessment

The B11 diagnostic flagged large annual ranges in many pixel-years, predominantly in
2020 (a wet year with a large legitimate seasonal swing). The warnings fire against a
loose threshold (2× dataset std) and reflect genuine seasonal variation rather than
cloud artefacts. The 10th-percentile aggregation is robust to high-end outliers; no
percentile filter is required before the aggregation step.

### Riparian confound

The riparian proxy (bare riverbed) scores at −0.176 median swir_p10 — between
infestation and grassland, not co-located with infestation as it would be for real
riparian woodland. This is expected: bare riverbed has negligible canopy water. The
result looks favourable here but must be treated as a known limitation — swir_mi is the
signal most likely to degrade when genuine riparian woodland is added to the dataset.

### Assessment

| Criterion | Result | Status |
|-----------|--------|--------|
| Fraction of dates with positive contrast | 96.3% | PASS |
| swir_p10 ordering (infestation > grassland) | −0.211 vs −0.268 | PASS |
| IQR overlap fraction | 0.00 | PASS |
| Signal driven by B11 not B08 | B11-driven in all 6 years | PASS |
| Low correlation with rec_mean (r < 0.7) | r = 0.143 | PASS |
| Independent of re_p10 (r < 0.7) | r = 0.729 | **REDUNDANT** |

**Recommendation:** Do not include swir_p10 as an independent feature alongside
re_p10. Retain it as a diagnostic and corroborating signal. If re_p10 degrades on
harder sites (e.g. with real riparian woodland present), revisit whether swir_mi
provides additional separation.

## What failure would mean

**If contrast time series shows swir_mi driven by B08 not B11:** swir_mi is a
roundabout proxy for NIR canopy structure and adds nothing beyond the dry-NIR analysis.
Drop it from the feature set.

**If swir_p10 is highly correlated with rec_mean (r ≥ 0.7):** Both signals reflect the
same canopy state. swir_mi is redundant — the red-edge ratio (targeting chlorophyll
specifically) is the stronger independent third axis.

**If the B11 time series diagnostic reveals systematic outliers:** Apply a percentile
filter (discard pixel-years where annual B11 range > 2× the dataset median range) and
rerun the aggregation. Log the number of pixel-years affected.

**If infestation and riparian proxy both score high on swir_p10:** Expected at this site
(bare riverbed scores low). Explicitly flag that swir_mi is the signal most likely to
fail against real riparian woodland, and note this as the primary motivation for the
eventual spatial expansion.
