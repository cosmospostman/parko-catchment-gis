# Recession Sensitivity and Greenup Timing Signals

## Why the current signals may not transfer

The five signals developed at Longreach all collapse the 2020–2025 time series into a
single per-pixel statistic — a mean, a floor percentile, or an inter-annual CV. That
averaging is appropriate when the goal is to characterise a pixel's typical state, and it
works well at Longreach because the contrast between classes is large: a dense Parkinsonia
patch on open floodplain vs. open grassland on gilgai clay.

The Longreach landscape has a specific structure that makes those statistics informative:

- The wet season brings heavy rainfall and occasional flooding, then the land dries
  completely. Grasses flush hard in the wet and senesce completely in the dry. The
  amplitude of that cycle is large, giving `rec_p` its discriminating power.
- The dry season is consistently dry. Inter-annual variation in dry-season greenness is
  driven almost entirely by root depth and canopy persistence. Parkinsonia's deep roots
  sustain canopy; grasses disappear. That contrast gives `nir_cv` its discriminating power.
- The absence class is grassland, whose spectral properties are strongly rainfall-dependent.
  Both `rec_p` and `nir_cv` measure properties that are high-contrast specifically because
  the negative class is shallow-rooted and rain-dependent.

On a more consistently watered riparian corridor — a perennial river or a system that
rarely fully dries — this contrast structure changes:

- The absence class may include deep-rooted native riparian trees (coolibah, eucalyptus)
  with permanent water access. These would also show low `nir_cv` (stable canopy) and
  low-amplitude NDVI (persistent greenness). Their values shift toward the Parkinsonia
  range.
- `rec_p` would compress for all vegetation classes — every riparian pixel stays green
  year-round, so the p90–p10 NDVI amplitude shrinks across the board. The signal that
  discriminated Parkinsonia from drying grassland becomes weak when nothing dries.
- `re_p10` and `swir_p10` are both floor statistics. On a perennial river, native
  riparian vegetation would also maintain elevated chlorophyll and canopy moisture
  floors — they would score similarly to Parkinsonia on both features.

The fundamental problem is that the current features measure **what a pixel looks like on
average**. They do not measure **how a pixel responds to changing water availability**. At
Longreach the absence class happens to respond strongly (grasses track rainfall), which
makes the average-state features discriminating. In a wetter landscape where the absence
class is also relatively stable, the average-state comparison loses power.

The two new signals proposed here target a different axis: **how sensitive is a pixel's
dry-season trajectory to the water it received in the preceding wet season?** This is a
within-pixel, year-to-year covariance rather than a cross-pixel comparison of means.

---

## Theory

### RecessionSensitivitySignal

**Core idea:** Fit a per-pixel per-year linear slope through NDVI during the dry-season
recession window (approximately April–September). This gives one slope value per
pixel-year. Across six years, examine how that slope varies as a function of how wet the
preceding wet season was.

**Wet-season moisture proxy:** Rather than using external rainfall data, use the pixel's
own peak NDWI (Normalised Difference Water Index, (B03 − B11) / (B03 + B11)) during the
wet season (December–March) of the same year. This is fully contained within the existing
parquet and is mechanistically more direct than station rainfall: it measures whether
*this pixel* got wet, not whether it rained at a gauge 10 km away. Flood-pulse
heterogeneity (different pixels at different elevations getting flooded in different years)
is captured automatically.

**Expected behaviour by class:**

| Class | Recession slope (mean) | Sensitivity to wet-season NDWI |
|---|---|---|
| Parkinsonia | Shallow (slow decline) | Low — deep roots sustain canopy regardless of wet season magnitude |
| Grassland | Steep (fast collapse) | High — good wet season → slower recession; poor wet season → faster crash |
| Perennial native riparian | Shallow (permanent water) | Near-zero — wet-season NDWI is consistently high every year; slope barely varies |

The distinguishing geometry:
- Parkinsonia vs. grassland: both slope (mean) and sensitivity differ
- Parkinsonia vs. perennial riparian: slope may be similar but **the mechanism differs** —
  Parkinsonia's shallow sensitivity comes from deep roots varying with annual conditions;
  perennial riparian's near-zero sensitivity comes from permanent water that never varies.
  These two cases may be distinguishable from the variance structure of the six per-year
  estimates even if their means are similar.

**Features produced per pixel:**

- `recession_slope` — mean dry-season NDVI slope across years (negative = declining;
  less negative = more persistent canopy)
- `recession_sensitivity` — Pearson correlation (or OLS slope) of per-year recession
  slope against per-year wet-season peak NDWI. Near zero = insensitive to wet season;
  strongly negative = steep recession in dry years, shallow in wet years.
- `recession_slope_cv` — coefficient of variation of per-year slopes across years.
  Captures inter-annual stability without needing to attribute it to moisture. Redundant
  with `recession_sensitivity` where moisture is the main driver, but informative where
  other factors (temperature, cloud cover) contribute.

**Minimum data requirement:** At least 3 years with sufficient observations in both the
wet-season window (for NDWI peak) and the recession window (for slope estimation). With
six years of S2 data at Longreach this is comfortable; flag pixels with fewer than 3
qualifying years as unreliable.

### GreenupTimingSignal

**Core idea:** For each pixel and year, find the day-of-year (DOY) at which NDVI peaks.
Compute the mean and inter-annual consistency of that peak DOY across years.

**Expected behaviour by class:**

| Class | Peak DOY (mean) | Peak DOY CV |
|---|---|---|
| Parkinsonia | Consistent — deep roots buffer timing from rainfall variability; likely peaks March–April based on existing red-edge and NDVI contrast findings | Low — same DOY ± a few weeks each year |
| Grassland | Tracks wet-season rainfall closely — peaks when the wet season is at its height, which shifts year to year | Higher CV — timing varies with monsoon onset |
| Perennial native riparian | Depends on species; may peak earlier or later, but permanent water access may also produce consistent timing | Unknown — to be measured |

**Relationship to existing signals:** The red-edge and NDVI contrast analyses already
show that March–April is consistently the highest-contrast window between Parkinsonia and
grassland across all six Longreach years. `GreenupTimingSignal` formalises this as an
explicit per-pixel DOY feature rather than a population-level observation.

**Features produced per pixel:**

- `peak_doy` — mean DOY of annual NDVI peak across years
- `peak_doy_cv` — coefficient of variation of peak DOY across years (lower = more
  consistent timing)
- `greenup_rate` — mean slope of NDVI on the rising limb (from seasonal trough to peak).
  This is the least reliable feature given sparse wet-season acquisitions; compute but
  flag as lower confidence.

**Caveat on cloud cover:** The wet season (December–February) is the cloud-heavy period
at Longreach, with ~3 usable acquisitions/month vs ~6/month in the dry. Peak DOY
estimation may be noisy for years where cloud cover obscures the actual peak. Mitigate
by: (1) using a smoothed NDVI curve (rolling median over a 30-day window) rather than
the raw maximum; (2) flagging years with fewer than 5 clean wet-season observations as
unreliable for that year's peak estimate; (3) checking that peak DOY estimates are
spatially coherent across adjacent pixels (high coherence → reliable estimate).

**Shared computation with RecessionSensitivitySignal:** Both signals require a smoothed
per-pixel per-year NDVI time series. This should be implemented as a shared private
kernel in `signals/_shared.py` (e.g. `annual_ndvi_curve`) rather than duplicated in each
class.

---

## Implementation plan

### Shared kernel: `annual_ndvi_curve`

Add to `signals/_shared.py`:

```
annual_ndvi_curve(df, min_obs_per_year, smooth_days) -> pl.DataFrame
```

Input: Polars observation-level DataFrame with `point_id`, `date`, `B08`, `B04`.
Output: smoothed per-pixel per-date NDVI (rolling median over `smooth_days`), with
`year`, `doy`, `month` columns added. Filtering sparse years by `min_obs_per_year`.

Smoothing approach: rolling median over a configurable window (default 30 days). Polars
`rolling_median` over sorted dates within each pixel group. This is more robust to
residual cloud outliers than a Gaussian kernel and requires no scipy dependency.

### `RecessionSensitivitySignal` — `signals/recession.py`

**Params dataclass:**
- `quality: QualityParams`
- `recession_start_month: int = 4` (April)
- `recession_end_month: int = 9` (September)
- `wet_start_month: int = 12` (December)
- `wet_end_month: int = 3` (March, wraps year boundary)
- `smooth_days: int = 30`
- `min_years: int = 3`

**`compute()` steps:**
1. Load and filter via `load_and_filter`.
2. Compute NDVI and NDWI per observation.
3. For each pixel-year:
   a. Recession slope: OLS slope of smoothed NDVI against DOY within the recession
      window. Require at least 5 observations in the window; otherwise mark as NaN
      for that year.
   b. Wet-season NDWI peak: maximum smoothed NDWI value within the wet window
      (handling December of year Y−1 through March of year Y as the same wet season).
4. Per pixel across years:
   a. `recession_slope` = mean of per-year slopes
   b. `recession_slope_cv` = CV of per-year slopes
   c. `recession_sensitivity` = Pearson r between per-year slope and per-year NDWI
      peak (requires ≥ `min_years` valid year pairs)
5. Return DataFrame: `[point_id, lon, lat, recession_slope, recession_slope_cv, recession_sensitivity, n_years]`.

**`diagnose()` figures:**
- Spatial map of `recession_sensitivity` coloured on a diverging scale (negative =
  sensitive to moisture, near-zero = insensitive)
- Distribution plot of all three features split by presence/absence class
- Scatter: per-year recession slope vs. per-year NDWI peak for a sample of presence
  and absence pixels overlaid (shows the different sensitivity gradients directly)

### `GreenupTimingSignal` — `signals/greenup.py`

**Params dataclass:**
- `quality: QualityParams`
- `search_start_month: int = 11` (November — allow early green-up)
- `search_end_month: int = 5` (May — allow late peak)
- `smooth_days: int = 30`
- `min_wet_obs: int = 5`
- `min_years: int = 3`

**`compute()` steps:**
1. Load and filter via `load_and_filter`.
2. Compute NDVI per observation.
3. For each pixel-year, within the search window:
   a. Apply smoothing kernel.
   b. Find DOY of maximum smoothed NDVI value.
   c. Flag year as unreliable if fewer than `min_wet_obs` clean observations in the
      search window.
   d. Compute greenup rate: OLS slope of smoothed NDVI from the seasonal trough
      (minimum NDVI in the 60 days before the peak) to the peak.
4. Per pixel across reliable years:
   a. `peak_doy` = mean peak DOY
   b. `peak_doy_cv` = CV of peak DOY
   c. `greenup_rate` = mean greenup rate (flag as lower confidence)
5. Return DataFrame: `[point_id, lon, lat, peak_doy, peak_doy_cv, greenup_rate, n_years, n_reliable_years]`.

**`diagnose()` figures:**
- Spatial map of `peak_doy` — do Parkinsonia pixels peak at a consistent and distinct DOY?
- Distribution plot of `peak_doy` and `peak_doy_cv` split by presence/absence
- Per-year dot plot: peak DOY per class per year (shows inter-annual consistency and
  whether the classes separate in time)

### `signals/__init__.py` updates

Add `RecessionSensitivitySignal` and `GreenupTimingSignal` to imports and `__all__`.
Do not add to `extract_parko_features` yet — these signals are experimental and require
validation before inclusion in the standard feature pipeline.

---

## Exploration and debugging plan at Longreach

The goal of this plan is not to confirm or reject the signals with a binary verdict, but
to understand what each signal is actually measuring, where it agrees with theory and
where it diverges, and what that tells us about how to tune or reframe it. Each step
produces diagnostic outputs that either confirm a theoretical assumption or reveal what
needs to change.

### Class labels

Rather than using only the sub-bbox training labels (which cover ~374 infestation and
~374 extension pixels), use the full 8×8 km scene scores from the existing logistic
regression classifier. Take:

- **Presence class:** pixels with `prob_lr` in the top 10% of the scene (~820 pixels)
- **Absence class:** pixels with `prob_lr` in the bottom 10% of the scene (~820 pixels)
- **Exclude** the middle 80% from class-labelled analysis — these are mixed or uncertain
  pixels. They are not discarded: several figures below use the full continuous `prob_lr`
  score rather than binary labels, which keeps all pixels in play and avoids a hard
  threshold artefact.

### Stage 1 — Understand the raw material: what does the NDVI time series look like?

Before fitting any curves or computing any features, examine the raw per-pixel NDVI
observations to test whether the theoretical picture (smooth annual wave, readable peak
and recession) matches reality at this site.

**Figure: mean NDVI time series by class**
Plot the scene-mean daily NDVI for each of the three classes (presence, absence, riparian
proxy) across 2020–2025. Show individual years in light lines, the 6-year mean in bold.
This is the most direct test of whether the theoretical waveform shape is present in the
data at all. Expected: presence pixels show a distinct post-wet plateau with a slow
recession into July–September; absence pixels show a sharp peak followed by a steep crash.
If both classes track each other closely, the entire approach needs rethinking before
implementation.

**Figure: observation density calendar**
For each year, plot the number of clean (SCL-filtered) acquisitions per 14-day DOY bin,
stacked for presence vs. absence pixels. This makes cloud-season data gaps explicit.
The wet season (Dec–Feb) cloud gap is the main risk for peak DOY estimation; this figure
shows exactly which years and DOY bins are sparse. Use it to decide whether the default
`min_wet_obs = 5` is achievable or needs relaxing.

**Figure: smoothing sensitivity — raw vs. smoothed NDVI for a sample of pixels**
Pick three presence pixels (high/medium/low `prob_lr` within the top decile) and three
absence pixels. Plot their raw NDVI observations as dots and the rolling-median smoothed
curve at three window widths (15, 30, 45 days). This answers the question: does the
smoothing window meaningfully change the shape of the curve, or is the signal robust to
window choice? If the smoothed peak DOY shifts by more than 14 days across window widths
for most pixels, the parameter is load-bearing and needs a principled choice rather than
a default.

### Stage 2 — Characterise the wet-season moisture proxy

The recession sensitivity signal rests on the claim that the per-year wet-season NDWI
peak varies enough across 2020–2025 to estimate sensitivity. This needs to be verified
before fitting any sensitivities.

**Figure: per-year scene-mean wet-season NDWI distribution**
Boxplot of peak wet-season NDWI per pixel, one box per year, for the full scene. If all
six years look similar (tight overlapping distributions), the sensitivity signal will be
unreliable because there is insufficient inter-annual variation to regress against. If
the years span a wide range (e.g. 2023 La Niña vs. 2019 drought conditions), the
sensitivity estimate will be meaningful.

**Figure: NDWI peak maps per year**
Six small spatial maps of the per-year wet-season NDWI peak across the 8×8 km scene.
Two things to check: (1) do the same pixels rank high/low each year (structural moisture
differences between pixels) or does the spatial pattern change year to year (flood
heterogeneity)? (2) Does the Thomson River corridor show elevated NDWI in most years,
consistent with being flood-pulse-fed? Structural differences between pixels will help
recession sensitivity; purely year-to-year variation will too — both are useful, but for
different reasons.

**Quantify NDWI inter-annual range per pixel**
Compute for each pixel: `ndwi_peak_range = max(annual NDWI peak) - min(annual NDWI peak)`
across years. Plot as a histogram and a spatial map. Pixels with low range (< 0.05) will
produce unreliable sensitivity estimates regardless of their class. Flag these during
signal computation. If a large fraction of pixels have low range, the sensitivity feature
needs a fallback (e.g. report `recession_slope` only, mark `recession_sensitivity` as
NaN).

### Stage 3 — Examine per-year recession slopes before averaging

The `recession_slope` feature is a mean across years, but what matters is whether the
*pattern* of slopes across years is consistent with the theory. Examine the per-year
estimates before collapsing them.

**Figure: per-year recession slope distributions by class**
Strip plot or violin, one panel per year (2020–2025), presence vs. absence side by side.
Theory predicts presence pixels are consistently less negative (shallower recession) in
every year, not just on average. If the class separation is strong in some years and
absent in others, that year-level information is itself diagnostic — it may reveal which
wet seasons produced the clearest signal and whether that tracks NDWI.

**Figure: recession slope vs. NDWI peak, per pixel coloured by `prob_lr`**
Scatter plot with wet-season NDWI peak on the x-axis and recession slope on the y-axis,
one point per pixel-year (so each pixel contributes six points). Colour by `prob_lr`.
This is the direct visualisation of the sensitivity hypothesis: if it holds, high-`prob_lr`
pixels should form a near-horizontal cloud (slope barely changes with NDWI), while
low-`prob_lr` pixels should form a negative slope cloud (steep recession when NDWI is
low, shallow when high). If both classes form the same cloud, sensitivity is not
discriminating. If the slopes are different in level but equally sensitive, then
`recession_slope` (mean) is the useful feature and `recession_sensitivity` is not adding
information.

**Figure: per-pixel recession sensitivity vs. `prob_lr`**
Scatter of the per-pixel Pearson r (recession slope vs. NDWI peak across 6 years) against
`prob_lr`. This collapses the per-year scatter into a single-pixel summary. The
theoretical prediction is a positive correlation: high `prob_lr` → sensitivity near zero;
low `prob_lr` → sensitivity strongly negative. The shape of this scatter reveals whether
the relationship is linear (good for a logistic regression feature), monotone but
non-linear, or noisy with a weak trend.

**Figure: spatial map of `recession_sensitivity`**
Map the per-pixel sensitivity across the 8×8 km scene. Expect the Thomson River corridor
to show a band of near-zero sensitivity (deep roots or perennial water) surrounded by
more negative values in grassland. If the map is spatially incoherent (salt-and-pepper),
the per-pixel sensitivity estimate has too much noise from six data points — the signal
may need pooling across spatial neighbours or a longer time series to be useful.

### Stage 4 — Examine per-year peak DOY estimates before averaging

**Figure: per-year peak DOY by class**
Strip plot of peak DOY per pixel, one panel per year, presence vs. absence. The key
question is whether the classes are consistently separated in DOY across years, or only
in some years. A result where the classes separate in 5 of 6 years with one anomalous
year is informative — identify the anomalous year and check whether it had unusual cloud
cover (observation density figure from Stage 1) or an anomalous wet season onset.

**Figure: example smoothed NDVI curves with identified peaks annotated**
For 6 pixels (3 presence, 3 absence, stratified by `prob_lr`), plot the smoothed NDVI
time series for all six years on one set of axes per pixel, with the identified peak DOY
marked as a vertical line per year. This is the most direct check that the peak-finding
algorithm is behaving sensibly — are the identified peaks plausible given the curve shape,
or are they picking noise spikes? If cloud-season gaps cause the smoothed curve to have
false peaks, they will be visible here.

**Figure: peak DOY uncertainty — distribution of per-pixel standard deviation across years**
Histogram of the per-pixel standard deviation of peak DOY across years, split by class.
This directly characterises how noisy the per-year peak estimates are. If the median SD
is > 30 days, the peak is not well-located and averaging across years produces a mean
with high uncertainty. If SD is < 14 days for most presence pixels, the timing is
genuinely consistent and the feature is reliable. Compare the SD distribution between
classes: theory predicts lower SD for presence (consistent deep-root buffering) than
absence (rainfall-driven variable timing).

**Figure: spatial map of `peak_doy`**
Map mean peak DOY across the scene. Expect spatial coherence along the river corridor.
If the map shows coherent spatial structure (the corridor is distinct from surrounding
grassland), the feature is capturing real spatial variation. If it is spatially random,
the per-pixel estimates are dominated by noise or the DOY difference between classes is
smaller than the estimation error.

### Stage 5 — Diagnose the riparian proxy case

The 39 riparian proxy pixels (top-10% NIR mean in the extension) are the best available
proxy for perennial water access. The theoretical prediction is that these pixels should
show near-zero `recession_sensitivity` (consistent moisture every year → stable slope
year to year) and a potentially distinct `peak_doy` from both Parkinsonia and grassland.

**Figure: recession slope vs. NDWI peak for riparian proxy pixels**
Overlay the 39 riparian proxy pixels on the scatter from Stage 3 (slope vs. NDWI peak,
coloured by class). Do they form a horizontal cloud (near-zero sensitivity, consistent
with permanent water) or do they follow the grassland pattern (sensitive)? If they show
near-zero sensitivity AND a similar recession slope level to Parkinsonia, the two cannot
be separated by this signal alone — but then examine whether `peak_doy` differs.

**Figure: riparian proxy NDVI time series**
Plot the mean daily NDVI for the 39 riparian proxy pixels alongside the presence and
absence class means, for 2020–2025. Is their waveform shape distinct from both? Do they
stay green through the dry season (similar to Parkinsonia) or do they show a flood-pulse
(green spike during inundation, drying out quickly)? This directly tests what ecological
class these pixels actually represent — which is not confirmed as "perennial riparian" but
is inferred from their high dry-season NIR.

### Stage 6 — Feature correlation with existing signals

**Table: Pearson r between new features and existing features**
Compute `(recession_slope, recession_slope_cv, recession_sensitivity, peak_doy, peak_doy_cv)`
vs. `(nir_cv, rec_p, re_p10, swir_p10)` across all pixels in the scene. Report as a
correlation matrix. Features with r > 0.8 against an existing feature are likely redundant
at this site; features with r < 0.4 are genuinely new information. The correlation
direction also matters: a strong negative correlation with `nir_cv` would confirm that
`recession_slope` is capturing the same deep-root persistence axis through a different
computational route.

**Figure: 2D scatter of new feature vs. most correlated existing feature, coloured by `prob_lr`**
For each new feature, plot it against its most correlated existing counterpart. If the
scatter is tight around a diagonal, the features are redundant. If there is structure in
the residuals (e.g. a cluster of pixels that `nir_cv` places incorrectly but
`recession_sensitivity` separates), that residual structure is the added value.

### Stage 7 — Parameter sensitivity

The signals have several free parameters whose defaults are choices, not truths. Before
locking them down, run a lightweight sweep.

**Recession window boundaries (`recession_start_month`, `recession_end_month`)**
Recompute `recession_slope` with start month ∈ {3, 4, 5} and end month ∈ {8, 9, 10}.
For each combination, report the median absolute difference in `recession_slope` between
presence and absence classes. Pick the window that maximises class separation. If the
optimal window is not April–September (the theoretical prior), note why — it may reveal
that the Longreach Parkinsonia recession starts earlier or ends later than expected.

**Smoothing window (`smooth_days`)**
Recompute peak DOY with smooth_days ∈ {15, 21, 30, 45}. Report per-pixel SD of peak DOY
across years for each window (from the Stage 4 figure). The best window minimises
within-class SD (more consistent estimates) while not over-smoothing the peak into a
plateau where DOY becomes indeterminate. If SD is similar across window choices, the
signal is robust; if it varies strongly, the choice is load-bearing.

**Minimum wet-season observations (`min_wet_obs`)**
Report the fraction of pixel-years that are flagged as unreliable at min_wet_obs ∈ {3, 5, 8}.
If raising the threshold from 5 to 8 drops 30% of pixels, the threshold is too strict for
this dataset's cloud characteristics. If it drops only 5%, raising it is safe and improves
estimate quality.

### What the diagnostics tell us

Each stage is designed to expose a specific assumption:

| Stage | Assumption being tested | What to change if it fails |
|---|---|---|
| 1 — raw time series | NDVI waveform is readable; theory matches data shape | Reconsider which index to use (NDWI, EVI); check SCL filter strictness |
| 2 — NDWI proxy | Sufficient inter-annual moisture variation exists | Fall back to recession_slope only; flag recession_sensitivity as site-limited |
| 3 — recession slopes | Class separation exists per year, not just on average | Widen/shift recession window; consider non-linear slope (breakpoint model) |
| 4 — peak DOY | Peak is reliably locatable given observation density | Relax min_wet_obs; try a different peak definition (e.g. centroid of top quartile DOYs) |
| 5 — riparian proxy | Riparian pixels have distinct water-access signature | Accept that Longreach cannot test this; defer to a perennial-river site |
| 6 — feature correlation | New features add information beyond existing signals | Retain only as backup for sites where existing signals weaken |
| 7 — parameter sensitivity | Default parameters are not strongly load-bearing | Tune per-site via YAML overrides (same pattern as `floor_percentile` for RedEdge) |
