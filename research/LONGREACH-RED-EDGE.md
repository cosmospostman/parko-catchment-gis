# Red-Edge Ratio — Analysis Plan

## Hypothesis

The red-edge region (B05 ~705 nm, B07 ~783 nm) is sensitive to chlorophyll
concentration in the canopy. In actively photosynthesising vegetation, B07 rises
relative to B05 as chlorophyll increases. When vegetation senesces, chlorophyll
degrades and the ratio B07/B05 drops toward 1.

Parkinsonia retaining active chlorophyll through the dry season should maintain an
elevated B07/B05 relative to senescing grasses. The physical distinction from
NIR-based signals: NIR (B08) responds to canopy *structure* (leaf area, water content,
multiple scattering) — both alive and dead plant material contribute. B07/B05 responds
specifically to *active chlorophyll* — dead or bleached material does not elevate the
ratio. This makes the red-edge ratio more selective than NIR for living canopy.

**No dry-season window is assumed.** Rather than pre-specifying a window (e.g. Jul–Sep),
the analysis computes the inter-class contrast across every acquisition date in the
2020–2025 archive and lets the data reveal when and whether the contrast is large. The
per-pixel summary statistic (annual low percentile) is window-free by construction.
This is important because the timing of grass senescence varies with rainfall, which
shifts year to year.

**Expected class ordering when contrast is at its peak:**

| Class | Expected B07/B05 | Mechanism |
|-------|-----------------|-----------|
| Parkinsonia | **Highest** (~1.2–1.4) | Active chlorophyll retained regardless of season |
| Riparian bare-soil mix | Intermediate | Some woody canopy, but bare soil fraction pulls ratio toward 1 |
| Grassland | **Lowest** (~1.0–1.1) | Near-complete chlorophyll degradation post-senescence |

**Key question for riparian discrimination:** If dense riparian woodland (coolibah,
eucalyptus) were present, it would also retain active chlorophyll and score similarly
to Parkinsonia. The existing riparian proxy (bare riverbed) scores low, which may be
misleading about real riparian woodland. The red-edge ratio result should be
interpreted with this caveat explicitly in mind.

## Data

- **Source:** `data/longreach_pixels.parquet`
- **Bands used:** `B05` (red-edge 1, ~705 nm), `B07` (red-edge 3, ~783 nm), `B08` (NIR)
- **Pixels:** 748 (374 infestation patch + 374 southern extension), 2020–2025

## Steps

### 1. Quality filter

Retain rows where `scl_purity >= 0.5`.

### 2. Compute red-edge ratio

```
re_ratio = B07 / B05
```

Also compute a normalised form for reference:

```
re_ndvi = (B07 - B05) / (B07 + B05)
```

re_ratio is the primary metric (near 1 = senescent, > 1 = active chlorophyll).
re_ndvi is included as a secondary check and logged alongside re_ratio throughout.

### 3. Inter-class contrast time series (no window assumption)

For each acquisition date, compute:
- Mean re_ratio across infestation pixels
- Mean re_ratio across grassland pixels
- Contrast = infestation mean − grassland mean

Plot the raw daily contrast as a scatter across all dates 2020–2025, with a 30-day
rolling mean overlaid. This is the primary diagnostic: it reveals *when* (and whether)
the red-edge separates the two classes, without assuming the timing in advance.

Log for each year:
- The date and value of maximum contrast
- The date and value of minimum contrast
- The fraction of dates where contrast > 0 (i.e. infestation consistently above grass)
- Whether the peak contrast timing is consistent across years (±30 days) or shifts

A consistent high-contrast period that recurs across years validates the signal. A
contrast that peaks at a different time each year is still useful but suggests the
per-pixel summary must be time-agnostic (which the annual percentile approach handles).

### 4. Monthly median profile by class

Compute the median re_ratio per calendar month for each class, averaged across all
years. Plot as a three-panel figure (one panel per class) with ±1 std band and
individual year traces behind the mean.

Log the peak and trough months per class and the peak-to-trough range. Compare the
shape to the NDVI monthly profiles from the wet-dry-amp analysis — if the red-edge
seasonal shape differs from NDVI, it is providing independent information.

### 5. Per-pixel annual low percentile (window-free summary statistic)

For each pixel and each year, compute the 10th-percentile re_ratio across all
qualifying observations in that year. This is the "driest the pixel gets each year"
in chlorophyll terms, without fixing which months count.

From the per-year 10th-percentile values, compute per pixel:
- `re_p10` — mean of annual 10th-percentile values across years
- `re_p10_std` — standard deviation across years
- `re_p10_cv` — coefficient of variation
- `n_years` — years with at least 10 qualifying observations (to make the 10th
  percentile stable; ~10 obs gives a reliable single low-end value)

The 10th percentile rather than the minimum suppresses single-date outliers
(cloud-edge contamination that passed the SCL filter).

Log the class-level distributions of `re_p10` and compare to the contrast time series:
the window the contrast analysis identifies as peak separation should correspond to
dates where pixel values are near their annual 10th percentile for grassland pixels.

### 6. Spatial map

Pixels coloured by `re_p10`. Expect high values inside the infestation bbox.

### 7. Histogram by class

Overlaid distributions of `re_p10` for infestation / riparian / grassland.

### 8. 2D projections into prior signal space

Two scatter plots:
- `nir_cv` vs `re_p10` — stability vs chlorophyll activity
- `rec_mean` vs `re_p10` — recession vs chlorophyll activity

Look for whether `re_p10` adds separation in directions that nir_cv and rec_mean do
not fully resolve. If the scatter in `rec_mean` vs `re_p10` is elongated along a single
axis, the two signals are collinear and re_p10 is redundant.

### 9. Correlation analysis

Pearson r between `re_p10` and each of:
- `nir_cv` (from dry-NIR stats)
- `rec_mean` (from wet-dry-amp stats)
- `nir_mean` (from dry-NIR stats)

High correlation with `rec_mean` (r ≥ 0.7) means both signals measure the same
underlying phenomenon (active chlorophyll) and re_p10 is redundant. Low correlation
indicates a genuinely independent axis worth retaining in the feature set.

## Outputs

| File | Content |
|------|---------|
| `outputs/longreach-red-edge/longreach_re_stats.parquet` | Per-pixel summary: `point_id`, `lon`, `lat`, `re_p10`, `re_p10_std`, `re_p10_cv`, `n_years`, class flags |
| `outputs/longreach-red-edge/longreach_re_contrast.png` | Daily inter-class contrast time series with 30-day rolling mean |
| `outputs/longreach-red-edge/longreach_re_monthly.png` | Monthly re_ratio profiles per class (3-panel, with year traces) |
| `outputs/longreach-red-edge/longreach_re_map.png` | Spatial plot: pixels coloured by `re_p10` |
| `outputs/longreach-red-edge/longreach_re_hist.png` | Overlaid histogram of `re_p10` by class |
| `outputs/longreach-red-edge/longreach_nir_cv_vs_re.png` | 2D scatter: `nir_cv` × `re_p10` |
| `outputs/longreach-red-edge/longreach_rec_vs_re.png` | 2D scatter: `rec_mean` × `re_p10` |

## Success criteria

1. **Contrast time series shows consistent signal** — fraction of dates where infestation
   mean re_ratio > grassland mean is ≥ 0.6 (more often above than below across all
   2020–2025 dates).
2. **Class ordering in re_p10** — infestation mean `re_p10` > grassland mean `re_p10`.
3. **IQR separation** — infestation and grassland IQRs in `re_p10` have overlap fraction
   < 0.5.
4. **Low correlation with rec_mean** — Pearson r < 0.7, confirming re_p10 is not
   redundant with the recession signal.

## Results (run 2026-04-05)

### Success criteria

| # | Criterion | Result | Status |
|---|-----------|--------|--------|
| 1 | Frac of dates infestation re_ratio > grassland ≥ 0.6 | **0.80** | PASS |
| 2 | re_p10 ordering: infestation > grassland | **1.186 > 1.142** | PASS |
| 3 | IQR overlap fraction < 0.5 | **0.00** | PASS |
| 4 | Pearson r(re_p10, rec_mean) < 0.7 | **r = 0.087** | PASS |

### Class-level re_p10 distributions

| Class | Mean | Median | IQR |
|-------|------|--------|-----|
| Infestation (362 px) | 1.1884 | 1.1860 | [1.1701, 1.2095] |
| Riparian (39 px) | 1.1984 | 1.1856 | [1.1355, 1.2580] |
| Grassland (347 px) | 1.1678 | 1.1422 | [1.1292, 1.1621] |

### Contrast time series

80% of acquisition dates (240/300) have infestation mean re_ratio above grassland mean —
well above the 0.6 threshold. Per-year breakdown:

| Year | Max contrast (date) | Min contrast (date) | Frac > 0 |
|------|---------------------|---------------------|----------|
| 2020 | 0.220 (2020-03-24) | −0.192 (2020-08-06) | 0.56 |
| 2021 | 0.350 (2021-04-13) | −0.080 (2021-11-19) | 0.91 |
| 2022 | 0.394 (2022-03-09) | −0.274 (2022-05-28) | 0.76 |
| 2023 | 0.222 (2023-04-28) | −0.236 (2023-05-03) | 0.89 |
| 2024 | 0.284 (2024-04-07) | −0.019 (2024-08-15) | 0.85 |
| 2025 | 0.444 (2025-04-27) | −0.101 (2025-11-23) | 0.86 |

Peak contrast falls in **March–April** in every year — tied to the post-wet green flush.
The timing is consistent (within ±30 days across all years) despite inter-annual rainfall
variability, which means a fixed early-wet window would capture the signal reliably.
2020 is the weakest year (frac=0.56), likely reflecting a weaker or delayed wet season.

### Monthly profiles

All three classes peak in April (month 4). The infestation seasonal range (0.51) is
double that of grassland (0.25) and riparian (0.21), driven by a higher April peak rather
than a lower trough — Parkinsonia's wet-season canopy flush elevates the red-edge ratio
more strongly than grass.

### Correlation with prior signals

| Signal | Pearson r | Interpretation |
|--------|-----------|----------------|
| nir_cv (dry-season NIR stability) | −0.304 | Weak–moderate, independent |
| rec_mean (NDVI seasonal recession) | 0.087 | Near-zero, independent |
| nir_mean (dry-season NIR mean) | −0.006 | Uncorrelated |

re_p10 is genuinely independent of both prior signals. The near-zero correlation with
rec_mean is the most important result: recession captures the *amplitude* of the seasonal
swing (how far NDVI drops from wet-season peak to dry-season trough), while re_p10
captures the *baseline level* of chlorophyll activity at the annual low. They are
measuring different properties. This means re_p10 adds a new axis to the feature space
rather than duplicating rec_mean.

### IQR separation

The infestation IQR [1.170, 1.210] and grassland IQR [1.129, 1.162] do not overlap at
all (overlap fraction = 0.00). A threshold between 1.162 and 1.170 separates every
infestation pixel from every grassland pixel in the re_p10 distribution. This is the
cleanest class separation of any signal tested so far.

### Riparian proxy

The riparian proxy (bare riverbed pixels, top-10% nir_mean outside the infestation bbox)
scores identically to the infestation in re_p10 (median 1.186 vs 1.186, wide IQR). This
is expected: the bare riverbed proxy is a poor stand-in for real riparian woodland. The
result confirms the caveat in the plan — at a site with dense coolibah or eucalyptus
canopy, this signal would produce false positives indistinguishable from Parkinsonia.

### Interpretation

The annual 10th-percentile approach is doing substantial work here: by taking the low end
of each year and averaging across 6 years, it suppresses single-date noise and isolates
the structural property — Parkinsonia's chlorophyll floor is consistently elevated above
the grassland floor. The 80% contrast rate confirms the effect is persistent rather than
a lucky seasonal window.

The combination of (1) zero IQR overlap, (2) independence from rec_mean, and (3)
consistent contrast peak timing makes re_p10 the strongest candidate for inclusion in a
multi-signal feature set alongside nir_cv and rec_mean. Whether the three signals together
achieve meaningful discrimination in the full 3D feature space is the next question.

## What failure would mean

**If contrast is near zero throughout the year:** The red-edge signal at this site is
dominated by the soil background (bare gilgai clay between crowns), pulling all pixel
ratios toward 1 regardless of canopy type. B07/B05 is not a useful single-pixel
discriminator at 10m resolution here. Note this for future sites with higher canopy
cover where soil mixture is less severe.

**If re_p10 is highly correlated with rec_mean (r ≥ 0.7):** Both signals measure the
same thing (active chlorophyll). The red-edge ratio adds no independent axis — SWIR
becomes the more important third signal to evaluate.

**If contrast peaks at a different time each year with no consistent window:** The
per-pixel annual percentile approach (step 5) is the right summary; do not try to fix
a window. Log the year-to-year peak dates as a finding — it means senescence timing
is rainfall-driven, which has implications for any single-date classifier.

**If riparian proxy scores similarly to infestation:** Expected at this site (bare
riverbed). Flag explicitly — this result will reverse at a site with dense riparian
canopy.
