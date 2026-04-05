# Dry-Season NIR Stability — Analysis Plan

## Hypothesis

Parkinsonia-dominated pixels have a persistent woody canopy year-round. In dry season
(June–October), surrounding grasses senesce and bare gilgai clay is exposed, causing NIR
reflectance to drop. Parkinsonia canopy sustains elevated NIR. Over five dry seasons
(2020–2024), high-canopy pixels should show both higher mean NIR and lower inter-annual
variability (CV) than low-canopy pixels.

## Data

- **Source:** `data/longreach_pixels.parquet`
- **Schema:** `point_id`, `lon`, `lat`, `date`, bands `B02…B12`, `scl_purity`, `aot`, `view_zenith`, `sun_zenith`
- **Pixels:** 374 (11 × 34 grid), ~387 observations each, 2020–2025

## Steps

### 1. Quality filter

Retain rows where `scl_purity >= 0.5` (at least half of the SCL contributing pixels are
vegetation/bare soil — not cloud or shadow). This is the existing quality column in the
dataset; no additional band-math needed.

### 2. Dry-season subset

Filter to months 6–10 (June–October inclusive). This is the S2-dense window (~6
acquisitions/month) and the period where grass senescence creates contrast.

### 3. Per-pixel, per-year dry-season B08 median

Group by `(point_id, year)` → `median(B08)`. Using median rather than mean to suppress
any residual cloud-contaminated outliers that passed the quality filter.

### 4. Per-pixel summary statistics (across years)

From the per-year medians, compute per pixel:
- `nir_mean` — mean of annual dry-season medians
- `nir_std` — standard deviation across years
- `nir_cv` — coefficient of variation (std / mean); lower = more stable year-to-year
- `n_years` — number of years with ≥ 5 qualifying observations (to flag sparse pixels)

### 5. Spatial plot

Scatter the 374 pixels on a lat/lon grid coloured by `nir_mean` and separately by
`nir_cv`. Expect:
- High `nir_mean` + low `nir_cv` → high-canopy-fraction (Parkinsonia) pixels
- Low `nir_mean` + high `nir_cv` → bare clay / grass pixels

Overlay the quicklook image outline from LONGREACH.md for visual validation.

### 6. Distribution check

Histogram of `nir_cv` across all 374 pixels. If the signal is real, expect a bimodal or
right-skewed distribution — a cluster near zero (stable canopy) and a tail of more
variable pixels.

### 7. Rank-order spot check

Sort pixels by `nir_mean` descending. Inspect the top and bottom 20 pixel coordinates
against the confirmed dense-infestation bbox (lon [145.4240, 145.4250], lat [-22.7640,
-22.7610]) to verify that the highest-NIR pixels fall where the canopy is densest in
the Queensland Globe imagery.

## Outputs

| File | Content |
|------|---------|
| `outputs/longreach-dry-nir/longreach_dry_nir_stats.parquet` | Per-pixel summary: `point_id`, `lon`, `lat`, `nir_mean`, `nir_std`, `nir_cv`, `n_years` |
| `outputs/longreach-dry-nir/longreach_dry_nir_map.png` | Spatial plot: pixels coloured by `nir_mean` |
| `outputs/longreach-dry-nir/longreach_dry_nir_cv_map.png` | Spatial plot: pixels coloured by `nir_cv` |
| `outputs/longreach-dry-nir/longreach_dry_nir_cv_hist.png` | Histogram of `nir_cv` |

## Success criteria

1. Pixels within the high-density bbox (LONGREACH.md §High-density scene) rank in the
   top quartile of `nir_mean`.
2. Those same pixels show `nir_cv` below the dataset median — confirming stability, not
   just elevation.
3. The spatial pattern in the map is coherent (not noise) — adjacent pixels should agree.

## What failure would mean

If `nir_cv` is uniformly low across all pixels, the signal exists but doesn't
discriminate (all pixels look stable — canopy fraction gradient is too small to detect
this way). Next step would be wet/dry amplitude (plan: LONGREACH-WET-DRY-AMP.md).

If `nir_mean` shows no spatial structure, the bbox may be too homogeneous and we need
external negative samples (bare/grass pixels outside the infestation).

---

## Results (run 2026-04-05)

**Dataset:** 748 pixels (374 infestation patch + 374 southern extension), 289,022 rows,
2020–2025. The southern extension (342 m, +34 pixel columns) was added after the initial
run on the infestation patch alone showed insufficient contrast — all 374 original pixels
are Parkinsonia-dominated, producing a unimodal CV distribution with no separable
negative population.

**Script:** `longreach/dry-season-nir.py`

### Numeric results

| Metric | Value |
|--------|-------|
| nir_mean range (all pixels) | 0.144 – 0.317 |
| nir_cv range (all pixels) | 0.009 – 0.191 |
| Dataset median CV | 0.067 |
| Infestation patch pixels | 362 |
| Infestation patch — nir_mean percentile | 57.5th |
| Infestation patch — nir_cv stability percentile | 73.2th (inverted) |
| Infestation patch pixels with CV ≤ dataset median | 325 / 362 (90%) |
| Spatial coherence (Pearson r, 8-neighbour mean) | 0.901 |

### Criterion outcomes

1. **[FAIL] nir_mean top-quartile** — infestation patch sits at the 57.5th percentile,
   not top quartile. This criterion was written assuming the infestation would be the
   highest-NIR population. With the southern extension included, the riparian feature
   near lat -22.765 produces the highest NIR values in the dataset, outscoring the
   Parkinsonia patch. Raw mean NIR is not a clean discriminator between Parkinsonia and
   riparian/dense woody vegetation.

2. **[PASS] nir_cv stability** — 90% of infestation pixels are more stable year-to-year
   than the dataset median. The grassland pixels in the southern extension have
   substantially higher CV (more variable between dry seasons), as expected from
   rain-dependent senescence. This is the stronger signal.

3. **[PASS] Spatial coherence** — r = 0.901, well above the 0.5 threshold. The CV
   pattern is spatially coherent, not noise.

### Interpretation

**CV discriminates Parkinsonia from grassland.** The inter-annual stability of dry-season
NIR is a real and spatially coherent signal that separates the infestation patch from the
surrounding grassland. This is consistent with the hypothesis: Parkinsonia's deep roots
sustain canopy through dry season year after year, while grasses vary with rainfall.

**nir_mean alone is not sufficient.** Riparian vegetation in the southern extension has
higher NIR than the Parkinsonia patch. Any classifier relying on mean NIR would confuse
dense riparian woody vegetation with Parkinsonia. CV corrects for this — riparian
vegetation is also variable (driven by seasonal flooding), which distinguishes it from
the consistently elevated NIR of Parkinsonia.

**Confounder: riparian pixels in the southern extension.** The water feature near
lat -22.765 introduces a high-NIR, potentially high-variability outlier population into
the negative sample set. These pixels should be masked or treated as a separate class
when training any classifier on this data.

### Next step

Wet/dry seasonal amplitude (`LONGREACH-WET-DRY-AMP.md`). Amplitude should further
separate Parkinsonia (low amplitude — persistent canopy through dry season) from both
grassland (high amplitude — complete senescence) and riparian vegetation (intermediate
amplitude — green in wet season from flooding, partially green in dry). This three-way
separation is not achievable from CV alone.
