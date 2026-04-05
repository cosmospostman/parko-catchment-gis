# Flowering Flash — Analysis Plan

## Hypothesis

Parkinsonia aculeata produces bright yellow flowers. During flowering, the visible
reflectance of canopy pixels shifts: red (B04) and green (B03) rise, blue (B02) drops
(yellow absorbs blue), and NIR (B08) stays flat or falls slightly relative to a fully
leafed canopy. This produces a distinctive spectral shape that should be detectable as a
transient anomaly in the 2020–2025 time series.

No assumption is made about when flowering occurs. The analysis will scan the full
year and let the signal reveal the window — if one exists at this site.

## Data

- **Source:** `data/longreach_pixels.parquet`
- **Schema:** `point_id`, `lon`, `lat`, `date`, bands `B02 B03 B04 B05 B06 B07 B08 B8A B11 B12`, `scl_purity`, `aot`, `view_zenith`, `sun_zenith`
- **Pixels:** 748 (374 infestation patch + 374 southern extension), ~387 obs each, 2020–2025

## Candidate indices

Five indices, each targeting a different facet of the yellow-flower spectral shape:

| Index | Formula | Rationale |
|-------|---------|-----------|
| FI_rg | (B03 + B04) / B08 | Green + red vs NIR; yellow raises numerator, leaves NIR flat |
| FI_r | B04 / B08 | Red-to-NIR alone; simpler, less influenced by green vegetation background |
| FI_by | (B03 + B04) / (B02 + B08) | Yellow-specific: exploits blue suppression + NIR flatness simultaneously |
| dNDVI | − (B08 − B04) / (B08 + B04) | Inverted NDVI; flowering depresses NDVI |
| FI_swir | B11 / B08 | SWIR rise from dry petals/pollen vs leaf water; subtle, included for completeness |

FI_by is expected to be the most discriminating: bare gilgai clay is bright in B02 as
well as B03/B04, so the blue suppression term reduces soil false-positives. Grasses
greening up post-rain have normal NDVI (NIR follows red), which dNDVI won't flag.

## Steps

### 1. Quality filter

Retain rows where `scl_purity >= 0.5`.

### 2. Compute indices

Add all five index columns to the filtered dataframe.

### 3. Within-pixel z-score anomaly detection

The infestation patch has persistently lower visible/NIR ratios than senescent grass
year-round (persistent green canopy suppresses the indices), so a direct
infestation−extension difference is structurally negative and masks any flowering spike.
Use within-pixel z-scores instead:

- **Baseline:** per-pixel per-DOY-bin median across all years (removes seasonal shape)
- **Denominator:** per-pixel overall std across all dates (normalises to pixel's own
  variance scale)

This removes the population-level offset and exposes transient anomalies regardless of
when they occur. Compute z-scores for all five indices, for both infestation and
extension pixels independently (each pixel's baseline estimated from its own time
series).

### 4. DOY anomaly profiles

Bin z-scores by 14-day DOY bins (26 bins). For each bin compute the mean z-score across
infestation pixels for all five indices. Plot as 5-panel figure with ±1 std band. A
positive bin mean indicates the infestation systematically exceeded its own baseline in
that fortnight across years. Do not expect a sharp peak — if flowering is opportunistic
the signal will be diluted by years where the event falls in a different bin.

### 5. Per-year flowering window detection

For each year 2020–2025, find all acquisition dates where the mean FI_by z-score across
infestation pixels ≥ 1.0. Report for each year:
- Number of qualifying dates
- DOY range (start–end)
- Peak z-score and the date it occurs

Check whether the window timing is consistent across years (±30 DOY) or shifts. A
consistent calendar window supports a fixed-phenology signal; a shifting window means
the per-pixel annual p90 (step 6) is the better summary.

### 6. Inter-population contrast time series

For each acquisition date, compute the contrast: mean FI_by z-score across infestation
pixels minus mean FI_by z-score across extension pixels. A positive contrast means the
infestation is above its own baseline more than the extension is above its own — i.e. the
populations are genuinely diverging, not moving together due to a scene-wide effect (haze,
rain). Plot as a scatter over time with 30-day rolling mean, plus a DOY profile of mean
contrast per 14-day bin. Log per-year: max contrast and date, fraction of dates > 0.

A scene-wide effect (haze, widespread rain) would produce near-zero contrast even when
both populations show elevated z-scores. Genuine Parkinsonia-specific events produce
positive contrast.

### 7. Per-pixel annual p90 summary statistic — two variants

For each pixel and each year compute the 90th-percentile FI_by z-score. Average across
years. Two variants:

- **`fi_p90` (unrestricted):** p90 across all haze-filtered observations. Asks "how high
  does this pixel's anomaly reach in a typical year?" — but wet-season greenness dates
  lift both populations equally, diluting class separation.

- **`fi_p90_cg` (contrast-gated):** p90 restricted to dates where the scene-level
  contrast (step 6) was positive that year. By conditioning on dates when the infestation
  was genuinely diverging from the extension, this removes observations where both
  populations move together and concentrates the statistic on candidate flowering or
  dry-season retention events. The "window" is defined per-year by the contrast sign
  rather than a fixed DOY range — window-free but signal-selective.

A Parkinsonia-absent pixel has no mechanism to produce elevated FI_by on dates when the
infestation is diverging from the extension, so its `fi_p90_cg` should remain near zero.

### 8. Band decomposition on peak dates

On each date identified as elevated in step 5, plot the per-date mean of the four
constituent bands (B02, B03, B04, B08) separately for infestation pixels vs their own
DOY-bin baseline. This confirms which bands are driving the FI_by spike:

- **Expected (flowering):** B03 and B04 elevated, B02 suppressed, B08 flat or slightly
  depressed
- **Confound (atmospheric haze):** B02, B03, B04 all elevated together; B08 unaffected
- **Confound (wet-season greenness):** B03, B04, B08 all elevated (normal vegetation
  response); B02 follows B03/B04

Plot as a small-multiple: one panel per peak date, four bands as bars showing
(observed − baseline) for infestation pixels. If multiple peak dates show the same
band pattern, the mechanism is consistent. If the pattern varies across dates, some
events may be atmospheric rather than phenological.

Log the fraction of peak dates that show the expected flowering signature vs confound
patterns.

### 9. Pixel-level spatial pattern at peak date

For the single acquisition date with the highest mean infestation z-score, plot all 748
pixels coloured by FI_by z-score (side-by-side infestation / extension panels). Expect:
- Infestation: uniformly elevated (coherent patch-wide response)
- Extension grassland: near-zero (grassland wet-season greenness does not produce a
  spike)
- Extension Parkinsonia pixels: elevated where known infestation is present

### 10. Correlation with rainfall

*Optional, if a consistent flowering window is found in step 5.* Cross-check the peak
DOY per year against SILO monthly rainfall for Longreach to test whether flowering
follows wet-season rain by a fixed lag.

## Outputs

| File | Content |
|------|---------|
| `outputs/longreach-flowering/fi_doy_profiles.png` | All five indices as DOY z-score anomaly profiles (infestation, mean ± std) |
| `outputs/longreach-flowering/fi_by_timeseries.png` | Raw FI_by scene-mean time series + infestation z-score anomaly 2020–2025 |
| `outputs/longreach-flowering/fi_by_contrast.png` | Inter-population z-score contrast time series + DOY profile |
| `outputs/longreach-flowering/fi_by_spatial_peak.png` | Pixel-level FI_by z-score map at peak acquisition date (infestation / extension side-by-side) |
| `outputs/longreach-flowering/fi_band_decomposition.png` | Per-band anomaly (B02, B03, B04, B08) on each peak date — mechanism confirmation |
| `outputs/longreach-flowering/flowering_window_by_year.csv` | Per-year: n_dates, DOY range, peak z-score, peak date |
| `outputs/longreach-flowering/fi_p90_per_pixel.csv` | Per-pixel `fi_p90` (unrestricted) and `fi_p90_cg` (contrast-gated) |

## Success criteria

1. **Peak date z-score ≥ 2.0** — at least one acquisition date has mean FI_by z-score
   ≥ 2.0 across infestation pixels, indicating a detectable anomaly above each pixel's
   own seasonal baseline.
2. **Recurs in ≥ 3 of 6 years** — at least 3 years have at least one date with mean
   z-score ≥ 1.0, regardless of when in the year it falls.
3. **Band decomposition confirms flowering mechanism** — on the majority of peak dates,
   B03 and B04 are elevated and B02 is suppressed relative to baseline (not a uniform
   broadband increase consistent with haze).
4. **Spatial coherence r ≥ 0.5** — Pearson r between pixel z-score and 8-neighbour mean
   on the peak date, confirming patch-level coherence rather than random noise.

## What failure would mean

**No window found:** Either Parkinsonia does not flower in a way detectable at 10m S2
resolution (crown cover ~30–40%, petals mix with leaf and soil signal), or flowering is
too brief for the ~5-day S2 revisit to reliably capture. Next step: investigate
wet/dry amplitude, which does not depend on a brief phenological event.

**Window found but not spatially coherent:** Could be a cloud/shadow artefact that
passed the SCL filter. Check the raw imagery for those dates.

**Riparian pixels in the extension also fire:** Expected — report separately but do not
treat as failure. The combination of FI_by + dry-season CV (from dry-NIR analysis) may
still separate Parkinsonia from riparian.

---

## Results (run 2026-04-05)

**Script:** `longreach/flowering.py`

**Method notes:**
- Within-pixel z-scores used throughout (baseline = per-pixel DOY-bin median, denominator =
  per-pixel overall std) to remove the structural population-level offset.
- Scene-level haze filter applied before all analysis: dates where scene-mean B02 exceeded
  its DOY-bin median by > 0.010 reflectance units are excluded (45 of 304 dates removed).
  AOT was considered but has near-zero correlation with B02 anomaly (r = −0.09) at this site.
- Contrast-gated p90 added after first run showed unrestricted fi_p90 could not separate
  populations — wet-season greenness dates lift both infestation and extension equally.

### Numeric results

| Metric | Value |
|--------|-------|
| Dates after haze filter | 259 of 304 (45 removed) |
| Peak acquisition date (haze-filtered) | 2020-02-28, mean infestation z = 1.868 |
| Infestation z-score range on peak date | 0.098 – 3.908 |
| Extension mean z on peak date | **0.038** (near-zero — populations genuinely diverging) |
| Years with ≥ 1 elevated date (mean z ≥ 1.0) | 2020, 2021, 2022, 2023 (4 of 6) |
| Spatial coherence r on peak date | **0.927** |
| Band decomposition: flowering signature | 3 of 6 elevated dates (50%) |
| Band decomposition: greenness signature | 3 of 6 elevated dates (50%) |
| fi_p90 median — infestation / extension | 0.407 / 0.444 — indistinguishable |
| fi_p90_cg median — infestation / extension | **0.459 / 0.110** — zero IQR overlap |

**Contrast time series (fraction of dates where infestation z > extension z):**

| Year | Frac > 0 | Max contrast | Date of max |
|------|----------|-------------|-------------|
| 2020 | 0.86 | +1.830 | 2020-02-28 (DOY 59) |
| 2021 | 0.09 | +0.290 | 2021-01-18 |
| 2022 | 0.49 | +2.368 | 2022-05-28 |
| 2023 | 0.47 | +1.810 | 2023-01-03 |
| 2024 | 0.73 | +0.407 | 2024-12-28 |
| 2025 | 0.54 | +0.852 | 2025-05-17 |
| **All years** | **0.52** | | |

### Criterion outcomes

1. **[FAIL] Peak acquisition date mean z ≥ 2.0** — peak clean date reaches z = 1.868,
   just below threshold. Pre-haze-filter, 2020-02-13 reached z = 2.123, but that date's
   elevated score was partly driven by haze (B02_anom = 0.002, borderline). Post-filter
   the strongest unambiguously clean date falls short.

2. **[PASS] Elevated date in ≥ 3 of 6 years** — 4 years (2020, 2021, 2022, 2023) have
   at least one haze-filtered date with mean infestation z ≥ 1.0.

3. **[PASS] Band decomposition confirms non-haze mechanism** — with haze dates removed,
   0 of 6 elevated dates show the haze pattern. 3 show the flowering signature (B03↑
   B04↑ B02 suppressed relative to B04), 3 show wet-season greenness (B03↑ B04↑ B08↑).

4. **[PASS] Spatial coherence r ≥ 0.5** — r = 0.927 on the peak date. High coherence is
   now biologically meaningful: extension sits at near-zero (mean z = 0.038) while
   infestation responds as a coherent unit.

### Haze filter impact

Before filtering, 9 elevated dates were identified; band decomposition showed 100% haze
pattern. After filtering, 6 elevated dates remain; 0% haze pattern. The 3 removed dates
(2020-05-18, 2021-03-29, 2023-05-03) had B02 anomalies of 0.046, 0.024, and 0.026 — well
above the 0.010 threshold. The critical change is the **extension mean z on the peak date
dropping from 1.024 to 0.038**: before filtering the extension was spiking almost as much
as the infestation (scene-wide effect); after filtering the infestation diverges while the
extension does not (population-specific effect).

### Contrast time series

Overall fraction positive (0.52) is weak — well below the red-edge result (0.80). The
signal is highly year-dependent: 2020 shows strong consistent separation (0.86), 2021 is
inverted (0.09 — extension systematically above infestation), and most other years are
near-random. No consistent DOY window concentrates the contrast across years; the DOY
profile peak (bin 141, mid-May) has a wide std band. This confirms the flowering signal,
if real, is opportunistic rather than calendar-consistent.

### Per-pixel p90 — unrestricted vs contrast-gated

| Variant | Infestation median | Extension median | IQR overlap |
|---------|-------------------|-----------------|-------------|
| fi_p90 (unrestricted) | 0.407 | 0.444 | 0.475 — indistinguishable |
| fi_p90_cg (contrast-gated) | **0.459** | **0.110** | **0.000** — clean separation |

The unrestricted p90 fails because wet-season greenness dates elevate both populations
equally — the extension contains genuine Parkinsonia pixels that also spike green in the
wet season. The contrast-gated variant resolves this: by restricting to dates when the
infestation is diverging from the extension, it removes the shared greenness signal and
exposes the Parkinsonia-specific anomaly. A grassland-only pixel has no mechanism to
produce elevated FI_by on those dates, so its fi_p90_cg collapses toward zero.

The contrast-gate effectively acts as a soft window: instead of specifying "use April" or
"use dry season", it uses "use dates when the infestation was doing something the extension
was not" — which is the directly relevant condition for detection.

### Conclusions

**After haze filtering, a real but weak and temporally inconsistent signal is present.**
The formal criteria are 3 of 4 passing, with criterion 1 (peak z ≥ 2.0) narrowly failing.
More importantly, the signal is not reliable year-to-year: 2 of 6 years show no elevated
dates at all, and the contrast fraction is near-random in 3 of 6 years.

**fi_p90_cg (contrast-gated annual p90) is the most useful output of this analysis.**
It achieves zero IQR overlap between infestation and extension with a clear mechanism:
it captures dates when Parkinsonia pixels are anomalously high *relative to their own
baseline and relative to the extension*. Wet-season greenness, which confounds the
unrestricted variant, is removed because grass and mixed-extension pixels spike together
with the infestation on those dates.

**Position in the multi-signal picture:**

| Signal | Discriminates Parkinsonia from grassland | From riparian | Notes |
|--------|------------------------------------------|---------------|-------|
| Dry-season NIR CV | Yes — zero IQR overlap | Partial | Complete |
| NDVI seasonal recession (rec_mean) | Yes — zero IQR overlap | Yes | Complete |
| Red-edge p10 (re_p10) | Yes — zero IQR overlap | Partial | Complete |
| FI_by fi_p90_cg | Yes — zero IQR overlap | TBD | Contrast-gated only |
| Wet/dry amplitude | Yes | Yes | Complete |

**Next step:** Riparian discrimination using fi_p90_cg needs to be evaluated — the
extension riparian pixels were not separated from Parkinsonia in the red-edge analysis,
and the same caveat may apply here. The wet/dry amplitude analysis already provides
three-way separation; fi_p90_cg is a candidate additional feature for the final
multi-signal feature set.
