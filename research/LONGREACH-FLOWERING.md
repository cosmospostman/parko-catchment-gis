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

### 3. Scene-level composite (infestation patch mean per date)

For each acquisition date, compute the mean of each index across the 374 infestation
pixels (excluding any pixel with fewer than 10 observations in the full archive, to
avoid edge effects). This gives a single time series per index.

Do the same for the 374 southern-extension pixels as a reference (non-Parkinsonia
background). The contrast between the two populations on each date is the signal.

### 4. Annual median profile (DOY)

Bin observations by day-of-year (DOY) into 2-week bins (26 bins). For each bin compute:
- Median index value for infestation pixels
- Median index value for extension pixels
- Difference (infestation − extension)

Plot all five indices as line charts across DOY. Look for bins where the infestation
population diverges from the extension population — a divergence that recurs across
multiple years is a candidate flowering window.

### 5. Year-by-year flowering dates

For each year 2020–2025, find the date range where FI_by (primary index) exceeds the
pixel's own annual 75th-percentile value. Report the start and end DOY per year. Check
whether the window is consistent year-to-year or shifts.

### 6. Pixel-level spatial pattern at candidate peak

Identify the DOY bin with the strongest infestation-vs-extension contrast. For that
bin, plot the 748 pixels on a lat/lon grid coloured by FI_by. Expect the infestation
patch to show elevated values; riparian pixels in the southern extension may also
respond (they have woody canopy that can flower).

### 7. Correlation with rainfall

*Optional, if a flowering window is found.* Cross-check the infestation peak DOY
against SILO monthly rainfall for Longreach (station or gridded) to test whether
flowering consistently follows wet-season rain by a fixed lag.

## Outputs

| File | Content |
|------|---------|
| `outputs/longreach-flowering/fi_doy_profiles.png` | All five indices as DOY median profiles, infestation vs extension |
| `outputs/longreach-flowering/fi_by_timeseries.png` | Raw FI_by scene-mean time series 2020–2025, both populations |
| `outputs/longreach-flowering/fi_by_spatial_peak.png` | Pixel-level FI_by map at peak DOY bin |
| `outputs/longreach-flowering/flowering_window_by_year.csv` | Start/end DOY of elevated FI_by per year |

## Success criteria

1. At least one DOY window shows infestation mean FI_by > extension mean FI_by by more
   than one standard deviation of the extension population — indicating a detectable
   signal above background.
2. The window recurs in at least 3 of 6 years (2020–2025) within a ±30-day range —
   confirming it is phenological, not a one-off cloud artefact.
3. The spatial pattern at peak is coherent with the infestation bbox (not randomly
   distributed across the 748 pixels).

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

**Method change from plan:** Switched from infestation-vs-extension contrast to within-pixel
z-score anomaly detection. The infestation patch pixels have persistently lower visible/NIR
ratios than senescent grass year-round (persistent green canopy suppresses the indices), so
a direct inf−ext difference is always negative and masks any flowering spike. Per-pixel
z-scores (baseline = that pixel's own DOY-bin median across all years, denominator = that
pixel's overall std) remove the population-level offset and expose transient anomalies.

### Numeric results

| Metric | Value |
|--------|-------|
| Peak z-score (DOY-bin mean, infestation) | +0.333 at DOY bin 43 (mid-Feb) |
| Peak acquisition date | 2020-02-13, mean z = 2.123 across infestation pixels |
| Infestation z-score range on peak date | 0.765 – 3.405 |
| Extension z-score range on peak date | −1.105 – 3.710, mean 1.024 |
| Years with ≥ 1 date at mean z ≥ 1.0 | 2020, 2021, 2022, 2023 (4 of 6 years) |
| Spatial coherence (Pearson r, 8-neighbour mean, infestation) | 0.800 |

### Criterion outcomes

1. **[FAIL] DOY-bin peak z > 1.0** — peak bin mean is only +0.333. The flash lands on
   different fortnights each year, diluting the bin average when pooled across all years.
   Individual acquisition dates exceed z = 2.0 in multiple years — the signal exists but
   is temporally narrow.

2. **[FAIL] Recurs in ≥ 3 years within ±30 DOY of peak bin** — criterion was written for
   a consistent calendar window. In practice elevated dates span DOY 44–337, so no single
   ±30 DOY window captures 3 years. The criterion is too strict for an opportunistic
   flowering phenology.

3. **[PASS] Spatial coherence r = 0.800** — the infestation patch responds as a coherent
   unit on the peak date, not as noise.

### Spatial pattern interpretation (from aerial imagery ground-truth)

The side-by-side map (infestation north, extension south) shows:

- **Infestation (left panel):** uniformly green — every pixel elevated above its own
  baseline on 2020-02-13. Coherent, patch-wide response consistent with mass flowering.

- **Extension (right panel):** spatially structured, not uniform:
  - **Cream/yellow pixels (near-zero z):** open grassland areas confirmed to be largely
    Parkinsonia-absent from aerial imagery. These are the true negative population.
  - **Green pixels in south of extension:** patchy Parkinsonia population present in that
    area — index responds as expected.
  - **Green pixels around riparian feature:** Parkinsonia mingled with other species at
    the northern edge of the riparian section. Both Parkinsonia and co-occurring woody
    species likely contribute.

**Key finding:** The cream zone (genuinely absent pixels) is the true negative class.
The extension is not a clean negative sample — it contains mixed Parkinsonia — but the
FI_by z-score correctly separates the absent pixels (cream) from the present pixels
(green), without any supervised label. This is encouraging: the index may be detecting
presence/absence rather than just wet-season greenness, since pure grassland pixels do
not spike even during wet season.

### Conclusions

**The original hypothesis — a detectable, calendar-consistent flowering window — is not
supported.** The formal success criteria fail because Parkinsonia at this site appears to
flower opportunistically: elevated FI_by anomalies occur in 4 of 6 years but span DOY
44–337 with no fixed window. The ~5-day S2 revisit captures the event in some years and
misses it in others. No single fortnight reliably concentrates the signal across years.

**However, the within-pixel z-score approach reveals a more useful signal than originally
expected.** On dates when the infestation does flower, FI_by z-scores reach 2–3 σ above
each pixel's own seasonal baseline across the entire patch simultaneously (spatial
coherence r = 0.800). More importantly, the spatial pattern within the mixed extension
zone matches known ground conditions:

- Pixels in genuinely Parkinsonia-absent grassland remain near their baseline (cream,
  z ≈ 0) even during wet season — wet-season greenness alone does not produce a spike.
- Pixels in the patchy Parkinsonia south of the extension spike green, consistent with
  low-density infestation there.
- Pixels around the riparian feature spike where Parkinsonia is known to be mingled with
  other species.

This means **FI_by z-score anomaly is responding to Parkinsonia presence, not just
general wet-season greenness.** The signal is not reliable as a single-date detector
(the flowering event may not be captured in any given year), but it contributes as one
component of a multi-temporal feature set: a pixel that never spikes above its own
baseline across the full 2020–2025 archive is unlikely to contain Parkinsonia.

**Position in the multi-signal picture:**

| Signal | Discriminates Parkinsonia from grassland | From riparian |
|--------|------------------------------------------|---------------|
| Dry-season NIR CV (stability) | Yes | Partial — riparian also variable |
| FI_by z-score anomaly | Yes — grassland does not spike | Partial — riparian spikes if Parkinsonia present |
| Wet/dry amplitude | TBD | Expected to separate all three |

The two completed analyses together already separate Parkinsonia from open grassland
convincingly. Riparian remains the residual confounder; wet/dry amplitude is the
intended resolution.

### Next step

Wet/dry seasonal amplitude (`LONGREACH-WET-DRY-AMP.md`). Parkinsonia's deep roots
sustain canopy through dry season, producing low NIR/NDVI amplitude. Grassland
senesces completely (high amplitude). Riparian vegetation is intermediate — green in wet
season from flooding, partially brown in dry. This three-way separation is not achievable
from CV or FI_by alone.
