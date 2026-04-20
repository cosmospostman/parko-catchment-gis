# Peak Day-of-Year Signal — Hypothesis and Investigation Plan

## Observation

The mean annual NDVI peak DOY map (s4d, recession-and-greenup exploration) shows that the
Longreach 8×8 km scene is dominated by early-peaking pixels (DOY ~50–100, February–March,
shown in deep purple). However, a spatially coherent cluster of late-peaking pixels (DOY
~150–250, May–September, shown in yellow/orange) is visible in the upper-right quadrant of
the scene (~145.47 E, ~22.72 S). This cluster corresponds to a known Parkinsonia infestation
surrounding an artificial cattle dam.

This is ecologically interpretable: vegetation with permanent or semi-permanent water access
(dam-fed Parkinsonia) stays green well into the dry season, so its NDVI peak occurs later
than the surrounding grassland, which peaks during or immediately after the wet season and
then rapidly senesces. The dam provides a water subsidy that decouples the vegetation's
greenness peak from the wet-season rainfall pulse.

---

## Why this matters for the generalisation problem

The existing signals (`rec_p`, `nir_cv`, `re_p10`) were developed on the Thomson River
riparian corridor, where Parkinsonia benefits from flood-pulse water access. The signals
measure *mean canopy state* — how green and stable a pixel is on average. They discriminate
well at Longreach because the contrast class (grassland) dries completely in the dry season.

The dam cluster is a different Parkinsonia growth context: **permanent point-source water,
not flood-pulse water**. The existing signals may still work here (the dam Parkinsonia is
probably also high `nir_cv`, high `re_p10`), but the *mechanism* is different and the peak
DOY signature is distinct: the dam pixels peak months later than both the typical Parkinsonia
(flood-pulse, peaks Feb–March with the river) and the grassland (peaks Dec–Feb with rainfall).

This raises the hypothesis:

> **Peak DOY is a signal of water-source type, not just Parkinsonia presence.** Early-peaking
> Parkinsonia is flood-pulse or rainfall-fed; late-peaking Parkinsonia is point-source or
> groundwater-fed. The two populations may require different features to discriminate from
> their respective absence classes. Peak DOY identifies which population a pixel belongs to,
> and therefore which feature regime applies.

If this is correct, peak DOY is not primarily a Parkinsonia detector — it is a
**water-access classifier** that partitions the scene into subpopulations with different
spectral discrimination strategies.

---

## Specific hypotheses to test

**H1 — Late-peaking pixels are enriched for high-`prob_lr` Parkinsonia.**
Among pixels with mean peak DOY > 150, the distribution of `prob_lr` should be shifted
toward higher values relative to the full scene. If the dam cluster is genuinely Parkinsonia,
the late-peaking population should have a higher Parkinsonia hit rate than the early-peaking
majority.

**H2 — The dam cluster is spatially tight and corresponds to mapped infrastructure.**
The late-peaking pixels should form a compact spatial cluster centred on the dam location,
not a diffuse scatter across the scene. Spatial compactness distinguishes a real ecological
signal from algorithm noise (broad plateaus causing random late-peak assignments).

**H3 — The NDVI time series shape of late-peaking pixels is structurally different from
early-peaking Parkinsonia.**
Late-peaking pixels should show a sustained NDVI plateau through April–August rather than
the typical post-wet-season recession. Their annual waveform should look like: moderate
greenness in the wet season (not the highest in the scene), rising or flat through Autumn,
peaking in the dry season, then declining in Spring as the dam drawdown or heat stress sets in.

**H4 — `rec_p` and `nir_cv` are less discriminating within the late-peaking subpopulation.**
If the absence class near the dam also has some water access (e.g. nearby gilgai clay
holding moisture), the mean-state features will compress. Within the late-peaking subset,
peak DOY consistency (`peak_doy_cv`) may be the better discriminator.

---

## Investigation plan

### Step 1 — Identify and characterise the late-peaking population

From the `green_stats` output of the recession-and-greenup pipeline:

- Extract pixels with `peak_doy` > 150 (approximately May onward).
- Plot their spatial distribution as a map overlay on s4d. Confirm whether they are
  concentrated in the upper-right dam cluster or scattered across the scene.
- Plot their `prob_lr` distribution alongside the full-scene distribution. Report the
  fraction with `prob_lr` > 0.7 in the late-peaking vs. early-peaking populations.

### Step 2 — NDVI time series for the late-peaking cluster

- Compute the mean daily NDVI for the late-peaking pixels (all years, 2020–2025).
- Plot alongside the presence class mean and absence class mean from Stage 1.
- Look for the expected dry-season plateau shape. If the curves show a mid-year NDVI
  peak (May–August) rather than a wet-season peak, H3 is supported.
- Also plot individual-pixel smoothed curves for a sample of the late-peaking pixels
  (same format as s4b) to check that the algorithm is finding real peaks, not plateau noise.

### Step 3 — Test H1 quantitatively

- Compute the enrichment ratio: P(prob_lr > 0.7 | peak_doy > 150) vs.
  P(prob_lr > 0.7 | peak_doy ≤ 150). A ratio > 2 would be meaningful enrichment.
- Also compute the converse: among pixels the original model scores as high Parkinsonia
  probability, what fraction are late-peaking? If it is > 5% of the presence class, the
  late-peaking population is not a marginal footnote but a meaningful subpopulation.

### Step 4 — Feature correlation within the late-peaking subset

- Within the late-peaking pixels only, compute Pearson r between `prob_lr` and each of:
  `rec_p`, `nir_cv`, `re_p10`, `peak_doy_cv`.
- Compare to the same correlations computed on the full scene (from Stage 6 of the
  recession-and-greenup exploration).
- If `rec_p` and `nir_cv` weaken (r drops toward zero) while `peak_doy_cv` holds or
  strengthens within the late-peaking subset, H4 is supported.

### Step 5 — Locate the dam in external data

- Cross-reference the cluster coordinates with OpenStreetMap / QLD DNRM infrastructure
  data or Google Earth to confirm the dam location.
- Record the approximate dam coordinates and extent for use as a reference point in
  future multi-site analysis.
- Note whether the dam is visible in the NDWI maps (s2b) as a persistent positive-NDWI
  pixel in dry years — permanent water surface should be identifiable.

---

## What success looks like

The investigation is worthwhile if it establishes one of two things:

1. **Peak DOY is a useful complementary feature** — the late-peaking Parkinsonia population
   is real, enriched for high `prob_lr`, and has a waveform shape that other features miss.
   This would motivate including `peak_doy` in the feature set as a detector of
   point-source-water Parkinsonia, distinct from the flood-pulse signature.

2. **Peak DOY identifies a generalisation gap** — there exists a Parkinsonia growth context
   (permanent water, late peak) that the current feature set handles only by coincidence
   (the mean-state features happen to be high). Knowing this gap exists motivates testing
   at dam/bore sites elsewhere before claiming the classifier generalises.

Either outcome advances the generalisation goal. A null result (the late-peaking pixels are
not enriched for Parkinsonia and the dam is a false lead) also closes a dead end cleanly.

---

## Implementation notes

- All required inputs already exist: `green_stats` parquet from the recession-and-greenup
  pipeline, `prob_lr` from the pixel ranking CSV, raw parquet for NDVI time series.
- No new signal computation is needed for Steps 1–4 — this is pure diagnostic analysis
  on existing outputs.
- Steps 1–4 should be implemented as a new exploration script:
  `longreach/peak-doy-explore.py`, writing outputs to
  `research/day-of-year-peak/longreach-peak-doy/`.
- Step 5 is manual cross-referencing; record findings as a note appended to this file.
