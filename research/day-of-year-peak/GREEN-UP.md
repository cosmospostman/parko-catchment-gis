# Green-Up Timing — Hypothesis and Investigation Plan

## Core observation

Parkinsonia's deep root system buffers its green-up timing from rainfall variability.
In flood-pulse systems (e.g. Longreach Thomson River corridor), Parkinsonia peaks
earlier than surrounding grassland and maintains canopy longer through the dry season.
Grassland tracks monsoon onset more closely, producing higher year-to-year variability
in peak DOY and a later, rainfall-dependent peak.

This is distinct from the dam/bore Parkinsonia thread (see DAM.md), where permanent
point-source water produces a late-peaking signature. The flood-pulse population is
the primary target here.

---

## Signal formulations

### 1 — Absolute peak DOY (`peak_doy`)

Per-pixel mean DOY of annual NDVI peak across reliable years. Already implemented in
`signals/greenup.py`.

**What the Longreach analysis showed:**
- Using `prob_lr` quantile labels: presence median SD ~35 days vs absence ~51 days —
  a 16-day gap. Likely optimistic (circular: `prob_lr` may encode timing implicitly).
- Using geometry labels (infestation bbox vs rest of scene): presence SD ~43 days vs
  absence ~45 days — a 2-day gap. Likely pessimistic (absence class is too heterogeneous,
  mixing grassland, bare soil, water, scrub).
- True discriminative power is somewhere between the two runs.
- Spatial map (s4d) shows coherent structure along the river corridor — the feature is
  tracking real spatial variation, not noise.
- Correlation with existing features: r ≈ −0.36 with `rec_p` — genuinely new information,
  not redundant.
- Pipeline already standardises features via `StandardScaler` before LR, so the 0–365
  scale is handled correctly.

**Caveat — DOY wraps.** Day 365 and day 1 are adjacent but numerically 364 apart.
At Longreach both classes peak well within DOY 50–200, so wrap-around is not a current
issue. Monitor at other sites.

**Caveat — plateau jitter.** Many Parkinsonia pixels have broad NDVI plateaus (DOY ~50–200)
where the peak-finder picks slightly different DOYs year to year. This inflates per-pixel
SD and degrades precision. Consider spatial smoothing of the feature before feeding to
the classifier.

### 2 — Relative greenup shift (neighbourhood contrast)

Per-pixel-year: `peak_doy - median(peak_doy of pixels within radius R)`, averaged across
years. Parkinsonia pixels with earlier peak timing should be consistently negative relative
to their local neighbourhood.

**Motivation:** Controls for inter-annual variation in flood/rainfall timing that shifts
everyone's peak together. The relative offset should be more stable across years and more
transferable across sites than the absolute DOY.

**Parameters:**
- Radius R: start at 500m. Sensitivity sweep over 250m, 500m, 1km planned.
- Vegetation gate: exclude pixels below a minimum NDVI amplitude threshold from the
  neighbourhood median to avoid bare soil contaminating the reference. Use a conservative
  threshold (e.g. annual NDVI amplitude < 0.10) so the gate transfers across sites without
  tuning.

**Known limitation — dense infestation cores:** If Parkinsonia dominates a neighbourhood,
the local median is pulled toward Parkinsonia timing, compressing the relative offset toward
zero. The signal is strongest at infestation edges and weakest in the dense core. This is
acceptable — it is complementary to the absolute `peak_doy` rather than a replacement,
acting as an edge/boundary detector. Dense cores are likely already captured by `rec_p`
and `nir_cv`.

---

## The riparian complication

Riparian native vegetation occupies a similar water-access niche to Parkinsonia and may
produce a similar greenup deviation from grassland. From Stage 1 of the recession-and-greenup
analysis, riparian pixels show: high dry-season NDVI floor + suppressed wet-season spike.
This is a different waveform shape from Parkinsonia (high dry floor + strong wet spike), but
both deviate from grassland in peak timing.

The relative greenup shift alone likely cannot separate Parkinsonia from riparian natives —
both would deviate from the grassland neighbourhood. A waveform shape feature is also needed:
specifically the ratio of wet-peak amplitude to dry-season floor, which RESULTS.md identified
as an unmeasured signal axis that would separate riparian from Parkinsonia.

**Literature gap:** Whether Parkinsonia's phenological signature has been characterised
against co-occurring native riparian species (not just grassland) in semi-arid floodplain
systems is unknown. The *Tamarix* literature (American southwest) is the most relevant
analogue — decades of remote sensing discrimination work in an ecologically similar invasive
phreatophyte system.

Research question for literature review:
> How does the seasonal NDVI phenology of *Parkinsonia aculeata* differ from co-occurring
> native riparian vegetation in semi-arid floodplain systems, and can remote sensing
> time-series features (peak timing, recession rate, dry-season canopy retention) discriminate
> between them?

---

## Implementation plan

### Step 1 — Validate absolute `peak_doy` with a cleaner comparison

Run the recession-and-greenup explore script with a third labelling scheme:
- Presence = infestation sub-bbox
- Absence = grassland sub-bbox only (not the full heterogeneous scene)

This gives a controlled comparison equivalent to the original training labels, without the
`prob_lr` circularity. Expected to recover most of the 16-day gap from the quantile analysis.

### Step 2 — Implement relative greenup shift signal

New signal: `signals/greenup_shift.py` (or extend `signals/greenup.py`).

Per pixel per year:
1. Find all pixels within radius R whose annual NDVI amplitude >= amplitude threshold.
2. Compute median peak DOY of that neighbourhood pool.
3. Compute offset = pixel peak DOY − neighbourhood median.
4. Average offset across reliable years → `peak_doy_shift`.
5. SD of offset across years → `peak_doy_shift_cv`.

### Step 3 — Sensitivity sweep

Sweep R over [250m, 500m, 1000m] and amplitude threshold over [0.08, 0.10, 0.15].
Report class separation (infestation bbox vs grassland bbox) for each combination.
Identify whether a stable optimum exists or whether the signal is highly parameter-sensitive.

### Step 4 — Literature review

Consult literature on Parkinsonia vs native riparian phenology before committing to the
waveform shape feature design. See research question above.

### Step 5 — Waveform shape feature (pending literature review)

Design a feature capturing the ratio of wet-peak amplitude to dry-season floor. This
targets the riparian separation axis identified in RESULTS.md. Implementation deferred
until the literature review informs whether this axis is already characterised and what
the expected direction of separation is.
