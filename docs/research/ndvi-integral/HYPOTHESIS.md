# NDVI Integral — Hypothesis and Investigation Plan

## Core idea

The area under the annual smoothed NDVI curve — the "large integral" or accumulated
greenness — captures something none of the current signals measure directly:

- `rec_p` measures *amplitude* (peak minus floor)
- `nir_cv` measures *variability* (inter-annual consistency of dry-season NIR)
- Neither measures total *duration × magnitude* of greenness across the year

A pixel that sits at NDVI 0.25 all year accumulates far more area than one that spikes
to 0.5 for three months and crashes to zero — even if their `rec_p` values are similar.

## Literature basis

The paper *Spatiotemporal Phenological Divergence and Remote Sensing Discrimination of
Parkinsonia aculeata in Semi-Arid Riparian Floodplains* describes this feature explicitly:

> "The 'large integral' of the seasonal NDVI curve, which sums the area under the
> greenness trajectory, is typically much higher for invasive thickets than for native
> woodlands. This is because P. aculeata occupies more of the 'phenological space'
> throughout the year, whereas native species have a more peaked and narrow greenness
> signature."

This is mechanistically grounded in Parkinsonia's green-stem strategy: photosynthetic
bark maintains a non-zero NDVI floor even during complete leaf drop, so the curve never
collapses to zero. Native grasses and semi-deciduous shrubs have narrow, high-amplitude
peaks that dominate `rec_p` but contribute little integrated area.

The same paper describes the HANTS-reconstructed Parkinsonia signature as
"low-amplitude, high-mean" — which is exactly the integral formulation: moderate peak,
high floor, sustained across the year.

## Relationship to existing signals

| Signal | What it measures | Integral overlap |
|---|---|---|
| `rec_p` | Peak − floor amplitude | Partial — high floor contributes to both |
| `nir_cv` | Inter-annual dry-season NIR variability | Low — different axis entirely |
| `re_p10` | Red-edge floor percentile | Partial — floor-sensitive but band-specific |
| `ndvi_integral` | Mean annual NDVI × year length | New — captures shape, not just endpoints |

Expected correlation with `rec_p`: moderate positive (high floor → more area). But two
pixels can have identical `rec_p` and very different integrals depending on curve width.
The integral is not redundant — it is expected to carry independent information about
curve shape.

## Hypotheses

**H1 — Parkinsonia pixels have a higher mean annual NDVI than grassland.**
The integral (normalised to mean NDVI per DOY) should be shifted upward for presence
pixels relative to absence, independently of `rec_p`.

**H2 — The integral adds information beyond `rec_p` and `nir_cv`.**
After regressing out `rec_p` and `nir_cv`, residual correlation between the integral
and `prob_lr` should remain positive and meaningful (r > 0.2).

**H3 — The integral separates Parkinsonia from native riparian species.**
Native riparian eucalypts (*E. camaldulensis*, *E. coolabah*) are flood-synchronised
with narrow peaked signatures. Their integral should be lower than Parkinsonia despite
similar dry-season floors, because their greenness is concentrated in a shorter wet-season
window. This would distinguish them where `rec_p` alone may not.

**H4 — The integral is stable across water-access contexts.**
Both flood-pulse Parkinsonia (earlier peak, Thomson River corridor) and point-source
Parkinsonia (later peak, dam/bore sites) should show elevated integrals relative to their
local grassland background, since the green-stem floor is present in both contexts. The
integral may therefore be more transferable across sites than peak-timing features.

---

## Implementation

### Feature definition

```
ndvi_integral = mean(ndvi_smooth)  across all DOYs in the year
```

Computed per pixel per year from the smoothed NDVI curve, then averaged across reliable
years (same `min_years` threshold as `GreenupTimingSignal`).

Normalisation: use the mean of the smoothed curve (equivalent to integral / year_length)
rather than the raw sum, so pixel-years with different observation counts are comparable.
The smoothed curve already interpolates across observation gaps, so the mean is consistent
regardless of cloud coverage — unlike a raw sum over observations.

Also compute `ndvi_integral_cv` (inter-annual CV of the per-year mean NDVI) as a
consistency measure, analogous to `peak_doy_cv`.

### Signal class

Standalone `signals/integral.py` — `NdviIntegralSignal`.

Rationale for standalone rather than adding to `greenup.py`:
- Conceptually closer to `rec_p` (a mean-state feature) than to `peak_doy` (timing)
- Single responsibility — easier to tune, disable, or replace independently
- The `_curve` parameter interface already supports curve sharing across signals without
  recomputation

The NDVI curve (`_curve`) is already computed by `GreenupTimingSignal` and
`RecessionSensitivitySignal` — the pipeline can pass it through to `NdviIntegralSignal`
without an additional smoothing pass. When added to `extract_parko_features`, the curve
should be computed once and shared.

### Parameters

```python
@dataclass
class Params:
    quality: QualityParams = field(default_factory=QualityParams)
    smooth_days: int = 30      # must match the curve used as input
    min_wet_obs: int = 5       # minimum obs per year to trust the mean estimate
    min_years: int = 3         # minimum reliable years for per-pixel stats
```

No new tunable parameters beyond what `GreenupTimingSignal` already uses — the integral
is a by-product of the same smoothed curve.

### Outputs

| Column | Description |
|---|---|
| `ndvi_integral` | Mean annual NDVI across reliable years |
| `ndvi_integral_cv` | CV of per-year mean NDVI across reliable years |
| `n_years` | Total pixel-year windows |
| `n_reliable_years` | Pixel-year windows passing `min_wet_obs` threshold |

---

## Investigation plan

### Step 1 — Implement and compute at Longreach

Implement `NdviIntegralSignal` in `signals/integral.py`. Run on the Longreach 8×8 km
scene using the cached NDVI curve from the recession-and-greenup pipeline. Produce:
- Spatial map of `ndvi_integral`
- Histograms by class (infestation bbox vs scene background)
- Histograms by class (`prob_lr` quantile labels for comparison)

### Step 2 — Correlation with existing features

Compute Pearson r between `ndvi_integral` and `[nir_cv, rec_p, re_p10, prob_lr]`.
Assess whether H2 holds — is the integral carrying independent information after
accounting for `rec_p`?

### Step 3 — Riparian separation

Compare `ndvi_integral` for the 9 riparian proxy pixels against presence and absence
class medians. If H3 holds, riparian should sit below presence despite similar `rec_p`.
(Sample of 9 is too small to validate, but the direction is informative.)

### Step 4 — Add to pipeline and re-score

Add `ndvi_integral` to `FEATURES` in `pipelines/longreach-8x8km.py`. Re-run the
classifier and compare the ranked output to the existing `prob_lr` scores. Check whether
the spatial pattern improves — particularly in the river corridor and at the dam cluster.

### Step 5 — Parameter sensitivity

The integral is largely parameter-free, but verify that the result is stable across
`smooth_days` in [15, 30, 45]. A feature that requires heavy smoothing to show separation
is less trustworthy than one that is robust across window choices.

---

## What success looks like

- `ndvi_integral` shows class separation at Longreach that is not fully explained by `rec_p`
- Adding it to the classifier improves or maintains discrimination without introducing
  spatial artefacts
- The feature is cheap enough (zero additional data, trivial computation) that even modest
  independent information justifies inclusion
- The riparian direction (Step 3) is consistent with H3, motivating future validation at a
  site with a larger riparian sample
