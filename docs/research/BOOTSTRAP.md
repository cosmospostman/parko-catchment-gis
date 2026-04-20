# Bootstrap Strategy: Seeding the Classifier from a Single Confirmed Patch

## Context

Science validation tests were run against a confirmed Parkinsonia infestation
on the coastal beach-ridge strand north of Pormpuraaw (Mitchell River mouth,
Gulf Plains QLD, ~141.58°E 14.66°S). This is the first real-data validation of
the spectral time series approach. Results diverged significantly from the
hypotheses in OVERVIEW.md, but a strong discriminating signal was found.

---

## What We Found

### The flowering-flush hypothesis does not hold at this site

The `flowering_index` (re_slope + nir_swir / 2) was designed expecting
Parkinsonia to appear as a dense flowering canopy: high red-edge slope (B07 >
B05) and high NIR relative to SWIR (B08 > B11). At the Pormpuraaw patch this
assumption fails:

| Component | Presence | Absence | Expected direction |
|---|---|---|---|
| re_slope (B07−B05)/(B07+B05) | +0.11 | +0.22 | pres > abs |
| nir_swir (B08−B11)/(B08+B11) | −0.31 | −0.17 | pres > abs |
| flowering_index (combined) | −0.10 | +0.04 | pres > abs |

The `nir_swir` term dominates and is negative at both labels because B11
(SWIR, 1610nm) runs at ~0.35 reflectance across the site — far above what a
vegetated canopy would produce. This is a sandy beach-ridge substrate signal,
not canopy. It swamps any flowering-flush response.

**This is not a code error.** The formula computes correctly. The mismatch is a
site-assumption failure: the index was designed for a mesic/riparian context
where Parkinsonia canopy is dense enough to suppress substrate reflectance. On
this open beach-ridge strand in dry season the substrate dominates.

### A strong discriminating signal exists — but it is inverted

The infestation site is spectrally *sparser* than the surrounding open
floodplain grassland during dry season (Jul–Oct):

| Index | Presence median | Absence median | p-value | Direction |
|---|---|---|---|---|
| NDVI | 0.243 | 0.450 | ~0 | pres < abs |
| NBR | −0.316 | −0.100 | ~0 | pres < abs |
| NDRE | 0.109 | 0.258 | ~0 | pres < abs |
| BSI (Bare Soil Index) | 0.266 | 0.092 | ~0 | **pres > abs** |
| FI_inv (−flowering_index) | 0.106 | −0.083 | ~0 | **pres > abs** |

All five indices discriminate at p≈0 (Mann-Whitney U, n=40+40 per label,
pooled Jul–Oct 2021/2022/2025).

### Ecological interpretation

The absence zones are denser floodplain grassland/sedge that maintains
continuous green cover into the dry season, masking the soil. The Parkinsonia
infestation sits on open sandy beach-ridge: the canopy is partially deciduous
in dry season, the understory is sparse, and bright sandy substrate is visible
between the trees — producing elevated SWIR, suppressed NIR, and high BSI.

This is a real and detectable feature of this specific site. Whether it
generalises to Parkinsonia infestations on other substrates (riverine clay
floodplains, rocky slopes, denser savanna) is unknown and is the central
question for the bootstrap phase.

---

## Current Pipeline Blockers

Two issues prevent training on the existing data:

**1. Feature representation mismatch.** All current features
(`peak_value`, `peak_doy`, `spike_duration`, etc.) are derived by detecting a
positive peak above `FLOWERING_THRESHOLD = 0.15`. The flowering index is
uniformly negative at this site — zero points produce waveform features, zero
rows reach the RF.

**2. Waveform gate too strict.** `extract_waveform_features` requires
`min_years=3` years of detected peaks. Even with a corrected index, with only
3 years of imagery (2021, 2022, 2025) there is no margin — one cloudy season
eliminates a point entirely.

---

## Recommended Direction

### Immediate: replace waveform features with per-season band statistics

Rather than detecting a peak event, summarise each point's dry-season (Jul–Oct)
spectral distribution directly. Candidate feature set:

- Median and IQR of: NDVI, NBR, NDRE, BSI, re_slope, nir_swir — per season
- Inter-annual variance of NDVI median (greenness stability signal from OVERVIEW.md)
- Optional: per-band median for B04, B05, B07, B08, B8A, B11

This sidesteps the threshold/peak machinery entirely, captures the signal that
is actually present, and lets the RF select which statistics matter. It is also
robust to the 3-year data limitation — a point only needs 1 usable season to
contribute a row.

### Bootstrap loop

1. Train RF on current 80 points (this patch only)
2. Run inference over broader Gulf Plains bbox (~136–142°E, 12–17°S)
3. Inspect candidate high-probability patches against Queensland Globe imagery
4. For each confirmed patch: digitise presence/absence zones, regenerate points,
   add to training set
5. Retrain and repeat

**Key constraint:** absence zones for each new patch should be chosen relative
to that patch's local backdrop — not imported from Pormpuraaw. The discriminating
contrast at each site is between the infestation and its immediate surroundings.
A riverine Parkinsonia stand on clay floodplain will need absence points from
that same floodplain, not from coastal grassland.

### Signals to watch as the training set diversifies

As new patches are added on different substrates, check whether:

- BSI remains elevated at presence (substrate-dependent — may flip on dark clay)
- NDVI suppression persists (more likely to generalise — canopy openness signal)
- NBR suppression persists (stress/litter signal — should generalise to most substrates)
- Inter-annual NDVI variance differs (greenness-persistence signal from OVERVIEW.md)

If NDVI suppression and NBR are consistently discriminating across multiple
substrate types, they are the strongest candidates for the generalised feature
set. BSI should be treated as a site-specific feature until proven otherwise.

### Longer term: revisit the flowering-flush hypothesis

The flowering flush signal may exist at sites where Parkinsonia canopy is dense
enough to suppress substrate SWIR — e.g. dense riverine infestations on dark
clay soils. It should not be abandoned, but it cannot be the primary signal for
a bootstrap classifier trained on this beach-ridge patch. Once training data
spans multiple substrate types, it will be possible to test whether any site
shows a positive flowering-index peak.
