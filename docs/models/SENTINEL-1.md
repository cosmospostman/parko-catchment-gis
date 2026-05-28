# Sentinel-1 SAR Backscatter — Design Notes

## Motivation

The spectral analysis (see docs/SWEEP.md) revealed that Parkinsonia's optical
signature is highly context-dependent — arid Parkinsonia and monsoonal Parkinsonia
look spectrally different in S2 bands, making a single generalised classifier
difficult to train. The core problem is that S2 reflectance is dominated by
landscape-level signals (soil brightness, background vegetation density, rainfall
regime) that swamp the species-level signal.

Sentinel-1 SAR backscatter is sensitive to canopy structure rather than reflectance
— specifically to woody stem density, branch architecture, and moisture content of
structural elements. These properties are more intrinsic to the plant and less
dependent on the surrounding landscape, making SAR a candidate for a more
transferable Parkinsonia signature.

## Why SAR could help

**Woody stem density:** Parkinsonia has a distinctive open, multi-stemmed architecture
with fine pinnate foliage. SAR backscatter (particularly C-band VV and VH
polarisations from S1) responds to the dielectric properties and geometry of woody
elements at scales of a few centimetres — roughly matching Parkinsonia's stem and
branch diameter. Dense woody stems produce characteristic double-bounce and volume
scattering signatures that differ from grass (low backscatter) and dense eucalypt
canopy (different geometry).

**Foliage structure:** Parkinsonia's bipinnate leaves produce a distinctive volume
scattering signature when wet. The cross-polarisation ratio (VH/VV) is sensitive to
canopy complexity and could distinguish Parkinsonia's fine-leafed structure from
broader-leafed native species.

**Moisture independence:** Unlike optical indices (NDVI, EVI), SAR backscatter is
not affected by cloud cover and responds to structural moisture rather than
chlorophyll. This means SAR observations are available year-round including during
the wet season when S2 is obscured — potentially providing phenological information
in the observation gaps that currently limit the temporal model.

**Climate-zone invariance:** Canopy structure is a plant-level property that is more
consistent across climate zones than absolute reflectance. A Parkinsonia stand in
arid Queensland and one in monsoonal Cape York will have similar woody architecture
even if their optical signatures differ. This is the strongest argument for SAR as
a generalisation feature.

## Sentinel-1 data characteristics

- **Spatial resolution:** 10m (IW mode GRD) — matches S2 pixel size exactly
- **Temporal resolution:** ~6-12 day revisit over Australia
- **Polarisations:** VV (vertical transmit, vertical receive) and VH (vertical
  transmit, horizontal receive)

### Per-observation features (instantaneous)

- **VH backscatter (dB)** — sensitive to volume scattering from canopy foliage and
  woody stems. Parkinsonia's fine pinnate foliage produces distinctive volume scattering.
- **VV backscatter (dB)** — sensitive to surface and double-bounce scattering from
  woody stems and soil. Higher in open/sparse canopy.
- **VH−VV ratio (dB)** — canopy complexity relative to surface roughness. High ratio
  indicates dense, complex canopy; low ratio indicates bare soil or smooth surfaces.
- **Radar Vegetation Index (RVI):** `4 × VH / (VV + VH)` — normalises out soil
  moisture effects, directly targets vegetation density. Well established for woody
  vegetation monitoring. Ranges 0 (bare) to 1 (dense canopy). Strong candidate for
  a climate-zone-invariant feature.
- **VH × VV** — product sensitive to combined volume and double-bounce scattering;
  sometimes more discriminative than either band alone for woody structure.

### Temporal/global features (derived from time series)

- **Mean VH / mean VV** — baseline backscatter level over the observation period.
  Captures average canopy density independent of seasonal fluctuation.
- **Temporal std of VH** — seasonal variability in volume scattering. Grassland
  fluctuates strongly with rainfall and senescence; dense woody canopy is more stable.
  Low std → structurally stable canopy → evidence of woody perennial vegetation.
- **Wet/dry season contrast (VH wet − VH dry)** — grass shows large contrast as it
  senesces in the dry season; Parkinsonia and native woodland retain more canopy.
  Could be a useful phenological proxy where optical data is cloud-obscured.
- **Mean RVI** — temporal mean of the Radar Vegetation Index; integrates vegetation
  density across the full observation period.

### Orbit geometry over Australia

Australia is predominantly covered under **IW mode, dual polarisation (VV+VH), descending passes**. This is confirmed by the ESA Sentinel-1 observation scenario map ([2019 reference](https://sentinels.copernicus.eu/documents/247904/3944045/Sentinel-1-Mode-Polarisation-Observation-Geometry-2019.jpeg)) and is consistent with STAC queries over training sites — e.g. Quaids (Cape York) shows 231 scenes across 2017–2024, all descending, all relative orbit 162, with zero ascending passes.

The practical implication is that **orbit-direction mixing is not a significant concern for Australian sites** — a given location is typically covered by a single relative orbit in a single direction. Per-pixel z-score normalisation (applied to VH and VV) is therefore sufficient to remove the incidence-angle offset introduced by that fixed orbit geometry. If a future site does show ascending/descending mixing, the `sat:orbit_state` and `sat:relative_orbit` fields are already captured in the STAC query and stored in the collector output, so orbit-stratified normalisation could be added if needed.

### Preprocessing note

The raw S1 GRD COGs from Element84 STAC are in linear power units and have no
embedded CRS or affine transform — georeferencing must be reconstructed from the
STAC item's `proj:transform` property. Conversion to dB (10 × log₁₀) is applied
before any analysis. Thermal noise removal and terrain correction are not applied
in the current diagnostic scripts — this is acceptable for a signal check but
should be addressed before production integration.

## Integration with TAM

SAR bands would be added to the per-observation feature vector alongside existing
S2 bands, provided S1 and S2 observations can be temporally co-registered. Since
S1 and S2 have different revisit schedules, options are:

1. **Nearest-in-time matching** — for each S2 observation, find the nearest S1
   acquisition within a time window (e.g. ±7 days) and append S1 bands. Straightforward
   but introduces temporal mismatch noise.

2. **Separate temporal streams** — run two parallel attention encoders (one for S2,
   one for S1) and fuse the pooled representations before the classification head.
   More expressive but more complex architecturally.

3. **S1-derived global features only** — compute temporal statistics of S1 backscatter
   (mean VH, VH/VV ratio, seasonal std) as additional global features rather than
   per-observation bands. Simpler integration, no temporal co-registration needed,
   slots directly into the existing global features pipeline.

Option 3 is the lowest-risk starting point — compute S1 summary statistics per pixel
over the training period and add them as global features alongside `nir_cv`, `rec_p`
etc. If those features improve AUC, the more complex integration (options 1 or 2)
becomes worth pursuing.

## Expected signal

- **Dense Parkinsonia stands:** elevated VH backscatter (volume scattering from
  fine foliage), moderate VV, high VH/VV ratio
- **Grassland:** low VH and VV, low VH/VV ratio — strong contrast with Parkinsonia
- **Native woodland/eucalypt:** high VH (dense canopy) but different VH/VV ratio
  due to different branch geometry — potentially discriminable from Parkinsonia
- **Bare soil:** low VH, higher VV (surface scattering) — easy negative
- **Water:** low VH and VV, specular reflection — easy negative

The most valuable discrimination is Parkinsonia vs native woodland — which is where
S2 optical features struggle most. SAR's sensitivity to canopy geometry rather than
chlorophyll may provide separation where optical bands cannot.

## Data access

Sentinel-1 GRD data is available via:
- **Element84 STAC** (same pipeline as S2) — S1 data is in the same catalog
- **Google Earth Engine** — S1 GRD collection with preprocessing applied
- **ASF DAAC** — full archive

Preprocessing required: thermal noise removal, radiometric calibration, terrain
correction (using DEM), conversion to dB scale. GEE applies these automatically;
STAC access would require a preprocessing step.

## Open questions

- Does S1 data coverage over the training sites have sufficient temporal density
  to compute meaningful statistics? (~6-12 day revisit should give 30+ observations
  per year in most areas
- Are there known SAR signatures for Parkinsonia in the literature? Worth checking
  before implementation
- How much does soil moisture variation affect S1 backscatter at these sites? Wet
  season vs dry season soil moisture could drive temporal variance that masks the
  structural signal
- Would S1 alone achieve reasonable AUC on Norman Road where S2 fails? This would
  be the clearest test of the hypothesis

## Implementation priority

Lower priority than the buffer zone extension (needed for temporal coherence and
local anomaly features) and lower priority than resolving the training data quality
issues identified in the spectral analysis. SAR is most valuable once the S2-based
model has been pushed to its ceiling — at that point SAR provides genuinely new
information rather than compensating for data quality problems.

First step when ready: collect S1 VH and VV time series for a subset of Norman Road
presence and absence pixels and run the spectral comparison (`tam/viz_bbox_compare.py`
extended for SAR bands) to verify whether the expected backscatter separation exists
in the data before committing to full pipeline integration.
