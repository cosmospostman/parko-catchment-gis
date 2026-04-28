# Local Anomaly Feature — Design Notes

## Concept

A Parkinsonia pixel embedded in a different vegetation matrix (grassland, savanna,
native woodland) will be a persistent spectral outlier relative to its immediate
neighbourhood. The deviation of a pixel's temporal trajectory from the local
neighbourhood mean is a landscape-relative signal that could transfer across climate
zones — even when absolute spectral values differ between arid and monsoonal contexts,
the *contrast* between Parkinsonia and its surroundings may be consistent.

This is complementary to the temporal coherence feature (see TEMPORAL-COHERENCE.md):
- **Coherence** detects dense monoculture patches — "this pixel is similar to its neighbours"
- **Local anomaly** detects Parkinsonia embedded in a different matrix — "this pixel
  is different from its neighbours"

## Signal logic

For each pixel, compute how much its per-band time series deviates from the mean
time series of its k nearest neighbours:

```
anomaly(pixel, band) = mean_over_time( |pixel_ts(band) - neighbourhood_mean_ts(band)| )
```

A Parkinsonia pixel in a grassland matrix would show persistent positive deviation
in NIR/red edge bands (higher NIR than surrounding grass) and potentially different
timing in NDVI green-up. A grassland pixel surrounded by other grassland would have
near-zero anomaly.

## Why this could transfer across climate zones

Unlike absolute spectral values, the local anomaly is relative to the surrounding
landscape. Even if arid Parkinsonia has lower absolute NDVI than monsoonal Parkinsonia,
both may be persistently higher NIR than their respective backgrounds. The contrast
signal is what transfers, not the absolute value.

## One-sided constraint

Local anomaly is also a one-sided signal — but in a different direction to coherence:

- **High anomaly in specific bands** → possibly Parkinsonia (or any spectrally
  distinct feature — see confounds below)
- **Low anomaly** → pixel is consistent with surroundings → likely background
  vegetation, but could also be Parkinsonia in a dense monoculture (where neighbours
  are also Parkinsonia)

The feature should not be used as negative evidence. Low anomaly does not mean absence.

## Combined signal with coherence

The two features are most powerful together:

| Coherence | Anomaly | Interpretation |
|-----------|---------|----------------|
| High | High | Coherent patch standing out from surrounding landscape → strong Parkinsonia signal |
| High | Low | Coherent patch blending with surroundings → likely background vegetation |
| Low | High | Isolated distinctive pixel → possible isolated plant, edge pixel, or noise |
| Low | Low | Consistent with surroundings → likely background vegetation |

The high coherence + high anomaly combination is the most discriminative signal for
large infestations in a contrasting matrix.

## Confounds

High local anomaly will also fire for:
- Bare soil or rock outcrops in vegetation matrix
- Water bodies at patch edges
- Agricultural fields adjacent to native vegetation
- Any spectrally distinct non-Parkinsonia feature

The model must learn which *kind* of anomaly is informative — which requires anomaly
to be computed per-band (and ideally per-season) rather than as a single scalar, so
the model can distinguish "high NIR anomaly in May" from "low NDVI anomaly year-round."

## Aggregation design

Per pixel, per band:

1. Compute neighbourhood mean time series (mean across k nearest neighbours, per DOY bin)
2. Compute signed deviation: `pixel_ts - neighbourhood_mean_ts` per DOY bin
3. Summarise across time — several options:
   - **Mean absolute deviation** — overall magnitude of anomaly, unsigned
   - **Mean signed deviation** — direction matters (positive = brighter than surroundings)
   - **Seasonal deviation** — separate wet-season and dry-season means, capturing
     phenological anomaly timing
   - **Peak deviation DOY** — when in the year is the pixel most anomalous

Signed deviation is probably most informative — Parkinsonia being brighter than
surroundings in NIR is a different signal to being darker, and the model should be
able to use the direction.

Seasonal decomposition (wet vs dry season deviation) adds the temporal dimension and
may be where the transfer signal lives — Parkinsonia's dry-season retention of green
biomass relative to deciduous surroundings could be consistent across climate zones
even when absolute values differ.

## Critical blocker: buffer zone pixels

Same as temporal coherence — the current training collector only fetches pixels within
labelled bboxes. Neighbourhood mean computed from within-bbox pixels is meaningless
for local anomaly because all neighbours share the same label. Buffer zone pixels
(unlabelled, fetched from surrounding landscape) are a prerequisite.

See TEMPORAL-COHERENCE.md § Critical blocker for the buffer zone implementation plan.

## Implementation path

Shares the buffer zone prerequisite with temporal coherence — implement both features
together once the collector is extended:

1. **Extend training collector** with buffer zone (shared with coherence feature)
2. **Compute neighbourhood mean time series** per pixel per band using buffer pixels
3. **Compute signed deviation** from neighbourhood mean, per band, per DOY bin
4. **Summarise** as wet-season mean deviation and dry-season mean deviation per band
   (2 × 13 bands = 26 features, or a subset of key bands)
5. **Add to `compute_global_features`** alongside coherence features
6. **Run diagnostic** — plot anomaly score distributions for presence vs absence
   per site before training

## Open questions

- Which bands carry the most transferable anomaly signal? Red edge (B06, B07, B8A)
  and NIR (B08) are the strongest candidates based on spectral analysis
- Wet/dry season split or finer temporal resolution (monthly anomaly)?
- Should anomaly be computed relative to the immediate neighbourhood (r=1 or r=2)
  or a larger landscape window? A larger window captures more background context but
  is more expensive and may include other Parkinsonia pixels in dense infestations
- For dense monoculture patches, interior pixels will have low anomaly (all neighbours
  are also Parkinsonia). Only edge pixels will show high anomaly relative to the
  surrounding matrix. This is a limitation for large infestations but may be acceptable
  if coherence handles those cases.

## Relationship to existing features

`nir_cv` (NIR coefficient of variation) already captures something related — high
temporal variability in NIR. Local anomaly adds the spatial dimension: not just "does
this pixel vary a lot over time" but "does it vary differently to its neighbours."
The two are complementary and should both be retained.
