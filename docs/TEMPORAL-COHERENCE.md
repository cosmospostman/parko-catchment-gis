# Temporal Coherence Feature — Design Notes

## Concept

Parkinsonia infestations form spectrally coherent patches — every pixel in a dense
stand is doing the same thing seasonally (green-up, leaf-drop, dry-season stress) at
the same time. Heterogeneous native vegetation (woodland, savanna, riparian mix) has
lower within-neighbourhood correlation because different species and structures respond
differently to rainfall and season.

The idea: compute how correlated a pixel's multi-band time series is with its nearest
neighbours, and use that coherence score as a global feature alongside the existing
`nir_cv`, `rec_p`, `dry_ndvi` etc.

## What we know works

- The concept is sound for dense, spatially aggregated infestations
- Percentile coherence (e.g. 75th percentile) within a wider radius (r=2 or r=3
  pixels) is the right aggregation — handles linear riparian strands, doesn't require
  choosing an absolute neighbour count, scales naturally with neighbourhood size
- The existing global features pipeline provides a clean integration point — no
  architectural changes needed
- Per-band coherence (all 13 bands) gives the model more signal to work with than a
  single aggregated score

## What we know doesn't work

- **Mean coherence across all neighbours** — diluted by non-Parkinsonia pixels in
  mixed or linear patches
- **Fixed 3×3 neighbourhood (r=1)** — misses riparian strands which may only be 1-2
  pixels wide; Parkinsonia neighbours get outnumbered by bank/water pixels
- **Treating low coherence as negative evidence** — coherence is a one-sided signal.
  High coherence → possibly Parkinsonia. Low coherence → could be anything including
  isolated Parkinsonia. The feature should be clipped at zero after normalisation so
  the model cannot use it as absence evidence.

## Aggregation design

Within a neighbourhood of radius r (e.g. r=2 → 5×5 = 24 neighbours):

1. Compute pairwise temporal correlation between target pixel and each neighbour,
   per band
2. Take the **Nth percentile** (e.g. 75th) of all pairwise correlations — this finds
   the coherent subset without being diluted by non-Parkinsonia neighbours and without
   requiring an absolute k
3. Produces one feature per band per radius, e.g. `tc_p75_r2_NDVI`, `tc_p75_r2_B08`

The percentile choice has an intuitive interpretation: "what coherence level does at
least 25% of this pixel's neighbourhood match or exceed?" This is robust to mixed
patches and linear strands.

## Limitations

- **Requires spatially aggregated infestations.** Isolated plants or very sparse
  infestations will have low coherence even if correctly labelled — the feature is
  uninformative for these cases, not misleading (one-sided).
- **Riparian strands** are the hardest case for fixed-radius neighbourhoods. A pixel
  in a 2-pixel wide riparian strand may only have 2-4 Parkinsonia neighbours out of
  24 in a 5×5 window. Percentile aggregation handles this better than mean, but very
  narrow strands remain challenging.
- **Graph-based coherence** (connected components of high-correlation pixels) would
  handle arbitrary infestation shapes naturally but assumes spatial connectivity and
  is significantly more complex to implement.
- **Does not help for sub-pixel mixing** — if Parkinsonia and other species are
  interleaved within individual pixels, the per-pixel time series is a spectral blend
  and coherence with neighbours will be moderate regardless of Parkinsonia proportion.

## Critical blocker: buffer zone pixels

The current training pixel collector only fetches pixels within labelled bboxes. Every
pixel's k nearest neighbours are therefore other labelled pixels from the same region
— the coherence score measures within-bbox homogeneity rather than landscape-scale
neighbourhood structure. This was confirmed by the diagnostic runs: all pixels within
a single bbox show coherence ≈ 1.0 regardless of land cover type.

**The buffer zone extension is a prerequisite for this feature to work.** Each bbox
needs a surrounding zone of unlabelled pixels fetched at collection time, wide enough
to support the chosen radius (at least r=3, i.e. 30m, but wider is better for
riparian corridors). These buffer pixels have no training label but provide the
neighbourhood context for coherence computation.

## Implementation path

In order of dependency:

1. **Extend the training collector** to fetch a buffer zone of unlabelled pixels
   around each bbox (at minimum r=3 pixels, ~30m)
2. **Implement percentile coherence** in `tam/core/global_features.py` alongside
   existing global features — compute per-band 75th percentile correlation within
   r=2 and r=3 radii
3. **Clip at zero after normalisation** — enforce the one-sided constraint so the
   model cannot learn "low coherence → absence"
4. **Run diagnostic** (`tam/viz_temporal_coherence.py`) with buffer pixels to verify
   presence/absence distributions separate before committing to a training run

## Open questions

- How wide does the buffer need to be? At minimum r=3 (30m) for neighbourhood
  computation, but riparian corridors may benefit from wider buffers
- Whether the feature helps at all in practice depends on infestation patch size
  relative to 10m pixel resolution — Norman Road and Frenchs likely yes, sparse
  arid sites (Stockholm, Wongalee) probably not
- Directional coherence (N-S, E-W, diagonal axes separately) could better capture
  linear riparian strands but adds complexity and more features

## Diagnostic tooling

- `tam/viz_temporal_coherence.py` — computes and plots coherence distributions per
  region or experiment, with Mann-Whitney significance test
- `tam/viz_bbox_compare.py` — spectral/temporal comparison between two regions or
  arbitrary bboxes, useful for verifying signal before adding regions to training
