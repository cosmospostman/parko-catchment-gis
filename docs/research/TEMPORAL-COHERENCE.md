# Temporal Coherence Feature — Design Notes

## Concept

Parkinsonia infestations form spectrally coherent patches — every pixel in a dense
stand is doing the same thing seasonally (green-up, leaf-drop, dry-season stress) at
the same time. The feature asks, for each pixel: **"is there at least one pixel within
30m that has an almost identical time series to mine?"**

This exploits phenological homogeneity as a proxy for monoculture. A Parkinsonia stand
is even-aged, single-species, and typically established from a single invasion event —
every plant responds identically. Mixed native woodland has species heterogeneity,
structural variation, and asynchronous individual responses even within the same species,
so inter-pixel correlations are moderate even among similar neighbours.

The feature is added as per-pixel global scalars alongside the existing `nir_cv`,
`rec_p`, `dry_ndvi` etc. — no architectural changes to TAM needed.

## What the feature is and is not doing

**It is:** detecting pixels that belong to a spectrally homogeneous patch — any
monoculture, not just Parkinsonia. Bare dirt, buffel grass, and dense native stands can
also score high.

**It is not:** directly discriminating Parkinsonia from other land cover. That job
belongs to the existing spectral-temporal features. Coherence just adds "does this pixel
belong to a homogeneous patch?" and the model learns to weight it in combination with
the rest of the feature set.

This means low coherence should never count as absence evidence — it just means the
feature is uninformative for that pixel. The feature is one-sided: high coherence is a
weak positive signal; low coherence is silence.

## Aggregation design

Within a neighbourhood of radius r=2 (5×5 = 24 neighbours):

1. Compute pairwise temporal Pearson correlation between target pixel and each neighbour,
   per band, over the full multi-year time series
2. Take the **95th percentile** of all 24 pairwise correlations — effectively the single
   most-correlated neighbour, asking "does at least one neighbour track me almost
   perfectly?"
3. Produces one feature per band: `tc_p95_r2_B02`, `tc_p95_r2_B08`, `tc_p95_r2_NDVI`,
   etc. (13 features total)

**Why 95th percentile:** empirical comparison of Norman Road presence bboxes
(compare_nr_p1_p6, compare_nr_p1_p7, compare_nr_p6_p7) shows max_sep between any two
Parkinsonia presence regions of 0.0003–0.0019 across all bands — essentially zero.
Pixels within a dense stand are so tightly synchronised that even a very selective
threshold should fire reliably. Mixed woodland neighbours will drop off steeply at the
high end of the correlation distribution. Native species likely cannot match this degree
of synchrony due to structural heterogeneity, individual variation in phenological
response, and asynchronous leaf flush (eucalypts in particular).

**Why per-band:** gives the model more signal to work with than a single aggregated
score. Different bands may show different coherence patterns across land cover types.

## What we know doesn't work

- **Mean coherence across all neighbours** — diluted by mixed-class neighbourhoods
- **Fixed 3×3 neighbourhood (r=1)** — misses riparian strands which may only be 1–2
  pixels wide; Parkinsonia neighbours get outnumbered by bank/water pixels
- **Treating low coherence as negative evidence** — one-sided signal; clip at zero after
  normalisation so the model cannot learn "low coherence → absence"
- **Subset/threshold gating** (keep only neighbours above correlation threshold, report
  subset size) — encodes density as a proxy for presence, which breaks for sparse or
  early-stage infestations
- **Residual coherence** (subtract regional mean before correlating) — unnecessary
  because the model already handles discrimination between Parkinsonia and other
  homogeneous land cover via existing features
- **Directional coherence** — adds complexity without clear benefit given percentile
  aggregation already handles linear riparian strands adequately

## Relationship to probability propagation

Coherence handles dense patches. Isolated individual Parkinsonia plants will have low
coherence (uninformative, not misleading) and are handled by a separate downstream
propagation step — see `docs/PROPAGATION.md`.

This cleanly separates two failure modes:
- TAM + coherence: find dense patches where temporal synchrony is strong
- Propagation: extend detections to isolated individuals near confirmed anchors

## Critical blocker: buffer zone pixels

The current training pixel collector only fetches pixels within labelled bboxes. Every
pixel's neighbours are therefore other labelled pixels from the same region — coherence
measures within-bbox homogeneity rather than landscape-scale neighbourhood structure.
All pixels within a single bbox show coherence ≈ 1.0 regardless of land cover type.

**The buffer zone extension is a prerequisite for this feature to work.** Each bbox
needs a surrounding zone of unlabelled pixels fetched at collection time, wide enough
to support r=2 plus margin (at minimum r=3, ~30m, but wider for riparian corridors).
These buffer pixels have no training label but provide the neighbourhood context for
coherence computation.

## Implementation path

In order of dependency:

1. **Extend the training collector** to fetch a buffer zone of unlabelled pixels
   around each bbox (at minimum r=3 pixels, ~30m)
2. **Implement 95th percentile coherence** in `tam/core/global_features.py` alongside
   existing global features — compute per-band correlation within r=2
3. **Clip at zero after normalisation** — enforce the one-sided constraint so the
   model cannot learn "low coherence → absence"
4. **Run diagnostic** (`tam/viz_temporal_coherence.py`) with buffer pixels to verify
   presence/absence distributions separate before committing to a training run

## Open questions

- How wide does the buffer need to be? At minimum r=3 (~30m) for neighbourhood
  computation, but riparian corridors may benefit from wider buffers
- Whether the feature helps at all in practice depends on infestation patch size —
  Norman Road and Frenchs likely yes, sparse arid sites (Stockholm, Wongalee) probably
  not; the feature will be uninformative rather than misleading for those sites
- The 95th percentile choice should be validated empirically once buffer pixels are
  available — sweep over {75, 90, 95} against known labels before committing

## Diagnostic tooling

- `tam/viz_temporal_coherence.py` — computes and plots coherence distributions per
  region or experiment, with Mann-Whitney significance test
- `tam/viz_bbox_compare.py` — spectral/temporal comparison between two regions or
  arbitrary bboxes, useful for verifying signal before adding regions to training
