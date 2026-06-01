# Probability Propagation — Design Notes

## System architecture

Detection is split into two stages with explicitly separated responsibilities:

**Stage 1 — TAM**: find dense Parkinsonia patches via spectral-temporal classification.
Dense patches are where the temporal signal is richest, training data is most abundant,
and management impact is highest. This is the focused objective of the model.

**Stage 2 — Propagation**: extend patch detections to isolated individuals via temporal
similarity to confirmed anchors from Stage 1.

Each stage has a distinct failure mode, a distinct diagnostic, and can be tuned
independently without affecting the other.

## Why TAM targets dense patches

An isolated Parkinsonia plant has no coherent neighbours. The temporal coherence feature
(see `docs/TEMPORAL-COHERENCE.md`) is uninformative for sparse or early-stage
infestations — not misleading, just silent. Trying to make the model detect isolated
individuals from spectral-temporal features alone would require it to learn a different,
weaker signal and would dilute the patch-detection objective.

Accepting this constraint clarifies the model's purpose and makes it testable: does TAM
reliably find dense stands? That question has a clean answer from labelled sites.

## Propagation concept

Once TAM produces a score map, high-confidence pixels ("anchors") seed a propagation
step that can reach isolated individuals:

1. Threshold the TAM score map to identify anchor pixels (e.g. `prob_tam > 0.8`)
2. For each non-anchor pixel within radius R, compute temporal correlation with each anchor
3. If correlation exceeds a threshold, propagate a fraction of the anchor's probability —
   weighted by distance, correlation, or both

This is a soft nearest-neighbour operation using temporal similarity as the distance
metric, seeded by the model's high-confidence predictions.

## Key design parameters

- **Anchor threshold**: confidence cutoff for TAM scores used as seeds. Higher = fewer
  but more reliable anchors; lower = more reach but more false positive propagation.
- **Radius R**: search radius for propagation. Larger than the coherence radius — looking
  for isolated individuals near a stand, not immediate neighbours. Likely 50–100m rather
  than the 30m used for coherence.
- **Correlation threshold**: minimum temporal similarity for probability transfer. Tuned
  diagnostically against known isolated-plant labels.
- **Propagation weight**: how much of the anchor's probability transfers. Can be
  distance-weighted, correlation-weighted, or a fixed fraction.

## Failure modes

**Stage 1 failure** (TAM misses a known dense stand): diagnose as a training data or
feature problem. Does not affect propagation design.

**Stage 2 failure** (propagation produces false positives): tighten the correlation
threshold or raise the anchor confidence cutoff. Does not require retraining TAM.

**Propagation is only as good as the anchor set.** False positive anchors from TAM will
propagate. This step works downstream of a well-calibrated model, not as a substitute
for one.

## Relationship to temporal coherence

Coherence (Stage 1 feature) and propagation (Stage 2 post-processing) handle the two
failure modes of Parkinsonia detection:

- Coherence helps TAM be more confident about dense patches by adding neighbourhood
  context to the spectral-temporal signal. It reinforces the model where the model is
  already strongest.
- Propagation extends detections outward from confirmed patches to isolated individuals
  that share the Parkinsonia phenology but lack coherent local neighbourhoods.

Neither step is trying to detect Parkinsonia alone — they are downstream of and dependent
on the core spectral-temporal classification.
