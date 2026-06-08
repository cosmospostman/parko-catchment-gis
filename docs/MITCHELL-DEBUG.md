# Mitchell False-Positive Debug — working document

## Current status (post-retrain, 2026-06-08)

v10 was retrained and rescored after fixing five compounding train/score pipeline bugs
(see **Resolved: Bugs 1–5** below). Results are meaningfully improved — overall false
positives are substantially reduced. Three residual problems remain:

1. **Residual mangrove/water false positives** — strands of mangroves and water patches
   still score as Parkinsonia, though at lower rates than before.
2. **Bare grass false positives** — some patches of bare/sparse grass are still
   classified as present.
3. **Missed presence bboxes** — some training presence bboxes are not being scored as
   present at all (false negatives on known-good sites).

---

## Next: diagnose the residual problems

Three candidate explanations, which can be distinguished by inspecting `prob_tam`
distributions from the `.scores.parquet` output:

### A. Threshold / calibration issue (most benign)
The model emits a continuous `prob_tam`; there is no scoring-side decision threshold.
"Present vs absent" is purely a UI display slider (`ranking.cutoff`, default 0 = show
everything). If residual FP pixels cluster at moderate probability (0.4–0.6) and known
Parkinsonia clusters at 0.8+, the fix is threshold calibration in the UI, not the model.

**Check**: pull `prob_tam` distributions from `.scores.parquet` for the three problem
categories and compare against true-positive Parkinsonia pixels. This single check
distinguishes "threshold story" from "fit story".

### B. Missed presence bboxes — quality mask or scoring gap
If presence bboxes are outright missing from scored output (not just low-probability),
they may be quality-masked out. Check whether they appear in `.scores.parquet` at all.
If present but low-probability, check whether those bboxes/years fall in geography or
seasonal windows under-represented in training (spatial generalization gap).

### C. Held-out fit on training regions
If FPs are genuinely high-confidence and missed-presence is genuinely near-zero (i.e.,
not a threshold story), check whether these problem pixels fall *inside or near* training
regions. If the model can't discriminate presence from absence on its own training
geography, the problem is upstream of generalization — a residual feature or fit issue,
not an OOD story.

**Class balance context**: training data is ~1.46:1 absence:presence at the pixel-year
level, corrected via `pos_weight = n_neg/n_pos` in `BCEWithLogitsLoss`. Val macro AUC
≈ 0.957, CVaR25 AUC ≈ 0.944 (from `outputs/models/tam-v10/train.log`). AUC is high
but there is no logged precision/recall at any fixed threshold — computing that from the
val parquets would be needed to quantify FP/FN rates operationally.

---

## Resolved: Bugs 1–5 (fixed in `12711af`, `3e918f7`; retrained + rescored `6f4ba88`)

Five compounding train/score pipeline bugs caused every pixel at inference to receive
wrong annual features, producing confident, uniform false positives across bare ground,
mangrove, and riparian cover types.

**Bug 1 — zero-substitution wiring (`12711af`)**: `annual_feat_mean`/`annual_feat_std`
were never threaded through the chunked-scoring call chain, so the model received an
all-zeros annual-feature vector for every pixel at inference.

**Bug 2 — aggregation-level mismatch (`12711af`)**: training aggregated annual features
per-pixel across *all years* (`group_by("point_id")`); inference computed them per
pixel-year. The saved z-scoring stats were calibrated on multi-year aggregates but
applied to single-year summaries.

**Bug 3 — quantile/std numerical method mismatch (`12711af`)**: training used Polars'
`interpolation="nearest"` quantiles and `ddof=1` std; inference's numba kernel uses
linear-interpolation percentiles and `ddof=0`. Fixed by aligning training to match
inference.

**Bug 4 — NaN handling mismatch (`12711af`)**: training passed raw floats into Polars
aggregations (where `NaN` behaves differently from `null`); inference's numba kernel
skips non-finite values. Fixed by converting `NaN → null` before aggregation in
`_compute_band_summaries`.

**Bug 5 — MAVI/CI_RE transcription bug (`3e918f7`)**: `add_spectral_indices` in
`analysis/constants.py` used wrong formulas for MAVI and CI_RE (using `B8A` instead of
`B08`/`B07`), introduced during a performance refactor. Fixed to match `MAVISignal`/
`CIRESignal` exactly; structural fix collapsed four independent "bands → indices"
implementations into one via the Unified Pixel Pipeline (UPP).

---

## What was ruled out earlier (training data is sound)

A full audit (`utils/bbox_audit.py`, `utils/contamination_check.py`) confirmed:
- Presence bboxes show textbook Parkinsonia spectral signature every year (2017–2025).
- Borderline riparian absence bboxes were eyeballed and confirmed non-Parkinsonia.
- Cloud/shadow contamination does not meaningfully move the features the model trains on.
- `scl_purity` quality gate is a structural no-op on this chunkstore (uniformly 1.0).

**Conclusion**: the training data and labels for Mitchell are sound. The original
false-positive problem was the pipeline bugs above, not labeling or contamination.

---

## Useful tools

- `utils/bbox_audit.py` — multi-year quality audit of training bboxes using the exact
  same `Signal.quality_mask` filter as the training pipeline.
- `utils/contamination_check.py` — quantifies whether cloud/shadow contamination
  meaningfully moves per-pixel annual features.
