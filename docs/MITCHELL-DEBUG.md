# Mitchell False-Positive Debug — investigation starting point

## Symptom

After training v10 on the new Mitchell regions, scoring confidently picks bare ground,
mangrove, and non-Parkinsonia riparian vegetation as Parkinsonia — despite the model
having been shown exactly these kinds of pixels as `absence` during training.

## What we've already ruled out

A multi-year quality audit of every Mitchell training bbox (`docs/MITCHELL-BBOX-AUDIT.md`,
produced by `utils/bbox_audit.py`, refactored to filter pixels using the exact same
`Signal.quality_mask` gate the training pipeline applies — so the audit measures the same
data the model trains on) found:

- **Presence bboxes look correct in every year (2017–2025)**: strong wet-season green-up
  (NDVI peak 0.71–0.97), deep wet–dry seasonal swing (Δ 0.49–0.83), and consistent VH/VV
  (−3.9 to −5.4 dB) — the textbook deciduous-shrub Parkinsonia signature, present every
  single year of every bbox. The audit's "✗ fail" verdicts on these bboxes are a
  thresholding artefact (IQR ceiling of 0.12 and dry-NDVI-floor lower bound of 0.20 were
  calibrated from a single 2025 snapshot and are too tight for the real multi-year range),
  not evidence of bad sites or bad labels.
- **Borderline absence bboxes (riparian) were eyeballed by the user on imagery** for the
  specific years where their seasonal metrics nudged into presence-like territory
  (`mitchell_absence_riparian_1` 2022, `mitchell_val_absence_riparian_1` 2017/2019). The
  user confirmed the vegetation is predominantly non-Parkinsonia, with at most minor
  subpixel mixing — expected at a riparian/Parkinsonia ecological boundary, not a
  labeling error.
- **Cloud/shadow contamination is real in the raw timeseries but does not meaningfully
  affect the features the model actually trains on.** `utils/contamination_check.py`
  reproduces v10's real feature-extraction path (`_compute_band_summaries` in
  `tam/core/train.py` — per-pixel `[p5, p95, std]` of NDVI/MAVI across the full multi-year
  S2 timeseries) and compares it computed with vs. without the audit's flagged
  "likely cloud/shadow" dates dropped. Result: **zero feature movement on all 3 sampled
  presence bboxes**, and only a small, localized effect (mean Δ≈0.002, max|Δ|≈0.055,
  affecting ~1–2% of pixels) on the one absence bbox where contamination was real and
  substantial (`mitchell_absence_mangrove_3`).
- **The `scl_purity` quality gate is a structural no-op on this chunkstore** — uniformly
  1.0 across every sampled region/year/tile combination, so it neither helps nor hurts.

In short: **the training data and labels for Mitchell appear sound.** The false-positive
problem is unlikely to be explained by bad bboxes, mislabeling, or unfiltered noise in the
features the model was shown.

## ROOT CAUSE FOUND & FIXED: two compounding train/inference "annual feature" bugs

(Note: these were called "global features" during the investigation — renamed to
"annual features" in the fix, since they are per-(pixel, year) statistical summaries,
not global pixel-level constants. See the rename rationale in
`tam/core/annual_features.py:1` and the implementation plan for the full identifier
rename table.)

Confirmed by reading the code (not guessing from naming) — **two distinct, compounding
bugs**, both now fixed:

### Bug 1 — zero-substitution wiring bug (primary cause, fixed)

The exact `tam.pipeline score ... --out-parquet` path
(`score_tiles_chunked` → `_score_tile_worker` → `score_tile_year` →
`score_pixels_chunked` → `_preprocess`) never threaded `annual_feat_mean`/
`annual_feat_std` through `score_tiles_chunked`'s signature, the worker-args tuple, or
`_score_tile_worker`'s call to `score_tile_year` — nor did `tam/pipeline.py`'s call site
pass them, even though `load_tam` already returns them. So `None` flowed all the way
down to `_preprocess`, whose gate
`if n_annual_features > 0 and annual_feat_mean is not None and annual_feat_std is not None`
evaluated **False**, leaving `annual_feats_np` as `None`. In `model.py`'s forward pass,
when `annual_feats is None` but `self.n_annual_features > 0`, the model **explicitly
substitutes an all-zeros tensor**. Effect: **every single pixel** at inference received
a constant all-zero annual-feature vector — a massive, uniform distribution shift on
~42 input dimensions where the model expected substantial per-pixel-varying z-scored
values. This was assessed as the dominant cause of "confidently wrong everywhere",
since it affects literally every pixel identically regardless of true cover type.

**Fix**: wired `annual_feat_mean`/`annual_feat_std` through the entire chunked-scoring
call chain (`score_tiles_chunked` signature, worker-args tuple, `_score_tile_worker`
unpacking + call, `score_tile_year`, `score_pixels_chunked`) and added them to
`pipeline.py`'s `score_tiles_chunked(...)` call site.

### Bug 2 — aggregation-level mismatch (secondary, compounding, fixed)

Even with Bug 1 fixed, a second, more subtle bug remained:

- **Training** (`_compute_band_summaries`, `tam/core/train.py`) used to group the
  *entire* `pixel_df` — which contains the same physical pixel sampled across all its
  years — by `point_id` **alone** (`group_by("point_id")`). So each pixel's
  `[p5, p95, std]` annual features were computed across its **entire multi-year
  history** (e.g. 2017–2025 combined), and every (pixel, year) training window for that
  pixel was given the *same* annual-feature vector (broadcast via the `point_id`-only
  lookup in `dataset.py`). The saved z-scoring stats (`tam_global_feat_stats.npz`) were
  `nanmean`/`nanstd` over this population of **per-pixel, multi-year-aggregated**
  summary vectors.

- **Inference** (`_preprocess` in `score.py`) splits the scoring pixel stream into
  windows on `pid_change | year_change` — i.e. **per pixel-year**, not per pixel —
  then computes `[p5, p95, std]` **per window** (a single year of observations), and
  normalizes using the training-time stats that were calibrated on multi-year
  aggregates.

**Why this produces confident, uniform false positives:** a single year of observations
has a narrower date range and far fewer samples than a 9-year aggregate, so its
`[p5, p95, std]` distribution is structurally different (typically tighter `std`,
narrower `p5`–`p95` spread). Normalizing single-year summaries with multi-year-
calibrated stats systematically shifts every pixel's annual features into a different
region of normalized space than the model was trained on — independent of true cover
type. That's exactly the "confidently wrong everywhere" symptom: bare ground, mangrove,
and riparian pixels all land in whatever normalized region the shift happens to map them
into, which apparently overlaps the Parkinsonia region.

**Fix**: changed `_compute_band_summaries` to group by `["point_id", "year"]` instead
of `point_id` alone — matching what `score` actually produces at inference (one row per
pixel-year, not one row per pixel broadcast across years) — and updated `dataset.py`'s
lookup to a composite `(point_id, year)` key. This was simpler than making inference
replicate multi-year aggregation (incompatible with `score`'s streaming/chunked
architecture), and arguably more semantically correct: a pixel's spectral character
*for a given year* is more meaningful than a blend across 9 years.

### Status

Both bugs are fixed in code (renamed `global_feature` → `annual_feature` throughout —
see `tam/core/annual_features.py`, `tam/core/train.py::_compute_band_summaries`,
`tam/core/dataset.py`, `tam/core/score.py`, `tam/core/model.py`, `tam/pipeline.py`).
**v10 must be retrained** — the existing `outputs/models/tam-v10` checkpoint's saved
`tam_global_feat_stats.npz` was computed at the wrong aggregation level (per-pixel
multi-year, not per-pixel-year) and is now incompatible with the fixed pipeline.
After retraining, re-run the scoring command from `docs/INCANTATIONS.md` and visually
confirm bare ground / mangrove / riparian areas are no longer confidently classified
as Parkinsonia.

This is very likely the actual root cause of the false-positive symptom — not a
labeling, contamination, or quality-gate issue (all ruled out above).

## Other directions (lower priority — only revisit if the above doesn't fully resolve it)

- **Class imbalance / decision threshold.** How many presence vs. absence examples did
  v10 actually see, and what threshold does `tam.pipeline score` use to call a pixel
  "Parkinsonia"? A skewed training set or a miscalibrated operating threshold could bias
  the model toward over-predicting presence regardless of feature quality.

- **Spatial generalization.** Are the false positives concentrated in areas geographically
  distant from any training bbox (terrain/soil/lighting the model has never seen), or do
  they also appear close to training sites? This distinguishes a generalization gap from a
  more fundamental fit problem.

- **Held-out fit on training regions.** Does the model discriminate presence from absence
  well on held-out pixels *within* the training regions themselves? If not, the problem is
  upstream of generalization — e.g. a fundamental fit or feature-pipeline issue — and
  inspecting new geography is a red herring.

## Useful tools produced during this investigation

- `utils/bbox_audit.py` — multi-year quality audit of Mitchell training bboxes; measures
  data using the exact same `Signal.quality_mask` filter the training pipeline applies.
  Regenerate with:
  ```
  python utils/bbox_audit.py --root /mnt/external/chunkstore \
      --training data/locations/training.yaml --prefix mitchell \
      --out docs/MITCHELL-BBOX-AUDIT.md
  ```
- `utils/contamination_check.py` — compares v10's actual per-pixel annual features
  (`[p5, p95, std]` of NDVI/MAVI, mirroring `_compute_band_summaries`) computed with vs.
  without suspected cloud/shadow-contaminated dates dropped, to quantify whether
  contamination meaningfully moves what the model trains on:
  ```
  python utils/contamination_check.py --root /mnt/external/chunkstore \
      --training data/locations/training.yaml --bbox-id <id> [<id> ...]
  ```
