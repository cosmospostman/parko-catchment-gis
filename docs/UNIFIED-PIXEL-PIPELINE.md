# Plan: Unify train/score band-summary computation onto one shared windowing path

## Context

`docs/MITCHELL-DEBUG.md` documents four compounding train/inference parity bugs in
the "annual feature" path — each one a place where training (Polars, whole-dataset,
`tam/core/train.py::_compute_band_summaries`) and scoring (numba kernels, streaming,
`tam/core/_preprocess_numba.py::compute_band_summaries` + `score.py::_preprocess`)
independently re-implemented "the same" per-pixel-year `[p5, p95, std]` statistic and
silently drifted apart (aggregation scope, quantile/std method, NaN handling). A
`PARITY_CASES` registry test (`tests/tam/test_train_score_parity.py`) now exists to
*catch* this class of bug — but it can only catch what someone thinks to register.

While discussing why two implementations exist at all, we found a **fifth, currently
live** instance of exactly this bug class — not yet in `MITCHELL-DEBUG.md`:

### New finding: MAVI / CI_RE transcription bug in `add_spectral_indices` (mode-dependent, live, root-caused)

Three implementations of the spectral indices feeding band-summary stats exist,
and one of them — the one used by scoring's *default* mode — has wrong formulas
for **MAVI** and **CI_RE**:

| Source | MAVI | CI_RE | Used by |
|---|---|---|---|
| `signals/mavi.py::MAVISignal`, `signals/ndre.py::CIRESignal` (canonical, documented) | `(B08-B04)/(B08+B04+B11)` | `(B07/B05)-1` | design source-of-truth; see docstrings explaining *why* (B11 = SWIR-1 "drawdown halo" detector; B07 = red-edge ratio variant) |
| `train.py::_compute_band_summaries` inline `_index_exprs` (`train.py:149-159`) | `(B08-B04)/(B08+B04+B11)` ✓ | `(B07/B05)-1` ✓ | training, always |
| `_preprocess_numba.py::extract_features` (`:521-531`) | `(B08-B04)/(B08+B04+B11)` ✓ | `(B07/B05)-1` ✓ | scoring, **S2-only** mode (`mixed=False`) |
| `analysis/constants.py::add_spectral_indices` (`:90-130`), via `dataset.py::prepare_s2_frame` | `(B8A-B11)/(B8A+B11)` ✗ | `(B8A/B05)-1` ✗ | scoring, **mixed S2+S1** mode (`mixed=True`, the *default* — `score.py:557`) |

**Root cause, found via `git log -p -S evi_d -- analysis/constants.py`**: commit
`f023003` ("More memory wrangling", 2026-05-26) inlined `add_spectral_indices` —
which previously *delegated* to `MAVISignal`/`CIRESignal` (the canonical
implementations, added in `c929739`) — into raw Polars expressions, for performance
(to let Polars fuse all six index computations into one scan). During that
transcription, **MAVI and CI_RE were miscopied**: MAVI's bands `(B08, B04, B11)`
became `(B8A, B11)` and CI_RE's `(B07, B05)` became `(B8A, B05)` — both errors
substitute `B8A` in a way that structurally mirrors the *adjacent* NDRE formula
`(B8A-B05)/(B8A+B05)` immediately above them in the signal-class pattern, i.e. a
copy-paste/band-substitution slip, not a deliberate alternate formulation. NDVI,
NDWI, and EVI were transcribed correctly (their formulas don't share that band
pattern). This is **not** a "which is correct" ambiguity — `MAVISignal`/`CIRESignal`
are the documented, hypothesis-driven definitions (see their docstrings: MAVI
deliberately puts B11/SWIR-1 in the denominator to detect a "soil-moisture drawdown
halo"; CI_RE deliberately uses B07 as "a ratio-form variant of NDRE, more sensitive
at high chlorophyll"), and `extract_features`+training already match them exactly.
**`add_spectral_indices` is simply wrong**, and has been corrupting MAVI/CI_RE in
the live, default (`mixed=True`) scoring path since 2026-05-26.

This is a sixth-ish variant of the exact bug class `MITCHELL-DEBUG.md` describes —
currently undetected because the parity registry only covers
`band_summaries_p5_p95_std` from a synthetic `feat` array (it never exercises the
index-*formula* step, only the windowing/aggregation step downstream of it).

**Why this keeps happening**: there are structurally *three* places in the
train/score path that know how to turn raw bands into spectral indices
(`_compute_band_summaries`'s inline Polars exprs, `extract_features`'s numba
arithmetic, `add_spectral_indices`'s inlined Polars exprs — itself a regression from
a fourth, *correct* place: the `Signal` subclasses it used to delegate to),
written/rewritten at different times for different performance contexts, with no
enforced single source of truth. Parity tests are a safety net for *known* shared
statistics; they don't prevent a transcription slip in a reimplementation next year.

### A fourth (UI-side) reimplementation: the pixel inspector

The UI's "pixel inspector" panel (`ui/src/components/panels/BBoxCard.svelte` →
`utils/pixel_timeseries.py::compute_timeseries`, backed by random-access reads via
`utils/pixel_reader.py::ChunkIndex`) is **yet another** independent
bands-to-indices implementation — this one in raw numpy, for random-access
bbox-timeseries display rather than batch training or streaming scoring:

```python
# utils/pixel_timeseries.py:99-100
ndvi = _safe_div(b08 - b04, b08 + b04)
mavi = _safe_div(b08 - b04, b08 + b04 + b11)
```

Two more findings here, of the same shape as the MAVI/CI_RE bug:

1. **It happens to have the correct MAVI formula** (matching `MAVISignal`/
   `extract_features`/training) — but only by chance, not by sharing code with any
   canonical source. The next person extending this file (e.g. to add NDRE/CI_RE/EVI
   to the inspector display, a natural next step given it already shows NDVI/MAVI)
   has nothing structural stopping them from making the exact same band-substitution
   slip that produced the `add_spectral_indices` bug.
2. **It applies no quality gating at all** — no `scl_purity >= 0.5`, no
   `source == "S2"` filter — unlike every train/score implementation
   (`_compute_band_summaries`, `extract_features`'s caller in `_read_raw_chunk`,
   `add_spectral_indices`/`prepare_s2_frame`). A user inspecting a pixel's timeseries
   can see NDVI/MAVI computed over cloud-contaminated or mismatched-source rows the
   model never sees — a *visualization-vs-model-input* mismatch, distinct from but
   structurally analogous to the train/score mismatches above (different consumers
   of "the same" raw data, silently disagreeing on how to interpret it).

This is lower-stakes than the live MAVI/CI_RE scoring bug — it's a diagnostic
display, not a model-input path, so a wrong value shown to a human doesn't corrupt
training or scoring. But it's good independent evidence for the thesis: **every**
hand-written reimplementation of "raw bands → spectral indices" is a latent
divergence waiting to happen, in any consumer, not just the train/score pair this
plan started from. The fix below should produce a single canonical primitive that
*all four* consumers — train, score, and the UI inspector — call into, not just the
two that happen to feed the model.

## The structural fix: make scoring's windowing/index path the *only* implementation

Rather than writing more parity tests (treating the symptom), restructure so
training **and** the UI pixel inspector call the **same numba primitives** scoring
uses — `extract_features` for index computation and `compute_band_summaries` for
window statistics — making a future divergence *impossible to write* rather than
merely *detectable*. This eliminates bugs 2, 3, 4 (already fixed, but by
hand-aligning implementations that can drift again), the live MAVI/CI_RE bug in
`add_spectral_indices`, and the latent risk in `pixel_timeseries.py`, all in one
move — and removes the long-term maintenance burden of keeping N independent
codepaths in numerical lockstep by reducing N to 1.

This is feasible because of a key structural fact uncovered in this session:
**training pixel-year groups and scoring pixel-year windows are the same thing** —
both are runs of `(point_id, year)`-identical rows in a stream sorted by
`(point_id, year, doy/date)`. `pixel_df_cache.parquet` is already built in that sort
order (confirmed: it's assembled tile-by-tile in `pipeline.py` and the same
`pid_change | year_change` boundary-detection scoring relies on — `score.py:585-594`
— would work identically over it). Scoring's boundary detection is ~10 lines of
numpy; there is no reason training can't run the identical 10 lines.

## Implementation plan

### 1. Extract a shared "raw bands → feature array + windows" helper

Currently this logic is inlined in `score.py::_preprocess`/`_read_raw_chunk`
(`score.py:438-548` builds the `feat`/`is_s1`/`pid_arr`/`year_arr` arrays;
`score.py:585-594` derives `boundaries`/`ends`). Factor the **boundary-detection**
piece (lines 585-594) out into a small standalone function, e.g.
`detect_pixel_year_windows(pid_arr, year_arr) -> (boundaries, ends)`, in
`tam/core/_preprocess_numba.py` or `score.py` (whichever has fewer import-cycle
issues — check whether `train.py` can import from `score.py` without circularity;
`_preprocess_numba.py` is the safer target since it's leaf-level).

This function has no Polars dependency and is trivially unit-testable in isolation
(feed it synthetic `pid_arr`/`year_arr`, assert window boundaries) — which also
means the *existing* parity-relevant logic in `_preprocess` shrinks, lowering its
own bug surface.

### 2. Build a `pixel_df` → `(feat, boundaries, ends)` adapter for the S2 band-summary case

Training needs: take `pixel_df` (already sorted `point_id, year, date` — confirmed
via `pipeline.py` tile-assembly order), filter to `source == "S2"` and
`scl_purity >= 0.5` (a row *filter*, matching how `prepare_s2_frame`/scoring's
`keep_series` gate works — `dataset.py:86`, `score.py:467-469` — rather than
`_compute_band_summaries`'s current in-aggregation `_qm`/`_safe_ratio` masking,
which produces `null` outputs inline; row-filtering is simpler and is what scoring
actually does), extract the 10 raw S2 bands as contiguous float32 numpy arrays, and
call `extract_features` (`_preprocess_numba.py:494`) to get the canonical `(N, 16)`
feature array — **the exact same function and column order scoring's S2-only path
uses**. Then call the new `detect_pixel_year_windows` on the filtered, sorted
`(point_id, year)` arrays.

**Order-preservation invariant — write this down explicitly, don't let it stay
implicit**: `detect_pixel_year_windows` is only correct if the `source`/
`scl_purity` row-filter does not disturb the `(point_id, year, date)` sort order
of `pixel_df`. Polars `.filter()` *is* order-preserving (it's a boolean mask, not
a join/sort/groupby), so this holds today — but it is exactly the kind of silent
structural assumption whose violation produced the bug class this whole plan
exists to eliminate (see "Why this keeps happening" above: undocumented shared
facts that the next refactor breaks without anyone noticing). Add a one-line
comment at the filter call site (e.g. "`.filter()` preserves row order — required
for the boundary-scan below; do not replace with a join/semi-join") and/or a
debug-mode assertion (`pixel_df["point_id"], pixel_df["year"]` non-decreasing
after filtering) so a future change that swaps the filter for something
order-disturbing fails loudly instead of silently producing wrong window
boundaries.

This becomes the single new code path that both:
- replaces `_compute_band_summaries`'s Polars index-expression block
  (`train.py:140-169`), and
- is (ideally) the *same* helper scoring's S2-only `_read_raw_chunk` path
  (`score.py:527-537`) calls — collapsing two call sites into one.

### 3. Replace `_compute_band_summaries`'s aggregation with `compute_band_summaries` (numba)

Once `(feat, boundaries, ends)` exist in the same shape `_preprocess` produces, call
`compute_band_summaries` (`_preprocess_numba.py:559`) directly — the exact kernel
scoring uses, with its linear-interpolation percentiles, population variance, and
explicit NaN exclusion already correct by construction. Slice the relevant
`V9_FEATURE_COLS` columns out of `extract_features`'s 16-column output (note:
`extract_features` produces `[B02..B12, NDVI, NDWI, EVI, MAVI, NDRE, CI_RE]` in a
fixed order that differs from `V9_FEATURE_COLS`'s order/membership — `dataset.py:35-38`
excludes B06 and EVI; this column selection/reorder is the one piece of new glue
code, and it is purely structural — a fixed index permutation, nothing
numerically sensitive).

Wrap the result back into the `[point_id, year, <col>_p5, <col>_p95, <col>_std]`
Polars DataFrame shape the three call sites (`train.py:787`, `_prep_worker.py:212`,
`pipeline.py:314`) currently expect, so their signatures don't need to change.

### 4. Delete what becomes dead

- `_compute_band_summaries`'s inline `_index_exprs`/`_safe_ratio`/`_qm` block
  (`train.py:140-169`) — superseded by `extract_features`.
- The `PARITY_CASES` entry `band_summaries_p5_p95_std` becomes a *structural*
  identity (same function called on both sides) rather than a numerical-coincidence
  check — it can be simplified to assert that train and score literally invoke the
  same kernel on the same input shape, or removed if redundant with a more direct
  "training reuses scoring's primitives" assertion. Keep the registry file/mechanism
  itself (per [[feedback_train_score_parity_tests]]) for *future* shared statistics
  that can't be unified this way.
- **Fix the MAVI/CI_RE transcription bug directly**: correct `add_spectral_indices`
  (`analysis/constants.py:117-123`) to use the canonical formulas —
  `mavi = _safe((b08 - b04) / (b08 + b04 + b11), b08 + b04 + b11)` and
  `cire = _safe(b07 / b05 - 1, b05)` (introducing `b07 = pl.col("B07")` alongside the
  other column refs at line 113-115) — matching `MAVISignal`/`CIRESignal`
  (`signals/mavi.py`, `signals/ndre.py`) and `extract_features`/training exactly. No
  design decision is needed: the canonical `Signal` subclasses are the documented
  source of truth (their docstrings explain *why* each formula uses the bands it
  does), and `extract_features` + training already match them — `add_spectral_indices`
  is the sole, accidentally-introduced outlier (root-caused above to commit `f023003`).
  Ideally also restore `add_spectral_indices` to *delegate* to the `Signal` classes
  (as it did pre-`f023003`) rather than re-inlining formulas Polars can still fuse —
  removing the third independent transcription site entirely, so there's exactly one
  place (`signals/`) that defines index formulas and the rest call into it.

### 5. Migrate the pixel inspector onto the same canonical index computation

Files: `utils/pixel_timeseries.py`

`compute_timeseries` (`pixel_timeseries.py:73-123`) currently hand-derives `ndvi`/
`mavi`/`vh_vv` from raw numpy arrays with no quality gating (see "A fourth (UI-side)
reimplementation" above). Replace its inline `_safe_div` index computation with a
call into whatever the unification produces as the canonical "raw bands → indices"
primitive — at minimum, route it through `add_spectral_indices` (once fixed in step 4
to either contain or delegate to the canonical `Signal` formulas) by converting the
queried `pa.Table` to a Polars DataFrame and calling `add_spectral_indices(df)` /
`prepare_s2_frame(df, scl_purity_min=0.5, feature_cols=[...])` rather than
hand-rolling `_safe_div` on raw column arrays. This:
- fixes the missing-quality-gate gap (the inspector would then show the *same*
  filtered values the model trains/scores on, which is arguably what a debugging tool
  *should* show — "what does the model actually see for this pixel"), and
- collapses the fourth independent index-formula transcription site into a single
  call to the shared primitive, so a future addition (e.g. NDRE/CI_RE/EVI to the
  inspector display) can't introduce a fifth divergence.

If `compute_timeseries`'s VH/VV dB computation (lines 102-106) has a train/score
analogue (`lin_to_db` in `dataset.py`, used by `prepare_s1_frame`), route that through
the shared `lin_to_db` too, for the same reason — one source of truth for "linear →
dB," not a third hand-rolled `10*log10`.

### 6. Update tests, docs, memory

- Extend/adjust `tests/tam/test_train.py::TestTT10S2ColsLoadedForNoiseFilter` and the
  `test_compute_band_summaries_groups_by_pixel_and_year` test to assert the new
  implementation still groups by `(point_id, year)` and matches `V9_FEATURE_COLS`
  shape — but the *numerical* parity assertion is no longer needed once both sides
  call the identical kernel (it becomes a tautology).
- Add a new `MITCHELL-DEBUG.md` section documenting the MAVI/CI_RE finding (call it
  "Bug 5") and the resolution.
- Update [[project_annual_feature_parity_bugs]] memory with the new bug + the
  structural fix (reuse, not parity-testing).

## Verification

1. Run the existing suite (`tests/tam/test_train.py`, `test_score.py`, `test_model.py`,
   `test_pixel_source.py`, `test_train_score_parity.py`) — all must still pass; the
   `band_summaries_p5_p95_std` parity case should now pass *trivially* (same code
   path on both sides).
2. Add/extend a test asserting `_compute_band_summaries`'s new implementation and
   `score`'s `_preprocess` band-summary computation produce bit-identical output
   given the same raw rows fed through each entry point — this is now a much
   stronger assertion than numerical closeness, since it's the same function.
3. **v10 must be retrained regardless** (already known from the prior bug fixes —
   `tam_annual_feat_stats.npz` is stale). Fold this retrain into whichever retrain
   the user schedules for the bug-3/bug-4 fixes already landed — no need for two
   separate retraining runs.
4. After retraining, re-run the `tam.pipeline score` command from
   `docs/INCANTATIONS.md` and visually confirm bare ground/mangrove/riparian areas
   are not confidently flagged Parkinsonia (the standing acceptance criterion from
   `MITCHELL-DEBUG.md`).
5. **Pixel inspector**: open the UI, select a bbox with known NDVI/MAVI behaviour
   (e.g. one of the audited Mitchell presence bboxes from `MITCHELL-BBOX-AUDIT.md`),
   and confirm the displayed timeseries values change in the expected direction after
   the quality-gating fix (cloud/shadow-contaminated dates should now be excluded —
   compare against `utils/contamination_check.py`'s findings for that bbox) and that
   `compute_timeseries`'s output still renders correctly end-to-end in
   `BBoxCard.svelte`.

## Note on blast radius of the MAVI/CI_RE fix

`add_spectral_indices` has been wrong since commit `f023003` (2026-05-26) — check
whether v10's training run predates or postdates that commit (`git log
--oneline -- analysis/constants.py` against the v10 training-run timestamp /
`outputs/models/tam-v10` checkpoint metadata) to determine whether v10's *training*
data ever passed through the broken `add_spectral_indices` path (it shouldn't have —
training uses `_compute_band_summaries`'s own correct inline exprs — but worth
confirming nothing else in the training prep pipeline calls `add_spectral_indices`
on the S2 feature columns that feed the model). The clearer impact is that **any
mixed-mode (`mixed=True`, the default) scoring run since 2026-05-26** — including
whatever produced the false-positive symptom under investigation in
`MITCHELL-DEBUG.md` — has been feeding the model corrupted MAVI/CI_RE values via
`prepare_s2_frame`. This is worth fixing and re-scoring *before* drawing final
conclusions about whether the four already-fixed annual-feature bugs fully explain
the false-positive symptom, since this is a fifth independent source of train/score
distribution shift that was active over the same period.
