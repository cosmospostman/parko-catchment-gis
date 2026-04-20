# Implementation Plan: Spectral Time Series Pipeline

This plan translates the architecture in `ARCHITECTURE.md` into discrete, sequential work
sessions. Each session has a clear entry condition (what must exist before it starts) and
a clear exit condition (what it produces). Sessions are ordered so that earlier ones
produce the inputs the next one needs — no session depends on anything not yet built.

The training pipeline is completed before inference work begins. Inference cannot be
meaningfully validated until a real trained model exists.

---

## Session 1 — Project skeleton and shared constants

**Goal:** Establish the directory structure and shared constants that every subsequent
session will import from. No logic, no tests — just scaffolding.

**Work:**
- Create `stage0/`, `analysis/timeseries/`, `analysis/primitives/`, `pipelines/` packages
- Create `analysis/constants.py`: band names, SCL clear values, quality profile sets
  (`Q_ATMOSPHERIC`, `Q_GEOMETRIC`, `Q_FULL`, `Q_CLOUD_ONLY`), `SPATIAL_VALIDATION_THRESHOLD`,
  `FLOWERING_WINDOW = (200, 340)`, `FLOWERING_THRESHOLD`
- Create `analysis/primitives/__init__.py`, `analysis/timeseries/__init__.py`
- Extend `tests/` with `tests/unit/` subdirs for new modules; confirm `pytest.ini` covers them

**Exit condition:** `pytest` collects with zero errors (no tests yet, just import checks).

---

## Session 2 — `Observation` and `ObservationQuality` data classes

**Goal:** Define the atomic unit of the pipeline and its quality abstraction. Everything
downstream consumes these types.

**Work:**
- `analysis/timeseries/observation.py`: `ObservationQuality` dataclass with `score(mask)`
  method; `Observation` dataclass
- Unit tests in `tests/unit/test_observation.py`:
  - `score()` with `mask=None` returns product of all five components
  - `score(Q_ATMOSPHERIC)` excludes `view_zenith`, `sun_zenith`, `greenness_z`
  - `score()` with any component = 0 returns 0
  - Quality components clamp correctly to [0, 1]

**Exit condition:** `pytest tests/unit/test_observation.py` passes.

---

## Session 3 — `ChipStore` protocol and `DiskChipStore`

**Goal:** Implement the storage abstraction that decouples downstream extraction code
from how chips were obtained.

**Work:**
- `stage0/chip_store.py`: `ChipStore` Protocol, `DiskChipStore` (reads from `inputs/`)
- Unit tests in `tests/unit/test_chip_store.py` using a small set of fixture `.tif` files
  already in `tests/fixtures/cog_chips/` (or create minimal synthetic ones):
  - `DiskChipStore.get()` returns correct array shape
  - Missing chip raises a clear `FileNotFoundError` with path context
- Document `StreamingChipStore` as a stub (interface only, not implemented yet — deferred
  until scale requires it)

**Exit condition:** `pytest tests/unit/test_chip_store.py` passes.

---

## Session 4 — Stage 0: async chip fetch

**Goal:** Implement the I/O-bound fetch layer that populates `inputs/` from the STAC
archive. This is the most expensive operation to reconstruct, so correctness and
idempotency matter more than speed tuning at this stage.

**Work:**
- `stage0/fetch.py`: `fetch_chips()` async function; semaphore-bounded concurrency
  (`max_concurrent=32` default); SCL pre-filter to skip wholly-clouded acquisitions;
  idempotent (skip existing chips)
- Wire into existing `utils/stac.py` for item queries and `utils/io.py` for COG reads
- `train.py load-testdata` subcommand: calls `fetch_chips()` with a small fixture point
  set (~5–10 points), writes `.fixture_commit` sentinel to `tests/fixtures/`
- `conftest.py` staleness check: `pytest_configure` exits with message if sentinel missing
  or stale (pipeline source files changed since last `load-testdata`)

**Exit condition:** `train.py load-testdata` completes; `inputs/` is populated;
`pytest` does not immediately exit with a missing-data error.

---

## Session 5 — Extraction primitive

**Goal:** Implement point extraction from staged chips. First layer of compute.

**Work:**
- `analysis/timeseries/extraction.py`: `extract_observations()` — reads chips via
  `ChipStore`, returns `list[Observation]`, one per usable acquisition
- Unit tests in `tests/unit/test_extraction.py` reading from `tests/fixtures/cog_chips/`:
  - Returns one `Observation` per acquisition with expected band keys
  - SCL-masked acquisitions are excluded
  - Band values are in expected range (not raw DN, not NaN)

**Exit condition:** `pytest tests/unit/test_extraction.py` passes against staged fixture chips.

---

## Session 6 — Quality scoring primitive

**Goal:** Implement per-observation quality scoring. Produces the weighted observations
that all downstream waveform and feature work consumes.

**Work:**
- `analysis/primitives/quality.py`: `ArchiveStats` dataclass (archive-wide greenness
  distribution); `score_observation()` pure function that populates all five quality
  components
- Unit tests in `tests/unit/test_quality.py` reading from `tests/fixtures/raw_observations.parquet`:
  - All quality components in [0, 1]
  - Clear, nadir, low-AOT observations score near 1.0
  - High-cloud / high-haze observations score low on the relevant components
  - `score_observation` is a pure function (returns new object, does not mutate input)

**Exit condition:** `pytest tests/unit/test_quality.py` passes.

---

## Session 7 — Waveform primitive and scientific assumption tests

**Goal:** Implement peak detection from a quality-weighted time series. This is the core
scientific claim of the pipeline. The scientific assumption tests live here.

**Work:**
- `analysis/timeseries/waveform.py`: `extract_waveform_features()` — takes
  `list[Observation]`, returns feature dict: `peak_value`, `peak_doy`, `spike_duration`,
  `peak_doy_mean`, `peak_doy_sd`, `years_detected`; returns `{}` if fewer than `min_years`
  of usable data
- `analysis/primitives/indices.py`: `flowering_index()` — canonical implementation shared
  by training and inference; numpy-compatible so `apply_index()` can vectorise it over rasters
- Unit tests in `tests/unit/test_waveform.py` reading from
  `tests/fixtures/scored_observations.parquet`:
  - **Scientific:** presence points have `peak_value > FLOWERING_THRESHOLD`
  - **Scientific:** `FLOWERING_WINDOW[0] <= peak_doy <= FLOWERING_WINDOW[1]` for presence points
  - **Scientific:** `median(presence peak_value) > median(absence peak_value)`
  - Edge case: fewer than `min_years` returns `{}`
  - Edge case: all-cloud acquisition sequence (all low quality) does not manufacture a peak

**Exit condition:** `pytest tests/unit/test_waveform.py` passes, including scientific assertions.

---

## Session 8 — Feature assembly primitive

**Goal:** Assemble the RF feature matrix from waveform features and structural inputs.

**Work:**
- `analysis/timeseries/features.py`: `assemble_feature_vector()` — flattens waveform
  features + structural features (`HAND`, `dist_to_water`) into a single dict; includes
  `mean_quality` as a feature
- Unit tests in `tests/unit/test_features.py` reading from
  `tests/fixtures/waveform_features.parquet`:
  - Feature vector contains all expected keys in the expected order
  - No NaN values in assembled output
  - `mean_quality` is in [0, 1]
  - Structural feature join does not silently drop rows (assert row count is preserved)

**Exit condition:** `pytest tests/unit/test_features.py` passes.

---

## Session 9 — Spatial validation primitive

**Goal:** Implement the validation gate that certifies a trained model for inference.

**Work:**
- `analysis/primitives/validation.py`: `ValidationResult` dataclass; `validate_spatial()`
  pure function computing AUC, precision, recall, calibration error, confusion matrix
- Unit tests in `tests/unit/test_spatial_validation.py` reading from
  `tests/fixtures/waveform_features.parquet`:
  - Returns `ValidationResult` with all fields populated
  - AUC is in [0, 1]
  - Fixture presence/absence separation exceeds `SPATIAL_VALIDATION_THRESHOLD`
    (confirms fixture data is discriminable — encodes the scientific claim as a gate)

**Exit condition:** `pytest tests/unit/test_spatial_validation.py` passes.

---

## Session 10 — Integration: full feature vector end-to-end

**Goal:** Verify that extraction → quality → waveform → feature assembly produces the
correct output when wired together in sequence. Also verifies the `feature_names` contract
is stable.

**Work:**
- `tests/integration/test_feature_pipeline.py`:
  - Run full chain from fixture chips → `Observation` list → quality-scored → waveform
    features → assembled feature vector
  - Assert feature vector schema matches a fixture `feature_names.json`
  - Assert row count matches point count (no silent drops)
  - Assert `mean_quality` column is present and in range
- Produce `tests/fixtures/feature_names_fixture.json` from this run as the schema reference

**Exit condition:** `pytest tests/integration/` passes.

---

## Session 11 — `train.py` orchestrator

**Goal:** Wire all primitives into the full training pipeline with parallelism,
checkpointing, and the model training + validation gate.

**Work:**
- `pipelines/train.py`: `run` subcommand — Stage 0 fetch, `ProcessPoolExecutor` with
  `_pool_size()`, `_worker_init()` loading `ArchiveStats`, incremental checkpoint writes
  (`observations_*.parquet` + `.progress` sidecar), waveform pass, feature assembly,
  RF fit, `validate_spatial()` gate, artefact writes
- Checkpoint resume: `--from-checkpoint observations` and `--from-checkpoint features`
- `drop-checkpoint` subcommand with `--yes` guard
- `model_{run_id}.pkl` written **last**, only after validation gate passes
- Pool sizing: `_pool_size()` from `utils/pipeline.py` (update formula if needed)

**Exit condition:** `train.py run` completes on the fixture point set; all five artefacts
in `outputs/` are written; `model_*.pkl` exists (validation passed).

---

## Session 12 — Inference pipeline: composite and feature stack

**Goal:** Implement the inference-side primitives that go from raster tiles to a feature
array matching the training schema.

**Work:**
- `analysis/timeseries/composite.py`: `quality_weighted_composite()` — takes raster chip
  stack + quality weights, returns single-value-per-pixel composite per band
- `analysis/timeseries/infer_features.py`: `assemble_infer_feature_stack()` — applies
  `feature_names` contract from training; calls `flowering_index` via `apply_index()`
  (vectorised over numpy arrays); joins HAND and `dist_to_water` rasters
- Unit tests in `tests/unit/test_composite.py`:
  - Quality-weighted composite gives higher weight to clear observations
  - Single-observation input returns that observation's values
- Unit tests in `tests/unit/test_infer_features.py`:
  - Feature stack column order matches `feature_names_fixture.json`
  - `flowering_index` called via `apply_index()` gives same result as the
    per-observation training call for matching inputs (shared primitive correctness check)

**Exit condition:** `pytest tests/unit/test_composite.py tests/unit/test_infer_features.py` passes.

---

## Session 13 — Integration: feature distribution check

**Goal:** Scientific assumption test for inference — confirm that the feature values at
known presence locations computed via the inference path are consistent with those computed
via the training path.

**Work:**
- `tests/integration/test_infer_feature_distribution.py`:
  - Load inference-path feature stack at a set of known presence fixture locations
  - Assert `flowering_index` feature values fall in the same range as training-path values
    for matching locations (within a tolerance)
  - Assert feature stack is not degenerate (no all-zero columns, no NaN columns)
  - Assert column order matches `feature_names_fixture.json`

**Exit condition:** `pytest tests/integration/test_infer_feature_distribution.py` passes.

---

## Session 14 — `infer.py` orchestrator and output rasters

**Goal:** Wire inference primitives into the full inference pipeline producing the
probability raster and companion confidence raster.

**Work:**
- `pipelines/infer.py`: `run --model-run-id {run_id}` subcommand — reads all five
  training artefacts, Stage 0 fetch for seasonal composite tiles, tile-parallel prediction,
  `RF.predict_proba()` across pixel array, writes probability raster and confidence raster
- `load-testdata` and `drop-checkpoint` subcommands (same pattern as `train.py`)
- Output rasters in the same CRS and extent as existing pipeline outputs (compatible with
  `07_priority_patches.py` and `08_change_detection.py` downstream)

**Exit condition:** `infer.py run` completes; probability raster and confidence raster are
written; raster is not degenerate (value distribution is not flat).

---

## Session 15 — Pipeline tests and operational hardening

**Goal:** Add pipeline-level tests covering failure modes that unit tests cannot reach:
worker crashes, network timeouts, checkpoint resume after partial run.

**Work:**
- `tests/pipeline/test_train_pipeline.py`:
  - Interrupt mid-run (kill a worker); confirm resume from `.progress` sidecar picks up
    where it left off without duplicating rows
  - Simulate network timeout in Stage 0; confirm idempotent retry works
  - Confirm `drop-checkpoint --yes` deletes the right files and nothing else
- `tests/pipeline/test_infer_pipeline.py`:
  - Confirm mismatched `feature_names` (wrong run ID) raises a clear error, not a
    silent wrong prediction
  - Confirm missing model file raises immediately with actionable message

**Exit condition:** `pytest tests/pipeline/` passes.

---

## Dependency graph

```
Session 1  (skeleton)
    └── Session 2  (Observation types)
            ├── Session 3  (ChipStore)
            │       └── Session 4  (Stage 0 fetch / load-testdata)
            │               └── Session 5  (extraction)
            │                       └── Session 6  (quality scoring)
            │                               └── Session 7  (waveform + science tests)
            │                                       └── Session 8  (feature assembly)
            │                                               └── Session 9  (validation primitive)
            │                                                       └── Session 10 (integration)
            │                                                               └── Session 11 (train.py)
            │                                                                       └── Session 12 (infer primitives)
            │                                                                               └── Session 13 (infer integration)
            │                                                                                       └── Session 14 (infer.py)
            │                                                                                               └── Session 15 (pipeline tests)
            └── (constants used throughout)
```

Each session's tests must pass before the next session begins. Sessions 1–10 are pure
primitive and test work — no orchestration, no CLI. The orchestrators (Sessions 11, 14)
are written last, after all the code they call is already tested.
