# Engineering Architecture: Spectral Time Series Pipeline

## Character of the problem

The new investigation is fundamentally **data analysis at points**, not GIS raster processing.
The current pipeline thinks in tiles, projections, and map outputs. The new approach thinks in
time series, observation quality, and feature matrices. The spatial coordinate of an ALA point
is an index into the satellite archive — not the primary dimension of analysis.

This distinction drives the architecture. The unit of work is:

```
(location, date) → (spectral values, quality weight)
```

Rather than:

```
(tile_bbox, date_range) → raster
```

GIS operations (HAND, riparian position, final probability raster) remain at the boundaries.
Everything between point extraction and RF prediction is tabular data analysis.

---

## Two pipelines, not one

The system is expressed as two top-level pipelines that share primitives but run on independent
schedules with independent triggers:

```
pipelines/
  train.py      # point extraction → waveform → feature matrix → RF fit
  infer.py      # raster composite → feature stack → RF predict → probability raster

analysis/
  primitives/   # shared: index functions, quality scoring, band math
  model.py      # shared: RF wrapper, feature names contract

stage0/
  fetch.py      # shared: ChipStore, async COG fetch, inputs/ staging
```

**Training** is triggered by new ground truth data becoming available (drone surveys, new ALA
records). It is point-sparse and time-deep: ~5,000 points × ~70 acquisitions, full waveform
history per point. It runs on an irregular cadence driven by data availability.

**Inference** is triggered by a new season's imagery becoming available plus an existing trained
model. It is spatially dense and time-shallow: every pixel in the catchment (~700M pixels at
10 m), current-season feature vector only, no waveform history. It runs annually.

These are independent triggers. Coupling them into a single pipeline would require skip-logic
or flags that amount to two pipelines with worse ergonomics. The waveform layer does not exist
in the inference path at all — inference computes season-summary features directly from a
composited raster stack.

### The training→inference contract

The handoff between pipelines is a versioned set of artefacts written by `train.py`:

```
outputs/
  model_{run_id}.pkl            # RF weights
  feature_names_{run_id}.json   # ordered feature list the model expects
  train_metrics_{run_id}.json   # OOB score, temporal CV results, feature importances
  train_manifest_{run_id}.json  # ALA points, year range, quality thresholds, index versions
```

`infer.py` takes `--model-run-id` and reads all four. The `feature_names` file is the critical
coupling — it ensures inference constructs the feature vector in exactly the same order and with
the same definitions as training. A silent column mismatch produces a plausible-looking but
invalid probability raster. The manifest provides the provenance chain: which model version
produced which raster, trained on what data, with what quality thresholds.

---

## Layers

### Training pipeline

```
┌─────────────────────────────────────────────────────┐
│  STAGE 0: FETCH                                     │
│  Async COG chip download to inputs/                 │
│  I/O-bound: saturates network, not CPU              │
│  Applies band selection + SCL pre-filter only       │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  EXTRACTION LAYER                                   │
│  Point → COG windowed read → raw band values        │
│  One observation per (point × acquisition)          │
│  Produces: raw observation records                  │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  QUALITY LAYER                                      │
│  Per-observation quality scoring                    │
│  w_scl × w_aot × w_vza × w_sza × w_greenness       │
│  Produces: weighted observation records             │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  WAVEFORM LAYER                                     │
│  Time series per point per year                     │
│  Peak detection, duration, timing, consistency      │
│  Produces: per-point feature vectors                │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  FEATURE LAYER                                      │
│  Assemble RF feature matrix                         │
│  Spectral + temporal + structural features          │
│  Produces: X matrix, y labels, sample weights       │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  MODEL LAYER                                        │
│  RF training                                        │
│  Temporal train/test split (hold out most recent)   │
│  Produces: model artefacts (see contract above)     │
└─────────────────────────────────────────────────────┘
```

### Inference pipeline

```
┌─────────────────────────────────────────────────────┐
│  STAGE 0: FETCH                                     │
│  Same ChipStore abstraction, raster footprint       │
│  Downloads seasonal composite tiles to inputs/      │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  COMPOSITE LAYER                                    │
│  Quality-weighted seasonal composite per band       │
│  Produces: raster feature stack (one value/pixel)   │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  FEATURE LAYER                                      │
│  Apply feature_names contract from training         │
│  Structural features (HAND, dist_to_water) joined   │
│  Produces: feature array matching training schema   │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  PREDICT LAYER  (GIS boundary)                      │
│  RF predict_proba across pixel array                │
│  Produces: probability raster + confidence raster   │
└─────────────────────────────────────────────────────┘
```

---

## Stage 0: Fetch

Stage 0 is an explicit, separate execution phase. It is **I/O-bound**, not CPU-bound, and must
not share a worker pool with compute stages.

### Design

```python
async def fetch_chips(
    points_or_bbox: GeoDataFrame | BoundingBox,
    items: list[pystac.Item],
    bands: list[str],
    window_px: int = 5,           # chip size; 1 for point extraction, larger for compositing
    inputs_dir: Path = Path("inputs/"),
    scl_filter: set[int] | None = SCL_CLEAR_VALUES,  # drop acquisitions that are all cloud
    max_concurrent: int = 32,     # semaphore bound — see concurrency note below
) -> None:
    """
    Download COG chips for all (item, band, point/tile) combinations.
    Applies only SCL pre-filter to skip wholly unusable acquisitions —
    no spectral computation, no quality scoring.
    Writes chips to inputs/{item_id}/{band}_{point_id}.tif
    """
    sem = asyncio.Semaphore(max_concurrent)

    async def fetch_one(item, band, point):
        async with sem:
            ...   # single COG range request + write

    await asyncio.gather(*[fetch_one(i, b, p) for i, b, p in work])
```

### Concurrency model

Fetch performance is **latency-bound, not bandwidth-bound**. Each COG range request to
Element84/S3 has ~50–150ms round-trip latency regardless of connection speed. A sequential
loop would take minutes for thousands of chips even on a fast connection; concurrent
requests amortise that latency across the work queue.

Concurrency is bounded by an `asyncio.Semaphore`, not a thread pool or process pool.
This limits simultaneous in-flight requests at the application level — before the OS
network stack or the remote server sees load — preventing socket exhaustion, connection
reset errors, and S3 rate-limit responses.

**Default `max_concurrent=32`** is chosen to:
- Keep the network pipe full on a 100 Mbps workstation connection
- Avoid overwhelming the OS network stack (thousands of open sockets)
- Stay well within Element84/S3 per-IP request rate limits
- Leave headroom for other processes sharing the connection

64–128 is appropriate for a production EC2 instance with a dedicated high-bandwidth
connection; do not raise the default without profiling. The value is a single knob —
do not replace the semaphore pattern with multiple thread pools or nested executors.

### load-testdata uses the same fetch path

`train.py load-testdata` calls `fetch_chips()` with the fixture point set — the same
function, same concurrency model, same `max_concurrent` default. The fixture work list
is simply smaller (~5–10 points vs. ~5,000), so staging completes in seconds on any
workstation. There is no separate implementation for test data fetching.

### ChipStore abstraction

Downstream stages access chips through a `ChipStore` interface, not by reading files directly.
This decouples the extraction primitives from whether data was staged to disk or must be
streamed on demand:

```python
class ChipStore(Protocol):
    def get(self, item_id: str, band: str, point_id: str) -> np.ndarray:
        """Return chip array. Implementation may read from disk or fetch live."""

class DiskChipStore:
    """Reads from inputs/ directory populated by Stage 0."""

class StreamingChipStore:
    """Issues COG range requests on demand. No disk required."""
```

`DiskChipStore` is the default. `StreamingChipStore` is available when inputs exceed local
storage capacity (e.g. national-scale inference where the full chip set would exceed EBS
budget). The extraction and composite primitives are identical in both modes.

### Storage sizing

For the training pipeline at current scope (~5,000 points × 70 acquisitions × 7 bands,
5×5 chips): ~2–5 GB. Fits comfortably on a laptop or small EBS volume.

For catchment-wide inference (~700M pixels, seasonal composite, 7 bands): significantly
larger. At this scale `StreamingChipStore` may be preferable to avoid staging the full
input set. The `inputs_dir` path is configurable; point it at an EBS mount in production.

---

## The core primitive: `Observation`

Every layer of the training pipeline operates on or produces observations. An observation
is a single acquisition at a single point — the atomic unit of the pipeline.

```python
@dataclass
class Observation:
    point_id:       str               # ALA record ID or synthetic absence ID
    date:           datetime
    bands:          dict[str, float]  # {"B03": 0.043, "B04": 0.021, ...}
    quality:        ObservationQuality
    meta:           dict              # view_zenith, sun_zenith, tile_id, etc.
```

This is a plain Python dataclass — no spatial geometry, no raster dependency. It serialises
trivially to a DataFrame row. All downstream layers consume DataFrames of these records.

---

## Quality as a multi-dimensional abstraction

Quality is not a single scalar. Different consumers care about different dimensions — a
consumer computing dry-season NDVI anomaly doesn't care how green the scene is (that's
the signal), but does care about cloud and smoke. A consumer computing the greenness
z-score itself must ignore the greenness component to avoid circularity.

`ObservationQuality` holds each component independently and computes weighted scalars
on demand, with the caller specifying which dimensions are relevant:

```python
@dataclass
class ObservationQuality:
    scl_purity:    float   # [0,1] fraction of clear pixels in local window
    aot:           float   # [0,1] inverse aerosol optical thickness (1 = clean)
    view_zenith:   float   # [0,1] inverse view zenith (1 = nadir)
    sun_zenith:    float   # [0,1] inverse sun zenith (1 = high sun)
    greenness_z:   float   # [0,1] inverse scene greenness z-score (1 = normal)

    def score(self, mask: set[str] | None = None) -> float:
        """
        Product of selected components. mask specifies which to include;
        if None, all components are used.

        Examples:
          obs.quality.score()
            → all components: general-purpose weight

          obs.quality.score(mask={"scl_purity", "aot"})
            → atmospheric clarity only: suitable for any spectral index

          obs.quality.score(mask={"scl_purity", "aot", "view_zenith", "sun_zenith"})
            → geometric/atmospheric only: suitable for NDVI anomaly where
              greenness IS the signal and must not be penalised

          obs.quality.score(mask={"scl_purity"})
            → cloud mask confidence only: suitable for structural features
              (texture, HAND) that are insensitive to haze or sun angle
        """
        components = {
            "scl_purity":  self.scl_purity,
            "aot":         self.aot,
            "view_zenith": self.view_zenith,
            "sun_zenith":  self.sun_zenith,
            "greenness_z": self.greenness_z,
        }
        active = {k: v for k, v in components.items()
                  if mask is None or k in mask}
        result = 1.0
        for v in active.values():
            result *= v
        return result
```

### Predefined quality profiles

```python
Q_ATMOSPHERIC   = {"scl_purity", "aot"}
Q_GEOMETRIC     = {"scl_purity", "aot", "view_zenith", "sun_zenith"}
Q_FULL          = None
Q_CLOUD_ONLY    = {"scl_purity"}
```

### How consumers use it

```python
peak_weight    = obs.quality.score(Q_FULL)       # flowering peak — penalise anomalous greenness
anomaly_weight = obs.quality.score(Q_GEOMETRIC)  # NDVI anomaly — greenness is the signal
texture_weight = obs.quality.score(Q_CLOUD_ONLY) # texture — only care about cloud
```

---

## Primitives that work in both single-thread and parallel contexts

The key design principle: **each primitive is a pure function from inputs to outputs**.
No shared state, no side effects beyond explicit I/O. This means:

- In single-thread mode: call primitives directly in a loop
- In parallel mode: map the same primitives across a pool of workers

The parallelism boundary is always at the **point level** — one point per task. Points have
no dependencies on each other. This is embarrassingly parallel.

### Primitives are shared between train and infer — by requirement, not convention

Training and inference must use identical implementations of every index function, quality
score, and band computation. This is a **correctness requirement**: if `flowering_index` is
computed slightly differently in each pipeline — different band order, a missing clamp, a
changed coefficient — the model is applied to a feature distribution that differs from the
one it was trained on. The probability raster will look plausible and be silently wrong.

The rule: **if you find yourself writing the same band math or index function twice, the
primitive is not general enough.** The canonical implementation lives in
`analysis/primitives/` and both pipelines import from there. Neither pipeline owns a copy.

The concrete case is `flowering_index`. During training it is called per-observation in a
loop. During inference it is called per-pixel across a numpy array. The same function
handles both:

```python
# training — called in a loop over Observation objects
peak_weight = flowering_index(obs.bands)

# inference — called vectorised across a raster tile
# (via np.vectorize or xarray.apply_ufunc wrapping the same function)
index_raster = apply_index(flowering_index, band_stack)
```

Same coefficients, same clamping, same definition. The `feature_names_{run_id}.json`
contract enforces that inference assembles features in the same order as training, but the
primitive sharing enforces that the values themselves are computed identically.

### Extraction primitive

```python
def extract_observations(
    point_id: str,
    geometry: Point,
    items: list[pystac.Item],
    bands: list[str],
    year_range: tuple[int, int],
    chip_store: ChipStore,
) -> list[Observation]:
    """
    Extract per-acquisition band values at a point via ChipStore.
    Never loads full scenes. Returns one Observation per usable acquisition.
    """
```

### Quality scoring primitive

```python
def score_observation(
    obs: Observation,
    archive_stats: ArchiveStats,
) -> Observation:
    """
    Populate obs.quality with per-component scores.
    Pure function — returns a new Observation with quality fully populated.
    """
```

`ArchiveStats` is computed once before the parallel loop and loaded into each worker
process via a `ProcessPoolExecutor` initializer (not pickled per-task):

```python
def _worker_init(archive_stats_path: Path) -> None:
    global _archive_stats
    _archive_stats = ArchiveStats.load(archive_stats_path)

with ProcessPoolExecutor(max_workers=_pool_size(),
                         initializer=_worker_init,
                         initargs=(stats_path,)) as pool:
    results = list(pool.map(process_point_task, point_tasks, chunksize=50))
```

### Waveform primitive

```python
def extract_waveform_features(
    observations: list[Observation],
    index_fn: Callable[[dict], float],
    window: tuple[int, int],
    quality_mask: set[str] | None = Q_FULL,
    min_quality: float = 0.3,
    min_years: int = 3,
) -> dict[str, float]:
    """
    From a list of observations at a single point, extract waveform features:
      - peak_value, peak_doy, spike_duration
      - peak_doy_mean, peak_doy_sd (consistency across years)
      - years_detected (fraction of years with spike above threshold)
    Returns {} if fewer than min_years of usable data.
    """
```

### Feature assembly primitive

```python
def assemble_feature_vector(
    point_id: str,
    waveform_features: dict[str, float],
    structural_features: dict[str, float],
    label: int,
    year_detected: int,
) -> dict:
    """
    Flatten all features into a single dict suitable for a DataFrame row.
    """
```

---

## Parallelism and resource management

### Pool sizing

Workers are sized to fit available memory, not blindly set to CPU count:

```python
import os, psutil

def _pool_size(mem_per_worker_gb: float = 2.0, cap: int = 16) -> int:
    """
    Size pool to available memory. Each worker holds COG buffers and
    observation arrays; 2 GB is a conservative per-worker budget.
    """
    available_gb = psutil.virtual_memory().available / 1e9
    mem_workers  = int(available_gb // mem_per_worker_gb)
    cpu_workers  = os.cpu_count() or 1
    return max(1, min(cpu_workers, mem_workers, cap))
```

On a 16 CPU / 32 GB machine with normal OS overhead (~4 GB used), this yields
`min(16, 14, 16) = 14` workers — leaving headroom to avoid OOM under load.

### Worker pools are separate by stage

Stage 0 (fetch) uses an async I/O pool sized for network concurrency (32–128).
Stages 1–N (compute) use a `ProcessPoolExecutor` sized by the formula above.
These pools must not be shared — saturating the network and saturating the CPU
simultaneously will cause OOM on this machine.

### Chunking

`pool.map` with a large iterable queues all tasks before any complete. Use `chunksize`
to bound queue depth and enable incremental progress reporting:

```python
results = list(pool.map(process_point_task, point_tasks, chunksize=50))
```

For 5,000 points with chunksize=50, at most ~100 task dicts are live in the queue
at any time rather than all 5,000.

---

## Data flow in practice

### Single-thread (development / exploration)

```python
points = load_ala_points(cluster="mitchell_dense", n=50)
items  = fetch_stac_items(bbox=points.total_bounds, years=range(2019, 2025))
store  = DiskChipStore(inputs_dir=Path("inputs/"))

records = []
for point_id, geom in points.itertuples():
    obs      = extract_observations(point_id, geom, items, BANDS, (2019, 2024), store)
    obs      = [score_observation(o, archive_stats) for o in obs]
    features = extract_waveform_features(obs, flowering_index, window=(150, 320))
    records.append(assemble_feature_vector(point_id, features, ...))

df = pd.DataFrame(records)
```

### Parallel training (production)

```python
# Stage 0: fetch all chips — async, I/O-bound, run before compute pool starts
await fetch_chips(points, items, BANDS, inputs_dir=Path("inputs/"))

# Stages 1–N: compute — CPU-bound, ProcessPoolExecutor
point_tasks = [{"point_id": p, "geom": g, "items": items, ...}
               for p, g in points.itertuples()]

with ProcessPoolExecutor(max_workers=_pool_size(),
                         initializer=_worker_init,
                         initargs=(stats_path,)) as pool:
    results = list(pool.map(process_point_task, point_tasks, chunksize=50))

df = pd.DataFrame([r for r in results if r is not None])
```

---

## Relationship to existing pipeline

| Existing module | Role in new pipeline |
|---|---|
| `utils/stac.py` | STAC queries for items covering ALA point clusters |
| `utils/pipeline.py` | `ProcessPoolExecutor` pattern reused; pool sizing updated |
| `utils/io.py` | COG windowed reads; wrapped by `DiskChipStore` / `StreamingChipStore` |
| `config.py` | STAC endpoints, CRS, band names |

The new pipeline adds:
- `stage0/fetch.py` — async chip fetch, `ChipStore` protocol and implementations
- `analysis/timeseries/` — extraction, quality, waveform, feature primitives
- `pipelines/train.py` — training orchestrator
- `pipelines/infer.py` — inference orchestrator

The existing `analysis/06_classifier.py` is eventually replaced by `pipelines/infer.py`,
which sources its feature matrix from the waveform pipeline rather than static seasonal
rasters. The output boundary (Step 07 patches, Step 08 change detection) is unchanged.

---

## Storage and checkpoints

Intermediate outputs are **checkpoints**, not caches. They represent accumulated compute
effort — network round-trips, quality scoring, waveform extraction — that is expensive to
repeat. The terminology is intentional: a cache can be evicted freely; a checkpoint is a
recovery point that should only be discarded deliberately.

### Stage 0 chips

```
inputs/                                        # configurable; EBS mount in production
  {item_id}/
    {band}_{point_id}.tif                      # 5×5 COG chip per (item, band, point)
```

Stage 0 is idempotent — re-running skips existing chips and resumes interrupted fetches.
This is the most expensive checkpoint to reconstruct (pure network time against the STAC
archive). For catchment-wide inference where chip volume exceeds local storage, use
`StreamingChipStore` to bypass disk entirely.

### Pipeline checkpoints

```
checkpoints/
  observations_{cluster}_{year_range}.parquet  # raw + scored observations (extraction work)
  features_{cluster}_{year_range}.parquet      # waveform features (waveform work)
```

Observations are written incrementally as batches complete (not accumulated in memory and
flushed at the end) — a worker crash loses at most one batch of 50 points. A sidecar
`observations_{cluster}_{year_range}.progress` file tracks completed `point_id`s so that
re-entry skips already-processed points rather than reprocessing or duplicating them.

Waveform features are cheap to recompute from the observations checkpoint when index
functions or quality thresholds change — but are checkpointed anyway to avoid repeating
the computation on every run.

### Checkpoint management

```
train.py --from-checkpoint observations    # skip Stage 0 + extraction, use existing chips
train.py --from-checkpoint features        # skip through to feature assembly
train.py --drop-checkpoint chips           # delete inputs/ for this run (confirms before deleting)
train.py --drop-checkpoint observations    # delete observations parquet + progress sidecar
train.py --drop-checkpoint all             # full reset (confirms before deleting)
```

`--drop-checkpoint` prints what will be deleted and requires `--yes` to proceed without
confirmation. Silently deleting hours of network work would be catastrophic.

### Training artefacts (train→infer contract)

```
outputs/
  model_{run_id}.pkl                  # RF weights — only written if validation gate passes
  feature_names_{run_id}.json         # ordered feature list the model expects
  train_metrics_{run_id}.json         # OOB score, temporal CV results, feature importances
  train_manifest_{run_id}.json        # ALA points, year range, quality thresholds, index versions, holdout region
  validation_report_{run_id}.json     # per-region AUC, precision/recall, calibration, confusion matrix
```

`infer.py --model-run-id {run_id}` reads all five. `feature_names` is the binding
contract between training and inference — column order and definitions must match exactly.

`model_{run_id}.pkl` is written **last**, only after spatial validation passes. A model
file's existence is proof it cleared the validation gate — `infer.py` can trust any model
it is handed.

---

## Quality score as a first-class citizen

Quality is carried through every layer, not discarded after masking:

- **Observations:** Each record carries `quality` float and `quality_components` dict
- **Waveform:** Peak detection is quality-weighted; low-quality observations don't
  suppress a genuine spike but also can't manufacture one
- **Feature matrix:** Mean quality score per point per year is itself a feature,
  allowing the RF to learn that noisier inputs produce less reliable signals
- **Output raster:** A companion confidence raster (mean quality of acquisitions
  informing each pixel's prediction) is produced alongside the probability raster

---

## Implementation strategy

The same philosophy applies independently to each pipeline (`train.py`, then `infer.py`).
The two pipelines are developed **sequentially** — inference cannot be meaningfully validated
until the training pipeline has produced a real model and established what the feature
distribution looks like from real data.

### Philosophy

1. **Write tests that verify core scientific assumptions.** For training: "presence points
   have detectable flowering peaks", "peak timing falls within the known window", "presence
   and absence peaks are separable". These tests encode domain knowledge as executable
   assertions. A failing test means either the code is wrong or the biological assumption
   was wrong — both are valuable findings.

2. **Write tests that verify remaining assumptions.** Quality scoring behaviour, edge cases
   (insufficient years, all-cloud acquisitions), feature assembly correctness. By this point
   most of the primitive code exists and has been exercised.

3. **Wire primitives into the pipeline.** The orchestration (Stage 0, parallelism, pool
   management, checkpointing) is written last. It does not emerge from the scientific tests
   and needs its own integration tests — but the primitives it calls are already verified.

4. **Each pipeline stage validates its inputs.** Schema validation (expected columns,
   dtypes) and scientific plausibility checks (quality scores in [0,1], DoY values in
   valid range) run at every stage boundary. The invariant to maintain: if stage 1 input
   validation passes and the primitives are correct, subsequent stage validations should
   not fail. Stages that join in external data (HAND raster, structural features) are
   exceptions — they introduce new failure modes that earlier stages cannot anticipate.

5. **Test data is loaded explicitly before running pytest.** The pipeline exposes a
   `load-testdata` subcommand that fetches and stages fixture chips and writes fixture
   parquet files. `pytest` then runs against what was staged. This is a setup operation,
   not an execution mode — the pipeline itself has only one mode: run for real.

### CLI interface

Each pipeline exposes two kinds of subcommand — data management and execution:

```
# Data management
train.py load-testdata                  # fetch and stage fixture data for pytest
train.py drop-checkpoint chips          # delete inputs/ (confirms before deleting)
train.py drop-checkpoint observations   # delete observations parquet + progress sidecar
train.py drop-checkpoint all            # full reset (confirms before deleting)

# Execution
train.py run                            # full catchment, production parallelism
train.py run --from-checkpoint observations  # resume from existing observations

# Same pattern for inference
infer.py load-testdata
infer.py drop-checkpoint all
infer.py run --model-run-id {run_id}
```

`load-testdata` is a prerequisite for `pytest`, not part of the pipeline execution path.
The analogy is `manage.py` in Django — administrative operations clearly separated from
the main run.

### Development sequence

**Training pipeline first:**

1. Scientific assumption tests + fixtures (waveform discriminability, peak timing, presence/absence separation)
2. Remaining primitive tests (quality scoring, edge cases, feature assembly)
3. `train.py` orchestrator: Stage 0 fetch, parallel extraction, checkpointing, `load-testdata` and `run` subcommands
4. Validate: `load-testdata` + `pytest` → `train.py run`

**Inference pipeline second:**

5. Scientific assumption tests for inference (feature distribution at known locations matches training, raster is not degenerate)
6. Remaining primitive tests (composite logic, feature stack assembly, spatial edge cases)
7. `infer.py` orchestrator: Stage 0 fetch, tile-parallel prediction, `load-testdata` and `run` subcommands
8. Validate: `load-testdata` + `pytest` → `infer.py run`

---

## Spatial validation

Temporal train/test splitting (hold out the most recent season) tests generalisation across
time at the *same locations*. It does not test whether the model generalises to *new places*
— which is what catchment-wide inference actually requires. A model can overfit to the
spatial characteristics of its training locations without that showing up in temporal CV.

Spatial validation uses a held-out geographic region — ALA records from a separate QLD
catchment (e.g. Flinders, Burdekin, Fitzroy) that shares the same species but was excluded
entirely from training. The holdout region is recorded in `train_manifest_{run_id}.json`
and enforced by region, not just by point ID, to prevent spatial leakage from nearby points
in the same cluster.

Drone survey data, when available, should be reserved as validation data in preference to
training data — spatially precise, high-confidence ground truth is more valuable as a
validation signal than as additional training signal, at least until enough exists to serve
both purposes.

### The validation primitive

Spatial validation is implemented as a primitive in `analysis/primitives/validation.py`:

```python
@dataclass
class ValidationResult:
    auc:               float
    precision:         float
    recall:            float
    calibration_error: float
    confusion_matrix:  np.ndarray
    region:            str

def validate_spatial(
    model,
    features:   pd.DataFrame,     # feature matrix for held-out region
    labels:     pd.Series,        # ground truth (1=presence, 0=absence)
    region:     str,
) -> ValidationResult:
    """
    Pure function. Runs model.predict_proba() against held-out features,
    computes validation metrics, returns ValidationResult.
    No side effects — callers decide what to do with the result.
    """
```

The threshold is defined once and imported by both callers:

```python
# analysis/constants.py
SPATIAL_VALIDATION_THRESHOLD = 0.85   # minimum AUC to certify a model for inference
```

### Two callers, one primitive

```
analysis/primitives/validation.py   ← validate_spatial(), pure function, unit tested
        │
        ├── tests/test_spatial_validation.py
        │       calls validate_spatial() against fixture data
        │       asserts result.auc > SPATIAL_VALIDATION_THRESHOLD
        │       fast, deterministic, catches regressions
        │
        └── pipelines/train.py  (final step)
                calls validate_spatial() against real held-out data, live STAC fetch
                writes validation_report_{run_id}.json
                refuses to write model_{run_id}.pkl if result.auc < SPATIAL_VALIDATION_THRESHOLD
```

The test and the pipeline gate are always in sync because they import the same threshold
constant and call the same function. Changing the threshold or the validation logic is a
single edit that propagates to both.

---

## Test pyramid

Nothing runs in the pipeline that isn't already tested as a primitive. The pipeline is pure
orchestration — it moves data between primitives and manages checkpoints, but contains no
logic of its own that could be wrong in an untested way.

```
┌─────────────────────────────────────────────────────────────────┐
│  PIPELINE TESTS   train.py run / infer.py run, live STAC fetch  │
│  Slow. Run before release, not on every commit.                 │
│  Tests: orchestration, Stage 0 fetch, checkpointing,            │
│         worker crashes, network timeouts, resume behaviour       │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│  INTEGRATION TESTS   fixture data, primitives wired together    │
│  Medium speed. Tests: full feature vector assembly end-to-end,  │
│  feature_names contract between train and infer paths           │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│  UNIT TESTS   primitives in isolation, fixture data             │
│  Fast. Always run. This is where scientific assumptions live.   │
│  Tests: extraction, quality scoring, waveform features,         │
│         feature assembly, spatial validation primitive           │
└─────────────────────────────────────────────────────────────────┘
```

Scientific assumption tests (`peak_value > THRESHOLD`, `200 <= peak_doy <= 340`,
`median(presence) > median(absence)`, `auc > SPATIAL_VALIDATION_THRESHOLD`) all live at
the unit level — fast, fixture-backed, always run. The integration and pipeline tiers add
coverage for wiring and operational failure modes, not new scientific logic.

---

## Testing strategy

### Test suite instead of exploration script

```python
def test_presence_points_have_detectable_peaks(presence_observations):
    for point_id, features in results:
        assert features["peak_value"] > FLOWERING_THRESHOLD

def test_peak_timing_is_within_expected_window(presence_observations):
    for point_id, features in results:
        assert 200 <= features["peak_doy"] <= 340

def test_absence_points_have_lower_peak_values(presence_absence_pairs):
    assert np.median(presence_peaks) > np.median(absence_peaks)
```

### Loading test data

Before running `pytest`, test data must be staged:

```
train.py load-testdata    # fetches fixture chips from STAC, writes fixture parquet files
pytest                    # runs against what was just staged — no network access
```

`load-testdata` is idempotent — re-running skips existing files. It is the only time
tests touch the network. After staging, `pytest` runs entirely offline.

### Layered fixtures

Each fixture is the output of the layer below it. Tests load from their corresponding
fixture and are isolated from failures in upstream layers:

```
tests/fixtures/
  cog_chips/
    B03_chip_001.tif
    SCL_chip_001.tif
    ...
  raw_observations.parquet
  scored_observations.parquet
  waveform_features.parquet
  .fixture_commit
```

| Test file | Reads from | Tests |
|---|---|---|
| `test_extraction.py` | `cog_chips/` | windowed reads, band indexing, SCL parsing |
| `test_quality.py` | `raw_observations.parquet` | quality scoring, component masks |
| `test_waveform.py` | `scored_observations.parquet` | peak detection, timing, duration |
| `test_features.py` | `waveform_features.parquet` | feature assembly, discriminability |
| `test_spatial_validation.py` | `waveform_features.parquet` | spatial holdout AUC gate |

### Staleness check

```python
def pytest_configure(config):
    if not SENTINEL_FILE.exists():
        pytest.exit("Test data missing. Run: train.py load-testdata")

    recorded_commit = SENTINEL_FILE.read_text().strip()
    changed = git_diff_names(recorded_commit, current_commit)
    stale = [f for f in changed if f in PIPELINE_SOURCES]
    if stale:
        pytest.exit(f"Test data stale — pipeline sources changed: {stale}\n"
                    "Run: train.py load-testdata")
```

`pytest_configure` rather than a test is deliberate: a failing test means "the code
is wrong"; a failing precondition means "the environment isn't ready".
