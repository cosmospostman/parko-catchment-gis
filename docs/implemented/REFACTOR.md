# TAM Architecture Refactor Plan

## Goal

Decouple training from inference so models can be trained on national-scale
labeled regions and applied independently to any area of interest. Support
multiple model experiments with clear provenance between code and checkpoints.

## What exists now (Phase 0 — complete)

| File | Purpose |
|------|---------|
| `training/locations/training_regions.yaml` | National labeled bbox registry |
| `training/regions.py` | Load / select `TrainingRegion` objects by ID |
| `utils/s2_tiles.py` | Fetch-and-cache AU S2 tile grid; bbox → tile ID resolution |
| `utils/training_collector.py` | `ensure_training_pixels` — fetch per-tile parquets for selected regions |

**Storage layout (training data):**
```
data/training/
  tiles/{tile_id}.parquet   — one parquet per S2 tile, all pixels within that tile
  index.parquet             — region_id → tile_id mapping (sidecar index)
  chips/{tile_id}/          — raw chip cache (gitignored)
data/cache/
  s2_tiles_au.gpkg          — cached AU S2 tile grid (gitignored, fetched once)
```

**CLI (available now):**
```bash
# Fetch pixels for specific regions
python -m utils.training_collector ensure \
    --regions longreach_presence longreach_absence \
    --start 2020-01-01 --end 2025-12-31

# Fetch pixels for all regions in training_regions.yaml
python -m utils.training_collector ensure --all \
    --start 2020-01-01 --end 2025-12-31
```

---

## Phase 1 — TAM train/score separation

**Problem:** `tam/pipeline.py:run()` conflates training and inference. Training
must be run to score, and scoring always uses the same location as training.

**Target structure:**
```
tam/
  core/
    dataset.py    ← move from tam/dataset.py
    model.py      ← move from tam/model.py
    train.py      ← move training loop from tam/train.py
    score.py      ← move _score_chunk, aggregate_year_probs, score_pixels_chunked
    config.py     ← move TAMConfig
  experiments/
    v1_spectral.py        ← first experiment: spectral bands only
    v2_spectral_fire.py   ← future: + NAFI fire history signal
  pipeline.py     ← thin CLI delegating to core + experiments
```

**Experiment dataclass** (`tam/core/experiment.py`):
```python
@dataclass
class Experiment:
    name: str                       # used to name checkpoint dir
    region_ids: list[str]           # selected from training_regions.yaml
    feature_cols: list[str]         # band columns to use (subset of BAND_COLS)
    model_kwargs: dict              # passed to TAMClassifier(...)
    train_kwargs: dict              # passed to train_tam(...)
```

**Checkpoint provenance:** save `experiment.name` into checkpoint metadata so
`load_tam()` can verify you're loading the right model variant.

**New CLIs:**
```bash
# Train a named experiment
python -m tam.pipeline train --experiment v1_spectral \
    --start 2020-01-01 --end 2025-12-31

# Score any location with an existing checkpoint
python -m tam.pipeline score --checkpoint outputs/models/tam-v1_spectral \
    --location longreach-8x8km --end-year 2024
```

**Key changes:**
- `label_pixels()` in `pipelines/common.py` needs to accept `TrainingRegion`
  list as an alternative to a `Location` object (both have bbox + role).
- `score_pixels_chunked()` moves to `tam/core/score.py` with no training deps.
- Training data loading reads from `data/training/tiles/` via the index, not
  from a per-location parquet.

---

## Phase 2 — Multi-signal experiments

**Problem:** Adding NAFI fire history (or other signals) requires injecting
extra feature columns into the dataset and model. Currently band columns are
hardcoded as `BAND_COLS`.

**Approach:**
- `Experiment.feature_cols` declares what goes into each model variant.
- `TAMDataset` accepts `feature_cols` parameter instead of assuming `BAND_COLS`.
- Signal computation (e.g. NAFI fire recency score per pixel) runs as a
  preprocessing step that appends columns to the tile parquet before training.
- Each signal lives in `signals/` (already partially implemented).

**New experiment example:**
```python
# tam/experiments/v2_spectral_fire.py
from tam.core.experiment import Experiment
from signals.nafi import compute_fire_recency

EXPERIMENT = Experiment(
    name="v2_spectral_fire",
    region_ids=["longreach_presence", "longreach_absence"],
    feature_cols=BAND_COLS + ["fire_recency"],
    model_kwargs={"n_features": len(BAND_COLS) + 1},
    train_kwargs={"n_epochs": 100, "patience": 15},
    preprocess=[compute_fire_recency],   # applied to tile df before training
)
```

---

## Phase 3 — National scoring

**Problem:** Scoring currently requires a named location with a pre-fetched
parquet. National scoring needs to iterate over all AU S2 tiles.

**Approach:**
- Extend `utils/training_collector.py` with an `ensure_inference_pixels` path
  that fetches an entire tile bbox (not just the labeled sub-bboxes).
- `tam score --tile 54LWH` fetches the full tile parquet if not present, then
  runs chunked inference → outputs a ranked pixel CSV + heatmap per tile.
- A national runner script iterates over all AU tiles, submitting each as a
  job (local multiprocess or remote).

---

## Decisions

1. **Inference loads locations as today.** `tam score --location longreach-8x8km`
   uses the existing `Location` abstraction and per-location parquet
   (`data/pixels/{id}/{id}.parquet`). No change to inference data loading.
   Tile-centric storage is training-only.

2. **Incremental parquet updates.** When a new year of data is available,
   `ensure_training_pixels` should append to existing tile parquets rather than
   re-fetching from scratch. Design:
   - The tile parquet stores a `date` column. On each run, query the max date
     already in the parquet and set `start = max_date + 1 day` for the fetch.
   - New observations are written to a dated shard
     (`{tile_id}.{start}_{end}.parquet`) then merged+sorted into the main
     parquet, preserving the pixel-sorted row-group structure that chunked
     inference depends on.
   - The shard `.done` sentinel pattern from `collect()` is reused so
     interrupted incremental fetches resume correctly.

3. **Deduplication at write time.** Cross-tile pixels (bbox straddles two tiles)
   must be deduplicated before the pixel enters any parquet. Strategy:
   - Each pixel is assigned a canonical tile: the tile whose centroid is closest
     to the pixel's lon/lat. `utils/s2_tiles.py` will expose
     `canonical_tile_for_point(lon, lat) -> str` for this.
   - `collect()` / `extract_item_to_df()` filters out any pixel whose canonical
     tile differs from the tile being written. This means each pixel appears in
     exactly one tile parquet — no post-hoc deduplication needed at training load
     time.
