# CHECKPOINTS — Mid-Step Tile Checkpointing

## Problem

Steps 01, 02 (_build_baseline), and 03 each process ~25,000 tiles via
`ThreadPoolExecutor`. The per-step sentinel is only written when the *entire*
step completes. Interrupting mid-run (e.g. to tune `TILE_WORKERS`) discards all
completed tile work and the step restarts from scratch.

## Fix: Skip Already-Written Tiles

Before processing a tile, check whether its scratch file already exists and has
non-zero size. If so, return the path immediately without reprocessing. On
restart the executor skips done tiles and only processes the remainder.

This is safe because:
- Tile paths are deterministic (`tile_{idx:05d}.tif`) and unique per step/year
- The scratch directory is only cleaned up after a successful merge
- A partially-written tile (from a crash mid-write) will be zero bytes or
  unreadable — treat any `OSError` or empty file on open as a cache miss and
  reprocess

### Change to `process_tile` in each script

```python
def process_tile(args):
    tile_idx, tile_bbox = args
    tile_path = scratch_dir / f"tile_{tile_idx:05d}.tif"

    # Resume: skip tiles already written in a previous interrupted run
    if tile_path.exists() and tile_path.stat().st_size > 0:
        return tile_path

    # ... existing processing logic ...
```

This is a 3-line change per script, all inside `process_tile`.

## Progress Logging

After each tile completes (whether processed or resumed from cache), log
progress at INFO level so the operator can see how far along the run is:

```python
def process_tile(args):
    tile_idx, tile_bbox = args
    tile_path = scratch_dir / f"tile_{tile_idx:05d}.tif"

    if tile_path.exists() and tile_path.stat().st_size > 0:
        logger.info("Tile %d/%d skipped (cached)", tile_idx + 1, n_tiles)
        return tile_path

    # ... existing processing logic ...

    logger.info("Tile %d/%d complete (%.1f%%)", tile_idx + 1, n_tiles,
                100 * (tile_idx + 1) / n_tiles)
    return tile_path
```

`n_tiles = len(tile_bboxes)` is captured in the closure before the executor
starts. The resume path logs `skipped (cached)` so it's clear on restart which
tiles were replayed from disk vs recomputed. Both lines are INFO so they appear
in the standard log without any config changes.

## Affected Scripts

1. `analysis/01_ndvi_composite.py` — `process_tile` inside `main()`
2. `analysis/03_flowering_index.py` — `process_tile` inside `main()`
3. `analysis/02_ndvi_anomaly.py` — `process_tile` inside `_build_baseline()`

## Scratch Directory Lifecycle

Currently the scratch directory is created with `mkdir(exist_ok=True)` and
removed after a successful merge. This already handles the resume case: if the
directory exists from a prior run, `exist_ok=True` lets it pass, and the
checkpoint check finds the existing tile files.

No change needed to the directory lifecycle.

## Note: Increase Default TILE_WORKERS

The current default of `TILE_WORKERS=8` is conservative. Based on the DO
droplet graphs (CPU and memory both well below capacity, network not saturated),
the bottleneck is concurrency to S3, not hardware. The default should be raised
to `32` in both the analysis scripts and `run.sh`.

The `--tile-workers` flag lets users override this per-run, so existing
behaviour on lower-spec machines is preserved.

## Tests

Add a test to `tests/unit/test_utils_tiling.py` (or a new
`tests/analysis/test_checkpointing.py`) that:

- Calls `process_tile` with a pre-existing tile file of non-zero size
- Asserts that `load_stackstac` is **not** called (i.e. the tile was skipped)
- Calls `process_tile` with a zero-byte tile file
- Asserts that `load_stackstac` **is** called (cache miss → reprocess)

## Implementation Order

1. `analysis/01_ndvi_composite.py` — add resume check, progress logging, raise default to 32
2. `analysis/03_flowering_index.py` — same
3. `analysis/02_ndvi_anomaly.py` — same
4. `run.sh` — raise default `TILE_WORKERS` from 4 to 32
5. Tests
