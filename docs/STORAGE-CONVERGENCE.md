# Storage pipeline convergence

Documents where the training and general (inference) parquet pipelines share code
and where they intentionally diverge.

## Current state

| Aspect | Training | General | Shared code |
|--------|----------|---------|-------------|
| **Collection entry point** | `ensure_training_pixels()` in `utils/training_collector.py` | `Location.fetch()` in `utils/location.py` | `pixel_collector.collect()` |
| **Region/location definition** | `data/locations/training.yaml` — labeled bbox regions with optional year pin | Individual `data/locations/<id>.yaml` files | — |
| **Storage path** | `data/training/tiles/{tile_id}.parquet` | `data/pixels/{location}/{year}/{tile_id}.parquet` | Both use S2 MGRS tile IDs as filenames |
| **Aggregation layer** | Per-region parquets → tile parquets (regions can share a tile) | One fetch per location-year directly to tile parquet | — |
| **Index** | Explicit sidecar `data/training/index.parquet` (region → tile) | Implicit via directory structure | — |
| **Point ID prefix** | `{region_id}_{row}_{col}` — label baked in at collection time | `px_{row}_{col}` — anonymous | Grid generation in `pixel_collector` |
| **Labeling strategy** | Eager — label assigned via region definition in `training.yaml` | Post-hoc — geometry intersection at scoring time via `label_pixels()` | — |
| **Date windowing** | `[min(year)-5, max(year)]` across tile's regions; `year` field required | Full calendar year per `year` argument | — |
| **S1 integration** | `append_s1_to_tile_parquet()` called on the region parquet after S2 collect | `append_s1_to_tile_parquet()` called on each tile parquet after S2 collect | `parquet_utils.append_s1_to_tile_parquet()` |
| **S1/S2 schema helpers** | — | — | `parquet_utils._extend_schema`, `_conform_table`, `_s1_df_to_arrow` |
| **Calibration** | Not applied | Optional tile harmonisation for multi-tile locations (`data/calibration/<id>.parquet`) | — |
| **Chip cache** | Shared tile-level cache via `tile_chips_path(tile_id)` | Single-tile: shared tile-level cache; multi-tile: per-location cache | `location.tile_chips_path()` |
| **TAM consumer** | `_cmd_train()` — reads tile paths via index lookup | `_cmd_score()` — discovers tile paths via `Location.parquet_tile_paths()` | — |

## Shared infrastructure

Everything below is used identically by both pipelines.

- **`pixel_collector.collect()`** — COG reads, 10 m pixel grid generation, shard
  management and checkpointing, NBAR correction, per-tile parquet writing. Both
  pipelines call it with the same signature; neither duplicates this logic.

- **`s1_collector.collect_s1()`** — S1 VH/VV fetch and chip caching. Called by
  `append_s1_to_tile_parquet`, which both pipelines use.

- **`parquet_utils.append_s1_to_tile_parquet()`** — atomically rewrites a tile
  parquet to add S1 rows, scanning all row groups for the pixel grid (idempotent).

- **`parquet_utils._extend_schema` / `_conform_table` / `_s1_df_to_arrow`** — PyArrow
  schema helpers for merging S2 and S1 tables into the combined schema.

- **`parquet_utils.sort_parquet_by_pixel()` / `ensure_pixel_sorted()`** — pixel-sort
  and sidecar management. Used by the TAM scorer on both training and inference data.

- **`parquet_utils._WRITE_OPTS`** — zstd compression and dictionary encoding settings
  applied to all parquet writes.

- **S2 MGRS tile IDs as filenames** — both pipelines name their output parquets
  `{tile_id}.parquet`. The tile ID is the natural unit of S2 data and provides a
  consistent key for joining training and inference data.

- **Chip cache layout** — single-tile locations (training regions and single-tile
  inference sites) share the same on-disk chip cache at
  `data/pixels/{tile_id}/{tile_id}.chips/` via `location.tile_chips_path()`.

- **Combined S2+S1 parquet schema** — both pipelines produce parquets with the same
  column layout: S2 rows have `source="S2"` with band columns populated and `vh`/`vv`
  null; S1 rows have `source="S1"` with `vh`/`vv` populated and band columns null.

- **STAC search via `utils/stac.search_sentinel2()`** — same endpoint and collection
  (`sentinel-2-l2a` on earth-search). Training caches results under
  `data/training/stac/`; inference re-queries (see future candidates below).

## Intentional divergences

The organisational split is load-bearing and not worth collapsing:

- **Training regions are small, labeled, and time-pinned.** They cover tens of pixels
  and need a `[year-5, year]` window to guard against post-clearance imagery. The
  tile-aggregation layer exists because many small regions land on the same S2 tile and
  need to be merged into one parquet for efficient training reads.

- **Inference locations are large and time-open.** They cover thousands of pixels and
  fetch full calendar years. No aggregation layer is needed.

- **Labeling strategy differs by use.** Training labels must be frozen at collection
  time (baked into `point_id`). Inference labels are applied post-hoc so the same
  pixel data can be relabeled as survey geometries are refined.

## Remaining duplication / future convergence candidates

- **STAC search caching** — training has an explicit pickle cache under
  `data/training/stac/`; the general pipeline re-queries on each run. Could share
  a common STAC cache layer.

- **Idempotency recovery** — both pipelines have the same "collect() returned []
  but parquets exist on disk" fallback, copied independently. Candidate for extraction
  into `pixel_collector` or a thin wrapper.
