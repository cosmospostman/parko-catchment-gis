# Fetch pipeline refactor

## Core insight

Both the location pipeline and the training pipeline are fetching a **named bbox +
years → sorted parquet per tile per year**. They share the same source data, the
same output schema, and the same downstream consumer (`score()`). The current dual
implementation is an accident of history, not a design requirement.

## Unified model: `FetchSpec`

A single dataclass replaces the `Location`/`TrainingRegion` split at the fetch layer:

```python
@dataclass(frozen=True)
class FetchSpec:
    id: str                        # "longreach" or "lake_mueller_presence"
    bbox: list[float]              # [lon_min, lat_min, lon_max, lat_max]
    years: list[int]
    point_id_prefix: str           # "px" for locations, region.id for training
    geometry: object | None = None # polygon mask (locations); None for training regions
    label: str | None = None       # "presence"/"absence" (training); None for locations
    out_dir: Path | None = None    # caller controls output root
    cache_dir: Path | None = None  # explicit override; training uses per-region subdir
```

A single `fetch_spec(spec, ...)` function runs the three-stage pipeline and writes
`<out_dir>/<year>/<tile_id>.parquet` regardless of whether the spec came from a
`Location` or a `TrainingRegion`.

```
fetch_spec(spec)
├── for year in spec.years (parallel):
│   ├── collect_s2(spec, year)  → <out_dir>/<year>/<tile_id>.s2.parquet
│   ├── collect_s1(spec, year)  → <out_dir>/<year>/<tile_id>.s1.parquet
│   └── merge_tile(s2, s1)      → <out_dir>/<year>/<tile_id>.parquet
```

## How existing entry points map onto FetchSpec

**`Location.fetch(years)`** — thin wrapper:
```python
def fetch(self, years, ...):
    spec = FetchSpec(
        id=self.id, bbox=self.bbox, years=years,
        point_id_prefix="px", geometry=self.geometry,
        out_dir=self.parquet_year_dir_root(),
        cache_dir=self.cache_dir(),
    )
    return fetch_spec(spec, ...)
```

**`training_collector._collect_one_region(region)`** — thin wrapper:
```python
spec = FetchSpec(
    id=region.id, bbox=region.bbox, years=region.years,
    point_id_prefix=region.id, label=region.label,
    out_dir=_REGION_DIR / region.id,
    cache_dir=tile_chips_path(tile_id) / region.id,  # per-region isolation preserved
)
tile_paths = fetch_spec(spec, items=tile_items, ...)
```

The training-specific logic that remains in `training_collector` — STAC sharing across
regions on the same tile, the tile rebuild, the index, incremental updates — is
genuinely training-specific and stays. `_collect_one_region` shrinks to a spec
construction + `fetch_spec` call.

## Changes by file

### New: `utils/fetch_spec.py`

Contains `FetchSpec` and `fetch_spec()`. The three-stage logic currently split across
`location.py`, `pixel_collector.py`, `s1_collector.py`, and `parquet_utils.py` is
orchestrated here.

### `pixel_collector.py` — `collect()`

- Output renamed to `<tile_id>.s2.parquet`
- Always returns tile paths (fix the ambiguous `return []` on cache-hit — return
  existing `.s2.parquet` paths instead)
- `sys.exit(1)` replaced with `FetchError(RuntimeError)`
- `calibration_out` moves to `fetch_spec()` as a post-step (multi-tile concern)

### `s1_collector.py` — new `collect_s1_for_tile()`

```python
def collect_s1_for_tile(
    s2_path: Path,
    bbox_wgs84: list[float],
    start: str,
    end: str,
    out_path: Path,
    cache_dir: Path | None = None,
    n_workers: int = 4,
) -> Path | None:
```

- Reads `(point_id, lon, lat)` from `s2_path` via PyArrow group_by
- Calls `_resolve_s1_items()` + `_collect_s1_shards()` + `_sort_s1_shards()` → `out_path`
- Idempotent: returns `out_path` immediately if it exists and is non-empty

### `parquet_utils.py` — new `merge_tile()`

```python
def merge_tile(
    s2_path: Path,
    s1_path: Path | None,
    out_path: Path,
) -> Path:
```

- No S1: copy `s2_path` → `out_path` tagging `source="S2"`
- With S1: calls existing `_merge_sorted_parquets(tag_s2_source=True)`
- Atomic write via `.merge_tmp.parquet` → rename
- Idempotent: skip if `out_path` exists with correct row count

`append_s1_to_tile_parquet` is **deleted**.

### `location.py` — `fetch()`

Replaced by a one-liner that constructs a `FetchSpec` and calls `fetch_spec()`.

### `training_collector.py` — `_collect_one_region()`

Reduced to: construct `FetchSpec`, call `fetch_spec()`, hand paths to the tile rebuild
step. The inline S2/S1 parallel thread logic and the `append_s1_to_tile_parquet` call
are deleted.

## What this fixes

| Concern | Resolution |
|---|---|
| Dual implementation of the same operation | One `fetch_spec()` used by both pipelines |
| S1 partial-failure idempotency | S2/S1 files are immutable; merge is atomic rename |
| Serial year loop | `fetch_spec()` parallelises over years with `ThreadPoolExecutor` |
| `collect()` returning `[]` on cache-hit | Fixed: always returns existing `.s2.parquet` paths |
| Inconsistent error handling | `FetchError` raised uniformly; callers decide how to handle |
| S1 cache dir asymmetry | `FetchSpec.cache_dir` is explicit; training keeps per-region isolation |
| Training regions don't benefit from polygon masking | `FetchSpec.geometry` is available to both |

## What stays separate

- **Tile rebuild and index** (`_rebuild_tile_parquet`, `_update_index`,
  `tile_ids_for_regions`) — training-specific, no location equivalent
- **STAC sharing across regions** on the same tile — training optimisation, stays in
  `training_collector` and passed into `fetch_spec()` via the `items=` parameter that
  `collect()` already supports
- **All internal shard/checkpoint/sort logic** in `pixel_collector` — untouched
- **`_collect_s1_shards`, `_sort_s1_shards`, `_merge_sorted_parquets`** — unchanged,
  just called from new homes

## Implementation sequence

1. Add `FetchError` in `pixel_collector.py`
2. Rename `collect()` output to `.s2.parquet`; fix `return []` on cache-hit
3. Add `collect_s1_for_tile()` in `s1_collector.py`
4. Add `merge_tile()` in `parquet_utils.py`; delete `append_s1_to_tile_parquet`
5. Create `utils/fetch_spec.py` with `FetchSpec` + `fetch_spec()`
6. Reduce `Location.fetch()` to a `FetchSpec` wrapper
7. Reduce `training_collector._collect_one_region()` to a `FetchSpec` wrapper
8. Delete dead code
