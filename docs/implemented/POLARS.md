# Polars Migration Plan

## Motivation

Benchmarked on `quaids/2025/55KCB.parquet` (82M rows, medium-sized location):

| Operation | pandas | Polars |
|---|---|---|
| Read 18 cols | 8.0s, **17 GB RAM** | 1.5s, 6.5 GB RAM |
| filter + groupby | OOM | 1.3s, ~2–3 GB peak |

At full-catchment scale, pandas in the score path will either OOM or require
manual chunking that Polars lazy evaluation makes unnecessary. This is a
performance requirement, not just codebase cleanliness.

Both Polars and DuckDB are already in `requirements.txt`. The goal is to
eliminate pandas entirely and operate on a single data layer.

---

## Scope

**Keep pandas out of scope (unchanged):**
- Analysis scripts under `analysis/` and `sweeps/` — exploratory, not
  production.
- `tam/eval_per_site.py`, `tam/viz_attention.py` — analysis tools, not the
  training or scoring production path.

**Phases 1–4 complete.** `tam/core/dataset.py` and `tam/core/train.py` are
the remaining production files; they are covered by Phase 5 below.

---

## Hard Problem to Solve First

Before writing any Phase 5 code, resolve this pattern — it runs through every
function in `train.py` and `dataset.py`.

### MultiIndex labels `(point_id, year)` in `train.py` / `dataset.py`

`train_tam` broadcasts pixel-level labels to pixel-year granularity and stores
them as a `pd.Series` with a `(point_id, year)` MultiIndex. This type is
threaded through `_broadcast_to_pixel_years`, `_apply_presence_filter`, and
`TAMDataset.__init__`. Every operation on it maps cleanly to a plain Python
`dict[tuple[str, int], float]`:

| pandas | dict |
|---|---|
| `lbl_py[~lbl_py.index.isin(drop_idx)]` | `{k: v for k, v in d.items() if k not in drop_set}` |
| `pd.concat([train_py, val_py])` | `{**train_py, **val_py}` |
| `(train_py_labels == 1).sum()` | `sum(v == 1.0 for v in d.values())` |
| `set(labels.index)` | `set(d.keys())` |
| `labels.index.get_level_values("point_id")` | `{k[0] for k in d}` |
| `labels.get((pid, yr), 0.0)` | `d.get((pid, yr), 0.0)` — **unchanged** |

The dict is small (training pixels only) and lookup is O(1). No Polars
equivalent needed — this is the right type for the job.

`TAMDataset` currently accepts both flat pixel-level `pd.Series` (from tests
and analysis scripts) and MultiIndex `pd.Series` (from `train_tam`). After the
migration it should accept:

- `dict[tuple[str, int], float]` — pixel-year labels (from `train_tam`)
- `dict[str, float]` — pixel-level labels (from tests; auto-broadcast against
  the pixel_df inside `__init__`)
- `None` — inference mode

Type detection is one line: `isinstance(next(iter(labels)), tuple)`. The
`_labels_are_pixel_year` branch in `__getitem__` stays identical; only the
construction and filtering code changes.

---

## Migration Phases

### Phase 1 — Utilities and signals (low risk, do first)

These files have no complex pandas idioms. Port them as a batch to establish
the pattern and remove the pandas/Polars boundary friction.

**Files:**
- `signals/base.py`, `signals/eval.py`
- `signals/ndvi.py`, `signals/ndre.py`, `signals/mavi.py`, `signals/ndsvi.py`
- `signals/s1.py`, `signals/s2_bands.py`, `signals/temporal.py`
- `utils/s1_collector.py`
- `utils/training_collector.py`

**Common pattern replacements:**

| pandas | Polars |
|---|---|
| `df.groupby(col).agg(...)` | `df.group_by(col).agg(...)` |
| `df.dropna(subset=[col])` | `df.drop_nulls(subset=[col])` |
| `df.sort_values([...])` | `df.sort([...])` |
| `pd.concat([a, b])` | `pl.concat([a, b])` |
| `df[col].values.astype(np.float32)` | `df[col].to_numpy()` |
| `df[mask]` | `df.filter(mask)` |
| `s.map(dict)` | `s.replace(dict)` or join on a lookup table |

**Acceptance:** existing signal tests pass unchanged; no `import pandas` remains
in these files.

### Phase 2 — `utils/pixel_collector.py` and `utils/tile_harmonisation.py`

Both already have partial Polars usage. Finish the job:

- `pixel_collector.py`: remove the lazy `import polars as _pl` and make it
  unconditional; replace the remaining pandas concat/groupby calls.
- `tile_harmonisation.py`: Phase 1 (tile counting) is already Polars. Port
  Phase 2 (overlap detection and median aggregation) using `join` + `group_by`.
  The weighted median has no built-in; use
  `pl.col(...).sort().slice(n//2, 1)` for an unweighted approximation or keep
  a small numpy helper for that one step.

### Phase 3 — `tam/core/score.py` (biggest win)

This is the full-catchment inference path. The OOM benchmark above came from
here. Port in this order:

1. **Parquet read:** replace `pd.read_parquet` / PyArrow row-group reads with
   `pl.scan_parquet(...).filter(pl.col("scl_purity") >= scl_purity_min)` —
   predicate pushdown means only clean observations are loaded into RAM.

2. **groupby point_id:** replace the pandas groupby-then-iterate pattern with
   `group_by("point_id").agg(...)` for the band summary computation, and
   `partition_by("point_id")` for building per-pixel windows.

3. **S1 stat lookups:** currently built as `dict[str, pd.Series]`. Replace with
   `dict[str, np.ndarray]` via `.to_numpy()` — no change to downstream tensor
   construction.

4. **Queue boundaries:** the reader → preprocessor → GPU pipeline passes raw
   DataFrames through queues. Convert to Polars at the read step; downstream
   code receives Polars from that point.

The numpy/torch interop at the end of each pixel window is unchanged — Polars
`.to_numpy()` is zero-copy for contiguous float32 arrays.

### Phase 4 — `tam/core/global_features.py` and `tam/pipeline.py`

**`global_features.py`:** straightforward groupby/agg replacements. The one
awkward case is `.idxmax()` for peak DOY — replace with:
```python
df.group_by("point_id").agg(
    pl.col("doy").sort_by("value").last().alias("peak_doy")
)
```

**`pipeline.py`:** replace `.categorical` dtype caching with Polars
`Categorical` dtype, which uses dictionary encoding internally. Replace
`.drop_duplicates()` with `.unique()`.

### Phase 5 — `tam/core/dataset.py`, `tam/core/train.py`, `tam/utils.py`

**Status:** `snap_s1_to_s2` has been removed from `dataset.py` — the `use_s1=True`
(snap) mode no longer exists. Only `use_s1=False`, `"mixed"`, and `"s1_only"` remain.
This eliminates the `merge_asof` hard problem entirely. What remains is the
MultiIndex label system and the pandas DataFrame plumbing.

Port in this order:

#### Step 1 — `tam/utils.py` (30 min, zero risk)

Three small functions — `label_pixels`, `summarise`, `save_pixel_ranking` — all
accept `pd.DataFrame`. Port to `pl.DataFrame`. No logic changes; pure API
translation. Update `pipeline.py` to remove the `scores.to_pandas()` adapter
introduced in Phase 4.

#### Step 2 — `despeckle_s1` in `dataset.py`

The rolling per-pixel median is the last isolated pandas operation. Replacement:

```python
s1 = s1.sort(["point_id", "date"])
for col in ("vh", "vv"):
    if col in s1.columns:
        s1 = s1.with_columns(
            pl.col(col)
              .rolling_median(window_size=window, min_periods=1, center=True)
              .over("point_id")
              .cast(pl.Float32)
              .alias(col)
        )
```

`despeckle_s1` is called from `dataset.py` (mixed/s1_only branches) and from
`score.py`'s `_compute_s1_despeckle_lookup` (which has a local pandas adapter).
Once `despeckle_s1` is Polars, remove the adapter in `score.py`.

#### Step 3 — label types (the core change)

Replace `pd.Series` labels throughout `train.py` and `dataset.py` with plain
dicts as described in **Hard Problem** above.

In `train.py`:

1. **Split functions** (`spatial_split`, `region_holdout_split`,
   `site_holdout_split`): change signatures from `pd.Series → pd.Series` to
   `dict[str, float] → dict[str, float]`. The split logic becomes set
   operations: `{k: v for k, v in labels.items() if k not in val_set}`.
   `spatial_split` still needs `pixel_coords` as a `pl.DataFrame` for the
   lat-sort, but only to extract a sorted list of point_ids — no pandas
   required.

2. **`_broadcast_to_pixel_years`**: replace the `merge + set_index` with:
   ```python
   def _broadcast_to_pixel_years(
       lbl: dict[str, float],
       pixel_years: pl.DataFrame,   # columns: point_id, year
   ) -> dict[tuple[str, int], float]:
       return {
           (row[0], row[1]): lbl[row[0]]
           for row in pixel_years.filter(
               pl.col("point_id").is_in(set(lbl))
           ).iter_rows()
       }
   ```

3. **`_apply_presence_filter`**: the `lbl_py` parameter becomes
   `dict[tuple[str, int], float]`. The `reindex` / `isin` pattern becomes
   dict comprehensions. The `drop_idx` set is built directly from the
   dry-VH computation without needing a pandas index at all.

4. **Counting helpers** (`px_counts`, `raw_counts`, `final_counts`,
   `pid_to_sc`): these are already `dict[tuple, int]` — only the construction
   changes. Replace `pd.Series(...).map(...).value_counts().to_dict()` with a
   plain `Counter`:
   ```python
   from collections import Counter
   raw_counts = Counter(_site_class(k[0]) for k in all_py)
   ```

5. **`pixel_df`** passed to `train_tam`: change type annotation from
   `pd.DataFrame` to `pl.DataFrame`. Update the SCL=6 exclusion (currently
   `drop(index=..., inplace=True)`) to a filter:
   ```python
   pixel_df = pixel_df.filter(
       ~((pl.col("source") == "S2") & (pl.col("scl") == 6))
   )
   ```
   Update `pixel_years` derivation, S1-slim extraction for the presence
   filter, and band summaries computation to use Polars group_by.

6. **`_compute_band_summaries`**: replace pandas groupby/quantile/std with:
   ```python
   grp = s2.group_by("point_id").agg([
       pl.col(c).quantile(0.05).alias(f"{c}_p5")  for c in cols
   ] + [
       pl.col(c).quantile(0.95).alias(f"{c}_p95") for c in cols
   ] + [
       pl.col(c).std().alias(f"{c}_std")           for c in cols
   ])
   ```

7. **`_load_or_compute_global_features`**: `compute_global_features` already
   returns `pl.DataFrame`. Cache I/O becomes `write_parquet` / `read_parquet`.
   The caller in `train_tam` passes it to `TAMDataset` as `global_features_df`;
   update that parameter type.

#### Step 4 — `TAMDataset.__init__` in `dataset.py`

With `pixel_df` now a `pl.DataFrame` and labels as dicts, the `__init__` body
needs these changes:

- Column filters (`df["source"] == "S2"`, `df["scl_purity"] >= min`) →
  `df.filter(...)`.
- `pd.concat([s2_rows, s1_rows])` → `pl.concat([s2_rows, s1_rows])`.
- `df.sort_values(...)` → `df.sort(...)`.
- `df["doy"] = pd.to_datetime(df["date"]).dt.day_of_year` →
  `df = df.with_columns(pl.col("date").cast(pl.Date).dt.ordinal_day().alias("doy"))`.
- `df[feature_cols].values.astype(np.float32)` →
  `df.select(feature_cols).to_numpy().astype(np.float32)`.
- `df.groupby(["point_id", "year"]).size()` for group boundaries →
  ```python
  sizes = (
      df.group_by(["point_id", "year"], maintain_order=True)
        .len()["len"]
        .to_numpy()
  )
  split_points = np.cumsum(sizes)[:-1]
  ```
- The `isinstance(labels.index, pd.MultiIndex)` branch detection →
  `isinstance(next(iter(labels)), tuple)` once labels are dicts.
- `global_features_df.reindex([pid for pid, *_ in self._windows])` →
  build a pid→row lookup dict from the `pl.DataFrame` once, then access by
  key in `__getitem__`.

The pixel z-score blocks in the mixed and s1_only branches use pandas
`groupby().agg(["mean", "std"])` + `reindex`. Replace with Polars group_by
+ join back, or with a numpy-based per-pixel stats pass (already the pattern
in `score.py`'s `_compute_s2_pixel_zscore_stats`).

`__getitem__` and `__len__` do not touch DataFrames and require no changes.

---

## What Not to Port

- **DuckDB in `utils/stac.py`:** DuckDB is the right tool for STAC catalogue
  queries (SQL over remote Parquet). Keep it.
- **numpy operations in `_preprocess_numba.py`:** already below the DataFrame
  layer; leave it alone.
- **Analysis scripts** (`analysis/`, `sweeps/`): one-off tools; not worth the
  churn.

---

## Testing Strategy

- Run the full test suite after each step. Phases 1–4 are complete and green.
- For Phase 5 Steps 1–2 (`tam/utils.py`, `despeckle_s1`): existing unit tests
  pass unchanged after the port. No new tests needed.
- For Phase 5 Steps 3–4 (label types + `TAMDataset`): the `tests/tam/`
  suite is the primary acceptance gate. The conftest `labels` fixture currently
  passes `pd.Series({"px_pres": 1.0, "px_abs": 0.0})`. Update it to
  `{"px_pres": 1.0, "px_abs": 0.0}` (plain dict) — every test that was passing
  before must still pass, providing the correctness guarantee for the label
  type change.
- For the full `train_tam` path: run a smoke training on a known small dataset
  (e.g. two regions, 10 epochs) and assert that `best_val_auc > 0` and the
  checkpoint loads without error. This does not need numerical exactness — only
  that the training loop completes and saves a valid checkpoint.
- Do not merge Phase 5 until `pytest tests/tam/` is green and the smoke
  training passes.

---

## Expected Outcome

After Phases 1–4 (complete):
- No top-level `import pandas` in `utils/`, `signals/`, `tam/core/score.py`,
  `tam/core/global_features.py`, or `tam/pipeline.py`.
- Full-catchment scoring fits in available RAM via lazy predicate pushdown.
- Score throughput improvement of ~3–5× on the IO/groupby path (based on the
  benchmark above).

After Phase 5 (remaining):
- No `import pandas` anywhere in `tam/` except analysis scripts
  (`eval_per_site.py`, `viz_attention.py`) which are intentionally out of scope.
- `TAMDataset` accepts `pl.DataFrame` for `pixel_df` and plain dicts for labels.
- `train_tam` accepts `pl.DataFrame` for `pixel_df`, `pixel_coords`.
- Single data-layer mental model across the entire production code path from
  parquet read to model checkpoint.
