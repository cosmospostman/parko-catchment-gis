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
- `tam/core/dataset.py` / `train.py` — the hot paths hand off to numpy/torch
  immediately; the DataFrame work is small (training sites only, not
  full-catchment). Port these only if profiling shows a bottleneck.
- Analysis scripts under `analysis/` and `sweeps/` — exploratory, not
  production.

**Everything else is in scope.** Priority order below.

---

## Hard Problems to Solve First

Before writing any migration code, resolve these two patterns — they appear
across multiple files and need a consistent solution.

### 1. `merge_asof` (S1 snap in `dataset.py`)

`snap_s1_to_s2()` uses `pd.merge_asof()` with a tolerance window to find the
nearest S1 observation within ±7 days of each S2 date. Polars has no direct
equivalent.

Replacement: sort both frames by date, use `join_asof` (added in Polars 0.19)
with `strategy="nearest"` and a `tolerance` of `7d`. Verify correctness against
the pandas version on a known tile before touching the broader codebase.

```python
s2.join_asof(
    s1.sort("date"),
    on="date",
    by="point_id",
    strategy="nearest",
    tolerance=timedelta(days=7),
)
```

### 2. MultiIndex labels `(point_id, year)` in `train.py` / `score.py`

Pandas `pd.Series` with a `MultiIndex` is used as a label lookup throughout
training and scoring. Replace with a plain Python `dict[tuple[str, int], float]`
built from a two-column Polars groupby result. No Polars equivalent needed —
the dict is small (training pixels only) and lookup is O(1).

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

### Phase 5 — `tam/core/dataset.py` and `tam/core/train.py` (defer)

Port these only after Phases 1–4 are complete and benchmarked. Reasons to
defer:

- These operate on training data (thousands of pixels), not full-catchment
  data (hundreds of millions), so the performance delta is small.
- `dataset.py` contains the S1 snap (`merge_asof`) and rolling despeckle
  — the two hard problems identified above. Having solved them in Phase 1
  (proof of concept on `utils/`) means the actual port will be lower risk.
- `train.py` has heavy MultiIndex label logic; the dict-based replacement
  needs to be proven correct in a staging environment before touching the
  training loop.

When the time comes:
- `despeckle_s1()` rolling window: `over("point_id")` + `.rolling_median()`
- MultiIndex labels: replace with `dict[tuple[str, int], float]` as described
  above
- `.drop(inplace=True)`: reassign (`pixel_df = pixel_df.drop(...)`)

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

- Run the full test suite after each phase. The existing tests use pandas
  fixtures; update them to Polars as files are migrated.
- For Phase 3 (`score.py`), run a side-by-side score on a known tile (e.g.
  `quaids/2025/55KBB.parquet`) and assert that output `prob_tam` values match
  to within 1e-5 before and after the port.
- Do not merge a phase until `pytest` is green and the side-by-side check
  passes.

---

## Expected Outcome

After Phases 1–4:
- No `import pandas` in `utils/`, `signals/`, `tam/core/score.py`,
  `tam/core/global_features.py`, or `tam/pipeline.py`.
- Full-catchment scoring fits in available RAM via lazy predicate pushdown.
- Score throughput improvement of ~3–5× on the IO/groupby path (based on the
  benchmark above).
- Single data-layer mental model for all production code paths.
