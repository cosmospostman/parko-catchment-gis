# TAMDataset Copy-on-Write Refactor

## Problem

When PyTorch forks DataLoader worker processes, each worker gets a copy of the
parent process memory via copy-on-write (CoW). For large numpy allocations this is
free — the OS shares the physical pages until a worker writes to them, and the band
data is read-only after construction so pages stay shared.

The problem is **Python's reference counting**. Every access to a Python object
increments/decrements its refcount, which is stored inline in the object header.
Reading a Python tuple or list entry *writes* to its memory page, triggering a
private copy even though the data didn't change.

`self._windows` is a Python list of ~200–700k tuples:
```python
(pid_str, year_int, bands_view, doy_view, src_view)
```
Each tuple header, the list array, and the string objects all live on Python-managed
heap pages. Worker 0 reading `self._windows[i]` dirties those pages. With 6–8
workers this multiplies, effectively copying the list and all its tuple objects
into each worker — even though the underlying numpy data is shared.

**Result:** Can't safely raise `n_workers` beyond 4 without excess RAM copies,
which limits DataLoader throughput on a fast GPU (e.g. 5060Ti).

---

## Solution: Struct-of-arrays layout

Replace the list-of-tuples with a small number of contiguous numpy arrays indexed
by window position. `__getitem__` reads from these arrays using integer arithmetic —
no Python object refcounting on the hot path.

### New instance variables (replacing `self._windows`)

| Variable | dtype | shape | replaces |
|---|---|---|---|
| `self._bands` | float32 | `(total_obs, n_features)` | `bands_view` across all windows |
| `self._doys` | int32 | `(total_obs,)` | `doy_view` across all windows |
| `self._sources` | int8 | `(total_obs,)` | `src_view` across all windows |
| `self._offsets` | int32 | `(n_windows,)` | start index into `_bands/_doys/_sources` |
| `self._lengths` | int32 | `(n_windows,)` | obs count per window |
| `self._pids` | object | `(n_windows,)` | `pid_str` per window — unavoidably Python heap |
| `self._years` | int32 | `(n_windows,)` | `year_int` per window |

`_global_feats` dict (keyed by pid string) is replaced by:

| Variable | dtype | shape |
|---|---|---|
| `self._global_feat_arr` | float32 | `(n_windows, n_global)` |

Lookup becomes `self._global_feat_arr[idx]` — a numpy row slice, no dict lookup.

---

## Implementation

### `tam/core/dataset.py`

**`__init__` — build phase (~line 446)**

Replace the `self._windows` construction loop:

```python
# --- current (list of tuples) ---
self._windows = []
for b, d, p, y, src in zip(bands_split, doy_split, ...):
    ...
    self._windows.append((p[0], int(y[0]), b[:n], d[:n], src[:n]))

# --- new (struct-of-arrays) ---
_w_pids    = []
_w_years   = []
_w_offsets = []
_w_lengths = []
_out_bands   = []
_out_doys    = []
_out_sources = []

write_ptr = 0
for b, d, p, y, src in zip(bands_split, doy_split, pid_split, yr_split, source_split):
    # existing min_obs_per_year / MIN_S1_OBS_PER_YEAR guards unchanged
    ...
    n = min(len(b), MAX_SEQ_LEN)
    _out_bands.append(b[:n])
    _out_doys.append(d[:n])
    _out_sources.append(src[:n])
    _w_pids.append(p[0])
    _w_years.append(int(y[0]))
    _w_offsets.append(write_ptr)
    _w_lengths.append(n)
    write_ptr += n

self._bands   = np.concatenate(_out_bands,   axis=0) if _out_bands else np.empty((0, len(feature_cols)), dtype=np.float32)
self._doys    = np.concatenate(_out_doys,    axis=0) if _out_doys  else np.empty(0, dtype=np.int32)
self._sources = np.concatenate(_out_sources, axis=0) if _out_sources else np.empty(0, dtype=np.int8)
self._offsets = np.array(_w_offsets, dtype=np.int32)
self._lengths = np.array(_w_lengths, dtype=np.int32)
self._pids    = np.array(_w_pids,    dtype=object)
self._years   = np.array(_w_years,   dtype=np.int32)
```

Note: `np.concatenate` on the output lists produces a new contiguous allocation.
The original `bands_all`/`doy_all`/`source_all` arrays can then be freed.

**Label filtering (~line 460)**

The existing `_windows` filter needs updating to work with parallel arrays:

```python
if labels is not None and self._labels_are_pixel_year:
    valid_py = set(labels.keys())
    keep = np.array([
        (self._pids[i], int(self._years[i])) in valid_py
        for i in range(len(self._pids))
    ], dtype=bool)
    # Rebuild _bands/_doys/_sources as a new contiguous array over kept windows
    kept_idx = np.where(keep)[0]
    new_bands   = np.concatenate([self._bands  [self._offsets[i]:self._offsets[i]+self._lengths[i]] for i in kept_idx], axis=0) if len(kept_idx) else self._bands[:0]
    new_doys    = np.concatenate([self._doys   [self._offsets[i]:self._offsets[i]+self._lengths[i]] for i in kept_idx], axis=0) if len(kept_idx) else self._doys[:0]
    new_sources = np.concatenate([self._sources[self._offsets[i]:self._offsets[i]+self._lengths[i]] for i in kept_idx], axis=0) if len(kept_idx) else self._sources[:0]
    new_offsets = np.concatenate([[0], np.cumsum(self._lengths[kept_idx])[:-1]]).astype(np.int32) if len(kept_idx) > 1 else np.array([0], dtype=np.int32)
    self._bands   = new_bands
    self._doys    = new_doys
    self._sources = new_sources
    self._offsets = new_offsets if len(kept_idx) else np.empty(0, dtype=np.int32)
    self._lengths = self._lengths[kept_idx]
    self._pids    = self._pids[kept_idx]
    self._years   = self._years[kept_idx]
```

**Global features — replace dict with 2D array (~line 503)**

```python
# Instead of: self._global_feats = {pid: normed[i] for i, (pid,*_) in enumerate(self._windows)}
# Use a 2D array indexed by window position:
self._global_feat_arr = normed.astype(np.float32)   # shape (n_windows, n_global)
```

**`__len__`**

```python
def __len__(self) -> int:
    return len(self._pids)
```

**`__getitem__`**

```python
def __getitem__(self, idx: int) -> TAMSample:
    pid    = self._pids[idx]         # one object array read — single CoW touch
    yr     = int(self._years[idx])
    start  = int(self._offsets[idx])
    n_obs  = int(self._lengths[idx])
    bands_np = self._bands  [start:start+n_obs].copy()   # copy so augmentation is safe
    doy_np   = self._doys   [start:start+n_obs].copy()
    src_np   = self._sources[start:start+n_obs].copy()
    n = n_obs
    # ... rest of augmentation logic unchanged ...
    gf = self._global_feat_arr[idx] if self._n_global > 0 else np.zeros(0, dtype=np.float32)
```

The `.copy()` calls on the slices are necessary because `__getitem__` mutates
`bands_np` in-place during noise injection and dropout. Without copying, concurrent
workers would race on the same memory. Each copy is at most `MAX_SEQ_LEN × n_features × 4`
bytes = 64 × 16 × 4 = 4 KB — negligible.

---

## Files changed

- `tam/core/dataset.py` — only file changed

## What doesn't change

- `TAMSample` namedtuple — unchanged
- `collate_fn` — unchanged
- Caller code in `train.py` — unchanged (`len(train_ds)`, `train_ds.band_stats`,
  `train_ds.global_feat_mean`, `train_ds.global_feat_std` all still valid)
- Score path in `score.py` — unchanged

---

## Expected outcome

- Worker CoW footprint: `_pids` (object array, one touch per `__getitem__`) is the
  only Python-heap structure on the hot path. All band/doy/source data lives in
  contiguous numpy allocations that stay clean across workers.
- Safe to raise `n_workers` to 6–8 on the 5060Ti without proportional RAM increase.
- ~4 KB allocation per `__getitem__` call (the `.copy()` slices) — negligible vs
  current tuple unpacking overhead.

---

## Verification

```bash
# Confirm no regression in training behaviour
LOG_LEVEL=DEBUG python -m tam.pipeline train --experiment v10 2>&1 | grep -E "RSS|epoch|val_cvar"

# Confirm workers don't blow memory — watch RSS stay flat after DataLoader creation
LOG_LEVEL=DEBUG python -m tam.pipeline train --experiment v10 2>&1 | grep RSS
```

With 8 workers, RSS after DataLoader creation should be similar to with 4 workers
(previously the dominant cost was CoW copies of `_windows` tuples).
