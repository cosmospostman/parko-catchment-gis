# S2 Radiometric Pipeline — Inter-tile Harmonisation

Horizontal banding in TAM inference output wherever two overlapping MGRS tiles
contribute observations to the same pixel.  Primary cases: Kowanyama (54LWH/54LWJ)
and Quaids (55KBB/55KCB).

---

## Root cause investigation

### Hypothesis 1 — BRDF/viewing-angle (NBAR)

Implemented a full NBAR correction (Roy et al. 2016 RossThick-LiSparse c-factors)
fetching per-acquisition 23×23 solar/view angle grids from `granule_metadata.xml`.

**Result:** NBAR alone didn't close the gap. Same-pixel same-day band ratios between
tiles still showed systematic offsets (e.g. B07 H/J ≈ 1.013–1.080), **uncorrelated
with VZA difference (r = 0.007)** — ruling out geometry as the root cause.

### Hypothesis 2 — Inter-sensor calibration drift (tile harmonisation)

S2A and S2B predominantly illuminate different tiles at Kowanyama and Quaids, so the
two tiles are effectively from different sensors.  Their relative calibration drifts
over time (B07 ratio: 1.019 in 2019 → 1.080 in 2025 at Kowanyama; B11 offset ~40%
at Quaids).

**Fix:** data-driven per-(tile, band, year) scale factors from same-pixel same-day
overlap observations.  Implemented in `utils/tile_harmonisation.py`.

### Hypothesis 3 — Structural duplicates (cross-tile dedup bug)

Pixels at MGRS tile boundaries were written twice to the parquet (once per tile),
entering TAMDataset as double-weighted observations with slightly inconsistent band
values.  Fixed in commit `f2482f6` (Apr 20 2026) by a post-write dedup pass inside
`collect()` that keeps the row with higher `scl_purity` for any `(point_id, date)` pair.

---

## Implemented architecture

Corrections are **baked into the pixel parquet** during the existing concat pass in
`collect()` (`utils/pixel_collector.py`), so every downstream consumer (training,
scoring, `describe`, `explore`) gets harmonised values automatically.

**Sequence:**

1. Fetch shards written and sorted (unchanged)
2. `tile_harmonisation.calibrate(sorted_shard_paths, out_path)` — scans shards to
   derive per-(tile, band, year) scale factors; writes `data/calibration/<loc>.parquet`
3. Concat pass applies `_apply_corrections()` inline then deduplicates (extended)

`Location.fetch()` in `utils/location.py` triggers this automatically for any location
whose bbox spans more than one MGRS tile.

**Key files:**

| File | Role |
|------|------|
| `utils/tile_harmonisation.py` | `calibrate()`, `load_corrections()` |
| `utils/pixel_collector.py` | `_apply_corrections()`, `_flush_write_buf()`, `collect(calibration_out=)` |
| `utils/location.py` | triggers calibration from `fetch()` |
| `data/calibration/<loc>.parquet` | correction table (small, ~KB) |

---

## Remaining issue — temporal coverage imbalance (Quaids)

After fixing the S2C regex bug (`S2[ABC]` not `S2[AB]` in `_TILE_ID_RE`) and
re-running with harmonisation, horizontal banding persists at the western edge of
the Quaids scene.

The remaining cause is **unequal temporal sampling**, not radiometric bias:

| Region | 55KBB dates/pixel | 55KCB dates/pixel |
|---|---|---|
| West edge (lon 145.19) | ~6 (dry/late season) | ~15 (wet/transitional) |
| East edge (lon 145.33) | ~11 | ~13 |

55KBB orbit tracks are narrower at the western edge, so the western section gets
fewer dry-season observations → different TAM input distribution → different
probability scores.  S2 orbit tracks run NNW→SSE, so coverage boundaries are roughly
E-W, producing horizontal bands at the transition zones.

`tile_harmonisation` corrects radiometric offsets; it cannot correct temporal
coverage imbalance.  No immediate fix — accept the artefact or use date-count
equalisation.

---

## What was tried and discarded

The full pre-implementation design docs are in `docs/superseded/`:
- `NBAR.md` — NBAR architecture (implemented, proved insufficient alone)
- `RADIOMETRIC-HARMONISATION.md` — initial harmonisation design (predates baking into parquet)
- `S2-HARMONISATION.md` — plan to bake corrections into concat pass (now implemented)
