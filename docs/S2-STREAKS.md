# Streaking in S2 tile overlap regions

The streaking manifests as a visible N-S band in the classifier output heatmap wherever only one of two overlapping S2 MGRS tiles contributes observations to a pixel. The Kowanyama scene (tiles **54LWH / 54LWJ**) is the primary case.

## Hypothesis 1 — BRDF / viewing-angle differences (NBAR)

The first hypothesis was that the two tiles see the landscape from different viewing angles and the resulting BRDF effect makes the same pixel look spectrally different between tiles. We designed and implemented a full NBAR correction ([docs/research/NBAR.md](research/NBAR.md)) based on Roy et al. 2016 RossThick-LiSparse c-factors, fetching per-acquisition 23×23 solar/view angle grids from `granule_metadata.xml` on the element84 S3 bucket and applying per-pixel c-factors in `extract_item_to_df()`.

**Result:** NBAR alone didn't close the gap. After applying it, same-pixel same-day band ratios between the two tiles were measured and found to still show a systematic offset (e.g. B07 H/J ratio ≈ 1.013–1.080), and crucially the offset was **uncorrelated with VZA difference** (r = 0.007), ruling out geometry as the root cause.

## Hypothesis 2 — Inter-sensor radiometric calibration drift (tile harmonisation)

The residual offset after NBAR was attributed to the fact that S2A and S2B satellites predominantly illuminate different tiles at Kowanyama — so tile H and tile J are effectively from different sensors. Their relative calibration drifts over time (B07 ratio: 1.019 in 2019 → 1.080 in 2025). We designed a data-driven inter-tile harmonisation approach ([docs/research/RADIOMETRIC-HARMONISATION.md](research/RADIOMETRIC-HARMONISATION.md)): compute per-(tile, band, year) scale factors from same-pixel same-day overlap observations, then apply them at feature-extraction time in `compute_features_chunked()`. The module `utils/tile_harmonisation.py` was implemented alongside this.

## Hypothesis 3 — Structural duplicates in the parquet (cross-tile dedup bug)

A separate but related problem was discovered: pixels at the boundary of two MGRS tiles were being written **twice** to the parquet — once per tile — because `_fetch_tile_items()` deduplicates granules within a single tile's result set but never compares across tiles. These duplicates entered `TAMDataset` as double-weighted observations with slightly inconsistent band values. This was fixed in commit `f2482f6` ("S2 Overlap bugfix", Apr 20 2026) by adding a post-write dedup pass inside `collect()` that keeps the row with higher `scl_purity` when two rows share the same `(point_id, date)`.

## Current state

The structural duplicate bug is fixed. The NBAR implementation and tile harmonisation module both exist. The stripe is not fully resolved — we know it has a radiometric calibration drift component that the harmonisation approach targets, but the pipeline needs a re-run with corrections applied to verify whether the stripe disappears in the output heatmap.

---

## Quaids investigation (tam-v3-2025-run2)

### Observed pattern

A horizontal streak at ~82% Parkinsonia probability runs east–west through the Quaids scene, with a gradual southward descent from west to east. The streak appears to terminate at the western end around lon 145.2146. Surrounding rows show different but equally uniform dominant probabilities (row `_0122` → 0.2526, row `_0123` → 0.8277, row `_0124` → 0.7038), giving the heatmap a banded horizontal appearance.

### Initial hypothesis: sub-pixel grid

The identical probability values shared by hundreds of pixels in the same grid row suggested they might share identical time-series — consistent with a sampling grid finer than 10 m causing multiple `point_id`s to fall within the same S2 pixel. This was ruled out.

### Actual cause: per-tile radiometric bias (55KBB vs 55KCB)

Joining the scored output with `tile_id` from the Quaids parquet revealed that the streak is entirely tile-driven. Within each grid row, 55KBB and 55KCB pixels are interleaved at the individual pixel level across the entire scene — there is no clean tile boundary. However, the two tiles produce systematically different model scores:

- **55KBB** pixels in row `_0123`: median prob = 0.8277, with 741 of 869 pixels sharing exactly that value
- **55KCB** pixels in the same row: median prob = 0.22, varied distribution

The root cause is a large radiometric offset between the tiles:

| Metric | 55KBB | 55KCB |
|--------|-------|-------|
| Median B11 | 0.325 | 0.232 |
| Mean obs/pixel | 211.7 | 135.1 |

B11 (SWIR-1) is the single most important feature in the TAM model. The ~40% higher B11 in 55KBB systematically inflates Parkinsonia probability for those pixels regardless of what vegetation is actually present. Because the pixel sampling grid assigns each `point_id` to exactly one tile, adjacent rows happen to land predominantly on different tiles, producing the horizontal banding pattern.

### Confirmation via the streak terminus

Examining the narrow N-S slice at the western streak terminus (lon ≈ 145.2146) confirmed that this is not a tile boundary — 55KBB and 55KCB pixels alternate row by row along every column at that longitude throughout the full north–south extent. The apparent "end" of the streak is simply where the scored scene bbox begins or where pixel density changes, not where one tile stops.

### Fix

Same as for Kowanyama: run `tile_harmonisation.calibrate()` on the Quaids parquet to derive per-(tile, band, year) scale factors from same-pixel same-day overlap observations, then apply them at feature-extraction time. The B11 offset at Quaids (~40%) is substantially larger than the ~1–8% seen at Kowanyama, so the correction is expected to have a proportionally larger impact.

---

## Quaids investigation (v9_spectral, May 2026)

### Observed pattern

Horizontal E-W banding across the full 55KCB-only section of the scene (right/east side). The left/west section (where both 55KBB and 55KCB overlap) scores cleanly.

### Root causes identified

**Bug 1 — S2C not recognized in `_TILE_ID_RE`:**

`utils/pixel_collector.py:69` had `r"^S2[AB]_(\d{2}[A-Z]{3})_"`. Sentinel-2C (operational 2025) item IDs like `S2C_55KBB_20250224_0_L2A` don't match, so `tile_id` was set to `""` for all S2C acquisitions. This caused the entire S2C 2025 fetch for quaids (70.3 M rows, 22 dates) to be written to a hidden file named `.parquet` in the year directory. The regex was fixed to `r"^S2[ABC]_"`.

**Consequence:** `parquet_tile_paths()` (iterdir + suffix check) correctly excluded `.parquet` from scoring, so the scoring run was unaffected by the S2C data. However, the fallback glob in `location.py:301` (`out_dir.glob("*.parquet")`) and `s1_tile_builder.py:45` both match `.parquet` on Linux. The orphaned `.parquet` should be deleted after re-fetching with the fixed regex.

**Bug 2 — Temporal coverage imbalance between 55KBB and 55KCB:**

The current 2025 fetch has no meaningful radiometric bias between tiles (B11 calibration scale_factor ≈ 1.0008; the historical 40% B11 difference was an older-data artifact). The horizontal banding comes instead from **unequal temporal sampling** across the scene:

| Region | 55KBB dates/pixel | 55KCB dates/pixel | Character |
|---|---|---|---|
| West edge (lon 145.19) | ~6 dates | ~15 dates | CB-dominant: wet/transitional season |
| East edge (lon 145.33) | ~11 dates | ~13 dates | More balanced: fuller year coverage |

55KBB contributes primarily dry/late-season acquisitions (Jun, Sep–Dec); 55KCB provides the wet and transitional season (Jan, Mar, Jun–Nov). Because 55KBB orbit tracks are narrower at the western edge of the quaids bbox, the western section gets fewer dry-season observations, giving a different model input distribution → different TAM probability. S2 orbit tracks run NNW→SSE, so the coverage boundaries are roughly E-W, producing horizontal bands at the observation-count transition zones.

The existing `tile_harmonisation` framework corrects radiometric offsets but not temporal coverage imbalances. No immediate fix exists for the temporal issue short of either date-count equalization or accepting the artifact.
