# Band Jitter — Making TAM Invariant to Tile Radiometric Offsets

## Problem

Sentinel-2 has systematic inter-tile radiometric offsets of roughly ±5–15% in raw
reflectance. These arise from differences in viewing geometry, atmospheric correction
residuals, and processing baseline across MGRS tiles.

Tile harmonisation (`tile_harmonisation.calibrate()`) can remove these offsets, but
only for tiles that are **contiguously connected** — i.e. share overlap pixels with a
reference tile. This is a hard constraint: calibration requires paired observations of
the same ground at the same time from two adjacent tiles.

The v4 training set has five disconnected tile components:

| Component | Tiles | Sites |
|---|---|---|
| Frenchs | 54LWH — 54LWJ | Frenchs Creek (presence + absence) |
| Maria Downs / Norman Road | 54KWC | Maria Downs, Norman Road |
| Mitchell River | 54LXH | Mitchell River absence |
| Quaids | 55KCB | Quaids absence |
| Unknown (empty tile\_id) | — | Lake Mueller, Barcoorah |

Because these components are disconnected, there is no cross-site calibration anchor.
Each component can only be normalised internally, which does nothing to remove
between-site offsets — the offsets that actually matter for generalisation.

The problem is worse at inference time. The model scores every pixel in a catchment
across many tiles, almost none of which appeared in training. Harmonisation corrections
cannot be computed for tiles with no overlap neighbour in the dataset. A model trained
on harmonised data would encounter uncorrected reflectances at inference — a
train/inference distribution mismatch that harmonisation cannot fix.

## Solution: per-window band jitter

Train the model to be invariant to a constant per-tile offset by injecting one during
training. At each `__getitem__` call, sample a single offset vector once per
(pixel, year) window and add it to every observation in that window before feeding it
to the model. The model sees a different "tile calibration" each time it sees a pixel,
and learns to ignore it.

This makes tile harmonisation **largely irrelevant** — the invariance is baked into
the weights rather than depending on a preprocessing step that may or may not be
computable for a given tile.

### Noise design

Two independent noise terms:

**Per-window offset** (`band_noise_std`, applied once per sequence):
- Models tile-level radiometric bias — the dominant failure mode
- Sampled as `N(0, σ_w)` in z-score normalised space, broadcast across all
  observations in the window
- Target σ_w: a 5–15% raw reflectance offset on B08 (mean ~0.3, std ~0.1) maps to
  ~0.5–1.5 normalised units; a value of **0.5** covers the realistic range

**Per-observation noise** (`band_obs_noise_std`, applied i.i.d. per observation):
- Models atmospheric correction uncertainty and measurement noise
- Much smaller: **~0.1** normalised units
- Optional; secondary priority

### Implementation

`band_noise_std` is already accepted as a parameter in `TAMDataset.__init__` (line 85)
with a comment describing its intent — it is a stub that was never wired up.

Changes needed:

1. **`tam/core/dataset.py`** — store `band_noise_std` in `__init__`; in `__getitem__`,
   after z-score normalisation and before padding, sample one offset per feature
   `N(0, band_noise_std)` and add it to all rows of `bands_np`

2. **`tam/core/config.py`** — add `band_noise_std: float = 0.5` to `TAMConfig`

3. **`tam/core/train.py`** — pass `band_noise_std=cfg.band_noise_std` to the training
   `TAMDataset`; val dataset keeps `band_noise_std=0` (no augmentation at eval time,
   matching the existing pattern for `doy_jitter`)

### What to tune

`band_noise_std` should be treated as a regularisation hyperparameter. Too small and
the model remains tile-sensitive; too large and it discards real spectral signal.
Start at 0.5 and include it in the hyperparameter search alongside `dropout` and `lr`.

## Relationship to S2-HARMONISATION.md

The harmonisation pipeline described in `S2-HARMONISATION.md` solved the immediate
stripe artefact (55KBB vs 55KCB at Quaids). Band jitter is the longer-term fix that
makes that correction unnecessary for the model itself.

Harmonisation corrections in the pixel parquets can be retained for signals-based
analysis (where the model is not involved), but they should not be considered a
prerequisite for TAM training or inference quality once band jitter is in place.
