# S2 Feature Integration — Approach

## Data source

Continue using Element 84 `sentinel-2-c1-l2a` on AWS via the existing STAC/stackstac pipeline. The collection uses a uniform processing baseline (PB 5.09/5.10) across the full archive, and stackstac applies the BOA_ADD_OFFSET automatically from STAC metadata. BRDF correction via `utils/nbar.py` remains on. Existing cloud masking (SCL + cloud cover filter) is retained unchanged.

Inter-tile radiometric harmonisation is not applied for training data. Training sites are spatially isolated across northern Australia with no contiguous tile overlap, so the calibration has nothing to anchor to. The z-score normalisation described below renders per-tile radiometric offsets second-order.

## Feature representation

For each band, per pixel, across the annual observation stack:

- **Z-scored time series** — each observation normalised by the pixel's own temporal mean and std. Preserves temporal shape (timing of green-up, dry-season retention, rate of senescence) while removing absolute level differences between tiles and sites.
- **p5** — robust dry-season floor; captures how much greenness is retained under stress.
- **p95** — robust wet-season peak.
- **std** — seasonality amplitude; high std = strong wet/dry swing, low std = year-round stability.

The summary statistics (p5/p95/std) are appended as additional inputs alongside the z-scored series, not as a replacement. They give the model explicit access to the phenological envelope without requiring it to infer it from the sequence.

Bands: B02, B03, B04, B05, B07, B08, B8A, B11, B12.

NDVI and NDWI are included as additional features alongside the raw bands — computed from raw reflectance first, then z-scored independently. They are ecologically motivated (NDVI dry-season floor is likely the strongest single Parkinsonia indicator) and provide pre-digested signal the model would otherwise have to derive from B04/B08. EVI is excluded; its additive denominator constant behaves differently under z-scoring and adds little over NDVI here. Whether indices improve over raw bands alone is an ablation to run during the S2-only evaluation phase.

## Model architecture — joint S1 + S2 input stream

S1 and S2 observations are interleaved into a single token sequence sorted by timestamp. Each token carries:

- Band values for its sensor (NaN/zero-padded for the other modality's bands)
- Day-of-year encoded as sin/cos pair (handles cyclicity, represents irregular spacing)
- Sensor flag token (S1 vs S2)

This avoids the need to snap S1 and S2 observations to a common time grid. The transformer attention mechanism learns cross-modal temporal relationships directly — for example, which S1 dry-season tokens are most informative given nearby S2 tokens. S2 cloud gaps are filled by S1 observations in the sequence, so the wet season is no longer a gap in the input.

## Evaluation strategy

Three sequential experiments on the same labelled sites (all have both S1 and S2 coverage):

1. **S2-only transformer** — establish S2 baseline; confirm the phenological signal is discriminating for Parkinsonia before adding complexity.
2. **S1-only (V8)** — existing model as reference point.
3. **Joint S1+S2** — interleaved sequence as described above; does the combined modality beat either alone?

Running all three on identical label sets gives a clean comparison with no confounding from different site distributions. If S2-only underperforms S1-only, that informs whether the joint model is worth the added complexity.
