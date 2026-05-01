# S1 Backscatter Normalisation

## Problem

Sentinel-1 backscatter varies systematically across sites and acquisition geometries in ways
that are unrelated to vegetation structure. Without normalisation, ML models can learn these
artefacts as signal — producing high in-distribution AUC while failing to generalise spatially.

**Empirically confirmed (2026-04-30):** NR VH mean = 17.68 dB vs Cloncurry = 19.37 dB —
a **1.69 dB offset** that reflects incidence angle differences between orbit tracks, not
real landscape differences. Within-pixel temporal std at NR is 1.57 dB. The presence vs
absence signal at NR is only **0.15 dB** — below noise. The model learns "low VH = presence"
as a site-level shortcut, producing spatially incoherent outputs at held-out sites
(Beaudesert prob_tam mean=0.594, std=0.099).

---

## Sources of Variability

### 1. Incidence Angle

The primary driver of site-level backscatter offsets. S1 incidence angle varies from 29° to
46° across its swath (~0.5–1 dB per degree), meaning identical land covers look different
across sites or orbits depending on which part of the swath they fall in.

**Correction approaches:**
- **Linear normalisation**: fit a first-order polynomial, normalise to a reference angle
  (typically 35° or 40°) (Widhalm et al., 2018)
- **Cosine squared normalisation**: Lambert's Law-based, used for large-scale mosaics
  (Haghighi, 2022)
- **Empirical per-pixel slopes**: account for surface roughness and vegetation structure
  varying the angle-backscatter relationship (Widhalm et al., 2018)

### 2. Ascending vs Descending Orbit

Different look directions and diurnal acquisition times (morning dew, vegetation water
content) produce systematic offsets between orbits. Mixing passes without correction adds
unexplained variance that the model may interpret as signal.

**Correction approaches:**
- Treat ascending and descending as separate features or compute monthly medians separately
  per orbit (Maleki et al., 2024)
- Global reprocessing using forward models (Fan et al., 2025)

### 3. Site-Level Baseline Offsets

Soil type, topography, and permanent vegetation structure cause pixel-level baseline
differences that are stable in time but vary across sites.

**Correction approaches:**
- **Per-pixel z-scoring**: subtract the pixel's multi-year mean, divide by std. Removes
  both incidence angle and site baseline effects, leaving only the seasonal shape
  (Park et al., 2019). Most tractable for our use case.
- **Anomaly detection**: express each observation relative to a reference image or the
  pixel's own historical mean (Massart et al., 2024; Haghighi, 2022)

---

## Findings: Orbit Mixing and Temporal Density

Checked training and transfer tiles (2026-04-30):

| Site | Years | Obs/pixel/year | Orbit pattern |
|---|---|---|---|
| NR | 2020–2025 | ~30 | 5, 7, 12d gaps — mixed orbits |
| Cloncurry | 2020–2025 | ~33 | 5, 7, 12d gaps — mixed orbits |
| Lake Mueller | 2016–2021 | ~26 | 12, 24d gaps — single orbit |
| Barcoorah | 2016–2021 | ~26 | 12, 24d gaps — single orbit |
| Stockholm | 2020–2025 | ~43 | mixed, 1–36d gaps |

**Orbit mixing confirmed** at NR and Cloncurry (training sites) — both ascending and
descending passes present. Lake Mueller and Barcoorah (transfer/holdout sites) are
single-orbit only, likely because their smaller bboxes fall in the footprint of only one
relative orbit track, plus coverage was reduced after S1B failure (Dec 2021).

**Year mismatch:** arid training sites (LM, Barcoorah) are year=2021 data; NR/Cloncurry
are 2020–2025. The model trains on multi-year time series but the 2021 arid sites have
lower temporal density and single-orbit data — a confound independent of climate zone.

**S1B failure timeline:** S1B failed 23 December 2021, reducing global revisit from 6-day
to 12-day until S1C launched April 2025. Sites acquired in 2021 (LM, Barcoorah) were
affected in the second half of the year.

---

## Recommendations for This Project

In priority order:

1. **Per-pixel z-scoring** ✓ *implemented* — normalise each pixel's VH/VV time series by its
   own multi-year mean and std before feeding to the transformer. This is the highest-leverage
   fix: removes site-level offsets and incidence angle effects simultaneously, leaving the model
   to learn seasonal curve shape rather than absolute backscatter levels. Enabled via
   `pixel_zscore=True` in TAMConfig; experiment `v8_s1_zscore` uses this with the best sweep
   hyperparams (lr=5e-6, dropout=0.7, d_model=64).

2. **Store orbit direction at fetch time** ✓ *implemented* — `sat:orbit_state` is now
   extracted from STAC item properties and stored as an `orbit` column in S1 collector
   output. Enables orbit-aware normalisation and prevents ascending/descending backscatter
   from being blended without correction. Existing parquet files pre-date this change and
   will have NaN for `orbit`.

3. **Re-fetch arid sites for more recent years** — Lake Mueller and Barcoorah currently
   have only 2021 data (single-orbit, lower density). Fetching 2023–2025 would give
   multi-orbit, higher-density time series consistent with NR/Cloncurry.

4. **Incidence angle as explicit feature** — add incidence angle as an additional band so
   the model can learn to discount it, rather than treating it as vegetation signal.

5. **Separate orbit statistics** — compute band_mean/band_std separately for ascending and
   descending passes, or treat orbit direction as a conditioning feature.

---

## Connection to Phase Shift Augmentation

The `doy_phase_shift=True` training augmentation (random full-year wraparound of the time
series) was introduced to force the model to learn curve shape rather than calendar-time
patterns. Per-pixel z-scoring is the complementary fix on the amplitude axis — phase shift
addresses the temporal dimension, z-scoring addresses the magnitude dimension.

Together they should produce a model that learns: "presence pixels have a stable, low-variance
VH curve regardless of when in the year or what the absolute backscatter level is."

---

## References

Fan, D. et al. (2025). A Sentinel-1 SAR-based global 1-km resolution soil moisture data
product. *Remote Sensing of Environment*, 318, 114579.

Gao, Q. et al. (2017). Synergetic Use of Sentinel-1 and Sentinel-2 Data for Soil Moisture
Mapping at 100 m Resolution. *Sensors*, 17(9), 1966.

Haghighi, M. H. (2022). Large-scale mapping of flood using Sentinel-1 radar remote sensing.
*ISPRS Archives*, XLIII-B3-2022, 1097–1102.

Maleki, S. et al. (2024). Machine Learning-Based Summer Crops Mapping Using Sentinel-1 and
Sentinel-2 Images. *Remote Sensing*, 16(23), 4548.

Massart, S. et al. (2024). Mitigating the impact of dense vegetation on the Sentinel-1
surface soil moisture retrievals over Europe. *European Journal of Remote Sensing*, 57.

Park, J.-W. et al. (2019). Classification of Sea Ice Types in Sentinel-1 SAR images.
*The Cryosphere Discussions*.

Widhalm, B. et al. (2018). Simplified Normalisation of C-Band SAR Data for Terrestrial
Applications in High Latitude Environments. *Remote Sensing*, 10(4), 551.
