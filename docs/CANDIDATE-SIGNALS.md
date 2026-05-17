# Candidate Signals for Future Investigation

Signals identified in the literature as biophysically promising for Parkinsonia discrimination but not yet investigated in this project. Each entry has enough context to scope a concrete analysis.

**Already investigated (not repeated here):**
- Red-edge ratio (B07/B05 → `re_p10`) — `docs/research/LONGREACH-RED-EDGE.md`
- SWIR moisture index (NDMI/swir_mi) — `docs/research/LONGREACH-SWIR.md`
- MAVI — `docs/MAVI.md`
- S1 SAR backscatter — V8 model (`docs/models/V8-TRAINING.md`)
- Temporal amplitude, peak DOY, recession — `docs/research/recession-and-greenup/`, `docs/research/day-of-year-peak/`

---

## Candidate index

| Signal | Formulation | Priority | Key uncertainty |
|---|---|---|---|
| NDRE / CI_RE | (B8A−B05)/(B8A+B05); (B07/B05)−1 | Medium | Redundancy with re_p10; B8A availability |
| SMA — PV/NPV/BS | Linear unmixing, 3 endmembers | High | Endmember selection; NPV fraction novelty |
| NDSVI + B11/B12 decoupling | (B11−B04)/(B11+B04); B12/B11 ratio | Medium | Redundancy with swir_mi |
| Explicit temporal phenometrics | σ_t, IQR, phase delay of NDVI | Medium-high | Phase delay requires rainfall reference |

---

## 1. NDRE / CI_RE — Red-Edge Chlorophyll Indices

### Hypothesis

Parkinsonia's stems are rich in chlorophyll-a and -b. When it drops its leaves under dry-season moisture stress, the stems maintain a distinct red-edge cliff — the sharp transition between visible red absorption and NIR reflection. Annual grasses and herbs have no photosynthetic stems; when they senesce, their red-edge signature collapses into a flat soil-litter spectrum. Evaluating the red-edge position in the late dry season (Jul–Sep) exposes sub-pixel green-stemmed woody perennials against a dead herbaceous background.

This is a different hypothesis from the existing `re_p10` investigation. `re_p10` used B07/B05 and captured the *wet-season chlorophyll flush* (March–April peak contrast). The *photosynthetic stem signal* is maximised in the **late dry season** when leaves are absent and grass is fully senescent — a window where re_p10 contrast was weakest.

### Formulation

```
NDRE  = (B8A - B05) / (B8A + B05)
CI_RE = (B07 / B05) - 1
```

- **B05** (~705 nm): red-edge 1, sensitive to chlorophyll concentration
- **B07** (~783 nm): red-edge 3, sensitive to canopy structure and chlorophyll
- **B8A** (~865 nm): narrow NIR plateau band — more sensitive to canopy properties than broad B08 (842 nm) because it avoids the atmospheric water vapour absorption edge

CI_RE is a ratio form (not normalised difference) that becomes more sensitive at high chlorophyll concentrations — better suited to dense or well-hydrated canopy. NDRE is the normalised version with easier cross-site comparison.

### Connection to what we know

The existing `re_p10` investigation used B07/B05 and found:
- Peak contrast March–April (wet-to-dry transition)
- IQR overlap fraction 0.00 at Longreach, but this was a grass-only absence class
- Independence from rec_mean confirmed (r = 0.087)

NDRE/CI_RE likely correlates highly with re_p10 in the wet season. The testable question is whether CI_RE *stays elevated* during Jul–Sep (late dry) when re_p10 falls — if stems retain chlorophyll while leaves are absent, CI_RE diverges from re_p10 at that window. That late-dry residual would be a new axis.

### Investigation sketch

1. Compute CI_RE time series at Landsend (sparse presence + grass absence labels)
2. Plot monthly CI_RE profiles for each class — look for a dry-season floor that stays elevated for presence but collapses for absence
3. Compute contrast fraction specifically in Jul–Sep vs Mar–Apr and compare to the re_p10 results
4. Check Pearson r between CI_RE annual p10 and re_p10; if r ≥ 0.7, the signals are redundant and the wet-season flush dominates both
5. Check B8A availability — it may not be in all training parquets (V9 uses B07 but not necessarily B8A)

### Priority

**Medium.** Fast to compute from existing parquets if B8A is available. Most value if the late-dry stem signal turns out to be independent of the early-wet re_p10 signal. If r ≥ 0.7, close the investigation quickly.

---

## 2. Spectral Mixture Analysis — PV/NPV/BS Fractional Cover

### Hypothesis

In a sparse infestation, absolute greenness (PV) of a single Parkinsonia canopy is heavily diluted within a 10 m pixel. However, across a multi-temporal profile, pure annual grass transitions entirely from ~100% PV in the wet season to ~100% NPV in the dry season. Sparse Parkinsonia creates an unmixing anomaly:

- A persistent, low-percentage **PV floor** year-round (photosynthetic stems and any retained leaves)
- A structurally stable **NPV fraction** from its dense branch and twig matrix that does not fully collapse in any season

The NPV trajectory is the novel axis. A grass pixel and a sparse-Parkinsonia pixel can have the same NDVI (both look green when grass is lush) but the Parkinsonia pixel has a meaningfully different NPV fraction because the woody skeleton is always present.

### Formulation

Three-endmember linear spectral unmixing across the full S2 optical stack:

```
ρ_pixel = f_PV * ρ_PV + f_NPV * ρ_NPV + f_BS * ρ_BS + ε

subject to: f_PV + f_NPV + f_BS = 1, all f ≥ 0
```

Endmembers selected from the scene itself (image-derived, not a fixed library) using NDVI × SWIR space to locate pure-grass, pure-dry-litter, and pure-soil pixels. The inversion is a constrained least-squares per observation.

Derived temporal features:
- **Multi-year PV IQR** — low for stable woody cover, high for volatile grass
- **Dry-season NPV floor** — annual p10 of NPV fraction during Jul–Sep
- **NPV seasonal amplitude** — difference between wet-season minimum NPV and dry-season maximum NPV

### Connection to what we know

PV trajectory will correlate with rec_mean (seasonal NDVI amplitude) and nir_cv (NIR stability) — these are not new. The novel axis is **NPV fraction stability**: a grass pixel transitions fully to NPV; a sparse Parkinsonia pixel does not, because the woody skeleton contributes persistent NPV that doesn't decompose or blow away between seasons.

This directly addresses the sub-pixel problem identified in OVERVIEW.md (§"Detection threshold: ~10% fractional cover"): SMA can detect the fractional-cover anomaly even when the pixel-level NDVI is dominated by background.

### Investigation sketch

1. Select endmembers from Longreach parquet using NDVI × B11 scatter: low-NDVI/high-B11 = bare soil, high-NDVI/low-B11 = PV, high-B11/mid-NDVI = NPV (dry grass litter)
2. Invert per observation; compute PV, NPV, BS time series per pixel
3. Compute dry-season NPV floor (annual p10, Jul–Sep) and NPV IQR for presence vs absence classes
4. Check whether NPV floor ordering (presence > absence) holds when PV profiles are similar — that's the discrimination signal
5. Correlation: test NPV floor against rec_mean, nir_cv, swir_p10; look for r < 0.7 on at least one axis

### Priority

**High.** Most conceptually novel relative to existing features — the only signal that explicitly separates woody skeleton from green canopy from bare soil. Computationally heavier than index-based approaches (requires the unmixing inversion per observation) but not expensive at the pixel-sample scale of existing Longreach data.

---

## 3. NDSVI + B11/B12 Structural Decoupling

### Hypothesis

Two related but distinct signals:

**NDSVI:** By substituting B04 (red, chlorophyll-absorbing) for B08 (NIR, structural) in the denominator, NDSVI makes the ratio sensitive to the chlorophyll-absorption side of the transition from green to senescent biomass. Parkinsonia's photosynthetic stems suppress the red reflectance rise during senescence (less chlorophyll degradation than grass) while B11 rises for both classes as canopy water depletes. The ratio therefore diverges between a structurally woody pixel and a senescing grass pixel in a way that swir_mi does not.

**B11/B12 decoupling:** B11 (~1610 nm) is primarily sensitive to liquid water content. B12 (~2200 nm) is primarily sensitive to dry cellulose and lignin absorption. In bare soil, B11 and B12 are tightly coupled — both respond to soil moisture and texture, tracking together through the season. In a woody skeleton (dense branches, persistent woody litter), B12 is elevated by dry structural carbon regardless of moisture conditions, while B11 varies with rainfall. The ratio B12/B11 (or its temporal variance) separates a structurally woody pixel from bare soil even when both look spectrally similar in NDVI or MAVI.

### Formulation

```
NDSVI     = (B11 - B04) / (B11 + B04)
B12_B11   = B12 / B11
```

- **B04** (665 nm): chlorophyll red absorption
- **B11** (1610 nm): liquid water + dry biomass sensitive
- **B12** (2190 nm): dry cellulose/lignin absorption, less water-sensitive than B11

### Connection to what we know

swir_mi = (B08−B11)/(B08+B11) was investigated at Longreach (LONGREACH-SWIR.md) and found redundant with re_p10 (r = 0.729). The shared variance was attributed to canopy structure — both are measuring the same underlying physiological state (active vs desiccated canopy). NDSVI is structurally different: it replaces B08 with B04, making it sensitive to *chlorophyll absorption* rather than *canopy structure*. Moderate redundancy with swir_mi is expected but not guaranteed.

B11/B12 decoupling is not captured by any existing feature and addresses a specific scenario: distinguishing bare/sparse soil from a woody skeleton when both have low NDVI.

### Investigation sketch

1. Compute NDSVI and B12/B11 ratio time series from Longreach parquet (all bands present)
2. Run the same contrast / p10 / correlation pipeline as LONGREACH-SWIR.md
3. Key test: Pearson r of NDSVI annual p10 against swir_p10 and re_p10; if both < 0.7, earns a place
4. For B12/B11: compute per-pixel temporal variance; test whether presence pixels show higher temporal B12/B11 variance than absence (the decoupling effect)
5. Test NDSVI specifically in the Jul–Sep window where swir_mi had weakest contrast

### Priority

**Medium.** Both indices are cheap to compute from existing parquets (B04, B11, B12 all collected). Likely partially redundant with swir_mi — the investigation can be scoped as a quick correlation check first; proceed to full analysis only if the r < 0.7 criterion holds.

---

## 4. Explicit Temporal Phenometrics

### Hypothesis

Annual grasses respond to rainfall pulses with large, rapid NDVI spikes followed by rapid decay. Parkinsonia, buffered by its deep root network, exhibits a flatter, delayed, and more stable response curve. Two pixel-level statistics capture this:

- **Temporal variance (σ_t, IQR):** Sparse woody invaders display significantly *lower* temporal NDVI variance than purely grass-dominated zones. The IQR of a rolling 12-month NDVI stack separates the volatile grass phenology from the buffered woody phenology without requiring knowledge of the rainfall calendar.
- **Phase delay:** Parkinsonia's green-up lags the rainfall peak (roots respond to the deep soil moisture front, not the surface pulse). Grasses respond near-instantly to surface rain. The offset between a pixel's NDVI peak DOY and the site's mean rainfall peak DOY separates deep-rooted woody plants from shallow-rooted grasses.

### Formulation

Per-pixel, per-year:
```
sigma_t    = std(NDVI) across qualifying observations in the year
ndvi_iqr   = p75(NDVI) - p25(NDVI) across qualifying observations in the year
phase_delay = DOY_of_peak_NDVI - DOY_of_peak_rainfall  (site-median rainfall from BoM gridded data)
```

Multi-year summaries: mean sigma_t, mean ndvi_iqr, mean phase_delay; std of phase_delay across years (calendar consistency).

These can be computed for any index (NDVI, MAVI, CI_RE) — NDVI is the most interpretable starting point.

### Connection to what we know

The TAM attention visualisation for V8 already shows the model attending to phenologically meaningful windows — wet-season at northern sites, Nov/Dec at Lake Mueller. This suggests the transformer is implicitly learning temporal buffering without being given explicit variance or phase features. Explicit phenometrics would:

1. Make the signal testable and interpretable independently of the model
2. Provide features for simpler classifiers (RF) in the V10 feature ablation
3. Allow pre-screening of training pixels by temporal variance quality (low obs count years produce artificially low σ_t)

rec_mean (wet-to-dry NDVI amplitude) captures a related but distinct property: the *total swing* rather than the *within-year variance*. A pixel could have high rec_mean (large wet-to-dry drop) but low σ_t if the transition is smooth. σ_t specifically captures the inter-observation volatility — whether the pixel fluctuates rapidly or plateaus.

nir_cv (dry-season NIR coefficient of variation) captures within-dry-season NIR stability. σ_t computed over the full annual window is a different statistic that includes the wet-to-dry transition shape.

### Investigation sketch

1. Compute per-pixel annual σ_t and NDVI IQR at Landsend (sparse presence + grass absence)
2. Check class ordering and IQR separation — expect presence < absence (lower variance)
3. Correlation test: σ_t against nir_cv, rec_mean; if r < 0.7 with both, σ_t adds a new axis
4. Phase delay: requires BoM AWAP monthly gridded rainfall at each pixel location; compute DOY-of-peak per pixel-year; compare to site's mean rainfall peak month
5. Test whether mean phase_delay differs between presence and absence; test inter-annual std of phase_delay (presence should be more calendar-consistent than grass)

### Priority

**Medium-high.** σ_t and NDVI IQR are cheap and can be computed immediately from existing parquets. Phase delay is the most biophysically novel but requires rainfall data — this is already flagged in OVERVIEW.md as a proposed feature ("rainfall-normalised anomaly") so the infrastructure question is known. Start with σ_t/IQR; tackle phase delay when rainfall data integration is prioritised.

---

## References

- Misra, G., Cawkwell, F., & Wingler, A. (2020). Status of Phenological Research Using Sentinel-2 Data: A Review. *Remote Sensing*, 12(17), 2760.
- Rusňák, T., et al. (2022). Detection of Invasive Black Locust (*Robinia pseudoacacia*) in Small Woody Features Using Spatiotemporal Compositing of Sentinel-2 Data. *Remote Sensing*, 14(4), 971.
- Toqeer, A. (2026). Remote Sensing of Woody Plant Encroachment: A Global Systematic Review. *Remote Sensing*, 18(3), 390.
- Bradshaw, T. M. (2022). Thesis: SWIR moisture indices for groundwater-dependent vegetation. University of Wyoming.
- Levick, S. R. (2021). Remote sensing of gamba grass in northern Australia. Resilient Landscapes Hub.
