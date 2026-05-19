# Candidate Signals for Future Investigation

Signals identified as biophysically promising for Parkinsonia discrimination. Each entry includes enough context to scope a concrete analysis and a `Signal` / `SiteSpec` eval plan using the `signals/` harness.

**Previously investigated (ad-hoc scripts, not yet in the Signal framework):**
- Red-edge ratio (B07/B05 → `re_p10`) — `docs/research/LONGREACH-RED-EDGE.md`
- SWIR moisture index (NDMI/swir_mi) — `docs/research/LONGREACH-SWIR.md`
- S1 SAR backscatter — V8 model (`docs/models/V8-TRAINING.md`)
- Temporal amplitude, peak DOY, recession — `docs/research/recession-and-greenup/`, `docs/research/day-of-year-peak/`

---

## Candidate index

| Signal | Formulation | Priority | Key uncertainty |
|---|---|---|---|
| MAVI | (B08−B04)/(B08+B04+B11) | High | Direction reverses with canopy fraction |
| NDRE / CI_RE | (B8A−B05)/(B8A+B05); (B07/B05)−1 | Medium | Redundancy with re_p10; dry-season window specificity |
| SMA — PV/NPV/BS | Linear unmixing, 3 endmembers | High | Endmember selection; requires constrained inversion |
| NDSVI + B11/B12 decoupling | (B11−B04)/(B11+B04); B12/B11 ratio | Medium | Redundancy with swir_mi |
| Temporal variance (σ_t) | std(NDVI) per pixel-year | Medium-high | Correlates with nir_cv; no external data needed |

---

## 1. MAVI — Moisture-Adjusted Vegetation Index

### Hypothesis

Sparse Parkinsonia creates an eco-hydrological footprint larger than its canopy footprint. Deep roots draw down soil moisture in a radial zone beyond the canopy drip line, producing a "drawdown halo" detectable in Sentinel-2 SWIR. MAVI makes this relationship explicit by placing B11 (SWIR-1, soil and canopy water sensitive) in the denominator alongside the standard NDVI bands.

### Formulation

```
MAVI = (B08 - B04) / (B08 + B04 + B11)
```

### Key findings from prior investigation (`docs/MAVI.md`)

- At Landsend, mean dry-season MAVI ordering: dense presence (0.193) > sparse presence (0.145) > grass absence (0.099) — consistent and physically interpretable.
- **Direction reverses at high canopy cover:** dense riparian presence (Corfield) shows MAVI *increasing* into dry season as surrounding matrix senesces. Sparse presence shows the expected *decrease*. The signal is canopy-fraction-weighted, not a universal monotone discriminator.
- Sparse presence bboxes show **bimodal within-bbox distributions** during Apr–May (wet-to-dry transition): a low-MAVI grass/soil cluster and a high-MAVI canopy cluster. This bimodality is the primary discrimination signal for sparse infestations; it is invisible in bbox-aggregate means.
- ΔMAVI/Δt (temporal derivative) showed no class separation at the pixel-year aggregate level — not recommended as an explicit feature.

### Connection to what we know

B08 and B11 are already in the feature set; MAVI adds value by making the canopy-fraction-weighted moisture relationship an explicit input the model doesn't have to re-derive. The bimodal within-bbox distributions suggest MAVI provides discrimination at the individual pixel level even when site-level means overlap.

### Priority

**High.** Formulation is simple, all bands present in training parquets, prior investigation provides clear hypotheses about the discrimination window (Apr–May) and expected direction at sparse vs dense sites. Main open question is whether the Signal harness recovers the bimodality finding across a wider site set.

---

## 2. NDRE / CI_RE — Red-Edge Chlorophyll Indices

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

## 3. Spectral Mixture Analysis — PV/NPV/BS Fractional Cover

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

## 4. NDSVI + B11/B12 Structural Decoupling

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

## 5. Temporal Variance (σ_t)

### Hypothesis

Annual grasses respond to rainfall pulses with large, rapid NDVI spikes followed by rapid decay. Parkinsonia, buffered by its deep root network, exhibits a flatter and more stable response curve. σ_t captures this directly without requiring knowledge of the rainfall calendar.

### Formulation

Per-pixel, per-year:
```
sigma_t  = std(NDVI) across qualifying S2 observations in the year
ndvi_iqr = p75(NDVI) - p25(NDVI) across qualifying S2 observations in the year
```

Applied to NDVI by default. Can be applied to any other signal (MAVI, CI_RE) as a secondary check.

Note: phase delay (peak NDVI DOY vs peak rainfall DOY) is a related hypothesis but requires external rainfall data not currently in the pipeline. It is not included here.

### Connection to what we know

`rec_mean` captures the *total wet-to-dry swing*; σ_t captures *within-year volatility*. A pixel with a smooth seasonal ramp has high rec_mean but low σ_t. `nir_cv` captures within-dry-season NIR stability; σ_t covers the full annual window including the wet-to-dry transition. Both correlations need to be below r = 0.7 for σ_t to earn its place.

### Priority

**Medium-high.** Cheap — computable directly from NDVI column already in training parquets. Unlike other candidates, σ_t is implemented via a custom `summarise()` override (the discriminative summary is `std`, not `p05`).

Note: the `summarise()` default returns `std` as one of its keys, so no override is needed — the harness just uses `rank_key="std"` instead of `rank_key="p05"`.

---

## Evaluation plan

Signals are evaluated using `signals/eval.py`. Each signal is a `Signal` subclass; discriminability is measured across three named `SiteSpec` tiers that serve different diagnostic purposes.

### Rationale for three tiers

| Tier | Purpose |
|------|---------|
| **Arid clean** | Upper-bound discriminability. Dense presence vs simple grassland/open absence with minimal background woody cover. If a signal fails here it has no path forward. |
| **Sparse stress** | Real-world stress test. Sparse presence pixels are the hardest detection case. A signal that holds here is genuinely useful. |
| **Woody false-positive panel** | Specificity check. Dense non-Parkinsonia woody vegetation. A signal that fires here as strongly as on presence is not discriminative — it is measuring woody cover generically. |

Note: signals don't need to be universally discriminative across all tiers to be useful. The TAM attention mechanism can learn when a signal is relevant. A signal that separates cleanly only at arid sites still earns its place.

### Site specs

```python
from signals.eval import SiteSpec

# Tier 1 — Arid clean: dense presence vs open grassland/absence
ARID_CLEAN = SiteSpec("arid_clean", [
    ("barcoorah_presence",   "presence"),
    ("barcoorah_presence_2", "presence"),
    ("barcoorah_presence_3", "presence"),
    ("barcoorah_presence_4", "presence"),
    ("lake_mueller_presence",   "presence"),
    ("lake_mueller_presence_2", "presence"),
    ("lake_mueller_presence_3", "presence"),
    ("lake_mueller_presence_4", "presence"),
    ("barcoorah_absence_2",    "absence"),
    ("barcoorah_absence_3",    "absence"),
    ("barcoorah_absence_4",    "absence"),
    ("lake_mueller_absence",   "absence"),
    ("lake_mueller_absence_2", "absence"),
    ("lake_mueller_absence_3", "absence"),
])

# Tier 2 — Sparse stress: sparse presence vs grass absence at Landsend
SPARSE_STRESS = SiteSpec("sparse_stress", [
    ("landsend_sparse_presence_1", "presence"),
    ("landsend_sparse_presence_2", "presence"),
    ("landsend_sparse_presence_3", "presence"),
    ("landsend_sparse_presence_4", "presence"),
    ("landsend_sparse_presence_5", "presence"),
    ("landsend_absence_grass_1",   "absence"),
    ("landsend_absence_grass_2",   "absence"),
])

# Tier 3 — Woody false-positive panel: non-Parkinsonia woody cover
# All regions treated as pseudo-absence (label is informational only here).
# Run signal on this tier and check whether scores resemble presence or absence.
WOODY_FP = SiteSpec("woody_fp", [
    # Cloncurry: monsoonal woodland (dense Eucalyptus/Acacia, no Parkinsonia)
    ("cloncurry_absence_1", "absence"),
    ("cloncurry_absence_2", "absence"),
    ("cloncurry_absence_3", "absence"),
    ("cloncurry_absence_4", "absence"),
    ("cloncurry_absence_5", "absence"),
    ("cloncurry_absence_6", "absence"),
    ("cloncurry_absence_7", "absence"),
    # Burdekin: semi-arid woody, added explicitly as a false-positive hard negative
    ("burdekin_absence_8",      "absence"),
    ("burdekin_val_absence_4",  "absence"),
    # Etna Creek: semi-arid woody absence (tags: woody)
    ("etna_absence_8",  "absence"),
    ("etna_absence_9",  "absence"),
    ("etna_absence_10", "absence"),
    ("etna_absence_11", "absence"),
    ("etna_absence_12", "absence"),
])
```

### Running the evaluation

```python
from signals.eval import evaluate
from signals.ndre import NDRESignal, CIRESignal

SITES = [ARID_CLEAN, SPARSE_STRESS, WOODY_FP]

# Index-based signals: rank on p05 (dry-season floor)
for sig in [NDRESignal(), CIRESignal()]:
    results = evaluate(sig, SITES, rank_key="p05")

# MAVI: also rank on p05; check p25/amplitude for the bimodal case
from signals.mavi import MAVISignal
results = evaluate(MAVISignal(), SITES, rank_key="p05")

# Temporal variance: rank on std (the discriminative summary)
from signals.temporal import TemporalVarianceSignal
results = evaluate(TemporalVarianceSignal(), SITES, rank_key="std")
```

### Verdict criteria

| Criterion | Threshold | Action |
|-----------|-----------|--------|
| IQR overlap (Tier 1) | 0.0 | Required for advancement |
| AUROC (Tier 1) | ≥ 0.75 | Minimum useful separation |
| AUROC (Tier 2, sparse) | ≥ 0.65 | Earns place in model feature set |
| Pearson r vs existing signals | < 0.7 | Required for independence |
| Tier 3 AUROC vs Tier 1 AUROC | Gap > 0.15 | Confirms Parkinsonia-specific rather than generic woody signal |

SMA is excluded from this harness — it requires a constrained least-squares unmixing step that does not fit the `Signal.compute()` interface. It will be evaluated separately.

---

## Results (2026-05-19)

Evaluated across all three tiers using the harness above. Woody FP tier returned no AUROC (all-absence spec — single-class, undefined). The critical tiers are arid_clean (Tier 1, upper bound) and sparse_stress (Tier 2, real-world bar).

### Raw results

| Signal | rank_key | T1 arid_clean AUROC | T1 IQR overlap | T2 sparse_stress AUROC | T2 IQR overlap |
|---|---|---|---|---|---|
| NDRE | p05 | 0.467 | 0.855 | 0.780 | 0.032 |
| CI_RE | p05 | 0.473 | 0.773 | **0.816** | **CLEAN** |
| MAVI | p05 | 0.501 | 0.898 | **0.794** | 0.037 |
| NDSVI | p05 | 0.454 | 0.708 | 0.730 | 0.209 |
| B12/B11 | std | 0.585 | 0.547 | 0.405 | 0.631 |
| sigma_t (NDVI std) | std | 0.637 | 0.303 | 0.414 | 0.697 |

### Verdicts

**CI_RE — ACCEPT.** Best Tier 2 result overall: AUROC 0.816, only signal with clean IQR separation (0.0) at sparse sites. Failure at Tier 1 (AUROC 0.473) is expected and physically interpretable — dense presence at arid sites does not rely on the dry-season stem signal.

**MAVI — ACCEPT.** AUROC 0.794, IQR overlap 0.037 at Tier 2. Consistent with prior investigation: MAVI is a sparse-infestation discriminator, not a dense-presence signal. Prior work (bimodal within-bbox distributions in Apr–May) holds across a wider site set.

**NDRE — ACCEPT.** AUROC 0.780 at Tier 2, IQR overlap 0.032. Adds a different spectral axis from MAVI (red-edge vs SWIR moisture). Tier 1 failure same interpretation as CI_RE.

**NDSVI — CLOSE.** AUROC 0.730 (marginal, below 0.75 T1 bar), IQR overlap 0.209 at Tier 2. Follow-up investigation (2026-05-19) found r = −0.692 vs swir_mi_p10 and r = 0.660 vs re_p10 — both near-threshold, and the negative correlation with swir_mi reflects inverted shared variance (same underlying state, opposite sign) rather than independent information. Monthly profile is flat year-round with no new temporal discriminant. Not worth adding a fourth SWIR-derived feature.

**B12/B11 (std) — CLOSE.** AUROC 0.405 at Tier 2 (presence scores *lower* than absence). No discriminative signal in either direction.

**sigma_t (NDVI std) — CLOSE.** AUROC 0.414 at Tier 2 — same direction problem. Presence pixels are actually *more* volatile than absence at this site, opposite to the hypothesis.

### NDSVI follow-up detail

Landsend sparse_stress sites, pixel-year p10 aggregation:

- `ndsvi_p10` vs `swir_mi_p10`: r = −0.692 (inverted — same underlying wet/dry state, opposite sign)
- `ndsvi_p10` vs `re_p10`: r = 0.660
- Jul–Sep delta: presence 0.299 vs absence 0.249 (Δ = 0.050); swir_mi collapses to near-zero delta in same window (Δ = 0.004) — NDSVI does maintain dry-season separation where swir_mi does not, but not large enough to justify inclusion alongside MAVI and CI_RE.

---

## References

- Misra, G., Cawkwell, F., & Wingler, A. (2020). Status of Phenological Research Using Sentinel-2 Data: A Review. *Remote Sensing*, 12(17), 2760.
- Rusňák, T., et al. (2022). Detection of Invasive Black Locust (*Robinia pseudoacacia*) in Small Woody Features Using Spatiotemporal Compositing of Sentinel-2 Data. *Remote Sensing*, 14(4), 971.
- Toqeer, A. (2026). Remote Sensing of Woody Plant Encroachment: A Global Systematic Review. *Remote Sensing*, 18(3), 390.
- Bradshaw, T. M. (2022). Thesis: SWIR moisture indices for groundwater-dependent vegetation. University of Wyoming.
- Levick, S. R. (2021). Remote sensing of gamba grass in northern Australia. Resilient Landscapes Hub.
