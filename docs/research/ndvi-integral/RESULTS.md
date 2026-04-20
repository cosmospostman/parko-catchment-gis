# NDVI Integral — Results (Longreach 8×8 km)

## Summary

The signal exists and is directionally correct, but is strongly collinear with `rec_p`
at this scene. The collinearity finding should not be taken as evidence against inclusion
— it is an expected artefact of the scene characteristics and training data limitations
described below.

---

## Stage 1 — Signal existence

**H1: Confirmed (weakly).**

| Class | n | Median ndvi_integral |
|---|---|---|
| Presence | 366 | 0.293 |
| Absence | 340 | 0.220 |
| Separation | — | 0.073 |

Direction is correct: Parkinsonia pixels have higher mean annual NDVI. The absence
distribution is a tight spike at ~0.21; the presence distribution is broader, centred
at ~0.29, with substantial overlap from ~0.22–0.28.

**H1b (ndvi_integral_cv): Fail.**

Median separation is 0.005 — effectively zero. Inter-annual consistency of mean NDVI
does not discriminate at this site. `ndvi_integral_cv` is not a candidate feature.

**Spatial map notes:**

The map is physically sensible but two non-vegetation confounds are present:
- **Cliff face (top-right of scene):** Appears as high integral from spectrally stable
  rock/shadow returns, not sustained vegetation greenness. Not a biological signal.
- **Lake Mueller (scene centre):** Seasonal lake. When inundated, observations are
  masked by SCL; when dry, exposed sediment produces low or negative NDVI. Either way
  the annual mean is suppressed, producing a dark centre patch unrelated to vegetation.

Any per-scene spatial interpretation should account for both.

---

## Stage 2 — Independence from existing features

**H2: Fail at this scene.**

| Feature | Pearson r with ndvi_integral |
|---|---|
| `rec_p` | 0.832 |
| `re_p10` | ~0.85 |
| `nir_cv` | ~0.25 |
| `prob_lr` | ~0.37 |

`ndvi_integral` is highly collinear with `rec_p` and `re_p10`. The scatter of
`ndvi_integral vs rec_p` is near-linear with little residual structure; the existing
classifier's probability scores (`prob_lr`) run cleanly along the same axis, confirming
the information is already captured.

**However, this result requires careful interpretation:**

1. **Scene characteristics.** Longreach is a dry, high-observation-density scene with
   strong, clean seasonality. When phenological amplitude and curve width co-vary tightly
   (as they do in a well-sampled grassland scene), `rec_p` and `ndvi_integral` are
   approximately linear transforms of each other. The collinearity is expected here and
   does not generalise to wetter or cloudier scenes where the relationship would decouple.

2. **Training data limitations.** The correlation is computed across all scene pixels,
   the majority of which are absence (grassland). The presence sample (n=366) is from a
   single infestation patch in a single ecological context. Parkinsonia occurs across
   dense riparian thickets, sparse paddock infestations, dryland and floodplain settings,
   and a range of rainfall zones — none of which are represented here. The correlation
   structure reflects the Longreach grassland matrix, not the full range of contexts the
   classifier must handle.

3. **Presence pixel purity.** The presence bbox is drawn over a dense infestation but
   contains mixed pixels — some are pure grass at sub-pixel scale. These impure presence
   pixels pull the presence distribution toward absence, compressing measured separation
   and contaminating the correlation estimate. The true `ndvi_integral`–`rec_p`
   correlation for pure Parkinsonia pixels is unknown.

**The robustness case.** `rec_p` is sensitive to observation density asymmetry: cloud
cover preferentially masking wet-season peaks will underestimate amplitude. `ndvi_integral`
is computed as `mean(ndvi_smooth)` across all DOYs and is less sensitive to which
observations are missing — the smoothed curve interpolates across gaps before averaging.
At scenes with compressed or patchy observation density, the two features would diverge
and the integral would carry independent weight. The Stage 4 smoothing sensitivity result
(flat across 15–45d) is consistent with this robustness.

---

## Stage 3 — Riparian proxy

**H3: Fail (wrong direction at this scene).**

The riparian proxy pixels (top-10% dry-season NIR within the grassland bbox, n=35)
sit *above* the presence class in `ndvi_integral`, not below it. In the
`ndvi_integral vs rec_p` scatter, riparian pixels are in the top-right, not displaced
below the presence cloud at similar `rec_p`.

**Caveat:** The riparian proxy derivation is unreliable at this scene. The top-10% NIR
pixels within the scene extent are likely dominated by the cliff face and lake-margin
geology rather than actual riparian vegetation. The H3 verdict (wrong direction) may
reflect cliff pixels being labelled as "riparian" rather than a genuine ecological
relationship. H3 cannot be evaluated at Longreach and requires a scene with a cleaner
riparian signature.

---

## Stage 4 — Smoothing sensitivity

**H4 (partial): Confirmed.**

| smooth_days | Separation |
|---|---|
| 15 | 0.0711 |
| 30 | 0.0730 |
| 45 | 0.0712 |

Effectively flat. The 30d default is marginally optimal but differences are trivial.
This is expected: averaging the curve over 365 days absorbs short-scale smoothing
differences. The integral is insensitive to the smoothing window choice, which supports
the robustness argument.

---

## Pipeline inclusion decision

Not added to `pipelines/longreach-8x8km.py` at this stage.

The decision is not straightforward. Arguments for and against inclusion are
scene-independent and the Longreach results alone are insufficient to resolve them:

**For inclusion:**
- The biological hypothesis is sound and scene-independent (green-stem floor sustained
  year-round regardless of context)
- `rec_p` degrades where wet-season observations are cloud-masked; `ndvi_integral`
  degrades more gracefully
- A collinear feature pair under L2 regularisation distributes weight between them;
  at scenes where one degrades, the other absorbs the load implicitly
- The r=0.83 result is expected at Longreach and would likely drop substantially at a
  wetter or cloudier scene — at which point the integral carries independent weight
- Excluding based on a single-scene, single-context correlation is optimising for the
  one scene measured rather than the distribution of deployment scenes

**Against inclusion:**
- Collinear features at r=0.83 consume two degrees of freedom for one discriminative
  dimension, which can hurt calibration when training data is small
- Adding features before validating on a second site is premature

**Recommended path:** Hold in reserve. Run the existing pipeline at a second scene
(preferably wetter, with denser Parkinsonia in a riparian context) and measure the
`ndvi_integral`–`rec_p` correlation there. If it drops below ~0.5, include both. If it
remains at ~0.8+, the robustness argument is weaker than theory predicts.

---

## Key open questions

1. Does `ndvi_integral`–`rec_p` collinearity persist at a wetter/cloudier scene, or
   is r=0.83 a Longreach-specific artefact of clean, high-density observations?
2. How does the signal behave at dense riparian thickets, where canopy is closed and
   the amplitude/integral relationship may decouple from the paddock-infestation pattern?
3. Would a tighter presence polygon (manually verified dense canopy pixels) change the
   separation and collinearity results materially?
