# Longreach — Research Summary

Dense Parkinsonia infestation on floodplain/gilgai clay at (-22.763, 145.425).
Each S2 pixel is a spectral mixture of Parkinsonia canopy (~30–40% crown cover) and
bare gilgai clay.  748 pixels: 362 infestation bbox + 374 grassland extension south.

See `LONGREACH.md` for site description and ALA sighting clusters.

---

## Signals evaluated and their verdicts

| Signal | Metric | IQR overlap | vs other features | Verdict |
|--------|--------|-------------|-------------------|---------|
| NIR inter-annual stability | `nir_cv` | 0.00 | independent of `rec_p` at r=−0.77 (both measure deep-root persistence) | **Primary feature** |
| Wet/dry NDVI amplitude | `rec_p` | 0.00 | r=−0.77 with `nir_cv` | **Primary feature — correlated but not redundant** |
| Red-edge ratio | `re_p10` (annual p10 of B07/B05) | 0.00 | r=0.087 with `rec_p` (independent) | **Strong supporting feature** |
| SWIR moisture index | `swir_p10` (p10 of (B08−B11)/(B08+B11)) | 0.00 | r=0.729 with `re_p10` (redundant) | Dropped — collinear with red-edge |
| Contrast-gated flowering index | `fi_p90_cg` | 0.00 vs grassland AND riparian proxy | riparian-specific discriminator | Retained as specialist feature |
| NDVI integral | `ndvi_integral` | weak separation | r=0.832 with `rec_p` at Longreach | Hold — test at second (wetter) site; if r drops below 0.5 include both |
| Day-of-year peak (absolute) | `peak_doy` | not discriminating | r=−0.36 with `rec_p` (partially independent) | Include as weak supporting feature |
| Peak DOY consistency | `peak_doy_cv` | 16-day gap presence vs absence | — | Include |
| Relative peak shift | `peak_doy_shift` | non-monotonic, context-dependent | — | **Do not implement** — not transferable |
| Recession sensitivity | `recession_sensitivity` | noise-dominated | — | **Rejected** — NDWI proxy broken for this site |
| Recession slope | `recession_slope` | r=−0.922 with `rec_p` | redundant | Optional `rec_p` replacement; no advantage |

---

## Effective classifier

**2D feature space: (nir_cv, rec_p).**  Mahalanobis distance does not improve when
adding `re_p10` as a third axis — it is useful for corroboration but not for
primary separation at Longreach.

Logistic regression on infestation/grassland end-members produces probability scores
that are monotone with visible crown density in Queensland Globe 20 cm imagery.

*Important:* zero IQR overlap is an **end-member artefact** — training labels are
pure infestation and pure grassland bboxes.  Real mixed pixels produce overlapping
distributions; classify via canopy-fraction gradient monotonicity, not IQR non-overlap.

---

## What did not work

- **Flowering spectral transient:** peak z-score reaches 1.868 (criterion was ≥ 2.0).
  Signal is real but weak and opportunistic (not calendar-fixed).  `fi_p90_cg` salvages
  a riparian-specific discriminator from this.
- **Day-of-year peak timing:** both presence and absence are in the same riparian
  corridor with similar flood-pulse water access → same timing.  The missing axis is
  waveform shape (wet-peak amplitude to dry-floor ratio), which is approximately
  what `rec_p` already captures.
- **NDWI-based recession sensitivity:** NDWI is negative/dry for ~99.8% of pixels
  (floodplain rarely inundates); variation is dry-soil reflectance noise.

---

## Open items from Stage 2

- **Stage 3 (second site validation):** does the 2D classifier transfer?  Do `rec_p`
  and `nir_cv` remain the dominant axes in a wetter / denser-riparian context?
- **Native riparian ground truth:** `fi_p90_cg` shows riparian discrimination promise
  but the riparian proxy used here is largely bare riverbed geology, not real riparian
  woodland.
- **Eastern expansion bbox** (`LONGREACH-EXPANSION.md`): data fetch for 2.1 km eastern
  extension (~40 k pixels of native riparian, scattered Parkinsonia on clay, gilgai
  mosaic) not yet completed.
- **NDVI integral at second site:** hold — collinearity with `rec_p` may be
  scene-specific (r=0.832 at Longreach; test if it drops below 0.5 elsewhere).

---

## Working notebooks

The individual signal analyses are in `docs/research/`:
`LONGREACH-DRY-NIR.md`, `LONGREACH-WET-DRY-AMP.md`, `LONGREACH-FLOWERING.md`,
`LONGREACH-RED-EDGE.md`, `LONGREACH-SWIR.md`, `LONGREACH-STAGE2.md`

Peak DOY and recession results are in `docs/research/day-of-year-peak/RESULTS.md`
and `docs/research/recession-and-greenup/RESULTS.md`.
