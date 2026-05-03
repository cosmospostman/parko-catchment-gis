# Kowanyama — Initial Site Run

## Site

Mitchell River delta, Gulf of Carpentaria coast.
Fetch bbox: `141.47528, −15.51051, 141.66922, −15.41736` (~21 km × 10 km).
Monsoonal climate; dry season May–October.

**Script:** `longreach/kowanyama-score.py`
**Outputs:** `outputs/scores/kowanyama/`

---

## Data fetch

S2 archive: 2020-05-01–2025-10-31, `--cloud-max 30`, dry-season-only date range.
145,210,023 rows × 249,744 pixels × 546 acquisition dates.
Zero rows dropped by the `scl_purity >= 0.5` quality filter — the fetch script
had already applied cloud filtering upstream.

All 249,744 pixels passed the `n_dry_years >= 5` and `n_amp_years >= 10`
coverage thresholds (median 6 years for both), indicating dense temporal
coverage across the full bbox.

---

## Results

### Probability distribution

| Statistic | Value |
|-----------|-------|
| Median | 0.756 |
| p10 | 0.000 |
| p90 | 1.000 |
| p99 | 1.000 |

The distribution is strongly bimodal: a large mass near prob = 1 and a tail
at prob = 0, with little in between. The probability map is almost entirely
dark green (prob ≈ 1) except for bare floodplain channels and water bodies
which score near zero. The spatial structure in the map is real — the
low-scoring channels are visible and correspond to bare/water features in
the WMS imagery — but the high scores are not discriminating Parkinsonia from
other vegetation.

### Classifier space

The Kowanyama pixel cloud occupies a completely different region of
`(nir_cv, rec_p)` space from the Longreach training populations:

| Population | nir_cv range | rec_p range |
|---|---|---|
| LR infestation centroid | ~0.05 | ~0.27 |
| LR grassland centroid | ~0.11 | ~0.21 |
| Kowanyama cloud (bulk) | 0–1.6 | 0–1.75 |

The Longreach training populations occupy a small cluster in the bottom-left
corner of the Kowanyama cloud. The decision boundary (prob = 0.5 contour)
bisects the Kowanyama cloud at `nir_cv` ≈ 0.6 — far to the right of both
Longreach centroids. The model is extrapolating well outside the training
feature range for the majority of Kowanyama pixels.

### Feature distributions

`rec_p` is the primary failure mode. Kowanyama `rec_p` ranges up to ~1.4,
versus a Longreach maximum of ~0.4. The monsoonal wet–dry cycle produces a
far larger NDVI amplitude across all vegetation types, including grasses. Any
vegetated pixel with a large wet–dry NDVI swing scores high — which at this
site is most of the landscape.

`nir_cv` is more promising. The Kowanyama `nir_cv` distribution overlaps the
Longreach training range at the low end (stable-canopy pixels), and there is
visible structure in the low-`nir_cv` portion of the classifier space plot.
Stable-canopy Parkinsonia should sit at low `nir_cv` relative to monsoonal
grasses, which may still hold here.

`re_p10` is shifted right relative to Longreach — consistent with higher
background chlorophyll in a wetter environment.

---

## Diagnosis

`rec_p` does not transfer to a monsoonal site without recalibration. The
feature encodes wet–dry NDVI amplitude, which is large for all vegetation at
Kowanyama, not just deep-rooted Parkinsonia. This is the opposite of the
compression risk flagged in the plan (where the concern was that rec_p might
be *small* for everything in a wet climate) — instead it is inflated for
everything.

`nir_cv` may still be useful as a primary discriminator. At a monsoonal site,
the signal of interest would be canopies that are *more stable* than the
surrounding seasonal grasses — Parkinsonia's deep roots should still decouple
it from rainfall variability, producing lower `nir_cv` than grassland even
if the absolute values shift. But this requires confirming with anchor pixels.

The Longreach-trained logistic regression boundary is not usable at this site
without site-specific calibration. The model is not malfunctioning — it is
being applied to a feature distribution it was never trained on.

---

## Anchor pixels

35 pixels within 100 m of the confirmed-presence anchor at (−15.457794, 141.535690).

| Statistic | nir_cv | rec_p | re_p10 | prob_lr |
|-----------|--------|-------|--------|---------|
| Median | 0.170 | 0.413 | 0.128 | 0.015 |
| Min | 0.131 | 0.376 | 0.119 | 0.000 |
| Max | 0.210 | 0.443 | 0.139 | 0.398 |

All 35 pixels score very low under the Longreach-trained model (prob 0.000–0.398).

### Where the anchor pixels sit in classifier space

`rec_p` (0.38–0.44) is modestly elevated above the Longreach infestation centroid
(~0.27) but not extreme — it is within a range that would be plausible at
Longreach. This is not the primary cause of the low scores.

`nir_cv` (0.13–0.21) is the failure mode. At Longreach, the infestation centroid
sits at `nir_cv` ≈ 0.05 and grassland at ≈ 0.11. The anchor pixels occupy the
range 0.13–0.21 — above the Longreach grassland centroid. The Longreach-trained
model interprets this as strongly non-Parkinsonia.

The pixel with the highest score (`px_0212_0203`, prob = 0.398) has the lowest
`nir_cv` in the anchor group (0.131), confirming that `nir_cv` is still the
primary discriminating axis — the Parkinsonia signal is present, but the absolute
values are shifted ~0.08–0.15 units higher than at Longreach.

### Interpretation

The elevated `nir_cv` at Kowanyama Parkinsonia pixels likely reflects the
monsoonal flood pulse: year-to-year variation in inundation depth and duration
causes canopy stress variation even in deep-rooted trees, raising their
inter-annual NIR variability above the Longreach baseline. The *relative* signal
(Parkinsonia more stable than surrounding grasses) may still hold, but the
absolute threshold needs to shift right by approximately 0.08–0.10 `nir_cv` units.

---

## Next steps

1. **nir_cv-only threshold scan** — score pixels on `nir_cv` alone with a
   site-calibrated threshold. The anchor pixels bracket a natural candidate:
   the best-scoring anchor has `nir_cv` = 0.131 and the group median is 0.170.
   Scan thresholds in the range 0.13–0.22 and compare the resulting spatial
   patterns against WMS imagery.

2. **rec_p normalisation** — normalise `rec_p` by the site-wide median to
   remove the climate-driven amplitude baseline. The relevant signal is whether
   Parkinsonia amplitude is *smaller* than its immediate neighbours (deep roots
   buffering the wet–dry swing), not its absolute value.

3. **Site-specific recalibration** — retrain the logistic boundary using the
   anchor pixels as the positive class and a matched set of high-`nir_cv`
   background pixels (e.g. `nir_cv` > 0.30, clearly seasonal grasses) as the
   negative class. Record the feature-space shift relative to Longreach
   centroids as a quantitative measure of cross-site offset.

4. **Absence anchor selection** — identify open bare floodplain interfluve
   pixels with no visible Parkinsonia crowns in the WMS imagery to anchor the
   negative class for recalibration.
