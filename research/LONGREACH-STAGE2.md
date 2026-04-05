# Longreach Signal Analysis — Stage 2

## Summary of Stage 1 findings

Five spectral/temporal signals were evaluated on 748 Sentinel-2 pixels (374 infestation
patch + 374 southern extension, 2020–2025 archive) at the Longreach confirmed Parkinsonia
site. All five produced zero IQR overlap between infestation and grassland classes when
summarised as per-pixel annual statistics. Three are independent axes; two are redundant.

### Feature inventory

| Feature | Signal | IQR overlap | Pearson r with re_p10 | Independent? |
|---------|--------|------------|----------------------|-------------|
| `nir_cv` | Dry-season inter-annual NIR stability | 0.00 | −0.304 | Yes |
| `rec_p` | Annual NDVI amplitude (p90 − p10, window-free) | 0.00 | 0.087 | Yes |
| `re_p10` | Annual red-edge low percentile (B07/B05, p10) | 0.00 | — | Yes (baseline) |
| `swir_p10` | SWIR moisture index annual low percentile | 0.00 | **0.729** | Redundant with re_p10 |
| `fi_p90_cg` | Contrast-gated annual flowering index p90 | 0.00 | TBD | TBD |

`rec_p` and `nir_cv` are partially correlated (r = −0.77) — both reflect deep-root
canopy persistence — but each passes zero IQR overlap independently, and their
combination has not been formally evaluated in 3D with `re_p10`.

### Mechanistic understanding

Each feature traces to a distinct property of Parkinsonia's deep-root physiology:

- **`nir_cv`** — canopy persists stably through every dry season regardless of rainfall
  year-to-year; grasses senesce to varying degrees
- **`rec_p`** — strong post-wet flush (deep-root sustained into April–May) produces a
  larger annual NDVI swing than shallow-rooted grasses or bare-soil riparian pixels
- **`re_p10`** — chlorophyll floor remains elevated at the annual low; senescent grasses
  collapse toward B07/B05 ≈ 1
- **`swir_p10`** — canopy water sustained year-round; corroborates re_p10 via a different
  physical channel (leaf water vs chlorophyll)
- **`fi_p90_cg`** — yellow flowering event produces a detectable FI_by anomaly in 4 of 6
  years; signal is opportunistic (not calendar-fixed), so the contrast-gated p90 is
  required to isolate it from shared wet-season greenness

### Phenological timing

The red-edge contrast peaks in **March–April** and the NDVI contrast in **April–May**,
consistent across all six years. This is the post-wet flush window driven by Parkinsonia's
deep roots sustaining canopy into the dry. Flowering, where detectable, occurs January–May
(opportunistic, follows wet-season rainfall). Dry-season discrimination (Jul–Sep) is driven
primarily by NIR CV and NDVI trough.

### What Stage 1 did not resolve

1. **3D feature space not evaluated** — pairwise 2D projections confirm independence but
   the joint `(nir_cv, rec_p, re_p10)` space has not been computed. Whether the nir_cv /
   rec_p correlation compresses the effective discriminating volume is unknown.

2. **Riparian class is a proxy, not a real population.** The "riparian" pixels in the
   southern extension are bare riverbed (no plant water, no active chlorophyll) — the
   opposite of real riparian woodland. All riparian-related discrimination results should
   be treated as provisional.

3. **Single site.** Every threshold, IQR, and correlation value is from one 820 × 820 m
   patch on gilgai clay floodplain near Longreach. Cross-site transfer is untested.

4. **`fi_p90_cg` riparian test not run.** The flowering analysis closed with this
   explicitly open.

---

## Revised classification framing

**Parkinsonia colonises riparian corridors.** It is not a confound to exclude; it is a
primary detection target in riparian settings. Sightings from the ALA cluster along
watercourses, and the ecological literature is consistent with Parkinsonia preferring
drainage lines and floodplains.

This changes the problem in several ways:

1. **"Riparian" is not a negative class.** The classifier should not be trained to predict
   low scores in riparian settings. A pixel in a riparian corridor that scores high on
   Parkinsonia features is likely a correct detection, not a false positive.

2. **Native-only riparian (coolibah, eucalyptus, no Parkinsonia) is a rare and hard
   class** in heavily infested catchments. We do not yet have labelled examples of it.
   Its spectral properties are likely intermediate — active chlorophyll (re_p10 elevated),
   permanent water access (swir_p10 elevated), but potentially different recession shape
   and NIR stability from Parkinsonia. `rec_p` is the most likely discriminator between
   Parkinsonia-dominated and native-dominated riparian, if such separation is needed at
   all.

3. **The classifier output should be framed as "Parkinsonia presence probability"** — a
   continuous score expected to be high in both open floodplain infestation and
   riparian Parkinsonia, and low only in Parkinsonia-free grassland or sparse shrubland.
   Hard presence/absence is not the right output for a mixed-pixel, mixed-community system.

4. **Zero IQR overlap between classes is an end-member artefact, not a realistic
   performance target.** The Longreach populations are artificially pure: a dense mature
   infestation patch on open floodplain, uniform grassland, and bare riverbed. Real survey
   pixels will be mixtures — Parkinsonia crowns over grassy understory, Parkinsonia
   scattered through coolibah, floodplain edge pixels that are half-grass half-Parkinsonia.
   At 10m resolution with ~30–40% crown cover, every pixel is already a spectral mixture.
   Overlap between class distributions at deployment is expected and ecologically correct.
   The right evaluation metric is whether the feature scores are **monotone with canopy
   fraction** — does the score increase predictably as Parkinsonia fraction increases within
   a pixel? The natural sub-pixel canopy fraction gradient already present in the infestation
   patch (documented in LONGREACH.md) provides a within-site test of this.

---

## Prioritised research directions for Stage 2

### Priority 1 — 3D feature space evaluation ✓ COMPLETE

**Script:** `longreach/feature-space.py`
**Outputs:** `outputs/longreach-feature-space/`

#### Results

**Class centroids:**

| Class | n | nir_cv | rec_p | re_p10 |
|-------|---|--------|-------|--------|
| Infestation | 362 | 0.047 | 0.273 | 1.188 |
| Grassland | 347 | 0.110 | 0.213 | 1.168 |
| Riparian proxy | 39 | 0.127 | 0.154 | 1.198 |

**Mahalanobis distances (3D pooled covariance):**

| Pair | d (3D) | d (nir_cv, rec_p) | d (nir_cv, re_p10) | d (rec_p, re_p10) |
|------|--------|-------------------|---------------------|-------------------|
| Infestation vs Grassland | 1.43 | 1.43 | 1.42 | 1.23 |
| Infestation vs Riparian  | 2.48 | 2.42 | 1.96 | 2.48 |
| Grassland vs Riparian    | 1.51 | 1.44 | 0.73 | 1.37 |

**Key finding: `re_p10` adds no incremental separation for Infestation vs Grassland.**
The 3D Mahalanobis distance (1.43) is identical to the `(nir_cv, rec_p)` 2D subspace
(1.43). The third axis does not contribute because `nir_cv` and `rec_p` already fully
capture the infestation–grassland axis; `re_p10` is orthogonal but not discriminating in
a direction those two miss.

`re_p10` does help for Grassland vs Riparian (d = 0.73 in the `(nir_cv, re_p10)` plane,
the weakest 2D projection for that pair), but the bare-riverbed riparian proxy makes this
result unreliable for real riparian woodland.

**Practical implication:** `(nir_cv, rec_p)` is the effective 2D classifier for
end-member separation at this site. `re_p10` should be retained in the feature set for
generalisability at new sites where the nir_cv/rec_p correlation may weaken, but is not
expected to improve classification accuracy at Longreach-like sites.

**Caveat — end-member populations.** The zero IQR overlap and clean Mahalanobis
separation reflect artificially pure training populations. At real survey sites, pixels
will be mixtures (Parkinsonia over grassland, Parkinsonia in riparian woodland, etc.) and
class distribution overlap is expected and ecologically correct. See the revised
classification framing section above.

#### Sanity check — probability score vs Queensland Globe 20cm imagery ✓ COMPLETE

**Script:** `longreach/probability-vs-imagery.py`
**Outputs:** `outputs/longreach-feature-space/longreach_prob_vs_imagery.png`,
`longreach_prob_deciles_imagery.png`, `longreach_pixel_ranking.csv`

A per-pixel Parkinsonia probability score was computed and overlaid on the 20cm WMS
imagery to verify that the feature-derived ranking tracks visible crown density.

**Result:** The spatial pattern matches the imagery closely. Top-decile pixels (prob ≥
0.99) fall within the dense Parkinsonia crown zone visible in the upper portion of the
scene. Bottom-decile pixels (prob ≤ 0.00) cluster around the bare riverbed and the most
open grassland in the southern extension. Mid-probability pixels sit at the
infestation–grassland boundary, consistent with mixed pixels at 10m resolution. The score
is monotone with visible crown density — the primary criterion for a presence probability
output.

**How the probability was calculated — and its limitations:**

The score is a logistic regression trained on the 362 infestation and 347 grassland
end-member pixels, with `(nir_cv, rec_p, re_p10)` standardised before fitting. The 39
riparian proxy pixels were excluded from training but scored at inference.

Key limitations to carry forward:

1. **Trained and evaluated on the same data.** No held-out test set. The near-perfect
   class separation (infestation median prob 0.948, grassland median 0.010) reflects
   end-member separability, not generalisation performance.

2. **Riparian excluded from training.** The model learned only Parkinsonia vs. grassland.
   Riparian scores are extrapolations, not trained predictions.

3. **Linear decision boundary.** Given the nir_cv / rec_p correlation (r = −0.77), the
   effective boundary is an oblique line in the 2D plane those two define. `re_p10`
   receives low weight for the same reason it added no Mahalanobis distance in 3D.

4. **Probabilities are uncalibrated.** They reflect the ~50/50 training class balance,
   not any real-world prior on landscape Parkinsonia fraction. They are best interpreted
   as a relative ranking or threshold-tuned detection score.

A simpler alternative — `score_raw`, the normalised sum of inverted `nir_cv` + `rec_p` +
`re_p10` — is also computed in the script and produces a near-identical spatial pattern
with no model involved. For the sanity-check purpose either works; for deployment the
calibration limitations above apply to both.

---

### Priority 2 — fi_p90_cg riparian test (no new data required)

**What:** Compute `fi_p90_cg` for the riparian proxy pixels already in the dataset
(extension pixels at highest nir_mean, used as proxy in the dry-NIR analysis). Compare
distributions: infestation vs grassland vs riparian proxy.

**Why second:** The flowering analysis left this explicitly open. It is a quick stats
addition to `longreach/flowering.py` (the riparian pixel indices are already defined in
the dry-NIR output parquet). If fi_p90_cg also achieves zero IQR overlap with riparian,
it becomes a stronger candidate reserve feature. If riparian scores high (as re_p10 did),
it is redundant with the known limitation.

---

### Priority 3 — Second site validation (data fetch required)

**What:** Fetch S2 time series for one of the ALA cluster sites — the -22.443, 144.652
cluster (n = 41, 2015–16) is the strongest candidate. Extend the bbox to include
confirmed-negative grassland pixels as was done at Longreach. Run the same
`(nir_cv, rec_p, re_p10)` feature extraction. Compare the Longreach-derived thresholds
against the new site's pixel distributions.

**Why third:** Cross-site transfer is the largest unknown. If the Longreach IQR thresholds
hold at a second site on different soils or vegetation structure, the features are
generalisable. If they shift, site-specific calibration will be required and the
classifier design must accommodate it.

**Note:** The ALA sightings are from 2015–16, predating the S2 archive. The infestation
likely persists or has expanded, but some patches may have been treated. Check the
Queensland Globe WMS for current canopy before fetching.

---

### Priority 4 — Phenological timing features

**What:** For each pixel and year, fit a smoothed NDVI curve (e.g. rolling median or
simple Gaussian kernel) and extract: DOY of annual NDVI peak, DOY of annual NDVI
trough, green-up rate (slope of the rising limb), recession rate (slope of the falling
limb). Compute per-pixel means across years.

**Why:** All Stage 1 features are amplitude/level statistics. Timing features are
mechanistically distinct — Parkinsonia's post-wet flush may peak at a consistent DOY
(deep roots decouple it somewhat from rainfall timing), while grasses track rainfall more
closely. The DOY of the re_ratio peak (consistently March–April across all years) suggests
a timing signal worth formalising.

**Caveat:** Timing features require dense temporal coverage to estimate curve shape
reliably. The ~3–6 observations/month cadence is borderline. Restrict to years with
≥ 20 qualifying observations and check that fitted DOY estimates are stable year-to-year
before using as features.

---

### Priority 5 — Native riparian woodland ground truth

**What:** Identify locations within the Galilee Basin East 2021 WMS coverage where dense
native riparian woodland (coolibah, eucalyptus) exists with no visible Parkinsonia crown
structure. Digitise a sample of pixels in QGIS. Fetch their S2 time series and extract
`(nir_cv, rec_p, re_p10)`.

**Why:** This is the class most likely to produce false positives under the current
feature set. Understanding its spectral behaviour is necessary before the classifier can
be deployed in riparian settings. This is a longer-horizon task requiring manual labelling
work; earlier priorities can proceed without it.

**Approach:** The 20cm WMS imagery resolves individual crowns. Parkinsonia crowns are
distinctive (rounded, pale-green canopy, consistent 1.5–3 m diameter against pale clay).
Native riparian woodland has irregular crown shapes, darker canopy, and typically a
closed canopy at the crown level rather than isolated crowns over bare soil. Patches
that look continuous and crown-heterogeneous with no isolated rounded crowns are the
target negative-riparian sample.

---

## Feature set recommended for Stage 2 classifier prototype

| Feature | Role | Justification |
|---------|------|--------------|
| `nir_cv` | Primary | Dry-season inter-annual stability; drives the main discriminating axis |
| `rec_p` | Primary | Annual NDVI amplitude; partially correlated with nir_cv but adds independent variance |
| `re_p10` | Reserve — generalisation only | Redundant at Longreach (adds zero Mahalanobis distance to the nir_cv/rec_p 2D space); retain for sites where the nir_cv/rec_p correlation weakens or soil/climate conditions differ |
| `fi_p90_cg` | Reserve | Include if Priority 2 confirms riparian separation |
| `swir_p10` | Reserve — diagnostic | Retain as corroboration; include only if re_p10 degrades at new sites |

**Effective dimensionality at Longreach:** 2D (`nir_cv`, `rec_p`). The 3D feature space
evaluation (Priority 1) confirmed that adding `re_p10` does not improve Infestation vs
Grassland separation at this site (Mahalanobis d unchanged at 1.43). The two primary
features capture the full discriminating axis available in this dataset.

**Classifier type:** Logistic regression or linear SVM as the first prototype — the
feature space is linearly separable in 2D, and interpretable coefficients directly link
back to the mechanistic understanding. A gradient-boosted tree is a natural follow-on
if linear separation proves insufficient at new sites.

**Training data available now:** 362 infestation pixels × 6 years = 2,172 pixel-years;
347 grassland pixels × 6 years = 2,082 pixel-years. Sufficient for a prototype. Riparian
class (native-only) requires Priority 5 before inclusion.
