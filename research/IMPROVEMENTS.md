# Candidate Signals: Evaluative Exploration

Signals identified in OVERVIEW.md as potentially discriminative for Parkinsonia that are
deferred until the core pipeline (flowering spike waveform) is running end-to-end.

Each signal should be introduced as an independent investigation: implement it, write a
test that asserts it produces discriminative features on real data, and use the test
result to decide whether the signal earns its place in the model or is abandoned. The
test is not just a correctness check — it is the evaluation criterion.

A signal worth keeping should demonstrate:
- the feature is measurably different between presence and absence points in the fixture dataset
- the difference is consistent across years (not a single wet/dry-year artefact)
- adding it to the RF feature set improves held-out AUC by a detectable margin

A signal that fails these criteria in the tests should be removed, not left as dead weight.

---

## 1. Dry-season greenness persistence (multi-year NDVI stability)

**Biological basis:** Parkinsonia is phreatophytic and maintains canopy greenness after
native vegetation senesces. Multi-year persistence is more reliable than single-season
anomaly — a pixel anomalously green across multiple dry seasons is far stronger evidence
than a single year.

**Feature to extract:** From the full observation sequence (outside the flowering window),
compute mean NDVI over dry-season acquisitions per year, and the count or fraction of
years where that mean exceeds a threshold. Also the coefficient of variation across years
(low CV = persistent, high CV = rainfall-coupled grass).

**Tests that would confirm this signal is useful:**
- Presence points have significantly higher mean dry-season NDVI across years than absence points
- Presence points have lower inter-annual CV of dry-season NDVI than grass-dominated absences
- These differences are maintained in wet-year data — the signal should be robust to individual
  wet years precisely because it is multi-year

**Implementation note:** `extract_waveform_features` currently only looks within
`FLOWERING_WINDOW`. A companion function would look at the complement — acquisitions
*outside* the window — and aggregate per-year NDVI statistics.

---

## 2. Post-wet-season green-up timing (early-dry-season peak)

**Biological basis:** Parkinsonia flushes rapidly after inundation, producing an
early-dry-season greenness peak (May–June, DOY ~120–180) before grasses senesce.
Timing of this NDVI peak may be detectably earlier than surrounding vegetation.

**Feature to extract:** DOY of maximum NDVI in the early-dry window (DOY 120–200) and
its value. Per-year values, then mean and SD across years.

**Tests that would confirm this signal is useful:**
- Presence points have a detectable NDVI peak in DOY 120–200 that absence points lack,
  or that peaks significantly earlier than absence points in the same year

**Caution:** This window overlaps with the post-wet flush of many riparian species. The
signal may not be discriminative without spatial context (HAND, distance to water).
Evaluate on its own first, then in combination.

---

## 3. Inter-annual NDVI amplitude variance (grasses vs. woody)

**Biological basis:** Native grasses have high inter-annual NDVI variance because they
are strongly rainfall-coupled. Parkinsonia is more stable year-to-year because it
accesses groundwater.

**Feature to extract:** Standard deviation or coefficient of variation of annual peak
NDVI (or dry-season mean NDVI) across all years in the archive.

**Tests that would confirm this signal is useful:**
- Presence points have significantly lower inter-annual NDVI variance than grass-dominated
  absence points
- The difference survives across a range of rainfall years (the stability signal is real,
  not an artefact of which years are in the archive)

**Note:** This overlaps with `peak_doy_sd` already in the waveform, but targets the
*amplitude* dimension rather than the *timing* dimension. Both may be needed, or one
may dominate — the test results will tell.

---

## 4. SWIR dry-season mean (B11/B12)

**Biological basis:** SWIR reflectance is sensitive to canopy water content. Dry grass
has very high SWIR; Parkinsonia's leaf-flush has distinctively low SWIR relative to the
senescent background. `flowering_index` already uses B11 at acquisition time, but a
*seasonal mean* SWIR may be a separate and more stable discriminator.

**Feature to extract:** Quality-weighted mean B11 (and optionally B12) over dry-season
acquisitions, aggregated per year then averaged across years.

**Tests that would confirm this signal is useful:**
- Presence points have lower mean dry-season SWIR than grass-dominated absence points
- The signal survives wet years (where grass SWIR may also be lower, reducing contrast)

**Implementation note:** Straightforward to add alongside existing band extraction — an
aggregation of an existing band over a different time window.

---

## 5. Rainfall-normalised NDVI anomaly

**Biological basis:** Raw NDVI anomaly is suppressed in wet years because background NDVI
rises to meet Parkinsonia's level. Normalising by a rainfall proxy should restore the
anomaly signal in wet years — directly addressing the 2025 problem.

**Feature to extract:** Per-year NDVI anomaly divided by a rainfall proxy. The simplest
proxy is the archive-wide mean NDVI for the same calendar window, already computable
from `ArchiveStats`.

**Tests that would confirm this signal is useful:**
- The normalised anomaly is more consistent across wet and dry years than the raw anomaly
  (lower inter-annual variance for presence points)
- In a simulated wet-year fixture, the normalised anomaly still separates presence from
  absence where the raw anomaly does not

**Caution:** This requires a catchment-wide reference — the feature is not computable
from a single point's observations alone. Design carefully to avoid data leakage: the
normalisation reference must be computed from the full archive *excluding* the point
being evaluated, analogous to how `ArchiveStats` is used in quality scoring.

---

## 6. Flowering spike calendar-locking (inter-annual peak DoY consistency)

**Biological basis:** Parkinsonia's flowering is photoperiod-triggered — peak DoY is
strongly calendar-locked year-to-year (low inter-annual SD). Native acacias flower more
opportunistically in response to rainfall and temperature, producing higher inter-annual
DoY variance. This is the primary temporal feature distinguishing Parkinsonia from
yellow-flowering Acacia co-occurring in the same landscape.

**Feature to extract:** Standard deviation of peak flowering index DoY across all years
with a detectable spike. Also the fraction of years in which a spike is detectable
(Parkinsonia should be consistently detectable; opportunistic flowerers will be absent in
some years).

**Tests that would confirm this signal is useful:**
- Presence points have significantly lower inter-annual SD of peak DoY than Acacia
  hard-negative points
- The fraction of years with a detectable spike is higher at presence points than at
  Acacia hard-negatives
- The signal is robust across wet and dry years (calendar-locking should be independent
  of rainfall)

**Primary absence class:** Acacia hard-negatives. This signal is specifically designed
to address the Acacia false-positive problem and should be evaluated primarily against
that class, not pooled absence points. Discriminability against grasses is expected but
is not the point of this signal.

---

## 7. Flowering spike sharpness (rise/fall rate and duration)

**Biological basis:** Parkinsonia's flowering is described in the literature as brief and
synchronised — a sharp spike with fast rise and fall. Native acacias that also produce
yellow flowering events tend to have longer, more diffuse flowering periods. Spike
duration (days above threshold) and rise/fall rate are therefore species-discriminative
independent of the index value at peak.

**Feature to extract:** Number of quality-passing observations above a threshold
surrounding the annual peak (spike duration proxy). Rise rate: difference between peak
value and the value two acquisitions prior, divided by days elapsed. Fall rate:
equivalent on the trailing side. Per-year values, then mean and SD across years.

**Tests that would confirm this signal is useful:**
- Presence points have shorter spike duration than Acacia hard-negatives
- Rise and fall rates are steeper at presence points than at Acacia hard-negatives
- The difference is consistent across years

**Primary absence class:** Acacia hard-negatives. This signal is specifically designed
to reduce Acacia false positives. Discriminability against grasses (which rarely produce
any detectable spike) is expected but not informative about this signal's purpose.

**Caution:** Cloud gaps can artifically shorten measured spike duration — a genuine
multi-week spike may appear as a single observation if surrounding acquisitions are
cloud-affected. Only count observations with quality weight above a threshold; treat
gaps as missing data, not as below-threshold.

---

## 8. Red-edge slope at flowering peak

**Biological basis:** OVERVIEW.md identifies B5/B6/B7 red-edge slope as a discriminator
for both Melaleuca and Acacia. The shape of the red-edge at the time of the flowering
peak reflects canopy density and leaf structure, which differ between Parkinsonia (fine,
semi-transparent pendant canopy) and dense-canopy natives. This is a structural signal
that persists even when broadband indices are similar.

**Feature to extract:** At the highest-quality acquisition within the annual flowering
window, compute the red-edge slope: (B7 − B5) / (centre wavelength difference), or
equivalently the ratio B6/B5 as a simpler proxy. Per-year values, then mean and SD
across years.

**Tests that would confirm this signal is useful:**
- Red-edge slope at peak differs significantly between presence points and Melaleuca
  hard-negatives
- Red-edge slope at peak differs significantly between presence points and Acacia
  hard-negatives
- The difference is consistent across years

**Primary absence class:** Melaleuca and Acacia hard-negatives. This signal targets
woody-vs-woody discrimination and should be evaluated against those classes. It is not
expected to add discriminative value against grasses beyond what existing signals provide.

---

## 9. Dry-season NDVI decline rate (Eucalyptus discriminator)

**Biological basis:** OVERVIEW.md explicitly names this: riparian Eucalyptus NDVI
typically declines through the dry season, while Parkinsonia's phreatophytic access to
groundwater maintains canopy greenness. The rate of NDVI decline between early dry
(July) and late dry (September) is therefore a direct discriminator for this species
pair. In wet years both may stay green — but the *rate of decline* should still differ
because the mechanisms are different.

**Feature to extract:** NDVI at the best-quality acquisition in a July window minus NDVI
at the best-quality acquisition in a September window, divided by days elapsed. Per-year
values, then mean and SD across years. A negative value (decline) is expected at
Eucalyptus absence points; near-zero or positive at presence points.

**Tests that would confirm this signal is useful:**
- Presence points show significantly lower (less negative) dry-season NDVI decline rate
  than Eucalyptus hard-negatives
- The difference is consistent across both wet and dry years

**Primary absence class:** Riparian Eucalyptus hard-negatives. This signal is targeted
at one specific false-positive source. It is also expected to discriminate against
grasses (which decline sharply), but that is not its primary purpose.

**Caution:** This requires two quality-passing acquisitions in separate calendar windows.
In years with heavy cloud cover through July or September, the feature may not be
computable — flag as missing rather than imputing.

---

## Evaluation approach

Once the pipeline is running end-to-end and producing real waveform features on fixture
data, each signal above can be investigated in isolation:

1. Implement the feature extraction function
2. Write a test asserting presence/absence discriminability on real fixture data
3. Add the feature to the RF and measure change in held-out AUC
4. Keep if AUC improves; remove if it does not

### Stratified evaluation by absence class

**This is critical.** A signal that works well against one absence class but not another
is not a failed signal — it is a conditionally useful signal. Pooling all absence points
masks this structure and leads to signals being incorrectly retained or discarded.

Each signal must be evaluated separately against:

| Absence class | Signals most relevant |
|---|---|
| Grass-dominated pixels | 1 (NDVI persistence), 3 (inter-annual amplitude variance), 4 (SWIR dry-season mean) |
| Riparian Eucalyptus | 9 (dry-season decline rate) |
| Melaleuca | 8 (red-edge slope) |
| Acacia / yellow-flowering natives | 6 (calendar-locking), 7 (spike sharpness), 8 (red-edge slope) |

A signal that discriminates well against grasses but not against Acacia should be
documented as a *grass-context signal*. In the RF, this means the signal is still
valuable — the model will learn to use it in combination with other features that
establish the landscape context (HAND, distance to water, background SWIR, texture).

The practical implication for training data: the absence point set must include
**hard negatives from each class** — not just randomly sampled background points, which
will be dominated by grasses and give an overly optimistic picture of signals 1, 3,
and 4. A representative absence set should include known Melaleuca stands, riparian
Eucalyptus stands, and Acacia stands from within the same tile.

### Landscape strata as named evaluation categories

Rather than abstracting landscape context into a single composite score (which would
collapse multiple independent dimensions and reinvent what the RF already does),
define a small set of named **landscape strata** as first-class categories in the
fixture dataset and evaluation reporting:

| Stratum | Definition | Key confounders |
|---|---|---|
| Riparian-woody | Low HAND (≤5m), woody matrix background | Melaleuca, Eucalyptus, Acacia |
| Riparian-grass | Low HAND (≤5m), grass/sedge matrix | Native sedges, annual grasses |
| Upland-infrastructure | High HAND (>5m), near bore drain or stock dam | Sparse Acacia, bare soil |
| Upland-dispersed | High HAND (>5m), no obvious water infrastructure | Isolated individuals, sparse cover |

Each absence hard-negative point should be assigned to a stratum. Per-stratum AUC
should be reported alongside the aggregate AUC for each signal and for the full RF
model. A model that achieves high aggregate AUC by performing well in riparian-grass
(easy) while failing in riparian-woody (hard) is not operationally useful.

This also guards against the HAND exclusion problem identified in OVERVIEW.md: if
upland-infrastructure points are absent from the training and evaluation fixture,
the model will never be tested on the case where HAND is not informative, and
performance on that stratum will be unknown until operational deployment.

### HAND vs. DEA WOfS inundation frequency as landscape context features

The current pipeline uses HAND as the primary indicator of riparian position. DEA also
provides **Water Observations from Space (WOfS)** — per-pixel inundation frequency
derived from the Landsat archive (1986–present), available as seasonal summaries
(wet/dry season separately) via the `dea-water-observations-statistics-landsat` product.

These two features measure different things and are complementary, not interchangeable:

| Feature | What it measures | Strengths | Limitations |
|---|---|---|---|
| HAND | Terrain height above nearest drainage | No cloud/vegetation interference; fine spatial resolution; captures terrain position regardless of observed water | Agnostic to actual hydrology; cannot distinguish bore drains from genuine upland; no temporal information |
| WOfS frequency | Fraction of clear observations where open water was detected (Landsat, 1986–present) | Directly observational; 40-year record; captures actual inundation history | Systematically underestimates inundation in **vegetated riparian pixels** — exactly the context where Parkinsonia grows; rivers <~50m wide and vegetated wetlands are under-detected |

**Recommendation:** use both as separate RF features, not as a choice between them.
A pixel that is low-HAND *and* has non-zero WOfS frequency is confidently riparian.
A pixel that is high-HAND with zero WOfS frequency but near mapped water infrastructure
is the upland-infrastructure case that neither feature captures well alone — which is
precisely why HAND must not be used as a hard exclusion mask (see OVERVIEW.md).

**Practical note:** WOfS dry-season frequency (`ga_ls_wo_fq_apr_oct_3`) may be more
informative than annual frequency for this application — it captures permanent or
semi-permanent water access rather than transient wet-season inundation, which is
more directly related to groundwater access and therefore Parkinsonia establishment.
