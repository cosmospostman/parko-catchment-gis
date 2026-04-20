# Recession Sensitivity and Greenup Timing — Longreach 8×8 km Results

Data: 2016–2021 Sentinel-2 imagery. This period is appropriate because the presence/absence training patches were drawn from 2021 imagery, so the satellite record is concurrent with the ground-truth labels.

---

## Stage 1 — Raw NDVI time series

**The waveform is present and readable, and the three-class separation is real.** s1a (reproduced as s5b with the riparian overlay) shows clear annual cycles across 2016–2021 with consistent class ordering throughout the record.

**Presence vs. absence:** Presence sits roughly 0.05–0.10 NDVI units above absence during the April–October dry season in every year without exception. The two classes converge at wet-season peaks — the separation is dry-season specific. This is the axis `rec_p` and `recession_slope` capture.

**Riparian vs. both:** The riparian class is not simply another high-NDVI class. During wet-season flush events (early 2017, late 2019, early 2020), riparian shows distinctly *lower* NDVI spikes than presence, while absence spikes hard. In the dry season, riparian holds a floor similar to presence. The pattern is: high dry-season floor + suppressed wet-season spike. This is ecologically coherent — pixels in or immediately adjacent to the river channel may be partially inundated (low NDVI from water surface) during flood events, while Parkinsonia on the surrounding floodplain gets the flood pulse and flushes green.

**This three-way structure identifies an unmeasured signal axis.** The distinguishing geometry across classes is:
- Presence: high dry-season floor, strong wet-season spike
- Absence: low dry-season floor, strong wet-season spike
- Riparian: high dry-season floor, **suppressed** wet-season spike

The ratio of wet-peak amplitude to dry-season floor — roughly the inverse of what `rec_p` measures — would separate riparian from Parkinsonia in a way that `nir_cv`, `rec_p`, and `recession_slope` all fail to do. None of the existing or proposed signals target this axis explicitly. The 9-pixel riparian sample at Longreach is too small to validate a signal from, but the pattern is a concrete design target for a perennial-river site where the riparian class is larger.

The dry-season gap between presence and absence is most visible from approximately DOY 100–250 (April–September) each year — consistent with the theoretical recession window. The six years are broadly comparable in amplitude, making cross-year statistics meaningful.

Observation density (s1b) is uniform year-round — no severe wet-season cloud gap bites into data availability. Every DOY bin has adequate coverage, which explains why `min_wet_obs` thresholds have zero effect (Stage 7c).

Smoothing (s1c) is well-behaved. The 15-day and 30-day curves are largely indistinguishable for both classes. Window choice is not load-bearing in the 15–30-day range.

---

## Stage 2 — NDWI moisture proxy

**The NDWI proxy has a structural problem that limits `recession_sensitivity`.** The boxplot (s2a) shows that for the bulk of pixels, peak wet-season NDWI sits around −0.35 to −0.45 in most years — these are dry soil or sparse vegetation values, not open water or saturated soil. The high whiskers reaching +0.75 to +1.0 are inundation outliers from the Thomson River channel itself.

The spatial maps (s2b) confirm this: positive NDWI (actual flooding) is confined to a small footprint around the river channel, and only in 2019 and 2020 does this footprint expand meaningfully across the scene. The other years show uniformly negative NDWI scene-wide.

The consequence is that year-to-year variation in NDWI for most pixels is not measuring flood-pulse intensity. It is measuring variation in dry-soil reflectance and sparse canopy cover year to year — a weaker and noisier signal than the flood-pulse proxy the theory assumed. Only 0.2% of pixels have NDWI range < 0.05, so the data is not flat; the problem is that the variation is in the wrong domain (dry negatives, not wet-to-dry transitions).

---

## Stage 3 — Recession slopes

**`recession_slope` is highly redundant with `rec_p`.** The correlation is r = −0.922 (correlation table). Theoretically this is unsurprising — `rec_p` is the p90–p10 NDVI amplitude, and a steeper recession slope produces a larger amplitude. Both features measure canopy persistence through the dry season via different arithmetic. Adding `recession_slope` to a model that already has `rec_p` does not add a new axis of information.

The per-year violin plots (s3a) show the right direction: presence is consistently less negative (shallower recession) than absence in every year from 2016–2021. The class medians are clearly separated, with absence roughly twice as negative as presence. The distributions overlap substantially — these are not clean separating features on their own — but the direction is reliable.

**`recession_sensitivity` does not work at this site.** The scatter (s3c) tells the story clearly: Pearson r between sensitivity and `prob_lr` is only 0.018, and the OLS slope is 0.057 — a nearly flat line through a vertical smear spanning the full range from −1 to +1. The signal is swamped by estimation noise from six data points per pixel. The spatial map (s3d) is salt-and-pepper with no coherent spatial structure.

The mechanism failure is visible in s3b: when recession slope is plotted against NDWI peak (coloured by `prob_lr`), both presence (green) and absence (red) pixels form the same broad fan. The theoretical prediction was that high-`prob_lr` pixels would form a flat horizontal cloud — they do not. At Longreach, Parkinsonia pixels show similar recession sensitivity to wet-season moisture as grassland pixels, likely because the NDWI proxy is not actually measuring the flood-pulse condition the theory assumed.

Riparian proxy statistics (Stage 5): median recession sensitivity for the 9 riparian pixels is 0.013, compared to presence 0.43 and absence 0.51 — directionally consistent with the theory (more decoupled from moisture), but the sample of 9 is too small to interpret.

---

## Stage 4 — Peak DOY (greenup timing)

**This is the most promising genuinely new signal.** The per-pixel peak DOY standard deviation (s4c) shows a meaningful class separation: presence median SD ≈ 35 days, absence median SD ≈ 51–53 days. This 16-day gap is substantially larger than the 14-day threshold flagged in the theory doc as meaningful. The histograms are shifted — presence pixels are concentrated at lower SD values with a clear mode around 25–35 days, while absence pixels have a broader distribution extending to high SD values.

The per-year strip plots (s4a) show presence pixels peaking earlier and more consistently than absence in most years. The class medians diverge most strongly in 2017, 2019, and 2021. The separation is not perfect in every year, but the directional signal is consistent.

The annotated curves (s4b) reveal the mechanism behind the noise: many high-probability presence pixels have broad NDVI plateaus from roughly DOY 50–200, where the curve is flat enough that the peak-finding algorithm picks slightly different DOYs year to year. This is biology (Parkinsonia maintaining high canopy through the dry season) but it produces algorithmic jitter in the DOY estimate. The wide plateau is itself a meaningful feature — it confirms canopy persistence — but it degrades the precision of a single peak DOY estimate.

The spatial map (s4d) shows spatially coherent structure: a band of earlier-peaking pixels (purple/blue, DOY ~50–100) overlying what appears to be the river corridor and surrounding Parkinsonia-dense areas, surrounded by later-peaking (orange/yellow, DOY ~150–200+) grassland. The coherence is reassuring — the feature is tracking real spatial variation rather than noise.

`peak_doy` correlation with existing features: r = −0.36 with `rec_p` and −0.35 with `nir_cv`. This is moderate correlation but not redundant — there is genuine independent information in the DOY axis.

`peak_doy_cv` correlation with `nir_cv`: r = 0.21. Higher NIR variability (grassland) → higher peak DOY variability. Directionally correct, partially overlapping with `nir_cv`.

---

## Stage 5 — Riparian proxy

**The 9-pixel sample is too small to validate a signal, but the pattern is informative.** The NDVI time series (s5b) — the same data shown in s1a — reveals that riparian pixels are not simply another high-NDVI class. See Stage 1 for the detailed interpretation. The key finding is that riparian holds a dry-season floor comparable to presence but shows a suppressed wet-season spike, distinguishing it from both presence and absence on an axis no existing signal captures.

The riparian proxy statistics from Stage 3 (median recession sensitivity 0.013 vs. presence 0.43 and absence 0.51) are directionally consistent with permanent water access decoupling the recession slope from wet-season forcing, but nine pixels is not enough to draw conclusions. Longreach cannot test the perennial riparian hypothesis at scale — that requires a site where the riparian class is substantially larger.

---

## Stage 6 — Feature correlation

| Feature | Most correlated existing | r | Verdict |
|---|---|---|---|
| `recession_slope` | `rec_p` | −0.922 | **Redundant** — same canopy-persistence axis |
| `recession_slope_cv` | anything | <0.01 | **Uninformative** — no meaningful correlation with any feature or `prob_lr` |
| `recession_sensitivity` | `re_p10` | +0.047 | **New information but non-discriminating** — not separating classes at this site |
| `peak_doy` | `rec_p` | −0.360 | **Genuinely new** — moderate independence from existing features |
| `peak_doy_cv` | `nir_cv` | +0.206 | **Partially new** — some overlap with `nir_cv` |

---

## Stage 7 — Parameter sensitivity

**The recession window finding is consistent with the theory and important.** The optimal window is May–October (start=5, end=10), with median class separation of 0.00050 — the strongest of all nine combinations. The default April–September from the spec (start=4, end=9) scores 0.00038, about 25% weaker. The improvement from pushing the window later is monotonic: every step from start=3 toward start=5 and from end=8 toward end=10 increases separation. This suggests the meaningful Parkinsonia recession at Longreach is later-season than the April theoretical prior. **The implementation default should be May–October.**

Smoothing window: the peak DOY SD is barely affected by window choice in the 15–45 day range. Presence SD shifts from ~36 days at 15d to ~36 days at 30d and ~36 days at 45d. Absence SD shifts from ~53 to ~51 to ~50 days. The class gap is stable. Window choice is not load-bearing; 30 days is a reasonable default.

`min_wet_obs`: all thresholds up to 8 leave 100% of pixel-years as reliable. Data density at Longreach is not a constraint. This parameter will only matter at sites with heavier cloud cover.

---

## Bottom line

**What to implement:**

- **`recession_slope` (May–October window):** Implement, but treat as a potential complement or replacement for `rec_p`, not an addition. They measure the same axis. `rec_p` is simpler computationally; `recession_slope` is more interpretable as a physical rate and may degrade more gracefully when the dry season is abbreviated or the growing season boundary shifts. Worth keeping for that flexibility.

- **`recession_sensitivity`:** Do not implement as a production signal. The NDWI proxy is not measuring the flood-pulse contrast the theory requires for most pixels — the bulk of the scene never floods, so year-to-year NDWI variation reflects soil moisture noise rather than a meaningful wet-season forcing variable. With only six data points per pixel, the per-pixel Pearson r is dominated by estimation noise. May be worth revisiting with 10+ years or at a site with genuine flood-pulse heterogeneity (e.g. a floodplain system where most pixels go from flooded to dry annually).

- **`peak_doy`:** Implement. It is the most genuinely novel feature (r ≈ −0.36 with the closest existing signal, not redundant), the spatial map shows coherent structure, and the class SD gap (35 vs. 51 days) is meaningful. The per-pixel SD is still substantial — most pixels have SD > 14 days — so raw `peak_doy` is noisy at the individual pixel level. Consider using a spatially smoothed version (neighbourhood mean) in the feature pipeline, which would reduce noise while preserving the spatial structure visible in s4d.

- **`peak_doy_cv`:** Include alongside `peak_doy` if it is implemented. It captures the consistency axis independently of the timing level and partially separates from `nir_cv` (r = 0.21, not redundant). The class separation in SD (s4c) directly motivates this feature.

- **`recession_slope_cv`:** Drop. No correlation with any existing feature or with `prob_lr`. Uninformative at this site.
