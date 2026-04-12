# Recession Sensitivity and Greenup Timing — Longreach 8×8 km Results

## Stage 1 — Raw NDVI time series

**The waveform is real but messy.** The three classes track each other closely most of the time (s1a), with only moderate separation. The 2023 La Niña year stands out — a very large NDVI spike across all classes that dwarfs the signal in other years. This is the most important observation from Stage 1: **2023 is a structural outlier that will distort any mean computed across years.**

Observation density (s1b) is surprisingly uniform — there's no severe wet-season gap. Every year has reasonable coverage across all DOY bins, which explains why `min_wet_obs` thresholds had zero effect (Stage 7c). The data is dense enough that the threshold never bites.

Smoothing (s1c) behaves well for the high-NDVI presence pixels but the low-NDVI absence pixels show the curves are noisier and more sensitive to window choice. The 15 and 30-day curves are largely similar — the signal is not strongly window-dependent at those scales.

---

## Stage 2 — NDWI moisture proxy

**This is where the recession sensitivity concept runs into its first structural problem.** The NDWI boxplot (s2a) shows that for most pixels, peak wet-season NDWI sits around −0.35 to −0.45 in most years — these are **dry soil / sparse vegetation values, not water**. The high whiskers reaching to +0.8 to +1.0 are outliers (inundated pixels in wet years). The bulk of the scene never gets wet enough for NDWI to be positive.

The s2b spatial maps confirm this: only a small footprint (the Thomson River channel itself) shows positive NDWI, and only in 2020, 2021, and 2023. In 2022, 2024, and 2025 the scene is overwhelmingly negative NDWI — the flood-pulse barely reached most pixels.

The s2c range map shows that those strongly inundated pixels (the red channel areas) have high NDWI range, which is good — but only 0.2% of pixels have range < 0.05. The bulk of the scene has adequate range, but that range is being driven by year-to-year variation in a very negative baseline, not by the difference between "flooded" and "dry." **The NDWI proxy is not measuring what you thought it was measuring.** It's measuring variation in dry-soil reflectance year to year, not flood-pulse intensity.

---

## Stage 3 — Recession slopes

**`recession_slope` is highly redundant with `rec_p`.** The correlation is r = −0.928 (s6 heatmap, correlation table). This makes theoretical sense: `rec_p` is the p90–p10 NDVI amplitude, and a steeper recession slope → larger amplitude. They are measuring the same axis — canopy persistence through the dry season — via slightly different arithmetic. You do not gain new information by adding `recession_slope` to a model that already has `rec_p`.

The per-year violin plots (s3a) show the right direction — presence is consistently less negative than absence in every year — but the distributions overlap heavily. The 2023 spike is visible here too: slopes in 2023 are more positive (less declining) for all classes, presumably because the NDVI was still elevated from the flood peak when the recession window opened.

**`recession_sensitivity` does not work at this site.** The scatter (s3c) tells the story clearly: the Pearson r between sensitivity and `prob_lr` is only −0.062, and while the trend line slopes slightly in the right direction, the cloud is essentially a vertical smear across the entire r range from −1 to +1. The signal is swamped by estimation noise from six data points. The spatial map (s3d) is salt-and-pepper — no coherent spatial structure.

The reason is visible in s3b: when you plot recession slope vs. NDWI peak coloured by `prob_lr`, both presence (green) and absence (red) pixels form the same fan shape — they both show steeper recession when NDWI is low. The theoretical prediction was that presence pixels would be a flat horizontal cloud. They are not. At Longreach, **even Parkinsonia pixels show recession sensitivity to wet season magnitude**, probably because the 8×8 km scene includes a mix of pure and mixed pixels, and because the NDWI proxy is measuring dry-soil reflectance rather than actual flood-pulse variation.

The riparian proxy (s5a) shows those 9 blue dots scattered throughout the presence+absence cloud — they are not a distinct cluster. Their median recession sensitivity is 0.013 vs. presence 0.43 and absence 0.51 (log line 98), but with only 9 pixels that is not interpretable.

---

## Stage 4 — Peak DOY (greenup timing)

**This signal is noisy but has a hint of real structure.** The per-year strip plots (s4a) show that presence peaks slightly earlier than absence in most years, but the distributions overlap massively. The median SD from s4c is ~39 days for presence vs. ~43 days for absence — a 4-day difference in consistency, which is smaller than the 14-day threshold the research doc flagged as meaningful. Both classes have peak DOY SD well above 30 days, meaning the per-year peak is not reliably located for most pixels.

The annotated curves (s4b) reveal why: **the smoothed NDVI curves for many pixels have broad plateaus rather than sharp peaks**, especially for the high-prob presence pixels. When the curve is flat for 2–3 months, the identified "peak" DOY can land anywhere in that plateau, giving large year-to-year variance not from biology but from algorithm sensitivity. The 2023 curves especially show this — a very high, flat top where the peak algorithm is essentially picking noise.

The spatial map (s4d) shows that most of the scene peaks around DOY 50–100 (February–April), with a band of later-peaking (yellow/orange, DOY 150–250) pixels in what looks like the central-southern gilgai clay area. That structure is spatially coherent, which is reassuring — the feature is capturing real spatial variation, not noise. But the class contrast is weak because the earlier-peaking presence pixels (DOY ~50–80) are not cleanly separated from many absence pixels that also peak early.

The `peak_doy_cv` correlation with `nir_cv` is +0.268 — moderate. Higher nir_cv (more variable NIR = grassland) → higher peak_doy_cv (more variable timing). That's directionally correct.

---

## Stage 5 — Riparian proxy

**The 9-pixel riparian proxy is too small to draw conclusions.** The NDVI time series (s5b) shows their mean closely tracks the presence class — they look like Parkinsonia, not like a distinct perennial water vegetation type. This is consistent with the OVERVIEW.md note that the Longreach extension area doesn't have a true perennial riparian class to test against. The Stage 5 assumption ("Longreach cannot test this") is confirmed.

---

## Stage 6 — Feature correlation

Summary of what's new vs. redundant:

| Feature | Most correlated existing | r | Verdict |
|---|---|---|---|
| `recession_slope` | `rec_p` | −0.928 | **Redundant** — same axis, different formula |
| `recession_slope_cv` | anything | <0.02 | **Uninformative** — no correlation with anything including `prob_lr` |
| `recession_sensitivity` | `re_p10` | +0.089 | **New but weak** — not discriminating at this site |
| `peak_doy` | `nir_cv` | −0.345 | **Genuinely new** — moderate independence |
| `peak_doy_cv` | `nir_cv` | +0.268 | **Partially new** — some overlap with nir_cv |

The scatter in s6b confirms this visually — `recession_slope` vs `rec_p` is a tight diagonal; `peak_doy` vs `nir_cv` is a diffuse cloud with no clear structure but also no single dominant axis.

---

## Stage 7 — Parameter sensitivity

**The recession window finding is important and contradicts the theory doc's prior.** The optimal window is May–October (start=5, end=10), not April–September. The class separation monotonically increases as you push the window later and extend it. This suggests the meaningful recession at Longreach happens later than expected — May onward, and the signal is still present through October. The April default in the spec should be revised to May.

The smoothing window result is benign — 15 vs. 30 days barely changes the SD, confirming the signal is not sensitive to this parameter in the range 15–30 days. Use 21 days as a compromise (slightly smoother than 15, not over-broadening like 45).

The `min_wet_obs=8` result (100% reliable at all thresholds) just means you have many observations per pixel-year at Longreach. This is a non-issue here but may matter at a denser-cloud site.

---

## Bottom line

**What to implement:**

- **`recession_slope` (May–October window):** Implement, but treat it as a potential replacement for or complement to `rec_p`, not an addition. They measure the same thing. `rec_p` is simpler; `recession_slope` is more interpretable. Worth keeping because it degrades more gracefully when the dry season is abbreviated.

- **`recession_sensitivity`:** Do not implement as a production signal for Longreach. The theoretical mechanism (sensitivity to flood-pulse intensity) is not separable from noise with 6 years at this site. May be worth revisiting with 10+ years or at a site with more dramatic inter-annual flood variability (e.g. a true flood-pulse system). **The NDWI proxy is measuring the wrong thing for most pixels** — the bulk of the scene doesn't flood, so year-to-year NDWI variation reflects soil moisture / sparse canopy variation, not the flood event the theory assumed.

- **`peak_doy`:** Implement cautiously. It's the most genuinely novel feature (r=−0.35 with `nir_cv`, not redundant), and the spatial map shows real structure. But the per-pixel SD is ~39–43 days — too large for the feature to be reliable at the individual pixel level. It may be useful as a **spatial smoothing target** (neighbourhood mean of peak_doy) rather than raw per-pixel. The 4-day difference in SD between classes is too small to use `peak_doy_cv` as a discriminating feature on its own.

- **`peak_doy_cv`:** Lower priority. Partially overlaps with `nir_cv`. Only include if `peak_doy` is included and you want the consistency axis.

- **`recession_slope_cv`:** Drop. No correlation with anything.
