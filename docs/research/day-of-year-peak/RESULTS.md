# Day-of-Year Peak — Results Summary

Investigation conducted at Longreach 8×8 km using a controlled labelling scheme:
- **Presence** — infestation sub-bbox (n=366 pixels, confirmed dense riverside Parkinsonia)
- **Absence** — grassland sub-bbox only (n=340 pixels, homogeneous grassland, no Parkinsonia)

This avoids both the `prob_lr` circularity and the heterogeneous-absence pessimism of
earlier labelling schemes. Outputs in `longreach-green-up/`.

---

## Stage 1 — Absolute peak DOY (`peak_doy`)

**Result: not discriminating at this site. |median sep| = 7d, wrong direction.**

Presence median DOY 104.8 vs Absence 97.8 — Parkinsonia peaks *later*, not earlier.
Per-year strip plots show heavy overlap every year with medians that jump around across
years. Per-pixel SD is virtually identical between classes (Presence med=43.9d,
Absence med=43.8d) — the classes are indistinguishable on timing variability.

**Why:** The infestation bbox and grassland bbox are physically adjacent (<300m apart)
along the same river corridor. Both have similar flood-pulse water access, so their
absolute peak timing is nearly identical. The original hypothesis — Parkinsonia peaks
earlier because its deep roots buffer it from rainfall variability — may hold at the
corridor-vs-upland scale, but not within this narrow same-corridor comparison.

The spatial map (s1d) confirms that real timing structure exists in the scene: the river
corridor as a whole peaks earlier (DOY ~50–100, purple) than the surrounding upland
matrix (DOY ~100–150, orange/yellow). But that contrast is between the corridor and
upland, not between Parkinsonia and grassland *within* the corridor.

---

## Stage 2 — Relative greenup shift (`peak_doy_shift`, R=250m)

**Result: 8d separation but wrong direction; signal is geometrically contingent.**

Presence median shift +5d, Absence median shift −3d — Parkinsonia pixels peak *later*
than their 250m neighbourhood, not earlier. Shift SD shows a mild Presence advantage
(13.5d vs 15.6d), meaning Parkinsonia's offset is slightly more consistent across years,
but the separation is small and the direction is not interpretable as an intrinsic signal.

**Why the direction inverts:** The infestation sits *inside* the early-peaking riparian
corridor. Its 250m neighbours are also riparian (early-peaking), so the neighbourhood
median is pulled early and Parkinsonia looks late by comparison. If the infestation were
embedded in a grassland matrix — surrounded by late-peaking upland pixels — the same
Parkinsonia phenology would produce a strongly *negative* shift.

**The fundamental problem:** `peak_doy_shift` measures contrast with the local
neighbourhood, so its sign and magnitude depend on what that neighbourhood contains,
not on any intrinsic property of Parkinsonia. A model trained on a riparian-matrix site
would learn a different shift direction than one trained on a grassland-matrix site.
The feature is not transferable across spatial contexts.

The per-year strip plot (s2d) confirms this: in most years both classes cluster near
zero with overlapping spreads and no consistent directional story.

---

## Stage 3 — Sensitivity sweep (R × amplitude gate)

| R | Gate | |sep| | Presence median | Absence median |
|---|---|---|---|---|
| 250m | 5% | 8.0d | +5d | −3d |
| 250m | 10% | 8.0d | +5d | −3d |
| 250m | 20% | 8.0d | +5d | −3d |
| 500m | 5–20% | 4.0d | +5d | +1d |
| 1000m | 5% | 8.0d | +1d | −7d |
| 1000m | 10% | 7.5d | +3d | −4.5d |
| 1000m | 20% | 7.0d | +6d | −1d |

**Amplitude gate has no effect** — separation is flat across 5%, 10%, 20% at every
radius. The gate can be treated as a fixed non-tunable parameter.

**Non-monotonic radius behaviour:** 250m (8d) → 500m (4d, collapse) → 1000m (8d,
recovery with flipped sign). The 500m collapse occurs because the neighbourhood bleeds
into upland grassland, muddying the reference. The 1000m recovery reflects a now-pure
upland reference, but the signal interpretation has changed — Absence is now the
earlier-peaking class relative to a broad upland neighbourhood.

This non-monotonic behaviour confirms that the shift signal is highly sensitive to the
spatial configuration of the infestation relative to its matrix — not a stable
intrinsic discriminator.

---

## Overall conclusion

Peak-timing features (absolute or relative) are **not discriminating** within the
riparian corridor at Longreach. The features track real spatial variation in the scene
(corridor vs upland), but cannot separate Parkinsonia from native riparian vegetation
within the corridor, which is the harder and more practically important problem.

**The missing axis is waveform shape**, not timing. Native riparian eucalypts have a
narrow wet-season spike (high amplitude, low integrated greenness); Parkinsonia has a
sustained green-stem floor year-round (moderate amplitude, high integrated greenness).
This distinction is:
- **Intrinsic** — independent of what neighbours are doing
- **Transferable** — present regardless of whether water access is flood-pulse or
  point-source (dam/bore)
- **Directly addressed** by `ndvi_integral` (see `research/ndvi-integral/HYPOTHESIS.md`)

Peak-timing signals may still add marginal value as ensemble features at sites where
the infestation is embedded in a grassland matrix, but should not be primary
discriminators in riparian-corridor contexts.

---

## Recommendation for feature pipelines

**`peak_doy` — include as a weak supporting feature.**
The recession-and-greenup investigation (same site, same labelling scheme) found
`peak_doy` has r = −0.36 with the nearest existing feature (`rec_p`). That is genuine
partial independence — a linear regression will extract real information from it even
though single-feature class separation is poor. The spatial map shows coherent structure
(corridor pixels cluster at DOY 50–100, upland at 100–150+), so the feature is not
noise. Expected coefficient weight is small; it will not carry a model on its own.

**`peak_doy_shift` — do not include.**
The shift feature is geometrically contingent: its sign and magnitude depend on what the
250m–1000m neighbourhood contains, not on any intrinsic property of Parkinsonia. A model
trained at a riparian-matrix site would learn the opposite coefficient to one trained at a
grassland-matrix site. This is a structural transferability problem, not a noise problem
that more data or tuning would fix.

**`peak_doy_cv` (per-pixel timing SD) — include alongside `peak_doy`.**
The recession-and-greenup results show a 16-day class gap in peak DOY standard deviation
(presence median ~35d, absence ~51d) that is partially independent of `nir_cv`
(r = 0.21). It captures timing consistency rather than timing level, and the two features
together span a broader axis than either alone.
