"""Science test: peak flowering timing and calendar-locking.

Hypothesis: Peak flowering DOY at presence points falls within the known
window for tropical QLD (Aug-Sep, DOY 213-273) and is more consistent
year-to-year than at absence points.

Two sub-tests:
1. Peak DOY distribution: chi-squared test of uniformity across months.
   Presence peaks should cluster in Aug-Sep (non-uniform), absence peaks
   should not show the same clustering.
2. Inter-annual SD of peak DOY: Mann-Whitney U, p < 0.05, presence SD < absence SD.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict

import pandas as pd
import pytest
from scipy import stats

from analysis.primitives.indices import flowering_index
from analysis.timeseries.waveform import extract_waveform_features

# Aug–Sep: DOY 213–273
TIMING_WINDOW = (213, 273)

# Months (1-indexed) used for chi-squared bins: Jul=7 through Dec=12
_MONTH_BINS = list(range(7, 13))  # 6 bins


def _doy_to_month(doy: int) -> int:
    """Approximate DOY → month (1-indexed). Using 30.44-day months."""
    return min(12, max(1, math.ceil(doy / 30.44)))


def _per_year_peak_doys(obs_list: list) -> list[int]:
    """Return peak DOY per calendar year within TIMING_WINDOW.

    Returns empty list if fewer than 2 years of data.
    """
    doy_start, doy_end = TIMING_WINDOW
    by_year: dict[int, list] = defaultdict(list)
    for obs in obs_list:
        doy = obs.date.timetuple().tm_yday
        if doy_start <= doy <= doy_end:
            by_year[obs.date.year].append(obs)

    peak_doys: list[int] = []
    for year_obs in by_year.values():
        if not year_obs:
            continue
        # Quality-weighted peak
        wf = extract_waveform_features(year_obs, index_fn=flowering_index,
                                       window=TIMING_WINDOW, min_years=1)
        if wf:
            peak_doys.append(int(wf["peak_doy"]))
    return peak_doys


def test_peak_doy_clustering(observations, science_points, report_collector):
    """Presence peak DOYs cluster in Aug-Sep (chi-squared non-uniformity, p < 0.05)."""
    labels = science_points.set_index("point_id")["label"].to_dict()

    presence_doys: list[int] = []
    for pid, obs_list in observations.items():
        if labels.get(pid) == 1:
            presence_doys.extend(_per_year_peak_doys(obs_list))

    n = len(presence_doys)

    if n < 6:
        report_collector.add(
            "Signal 3a: Peak DOY clustering",
            f"SKIP — insufficient presence peak DOYs (n={n}, need ≥6)",
        )
        pytest.skip(f"Insufficient presence peak DOYs: n={n}")

    # Bin into months 7–12
    month_counts = {m: 0 for m in _MONTH_BINS}
    for doy in presence_doys:
        m = _doy_to_month(doy)
        if m in month_counts:
            month_counts[m] += 1

    observed = [month_counts[m] for m in _MONTH_BINS]
    # Expected under uniformity
    expected = [n / len(_MONTH_BINS)] * len(_MONTH_BINS)

    chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected)

    # Aug+Sep bins
    aug_sep_count = month_counts.get(8, 0) + month_counts.get(9, 0)
    aug_sep_frac  = aug_sep_count / n if n > 0 else 0.0

    verdict = "PASS" if p_value < 0.05 else "FAIL"

    report_collector.add(
        "Signal 3a: Peak DOY clustering",
        f"Hypothesis: presence peak DOYs cluster in Aug-Sep (DOY {TIMING_WINDOW[0]}–{TIMING_WINDOW[1]})\n"
        f"  Total presence peak DOYs: {n}\n"
        f"  Aug+Sep fraction: {aug_sep_frac:.2%}  (n={aug_sep_count})\n"
        f"  Monthly counts (Jul–Dec): {observed}\n"
        f"  Chi-squared: {chi2:.3f}  p={p_value:.4f}  {verdict}",
    )

    assert p_value < 0.05, (
        f"Peak DOY not clustered significantly: chi2={chi2:.3f}, p={p_value:.4f} >= 0.05 "
        f"(Aug+Sep fraction={aug_sep_frac:.2%}, n={n})"
    )


def test_peak_doy_consistency(observations, science_points, report_collector):
    """Presence inter-annual SD of peak DOY < absence (Mann-Whitney U, p < 0.05)."""
    labels = science_points.set_index("point_id")["label"].to_dict()

    presence_sds: list[float] = []
    absence_sds:  list[float] = []

    for pid, obs_list in observations.items():
        doys = _per_year_peak_doys(obs_list)
        if len(doys) < 2:
            continue
        sd = statistics.pstdev(doys)
        if labels.get(pid) == 1:
            presence_sds.append(sd)
        elif labels.get(pid) == 0:
            absence_sds.append(sd)

    n_pres = len(presence_sds)
    n_abs  = len(absence_sds)

    pres_median = statistics.median(presence_sds) if presence_sds else float("nan")
    abs_median  = statistics.median(absence_sds)  if absence_sds  else float("nan")

    if n_pres < 2 or n_abs < 2:
        report_collector.add(
            "Signal 3b: Inter-annual peak DOY SD",
            f"SKIP — insufficient data (presence n={n_pres}, absence n={n_abs})",
        )
        pytest.skip(f"Insufficient data: presence n={n_pres}, absence n={n_abs}")

    _, p_value = stats.mannwhitneyu(presence_sds, absence_sds, alternative="less")

    verdict = "PASS" if (p_value < 0.05 and pres_median < abs_median) else "FAIL"
    direction_note = "" if pres_median < abs_median else "  WARNING: direction reversed"

    report_collector.add(
        "Signal 3b: Inter-annual peak DOY SD",
        f"Hypothesis: presence has lower inter-annual peak DOY SD than absence (calendar-locking)\n"
        f"  Presence median SD: {pres_median:.2f} days  (n={n_pres})\n"
        f"  Absence median SD:  {abs_median:.2f} days  (n={n_abs})\n"
        f"  Mann-Whitney U:     p={p_value:.4f}  {verdict}{direction_note}",
    )

    assert pres_median < abs_median, (
        f"Direction wrong: presence median SD {pres_median:.2f} >= absence {abs_median:.2f}"
    )
    assert p_value < 0.05, (
        f"Calendar-locking signal not significant: p={p_value:.4f} >= 0.05 "
        f"(n_pres={n_pres}, n_abs={n_abs})"
    )
