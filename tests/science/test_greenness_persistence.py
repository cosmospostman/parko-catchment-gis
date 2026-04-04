"""Science test: dry-season greenness persistence.

Hypothesis: Presence points have higher mean dry-season NDVI (Jul-Sep,
DOY 182-273) than absence points, and lower inter-annual variance.

Two sub-tests:
1. Mean dry-season NDVI: Mann-Whitney U, p < 0.05, presence > absence
2. Inter-annual NDVI variance: Mann-Whitney U, p < 0.05, presence < absence
"""

from __future__ import annotations

import statistics
from collections import defaultdict

import pandas as pd
import pytest
from scipy import stats

# Jul–Sep: DOY 182–273
DRY_SEASON_DOY = (182, 273)


def _ndvi(bands: dict[str, float]) -> float:
    b08 = bands.get("B08", 0.0)
    b04 = bands.get("B04", 0.0)
    denom = b08 + b04
    if denom == 0:
        return 0.0
    return (b08 - b04) / denom


def _dry_season_ndvi_stats(obs_list: list) -> tuple[float | None, float | None]:
    """Return (mean_ndvi, interannual_variance) for dry-season observations.

    Groups by year, computes per-year mean NDVI within the dry window, then
    returns the overall mean and the variance across years.

    Returns (None, None) if fewer than 2 years of dry-season data exist.
    """
    doy_start, doy_end = DRY_SEASON_DOY

    by_year: dict[int, list[float]] = defaultdict(list)
    for obs in obs_list:
        doy = obs.date.timetuple().tm_yday
        if doy_start <= doy <= doy_end:
            by_year[obs.date.year].append(_ndvi(obs.bands))

    if len(by_year) < 2:
        return None, None

    yearly_means = [statistics.mean(vals) for vals in by_year.values() if vals]
    if len(yearly_means) < 2:
        return None, None

    overall_mean = statistics.mean(yearly_means)
    interannual_var = statistics.variance(yearly_means)
    return overall_mean, interannual_var


def test_dry_season_ndvi_mean(observations, science_points, report_collector):
    """Presence mean dry-season NDVI > absence (Mann-Whitney U, p < 0.05)."""
    labels = science_points.set_index("point_id")["label"].to_dict()

    presence_means = []
    absence_means  = []

    for pid, obs_list in observations.items():
        mean_ndvi, _ = _dry_season_ndvi_stats(obs_list)
        if mean_ndvi is None:
            continue
        if labels.get(pid) == 1:
            presence_means.append(mean_ndvi)
        elif labels.get(pid) == 0:
            absence_means.append(mean_ndvi)

    n_pres = len(presence_means)
    n_abs  = len(absence_means)

    pres_median = statistics.median(presence_means) if presence_means else float("nan")
    abs_median  = statistics.median(absence_means)  if absence_means  else float("nan")

    if n_pres < 2 or n_abs < 2:
        report_collector.add(
            "Signal 2a: Dry-season NDVI mean",
            f"SKIP — insufficient data (presence n={n_pres}, absence n={n_abs})",
        )
        pytest.skip(f"Insufficient data: presence n={n_pres}, absence n={n_abs}")

    _, p_value = stats.mannwhitneyu(presence_means, absence_means, alternative="greater")

    verdict = "PASS" if (p_value < 0.05 and pres_median > abs_median) else "FAIL"
    direction_note = "" if pres_median > abs_median else "  WARNING: direction reversed"

    report_collector.add(
        "Signal 2a: Dry-season NDVI mean",
        f"Hypothesis: presence has higher mean dry-season NDVI than absence (DOY {DRY_SEASON_DOY[0]}–{DRY_SEASON_DOY[1]})\n"
        f"  Presence median: {pres_median:.4f}  (n={n_pres})\n"
        f"  Absence median:  {abs_median:.4f}  (n={n_abs})\n"
        f"  Mann-Whitney U:  p={p_value:.4f}  {verdict}{direction_note}",
    )

    assert pres_median > abs_median, (
        f"Direction wrong: presence median {pres_median:.4f} <= absence median {abs_median:.4f}"
    )
    assert p_value < 0.05, (
        f"Dry-season NDVI signal not significant: p={p_value:.4f} >= 0.05 "
        f"(n_pres={n_pres}, n_abs={n_abs})"
    )


def test_dry_season_ndvi_variance(observations, science_points, report_collector):
    """Presence inter-annual NDVI variance < absence (Mann-Whitney U, p < 0.05)."""
    labels = science_points.set_index("point_id")["label"].to_dict()

    presence_vars = []
    absence_vars  = []

    for pid, obs_list in observations.items():
        _, iav = _dry_season_ndvi_stats(obs_list)
        if iav is None:
            continue
        if labels.get(pid) == 1:
            presence_vars.append(iav)
        elif labels.get(pid) == 0:
            absence_vars.append(iav)

    n_pres = len(presence_vars)
    n_abs  = len(absence_vars)

    pres_median = statistics.median(presence_vars) if presence_vars else float("nan")
    abs_median  = statistics.median(absence_vars)  if absence_vars  else float("nan")

    if n_pres < 2 or n_abs < 2:
        report_collector.add(
            "Signal 2b: Dry-season NDVI inter-annual variance",
            f"SKIP — insufficient data (presence n={n_pres}, absence n={n_abs})",
        )
        pytest.skip(f"Insufficient data: presence n={n_pres}, absence n={n_abs}")

    # "presence variance < absence" → alternative="less"
    _, p_value = stats.mannwhitneyu(presence_vars, absence_vars, alternative="less")

    verdict = "PASS" if (p_value < 0.05 and pres_median < abs_median) else "FAIL"
    direction_note = "" if pres_median < abs_median else "  WARNING: direction reversed"

    report_collector.add(
        "Signal 2b: Dry-season NDVI inter-annual variance",
        f"Hypothesis: presence has lower inter-annual NDVI variance than absence\n"
        f"  Presence median var: {pres_median:.6f}  (n={n_pres})\n"
        f"  Absence median var:  {abs_median:.6f}  (n={n_abs})\n"
        f"  Mann-Whitney U:      p={p_value:.4f}  {verdict}{direction_note}",
    )

    assert pres_median < abs_median, (
        f"Direction wrong: presence median variance {pres_median:.6f} >= absence {abs_median:.6f}"
    )
    assert p_value < 0.05, (
        f"NDVI variance signal not significant: p={p_value:.4f} >= 0.05 "
        f"(n_pres={n_pres}, n_abs={n_abs})"
    )
