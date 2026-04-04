"""Science test: flowering flush signal.

Hypothesis: Presence points produce a higher quality-weighted flowering index
peak than absence points within the Aug-Oct window (DOY 213-304).

Statistical test: Mann-Whitney U (non-parametric; no normality assumption).
Assertion: p < 0.05 AND median presence peak > median absence peak.
"""

from __future__ import annotations

import statistics

import pandas as pd
import pytest
from scipy import stats

from analysis.primitives.indices import flowering_index
from analysis.timeseries.waveform import extract_waveform_features

# Aug–Oct: DOY 213–304
FLOWERING_WINDOW = (213, 304)


def _peak_values(obs_by_point: dict, labels: dict[str, int], label: int) -> list[float]:
    """Extract peak_value for all points with the given label."""
    peaks = []
    for pid, obs_list in obs_by_point.items():
        if labels.get(pid) != label:
            continue
        wf = extract_waveform_features(obs_list, index_fn=flowering_index, window=FLOWERING_WINDOW)
        if wf:
            peaks.append(wf["peak_value"])
    return peaks


def test_flowering_signal(observations, science_points, report_collector):
    """Presence peak_value > absence peak_value within Aug-Oct window (p < 0.05)."""
    labels = science_points.set_index("point_id")["label"].to_dict()

    presence_peaks = _peak_values(observations, labels, label=1)
    absence_peaks  = _peak_values(observations, labels, label=0)

    n_pres = len(presence_peaks)
    n_abs  = len(absence_peaks)

    pres_median = statistics.median(presence_peaks) if presence_peaks else float("nan")
    abs_median  = statistics.median(absence_peaks)  if absence_peaks  else float("nan")

    if n_pres < 2 or n_abs < 2:
        report_collector.add(
            "Signal 1: Flowering flush",
            f"SKIP — insufficient data (presence n={n_pres}, absence n={n_abs})",
        )
        pytest.skip(f"Insufficient data for Mann-Whitney: presence n={n_pres}, absence n={n_abs}")

    stat, p_value = stats.mannwhitneyu(presence_peaks, absence_peaks, alternative="greater")

    verdict = "PASS" if (p_value < 0.05 and pres_median > abs_median) else "FAIL"
    direction_note = "" if pres_median > abs_median else "  WARNING: direction reversed"

    report_collector.add(
        "Signal 1: Flowering flush",
        f"Hypothesis: presence points have higher mean peak_value than absence (DOY {FLOWERING_WINDOW[0]}–{FLOWERING_WINDOW[1]})\n"
        f"  Presence median: {pres_median:.4f}  (n={n_pres})\n"
        f"  Absence median:  {abs_median:.4f}  (n={n_abs})\n"
        f"  Mann-Whitney U:  p={p_value:.4f}  {verdict}{direction_note}",
    )

    assert pres_median > abs_median, (
        f"Direction wrong: presence median {pres_median:.4f} <= absence median {abs_median:.4f}"
    )
    assert p_value < 0.05, (
        f"Flowering signal not significant: p={p_value:.4f} >= 0.05 "
        f"(presence median={pres_median:.4f}, absence median={abs_median:.4f}, "
        f"n_pres={n_pres}, n_abs={n_abs})"
    )


def test_flowering_signal_by_year(observations_by_year, science_points, report_collector):
    """Per-year flowering signal — reported but not pass/fail gated."""
    labels = science_points.set_index("point_id")["label"].to_dict()

    year_summaries: list[str] = []

    for year, obs_by_point in sorted(observations_by_year.items()):
        presence_peaks = _peak_values(obs_by_point, labels, label=1)
        absence_peaks  = _peak_values(obs_by_point, labels, label=0)

        n_pres = len(presence_peaks)
        n_abs  = len(absence_peaks)

        if n_pres < 2 or n_abs < 2:
            year_summaries.append(
                f"  {year}: SKIP (presence n={n_pres}, absence n={n_abs})"
            )
            continue

        pres_median = statistics.median(presence_peaks)
        abs_median  = statistics.median(absence_peaks)
        _, p_value = stats.mannwhitneyu(presence_peaks, absence_peaks, alternative="greater")

        direction = "pres > abs" if pres_median > abs_median else "pres <= abs (reversed)"
        year_summaries.append(
            f"  {year}: presence={pres_median:.4f} (n={n_pres}), "
            f"absence={abs_median:.4f} (n={n_abs}), "
            f"p={p_value:.4f}, {direction}"
        )

    report_collector.add(
        "Signal 1: Flowering flush (per year)",
        "\n".join(year_summaries) if year_summaries else "  No per-year data available",
    )
    # No assertion: per-year results are diagnostic, not pass/fail
