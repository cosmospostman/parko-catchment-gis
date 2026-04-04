"""Science test: quality weighting effectiveness.

Hypothesis: Quality-weighted peak detection produces a lower false-peak rate
than raw maximum detection on the same observations.

Method: For each absence point, compare:
- Raw max of flowering index within window (unweighted)
- Quality-weighted peak (as per extract_waveform_features)

A "false peak" is a peak value above FLOWERING_THRESHOLD that occurs in a
single low-quality observation (quality < 0.4).

Assertion: Weighted method produces fewer false peaks than raw max.
"""

from __future__ import annotations

import pytest
from scipy import stats

from analysis.constants import FLOWERING_THRESHOLD, FLOWERING_WINDOW
from analysis.primitives.indices import flowering_index

_LOW_QUALITY_THRESHOLD = 0.4


def _raw_false_peaks(obs_list: list) -> int:
    """Count false peaks using raw (unweighted) max within FLOWERING_WINDOW.

    A false peak: max raw index value >= FLOWERING_THRESHOLD, AND that maximum
    occurs in a single observation whose quality score < LOW_QUALITY_THRESHOLD.
    """
    doy_start, doy_end = FLOWERING_WINDOW
    window_obs = [
        obs for obs in obs_list
        if doy_start <= obs.date.timetuple().tm_yday <= doy_end
    ]
    if not window_obs:
        return 0

    # Find observation with max raw index
    best = max(window_obs, key=lambda o: flowering_index(o.bands))
    raw_value = flowering_index(best.bands)

    if raw_value < FLOWERING_THRESHOLD:
        return 0

    # It's a peak — is it driven by a single low-quality observation?
    if best.quality.score() < _LOW_QUALITY_THRESHOLD:
        return 1
    return 0


def _weighted_false_peaks(obs_list: list) -> int:
    """Count false peaks using quality-weighted detection.

    A false peak: max quality-weighted index >= FLOWERING_THRESHOLD, AND the
    observation driving that maximum has quality < LOW_QUALITY_THRESHOLD.
    """
    doy_start, doy_end = FLOWERING_WINDOW
    window_obs = [
        obs for obs in obs_list
        if doy_start <= obs.date.timetuple().tm_yday <= doy_end
    ]
    if not window_obs:
        return 0

    weighted_vals = [
        (obs, flowering_index(obs.bands) * obs.quality.score())
        for obs in window_obs
    ]
    best_obs, best_weighted = max(weighted_vals, key=lambda x: x[1])

    if best_weighted < FLOWERING_THRESHOLD:
        return 0

    if best_obs.quality.score() < _LOW_QUALITY_THRESHOLD:
        return 1
    return 0


def test_quality_weighting_reduces_false_peaks(observations, science_points, report_collector):
    """Weighted method produces fewer false peaks than raw max on absence points."""
    labels = science_points.set_index("point_id")["label"].to_dict()

    raw_fps      = 0
    weighted_fps = 0
    n_absence    = 0

    per_point: list[tuple[str, int, int]] = []  # (pid, raw_fp, weighted_fp)

    for pid, obs_list in observations.items():
        if labels.get(pid) != 0:
            continue
        n_absence += 1
        rfp = _raw_false_peaks(obs_list)
        wfp = _weighted_false_peaks(obs_list)
        raw_fps      += rfp
        weighted_fps += wfp
        per_point.append((pid, rfp, wfp))

    if n_absence == 0:
        report_collector.add(
            "Signal 4: Quality weighting false-peak rate",
            "SKIP — no absence points found",
        )
        pytest.skip("No absence points available")

    reduction = raw_fps - weighted_fps

    if raw_fps == 0:
        # Nothing to compare — report and skip assertion
        report_collector.add(
            "Signal 4: Quality weighting false-peak rate",
            f"INFO — no raw false peaks detected on {n_absence} absence points; "
            "quality filter effect cannot be measured",
        )
        pytest.skip("No raw false peaks found — cannot measure weighting effect")

    # One-sided binomial test: weighted_fps < raw_fps under H0 (p_success=0.5 per point)
    # Treat each point where raw_fp=1 as a trial; weighted_fp=0 = "success"
    trials   = raw_fps  # points where raw detected a false peak
    successes = raw_fps - weighted_fps  # points where weighting suppressed it

    p_value: float
    if trials > 0:
        result = stats.binomtest(successes, trials, p=0.5, alternative="greater")
        p_value = result.pvalue
    else:
        p_value = 1.0

    verdict = "PASS" if weighted_fps < raw_fps else "FAIL"

    report_collector.add(
        "Signal 4: Quality weighting false-peak rate",
        f"Hypothesis: quality-weighted detection produces fewer false peaks than raw max\n"
        f"  Absence points tested:   {n_absence}\n"
        f"  Raw false peaks:         {raw_fps}\n"
        f"  Weighted false peaks:    {weighted_fps}\n"
        f"  Reduction:               {reduction}\n"
        f"  Binomial p:              {p_value:.4f}  {verdict}",
    )

    assert weighted_fps < raw_fps, (
        f"Quality weighting did not reduce false peaks: "
        f"raw={raw_fps}, weighted={weighted_fps} on {n_absence} absence points"
    )
