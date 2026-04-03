"""Stage 7: waveform feature extraction from a quality-weighted time series.

This module implements the core scientific claim of the pipeline:

    A quality-weighted spectral time series at a Parkinsonia presence point
    produces a detectable peak within the known flowering window (DOY 200–340),
    and that peak is systematically higher than at absence points.

The waveform primitive is a pure function — no I/O, no global state. It operates
on a list of scored Observations (output of Stage 6) for a single point and
returns a feature dict. Returns {} when there is insufficient usable data, which
the caller (feature assembly) treats as a missing row rather than raising.

Algorithm
---------
For each calendar year in the observation sequence:

1. Filter to observations with quality score >= min_quality (using quality_mask).
   This removes low-quality acquisitions that could suppress or manufacture a peak.

2. Compute index_fn for each usable observation in that year to get a
   quality-weighted time series: value × quality_weight.

3. Find the maximum weighted value within the flowering window (DOY window).
   This is the candidate peak for that year.

4. A year counts as "detected" if its peak_value >= FLOWERING_THRESHOLD.

Across all detected years:
- peak_value      : maximum peak across all detected years
- peak_doy        : DOY of that maximum
- spike_duration  : mean number of consecutive DOYs with value > threshold/2
                    (proxy for spike width, averaged across detected years)
- peak_doy_mean   : mean peak DOY across detected years (phenological consistency)
- peak_doy_sd     : std dev of peak DOY (lower = more consistent timing)
- years_detected  : count of years with detectable peak

Quality weighting
-----------------
Each observation's contribution is scaled by its quality score:

    weighted_value = index_fn(obs.bands) * obs.quality.score(quality_mask)

This means a genuine peak in a high-quality observation is preferred over
a noisy spike in a low-quality one. It does NOT mean low-quality observations
are excluded — they are down-weighted. The min_quality threshold only cuts
observations so poor that they contribute noise rather than signal.

The quality_mask defaults to Q_FULL (all five components), which penalises
geometrically anomalous and spectrally unusual acquisitions. Use Q_GEOMETRIC
when greenness itself is the signal and should not be penalised.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Callable

from analysis.constants import FLOWERING_THRESHOLD, FLOWERING_WINDOW, Q_FULL
from analysis.timeseries.observation import Observation


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_waveform_features(
    observations: list[Observation],
    index_fn: Callable[[dict[str, float]], float],
    window: tuple[int, int] = FLOWERING_WINDOW,
    quality_mask: set[str] | None = Q_FULL,
    min_quality: float = 0.3,
    min_years: int = 3,
) -> dict[str, float]:
    """Extract waveform features from a quality-weighted time series.

    Parameters
    ----------
    observations:
        All scored Observations for a single point (output of Stage 6).
        May span multiple years. Order does not matter — grouped internally
        by calendar year.
    index_fn:
        Spectral index function: bands dict → float. Typically flowering_index
        from analysis.primitives.indices. Passed in rather than imported to
        keep the waveform primitive decoupled from a specific index.
    window:
        (doy_start, doy_end) — inclusive DOY range for peak search.
        Defaults to FLOWERING_WINDOW (200, 340).
    quality_mask:
        Quality profile passed to obs.quality.score(). Defaults to Q_FULL
        (all five components). Pass Q_GEOMETRIC when computing NDVI anomaly
        where greenness must not be penalised.
    min_quality:
        Minimum quality score to include an observation in the time series.
        Observations below this are too noisy to contribute meaningful signal.
        Default 0.3 (empirically chosen; preserves ~85% of clear observations).
    min_years:
        Minimum number of calendar years with usable data to return features.
        Returns {} if fewer years are available — prevents the RF from training
        on near-empty time series that happen to have a single-year spike.

    Returns
    -------
    dict[str, float]
        Feature dict with keys: peak_value, peak_doy, spike_duration,
        peak_doy_mean, peak_doy_sd, years_detected.
        Returns {} if fewer than min_years of usable data exist, or if the
        all-years candidate set is empty (e.g. all acquisitions are clouded).
    """
    doy_start, doy_end = window

    # --- Group observations by calendar year --------------------------------
    by_year: dict[int, list[Observation]] = defaultdict(list)
    for obs in observations:
        by_year[obs.date.year].append(obs)

    if len(by_year) < min_years:
        return {}

    # --- Per-year peak detection -------------------------------------------
    peak_values: list[float] = []
    peak_doys: list[int] = []
    durations: list[float] = []

    for year, year_obs in sorted(by_year.items()):

        # Filter to usable observations within the flowering window
        usable = [
            obs for obs in year_obs
            if obs.quality.score(quality_mask) >= min_quality
            and doy_start <= obs.date.timetuple().tm_yday <= doy_end
        ]

        if not usable:
            continue

        # Compute quality-weighted index values
        weighted = [
            (obs, index_fn(obs.bands) * obs.quality.score(quality_mask))
            for obs in usable
        ]

        # Find the peak within this year's window
        best_obs, best_value = max(weighted, key=lambda x: x[1])
        best_doy = best_obs.date.timetuple().tm_yday

        if best_value < FLOWERING_THRESHOLD:
            # No detectable peak this year — year does not count
            continue

        peak_values.append(best_value)
        peak_doys.append(best_doy)

        # Spike duration: count observations (within the window) with
        # weighted value >= threshold / 2 (half-power width proxy)
        half_threshold = FLOWERING_THRESHOLD / 2.0
        above_half = sum(1 for _, wv in weighted if wv >= half_threshold)
        durations.append(float(above_half))

    years_detected = len(peak_values)

    if years_detected < min_years:
        return {}

    # --- Aggregate across detected years ------------------------------------
    # peak_value / peak_doy: the best single-year peak
    best_idx = peak_values.index(max(peak_values))
    peak_value = peak_values[best_idx]
    peak_doy = peak_doys[best_idx]

    spike_duration = statistics.mean(durations)
    peak_doy_mean = statistics.mean(peak_doys)
    peak_doy_sd = statistics.pstdev(peak_doys) if len(peak_doys) > 1 else 0.0

    return {
        "peak_value": peak_value,
        "peak_doy": float(peak_doy),
        "spike_duration": spike_duration,
        "peak_doy_mean": peak_doy_mean,
        "peak_doy_sd": peak_doy_sd,
        "years_detected": float(years_detected),
    }
