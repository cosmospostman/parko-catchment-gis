"""Stage 8: feature vector assembly from waveform and structural inputs.

This module assembles the final RF feature vector for a single training point.
It is a pure function — no I/O, no global state.

Feature vector layout
---------------------
The assembled dict contains exactly these keys, in this order:

    Waveform features (from extract_waveform_features):
        peak_value, peak_doy, spike_duration,
        peak_doy_mean, peak_doy_sd, years_detected

    Structural features (caller-supplied):
        HAND, dist_to_water

    Quality summary:
        mean_quality

The ordering is stable because Python dicts preserve insertion order (3.7+)
and the caller passes structural_features as a dict. The training orchestrator
records the key order to feature_names_{run_id}.json so inference can
reconstruct the same order without hard-coding it here.

mean_quality
------------
The mean of quality scores (Q_FULL, i.e. mask=None) across all observations
passed in — not just the usable subset used for waveform extraction. This
captures the overall observing conditions for the point, which the RF can
use to learn that noisier inputs produce less reliable signals.

If observations is empty, mean_quality defaults to 0.0.

Structural features
-------------------
HAND (Height Above Nearest Drainage) and dist_to_water are GIS-derived
per-point values. They are passed in rather than looked up internally so
that this primitive remains pure and testable without GIS I/O.

Row-count contract
------------------
assemble_feature_vector does not drop rows. It always returns a dict.
If waveform_features is empty (i.e. extract_waveform_features returned {}),
the caller must decide whether to skip the point — the contract here is
that we never silently discard a row. The caller should check for {} from
extract_waveform_features before calling assemble_feature_vector, but if
it is called with empty waveform features it raises ValueError rather than
quietly returning a partial vector.
"""

from __future__ import annotations

import math
import statistics
from typing import Sequence

from analysis.constants import Q_FULL
from analysis.timeseries.observation import Observation

# ---------------------------------------------------------------------------
# Required waveform feature keys (defines expected input contract)
# ---------------------------------------------------------------------------

WAVEFORM_KEYS: tuple[str, ...] = (
    "peak_value",
    "peak_doy",
    "spike_duration",
    "peak_doy_mean",
    "peak_doy_sd",
    "years_detected",
)

# Required structural feature keys
STRUCTURAL_KEYS: tuple[str, ...] = ("HAND", "dist_to_water")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assemble_feature_vector(
    waveform_features: dict[str, float],
    structural_features: dict[str, float],
    observations: Sequence[Observation],
    quality_mask: set[str] | None = Q_FULL,
) -> dict[str, float]:
    """Assemble the RF feature vector for a single point.

    Parameters
    ----------
    waveform_features:
        Output of extract_waveform_features(). Must contain all keys in
        WAVEFORM_KEYS. Raises ValueError if empty or missing keys.
    structural_features:
        GIS-derived per-point values. Must contain at least the keys in
        STRUCTURAL_KEYS. Extra keys are included in the output as-is,
        preserving the order they appear in structural_features.
    observations:
        All scored Observations for this point (output of Stage 6).
        Used to compute mean_quality. May be empty — mean_quality will be 0.0.
    quality_mask:
        Quality profile passed to obs.quality.score(). Defaults to Q_FULL
        (all five components), consistent with the waveform primitive.

    Returns
    -------
    dict[str, float]
        Flat feature dict with all waveform keys, all structural keys,
        and mean_quality. Key order is:
            waveform keys (WAVEFORM_KEYS order)
            → structural keys (structural_features insertion order)
            → mean_quality

    Raises
    ------
    ValueError
        If waveform_features is empty (caller should check for {} from
        extract_waveform_features before calling this function).
    ValueError
        If waveform_features is missing any required key from WAVEFORM_KEYS.
    ValueError
        If structural_features is missing any required key from STRUCTURAL_KEYS.
    """
    if not waveform_features:
        raise ValueError(
            "assemble_feature_vector called with empty waveform_features. "
            "Check extract_waveform_features returned a non-empty dict before "
            "calling assemble_feature_vector."
        )

    missing_waveform = [k for k in WAVEFORM_KEYS if k not in waveform_features]
    if missing_waveform:
        raise ValueError(
            f"waveform_features is missing required keys: {missing_waveform}. "
            f"Got keys: {list(waveform_features.keys())}"
        )

    missing_structural = [k for k in STRUCTURAL_KEYS if k not in structural_features]
    if missing_structural:
        raise ValueError(
            f"structural_features is missing required keys: {missing_structural}. "
            f"Got keys: {list(structural_features.keys())}"
        )

    # Compute mean quality across all observations
    if observations:
        scores = [obs.quality.score(quality_mask) for obs in observations]
        mean_quality = statistics.mean(scores)
    else:
        mean_quality = 0.0

    # Assemble in stable order: waveform → structural → mean_quality
    result: dict[str, float] = {}
    for key in WAVEFORM_KEYS:
        result[key] = float(waveform_features[key])
    for key, value in structural_features.items():
        result[key] = float(value)
    result["mean_quality"] = mean_quality

    return result
