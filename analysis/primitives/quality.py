"""Quality scoring primitive: per-observation greenness z-score component.

Stage 5 (extraction) populates four of the five ObservationQuality components
from chip data: scl_purity, aot, view_zenith, sun_zenith. The fifth component,
greenness_z, requires an archive-wide reference distribution and so cannot be
computed during extraction.

This module provides:

    ArchiveStats
        Archive-wide statistics for the greenness index (NDVI), computed once
        before the parallel extraction loop and passed to every worker.

    score_observation(obs, archive_stats) -> Observation
        Pure function. Takes an Observation with greenness_z=1.0 (the Stage 5
        placeholder) and returns a new Observation with greenness_z replaced by
        the inverse greenness z-score derived from ArchiveStats.

        Returns a new object — does not mutate the input.

Greenness index
---------------
NDVI = (B08 - B04) / (B08 + B04), clamped to [-1, 1].

This is the simplest, most stable vegetative index. The flowering_index
(Session 7) may use a richer expression, but greenness_z specifically measures
whether the scene's overall greenness is anomalous relative to the archive.
NDVI is the appropriate index for that purpose.

greenness_z component
---------------------
    raw_z  = (ndvi - archive_stats.mean) / archive_stats.std
    score  = 1 / (1 + |raw_z|)     clamped to [0, 1]

score = 1.0 when ndvi == archive_mean (not anomalous).
score → 0   when ndvi is far from the mean (spectrally anomalous).

The logistic-style transform avoids hard cutoffs and keeps the component
continuous and differentiable, which is appropriate for a multiplicative
quality weight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Sequence

from analysis.timeseries.observation import Observation, ObservationQuality


# ---------------------------------------------------------------------------
# ArchiveStats
# ---------------------------------------------------------------------------

@dataclass
class ArchiveStats:
    """Archive-wide statistics for the greenness index (NDVI).

    Parameters
    ----------
    mean:
        Mean NDVI across all usable observations in the archive.
    std:
        Standard deviation of NDVI across the same population.
        Must be > 0. If the archive is degenerate (all observations have
        identical NDVI), std is floored to a small positive value so
        score_observation does not divide by zero.
    """

    mean: float
    std: float

    _STD_FLOOR: float = 1e-6  # class-level constant, not a field

    def __post_init__(self) -> None:
        if self.std <= 0.0:
            self.std = self._STD_FLOOR

    @classmethod
    def from_observations(cls, observations: Sequence[Observation]) -> "ArchiveStats":
        """Compute ArchiveStats from a sequence of Observations.

        Only observations that contain both B08 and B04 contribute.
        Raises ValueError if fewer than 2 usable observations are found.
        """
        ndvi_values = [
            _ndvi(obs.bands)
            for obs in observations
            if "B08" in obs.bands and "B04" in obs.bands
        ]
        if len(ndvi_values) < 2:
            raise ValueError(
                f"Cannot compute ArchiveStats: need at least 2 observations with "
                f"B08 and B04 bands, got {len(ndvi_values)}."
            )
        return cls(mean=mean(ndvi_values), std=stdev(ndvi_values))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ndvi(bands: dict[str, float]) -> float:
    """Compute NDVI = (B08 - B04) / (B08 + B04), clamped to [-1, 1].

    Returns 0.0 if denominator is zero (avoids division by zero at very dark
    or saturated pixels).
    """
    b08 = bands.get("B08", 0.0)
    b04 = bands.get("B04", 0.0)
    denom = b08 + b04
    if denom == 0.0:
        return 0.0
    return max(-1.0, min(1.0, (b08 - b04) / denom))


def _greenness_score(ndvi: float, archive_mean: float, archive_std: float) -> float:
    """Convert an NDVI value to a greenness_z quality component in [0, 1].

    score = 1 / (1 + |z|)   where z = (ndvi - mean) / std

    score == 1.0 at z==0 (no anomaly).
    score → 0   as |z| → ∞ (strong anomaly in either direction).
    """
    z = (ndvi - archive_mean) / archive_std
    return 1.0 / (1.0 + abs(z))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_observation(obs: Observation, archive_stats: ArchiveStats) -> Observation:
    """Return a new Observation with greenness_z populated from ArchiveStats.

    All other quality components (scl_purity, aot, view_zenith, sun_zenith)
    are carried over unchanged from the input Observation. This function only
    replaces the greenness_z placeholder (1.0) set by Stage 5 extraction.

    Parameters
    ----------
    obs:
        An Observation as returned by extract_observations(). Its
        quality.greenness_z is expected to be 1.0 (the Stage 5 sentinel).
    archive_stats:
        Archive-wide NDVI statistics. Typically loaded once per run from
        a pre-computed ArchiveStats and passed to each worker.

    Returns
    -------
    Observation
        A new Observation with a new ObservationQuality where greenness_z
        has been replaced. The input is not mutated.

    Notes
    -----
    If the observation lacks B08 or B04, NDVI defaults to 0.0, which maps
    to a greenness_z of 1/(1 + |z|) where z = (0.0 - mean)/std. This is a
    mild penalty when the archive mean is non-zero, which is appropriate —
    a missing band is a data-quality issue, not a spectral-normality signal.
    """
    ndvi = _ndvi(obs.bands)
    gz = _greenness_score(ndvi, archive_stats.mean, archive_stats.std)

    new_quality = ObservationQuality(
        scl_purity=obs.quality.scl_purity,
        aot=obs.quality.aot,
        view_zenith=obs.quality.view_zenith,
        sun_zenith=obs.quality.sun_zenith,
        greenness_z=gz,
    )
    # Construct a new Observation, sharing the same bands and meta dicts
    # (dicts are not deep-copied — callers should treat observations as immutable).
    return Observation(
        point_id=obs.point_id,
        date=obs.date,
        bands=obs.bands,
        quality=new_quality,
        meta=obs.meta,
    )
