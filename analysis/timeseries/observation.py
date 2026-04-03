"""Atomic unit of the spectral time series pipeline.

An Observation is a single satellite acquisition at a single point.
ObservationQuality holds per-component quality scores and computes
weighted scalars on demand via score(mask).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ObservationQuality:
    """Per-component quality scores for one observation.

    Each component is clamped to [0, 1] on assignment via __post_init__.

    Components
    ----------
    scl_purity  : fraction of clear pixels in the local chip window (SCL mask)
    aot         : inverse aerosol optical thickness  (1 = clean air)
    view_zenith : inverse view zenith angle          (1 = nadir)
    sun_zenith  : inverse sun zenith angle           (1 = high sun)
    greenness_z : inverse scene greenness z-score    (1 = spectrally normal)
    """

    scl_purity: float
    aot: float
    view_zenith: float
    sun_zenith: float
    greenness_z: float

    def __post_init__(self) -> None:
        self.scl_purity = float(max(0.0, min(1.0, self.scl_purity)))
        self.aot = float(max(0.0, min(1.0, self.aot)))
        self.view_zenith = float(max(0.0, min(1.0, self.view_zenith)))
        self.sun_zenith = float(max(0.0, min(1.0, self.sun_zenith)))
        self.greenness_z = float(max(0.0, min(1.0, self.greenness_z)))

    def score(self, mask: set[str] | None = None) -> float:
        """Return the product of the selected quality components.

        Parameters
        ----------
        mask:
            Set of component names to include in the product.
            If None (i.e. Q_FULL), all five components are used.

        Examples
        --------
        obs.quality.score()                          # all components
        obs.quality.score({"scl_purity", "aot"})     # Q_ATMOSPHERIC
        obs.quality.score({"scl_purity"})            # Q_CLOUD_ONLY
        """
        components = {
            "scl_purity": self.scl_purity,
            "aot": self.aot,
            "view_zenith": self.view_zenith,
            "sun_zenith": self.sun_zenith,
            "greenness_z": self.greenness_z,
        }
        active = {k: v for k, v in components.items() if mask is None or k in mask}
        result = 1.0
        for v in active.values():
            result *= v
        return result


@dataclass
class Observation:
    """A single satellite acquisition at a single geographic point.

    This is the atomic unit consumed by every layer of the training pipeline.
    It carries no spatial geometry — the point_id is the spatial index.

    Attributes
    ----------
    point_id : ALA record ID or synthetic absence ID.
    date     : Acquisition datetime (UTC).
    bands    : Mapping of band name → surface reflectance value,
               e.g. {"B03": 0.043, "B04": 0.021, ...}.
    quality  : Per-component quality scores for this acquisition.
    meta     : Ancillary metadata: view_zenith (degrees), sun_zenith (degrees),
               tile_id (str), and any other acquisition-level fields.
    """

    point_id: str
    date: datetime
    bands: dict[str, float]
    quality: ObservationQuality
    meta: dict = field(default_factory=dict)
