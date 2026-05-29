"""utils/nbar.py — Roy et al. 2016 BRDF c-factor correction for Sentinel-2.

Implements the RossThick-LiSparse-R kernel model using the S2 coefficients
from Roy et al. 2016 (Table 1).  All kernel functions are vectorised over
numpy arrays of shape (N,).

Reference
---------
Roy, D.P. et al. (2016). "Characterization of Landsat-7 to Landsat-8
reflective wavelength and normalized difference vegetation index continuity."
Remote Sensing of Environment, 185, 57-70.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# BRDF coefficients (Roy et al. 2016, Table 1 — Sentinel-2 equivalent bands)
# ---------------------------------------------------------------------------

BRDF_COEFFICIENTS: dict[str, dict[str, float]] = {
    "B02": {"fiso": 0.0774, "fgeo": 0.0079, "fvol": 0.0372},
    "B03": {"fiso": 0.1306, "fgeo": 0.0178, "fvol": 0.0580},
    "B04": {"fiso": 0.1690, "fgeo": 0.0227, "fvol": 0.0574},
    "B05": {"fiso": 0.2085, "fgeo": 0.0256, "fvol": 0.0845},
    "B06": {"fiso": 0.2316, "fgeo": 0.0273, "fvol": 0.1003},
    "B07": {"fiso": 0.2599, "fgeo": 0.0294, "fvol": 0.1197},
    "B08": {"fiso": 0.3093, "fgeo": 0.0330, "fvol": 0.1535},
    "B8A": {"fiso": 0.3430, "fgeo": 0.0453, "fvol": 0.1154},  # proxy from B11
    "B11": {"fiso": 0.3430, "fgeo": 0.0453, "fvol": 0.1154},
    "B12": {"fiso": 0.2658, "fgeo": 0.0387, "fvol": 0.0639},
}

# Nadir target geometry (Roy 2016 standard)
TARGET_SZA_DEG = 45.0
TARGET_VZA_DEG = 0.0


# ---------------------------------------------------------------------------
# Kernel functions (inputs in radians, vectorised over N pixels)
# ---------------------------------------------------------------------------

def _kvol(sza: np.ndarray, vza: np.ndarray, raa: np.ndarray) -> np.ndarray:
    """RossThick volumetric scattering kernel."""
    cos_phase = (
        np.cos(sza) * np.cos(vza)
        + np.sin(sza) * np.sin(vza) * np.cos(raa)
    )
    # Clamp to [-1, 1] to guard against floating-point noise
    cos_phase = np.clip(cos_phase, -1.0, 1.0)
    phase = np.arccos(cos_phase)
    return ((np.pi / 2.0 - phase) * np.cos(phase) + np.sin(phase)) / (
        np.cos(sza) + np.cos(vza)
    ) - np.pi / 4.0


def _kgeo(sza: np.ndarray, vza: np.ndarray, raa: np.ndarray) -> np.ndarray:
    """LiSparse-R geometric scattering kernel."""
    # Crown relative height/width ratio b/r = 1, h/b = 2 (standard LiSparse)
    tan_sza = np.tan(sza)
    tan_vza = np.tan(vza)
    cos_raa = np.cos(raa)

    # Overlap (shadow) calculation
    d2 = tan_sza**2 + tan_vza**2 - 2.0 * tan_sza * tan_vza * cos_raa
    d = np.sqrt(np.maximum(d2, 0.0))

    cos_t_num = 2.0 * np.sqrt(d2 + (tan_sza * tan_vza * np.sin(raa)) ** 2)
    cos_t = np.clip(cos_t_num / (1.0 / np.cos(sza) + 1.0 / np.cos(vza)), -1.0, 1.0)
    t = np.arccos(cos_t)
    overlap = (1.0 / np.pi) * (t - np.sin(t) * np.cos(t)) * (
        1.0 / np.cos(sza) + 1.0 / np.cos(vza)
    )
    overlap = np.maximum(overlap, 0.0)

    cos_xi = np.cos(sza) * np.cos(vza) + np.sin(sza) * np.sin(vza) * cos_raa
    cos_xi = np.clip(cos_xi, -1.0, 1.0)

    return overlap - 1.0 / np.cos(sza) - 1.0 / np.cos(vza) + 0.5 * (1.0 + cos_xi) * (
        1.0 / np.cos(sza) + 1.0 / np.cos(vza) - overlap
    )


def _brdf(
    sza: np.ndarray,
    vza: np.ndarray,
    raa: np.ndarray,
    fiso: float,
    fvol: float,
    fgeo: float,
) -> np.ndarray:
    return fiso + fvol * _kvol(sza, vza, raa) + fgeo * _kgeo(sza, vza, raa)


# BRDF(target geometry) is a scalar per band — precomputed once at module load.
# Target is sza=45°, vza=0, raa=0 for every pixel and every scene.
def _brdf_scalar(sza_rad: float, vza_rad: float, raa_rad: float,
                 fiso: float, fvol: float, fgeo: float) -> float:
    s = np.array([sza_rad])
    v = np.array([vza_rad])
    r = np.array([raa_rad])
    return float(_brdf(s, v, r, fiso, fvol, fgeo)[0])

_TARGET_SZA_RAD = np.deg2rad(TARGET_SZA_DEG)
BRDF_TARGET: dict[str, float] = {
    band: _brdf_scalar(_TARGET_SZA_RAD, 0.0, 0.0, **coef)
    for band, coef in BRDF_COEFFICIENTS.items()
}


# ---------------------------------------------------------------------------
# C-factor
# ---------------------------------------------------------------------------

def c_factor(
    sza_deg: np.ndarray,
    vza_deg: np.ndarray,
    raa_deg: np.ndarray,
    band: str,
) -> np.ndarray:
    """Per-pixel BRDF c-factor = BRDF(target geometry) / BRDF(observed geometry).

    Parameters
    ----------
    sza_deg, vza_deg, raa_deg : shape (N,) arrays in degrees
        Solar zenith, view zenith, relative azimuth (saa - vaa).
    band : S2 band name, e.g. "B05"

    Returns
    -------
    np.ndarray shape (N,), clamped to [0.5, 2.0].
    """
    coef = BRDF_COEFFICIENTS[band]

    sza = np.deg2rad(sza_deg)
    vza = np.deg2rad(vza_deg)
    raa = np.deg2rad(raa_deg)

    return _c_factor_rad(sza, vza, raa, coef, band=band)


def c_factor_rad(
    sza_rad: np.ndarray,
    vza_rad: np.ndarray,
    raa_rad: np.ndarray,
    band: str,
) -> np.ndarray:
    """Like c_factor() but accepts inputs already in radians.

    Use this when processing multiple bands with shared angle arrays to avoid
    redundant deg2rad conversions.
    """
    return _c_factor_rad(sza_rad, vza_rad, raa_rad, BRDF_COEFFICIENTS[band], band=band)


def _c_factor_rad(
    sza: np.ndarray,
    vza: np.ndarray,
    raa: np.ndarray,
    coef: dict,
    band: str = "",
) -> np.ndarray:
    # Target BRDF is a precomputed scalar — no N-element array needed.
    brdf_target = BRDF_TARGET[band] if band else _brdf_scalar(_TARGET_SZA_RAD, 0.0, 0.0, **coef)
    brdf_obs    = _brdf(sza, vza, raa, **coef)

    safe_denom = np.where(brdf_obs < 1e-6, 1.0, brdf_obs)
    cf = brdf_target / safe_denom

    return np.clip(cf, 0.5, 2.0)
