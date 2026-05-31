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

def _kvol_from_trig(
    cos_sza: np.ndarray, sin_sza: np.ndarray,
    cos_vza: np.ndarray, sin_vza: np.ndarray,
    cos_raa: np.ndarray,
) -> np.ndarray:
    """RossThick kernel from precomputed trig values."""
    cos_phase = np.clip(cos_sza * cos_vza + sin_sza * sin_vza * cos_raa, -1.0, 1.0)
    phase = np.arccos(cos_phase)
    return ((np.pi / 2.0 - phase) * cos_phase + np.sin(phase)) / (cos_sza + cos_vza) - np.pi / 4.0


def _kgeo_from_trig(
    cos_sza: np.ndarray, sin_sza: np.ndarray,
    cos_vza: np.ndarray, sin_vza: np.ndarray,
    cos_raa: np.ndarray, sin_raa: np.ndarray,
    tan_sza: np.ndarray, tan_vza: np.ndarray,
) -> np.ndarray:
    """LiSparse-R kernel from precomputed trig values."""
    sec_sza = 1.0 / cos_sza
    sec_vza = 1.0 / cos_vza
    d2 = tan_sza**2 + tan_vza**2 - 2.0 * tan_sza * tan_vza * cos_raa
    cos_t = np.clip(
        2.0 * np.sqrt(np.maximum(d2 + (tan_sza * tan_vza * sin_raa) ** 2, 0.0))
        / (sec_sza + sec_vza),
        -1.0, 1.0,
    )
    t = np.arccos(cos_t)
    overlap = np.maximum((1.0 / np.pi) * (t - np.sin(t) * cos_t) * (sec_sza + sec_vza), 0.0)
    cos_xi = np.clip(cos_sza * cos_vza + sin_sza * sin_vza * cos_raa, -1.0, 1.0)
    return overlap - sec_sza - sec_vza + 0.5 * (1.0 + cos_xi) * (sec_sza + sec_vza - overlap)


def _kvol(sza: np.ndarray, vza: np.ndarray, raa: np.ndarray) -> np.ndarray:
    """RossThick volumetric scattering kernel."""
    return _kvol_from_trig(np.cos(sza), np.sin(sza), np.cos(vza), np.sin(vza), np.cos(raa))


def _kgeo(sza: np.ndarray, vza: np.ndarray, raa: np.ndarray) -> np.ndarray:
    """LiSparse-R geometric scattering kernel."""
    return _kgeo_from_trig(
        np.cos(sza), np.sin(sza), np.cos(vza), np.sin(vza),
        np.cos(raa), np.sin(raa), np.tan(sza), np.tan(vza),
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
    brdf_target = BRDF_TARGET[band] if band else _brdf_scalar(_TARGET_SZA_RAD, 0.0, 0.0, **coef)
    kvol, kgeo  = compute_kernels(sza, vza, raa)
    brdf_obs    = coef["fiso"] + coef["fvol"] * kvol + coef["fgeo"] * kgeo
    safe_denom  = np.where(brdf_obs < 1e-6, 1.0, brdf_obs)
    return np.clip(brdf_target / safe_denom, 0.5, 2.0)


def compute_kernels(
    sza_rad: np.ndarray,
    vza_rad: np.ndarray,
    raa_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (kvol, kgeo) from shared trig — each trig function called once.

    Use when the same (sza, vza, raa) geometry is needed for both kernels,
    e.g. in c_factor_rad() or when building a vectorised per-band correction.
    """
    cos_sza = np.cos(sza_rad); sin_sza = np.sin(sza_rad)
    cos_vza = np.cos(vza_rad); sin_vza = np.sin(vza_rad)
    cos_raa = np.cos(raa_rad); sin_raa = np.sin(raa_rad)
    tan_sza = sin_sza / cos_sza
    tan_vza = sin_vza / cos_vza
    kvol = _kvol_from_trig(cos_sza, sin_sza, cos_vza, sin_vza, cos_raa)
    kgeo = _kgeo_from_trig(cos_sza, sin_sza, cos_vza, sin_vza, cos_raa, sin_raa, tan_sza, tan_vza)
    return kvol, kgeo


def c_factor_from_kernels(
    kvol: np.ndarray,
    kgeo: np.ndarray,
    band: str,
) -> np.ndarray:
    """C-factor from precomputed kernel arrays.

    Pair with compute_kernels() to avoid recomputing trig when both kernels
    share the same (sza, vza, raa) geometry.
    """
    coef = BRDF_COEFFICIENTS[band]
    brdf_obs = coef["fiso"] + coef["fvol"] * kvol + coef["fgeo"] * kgeo
    safe_denom = np.where(brdf_obs < 1e-6, 1.0, brdf_obs)
    return np.clip(BRDF_TARGET[band] / safe_denom, 0.5, 2.0)
