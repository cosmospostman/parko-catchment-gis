"""Shared constants for the spectral time series pipeline.

All band names, quality profiles, SCL masks, and scientific thresholds
are defined here and imported by every other module. Edit once, applies everywhere.
"""

# ---------------------------------------------------------------------------
# Sentinel-2 band names used throughout the pipeline
# ---------------------------------------------------------------------------

BANDS: list[str] = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
SPECTRAL_INDEX_COLS: list[str] = ["NDVI", "NDWI", "EVI"]

SCL_BAND = "SCL"
AOT_BAND = "AOT"
VZA_BAND = "VZA"  # view zenith angle
SZA_BAND = "SZA"  # sun zenith angle

# ---------------------------------------------------------------------------
# SCL (Scene Classification Layer) clear values
# SCL class codes that indicate usable (non-cloud, non-shadow) pixels:
#   4 = vegetation, 5 = bare soil, 6 = water, 7 = unclassified,
#   11 = snow/ice (retained — not cloud)
# ---------------------------------------------------------------------------

SCL_CLEAR_VALUES: set[int] = {4, 5, 6, 7, 11}

# ---------------------------------------------------------------------------
# Quality profile masks — passed to ObservationQuality.score(mask=...)
#
# Q_ATMOSPHERIC  : cloud + haze only; use for any spectral index computation
# Q_GEOMETRIC    : cloud + haze + geometry; use for anomaly where greenness IS signal
# Q_FULL         : all five components; use for flowering peak detection
# Q_CLOUD_ONLY   : SCL purity only; use for structural features (HAND, texture)
# ---------------------------------------------------------------------------

Q_ATMOSPHERIC: set[str] = {"scl_purity", "aot"}
Q_GEOMETRIC: set[str] = {"scl_purity", "aot", "view_zenith", "sun_zenith"}
Q_FULL: None = None  # all five components
Q_CLOUD_ONLY: set[str] = {"scl_purity"}

# ---------------------------------------------------------------------------
# Spatial validation gate
# Minimum AUC on held-out spatial region required to certify a model for inference.
# train.py refuses to write model_{run_id}.pkl if validation falls below this.
# ---------------------------------------------------------------------------

SPATIAL_VALIDATION_THRESHOLD: float = 0.85

# ---------------------------------------------------------------------------
# Flowering phenology window and detection threshold
#
# FLOWERING_WINDOW : (doy_start, doy_end) — expected day-of-year range for
#                    Parkinsonia aculeata peak greenness in north Queensland.
#                    200 = ~19 Jul, 340 = ~6 Dec.
# FLOWERING_THRESHOLD : minimum flowering index value to count as a detected peak.
# ---------------------------------------------------------------------------

FLOWERING_WINDOW: tuple[int, int] = (200, 340)
FLOWERING_THRESHOLD: float = 0.15


# ---------------------------------------------------------------------------
# Spectral index computation
# ---------------------------------------------------------------------------

def add_spectral_indices(df: "pd.DataFrame") -> "pd.DataFrame":
    """Return df with NDVI, NDWI, EVI columns appended, derived from raw bands."""
    import numpy as np
    b02 = df["B02"].values.astype("float32")
    b03 = df["B03"].values.astype("float32")
    b04 = df["B04"].values.astype("float32")
    b08 = df["B08"].values.astype("float32")
    df = df.copy()
    denom = b08 + b04
    df["NDVI"] = np.divide(b08 - b04, denom, out=np.zeros_like(denom), where=denom != 0)
    denom = b03 + b08
    df["NDWI"] = np.divide(b03 - b08, denom, out=np.zeros_like(denom), where=denom != 0)
    evi_denom = b08 + 6 * b04 - 7.5 * b02 + 1
    df["EVI"]  = np.divide(2.5 * (b08 - b04), evi_denom, out=np.zeros_like(evi_denom), where=evi_denom != 0)
    return df
