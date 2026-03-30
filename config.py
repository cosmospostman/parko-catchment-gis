"""config.py — central configuration for the Parkinsonia GIS pipeline.

Imported by every script. Fails loudly at import time if required env vars are missing.
All scientific parameters are defined as module-level constants (not from env).
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Path constants — sourced from environment (set by config.sh)
# ---------------------------------------------------------------------------

def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        raise KeyError(
            f"Required environment variable '{name}' is not set. "
            f"Did you forget to run 'source config.sh' before this script?"
        )
    return value


BASE_DIR = Path(_require_env("BASE_DIR"))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", str(BASE_DIR / "cache")))
WORKING_DIR = Path(os.environ.get("WORKING_DIR", str(BASE_DIR / "working")))
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", str(BASE_DIR / "outputs")))
LOG_DIR = Path(os.environ.get("LOG_DIR", str(BASE_DIR / "logs")))
CATCHMENT_GEOJSON = Path(os.environ.get("CATCHMENT_GEOJSON", str(BASE_DIR / "mitchell_catchment.geojson")))
CODE_DIR = Path(_require_env("CODE_DIR"))

# ---------------------------------------------------------------------------
# Runtime constants — sourced from environment
# ---------------------------------------------------------------------------

YEAR: int = int(_require_env("YEAR"))
COMPOSITE_START: str = os.environ.get("COMPOSITE_START", "05-01")
COMPOSITE_END: str = os.environ.get("COMPOSITE_END", "10-31")

# ---------------------------------------------------------------------------
# Analysis constants
# ---------------------------------------------------------------------------

TARGET_CRS: str = "EPSG:7855"
TARGET_RESOLUTION: int = 10
CLOUD_COVER_MAX: int = 20
COMPOSITE_BANDS: list = ["blue", "green", "red", "nir", "nir08", "swir16", "swir22"]

FLOWERING_WINDOW_START: str = "08-01"
FLOWERING_WINDOW_END: str = "10-31"
FLOOD_SEASON_START: str = "01-01"
FLOOD_SEASON_END: str = "05-31"

BASELINE_START_YEAR: int = 1986
PROBABILITY_THRESHOLD: float = 0.6
MIN_PATCH_AREA_HA: float = 0.25
RF_N_ESTIMATORS: int = 500
SPATIAL_BLOCK_SIZE_KM: int = 50
NAN_FRACTION_MAX: float = 0.20

CATCHMENT_MEDIAN_NDVI_MIN: float = 0.15
CATCHMENT_MEDIAN_NDVI_MAX: float = 0.50

NDVI_ANOMALY_MIN_MEAN: float = -0.05
NDVI_ANOMALY_MAX_MEAN: float = 0.05
NDVI_ANOMALY_MIN_STD: float = 0.03
NDVI_ANOMALY_MAX_STD: float = 0.20

FLOWERING_ANOMALY_CORRELATION_MAX: float = 0.70

TARGET_OVERALL_ACCURACY: float = 0.85
TARGET_RECALL: float = 0.80
CHANGE_DETECTION_MEAN_TOLERANCE: float = 0.05

# ---------------------------------------------------------------------------
# Data source constants
# ---------------------------------------------------------------------------

STAC_ENDPOINT_ELEMENT84: str = "https://earth-search.aws.element84.com/v1"
STAC_ENDPOINT_CDSE: str = "https://catalogue.dataspace.copernicus.eu/stac"
DEA_S3_BUCKET: str = "dea-public-data"
ALA_API_BASE: str = "https://api.ala.org.au"

S2_COLLECTION: str = "sentinel-2-l2a"
S1_COLLECTION: str = "sentinel-1-grd"
DEA_COLLECTION: str = "ga_ls_ard_3"
FC_COLLECTION: str = "ga_ls_fc_3"

ALA_SPECIES_QUERY: str = "Parkinsonia aculeata"

# ---------------------------------------------------------------------------
# Output filename template functions
# ---------------------------------------------------------------------------


def ndvi_median_path(year: int) -> Path:
    """Return the path for the NDVI median composite raster for the given year."""
    return OUTPUTS_DIR / str(year) / f"ndvi_median_{year}.tif"


def ndvi_anomaly_path(year: int) -> Path:
    """Return the path for the NDVI anomaly raster for the given year."""
    return OUTPUTS_DIR / str(year) / f"ndvi_anomaly_{year}.tif"


def flowering_index_path(year: int) -> Path:
    """Return the path for the flowering index raster for the given year."""
    return OUTPUTS_DIR / str(year) / f"flowering_index_{year}.tif"


def flood_extent_path(year: int) -> Path:
    """Return the path for the flood extent vector file for the given year."""
    return OUTPUTS_DIR / str(year) / f"flood_extent_{year}.gpkg"


def probability_raster_path(year: int) -> Path:
    """Return the path for the Parkinsonia probability raster for the given year."""
    return OUTPUTS_DIR / str(year) / f"probability_raster_{year}.tif"


def priority_patches_path(year: int) -> Path:
    """Return the path for the priority patches vector file for the given year."""
    return OUTPUTS_DIR / str(year) / f"priority_patches_{year}.gpkg"


def change_detection_path(year: int) -> Path:
    """Return the path for the change detection raster for the given year."""
    return OUTPUTS_DIR / str(year) / f"change_detection_{year}.tif"


def verification_report_path(year: int) -> Path:
    """Return the path for the verification report JSON for the given year."""
    return OUTPUTS_DIR / str(year) / f"verification_report_{year}.json"


def ndvi_baseline_path() -> Path:
    """Return the path for the cached NDVI baseline median raster."""
    return CACHE_DIR / "ndvi_baseline_median.tif"
