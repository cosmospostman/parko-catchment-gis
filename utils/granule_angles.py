"""utils/granule_angles.py — fetch and interpolate S2 granule angle grids.

Fetches the granule_metadata.xml asset for a pystac Item, parses the 23×23
Sun/View angle grids (5 km spacing), and interpolates them to arbitrary pixel
coordinates.

The XML is fetched once per item and cached in memory via lru_cache.
"""

from __future__ import annotations

import logging
import threading
import time
import warnings

# nanmean over all-NaN grid cells (detectors with no coverage) is expected;
# suppress the warning globally for this module — catch_warnings is not
# thread-safe so a per-call filter would race with concurrent workers.
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning,
                        module=__name__)
import xml.etree.ElementTree as ET
from functools import lru_cache

import numpy as np
import requests
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)

# S2 granule angle grid: 23×23 nodes at 5000 m spacing
_GRID_STEP_M = 5000.0
_GRID_SIZE   = 23


_xml_cache: dict[str, dict | None] = {}
_xml_cache_lock = threading.Lock()


def _fetch_and_parse_xml(item_id: str, xml_url: str) -> dict | None:
    """Fetch and parse granule_metadata.xml; cached by item_id.

    Returns a dict:
        {
          "sun_zen":  np.ndarray shape (23, 23),
          "sun_az":   np.ndarray shape (23, 23),
          "view_zen": dict[bandId -> np.ndarray (23,23)],   # mean across detectors
          "view_az":  dict[bandId -> np.ndarray (23,23)],
          "ul_utm":   (easting, northing)  — upper-left corner of angle grid in UTM
        }
    or None on failure.
    """
    with _xml_cache_lock:
        if item_id in _xml_cache:
            return _xml_cache[item_id]

    root = None
    last_exc = None
    for attempt in range(4):
        try:
            with requests.get(xml_url, timeout=60, stream=True) as resp:
                resp.raise_for_status()
                content = resp.raw.read(decode_content=True)
                expected = resp.headers.get("Content-Length")
                if expected and len(content) != int(expected):
                    raise IOError(
                        f"Truncated response: got {len(content)} bytes, "
                        f"expected {expected}"
                    )
            root = ET.fromstring(content)
            break
        except (requests.RequestException, IOError) as exc:
            last_exc = exc
        except ET.ParseError as exc:
            last_exc = exc
        time.sleep(2 ** attempt)

    if root is None:
        logger.warning("granule_angles: failed to fetch/parse %s after retries: %s", item_id, last_exc)
        with _xml_cache_lock:
            _xml_cache[item_id] = None
        return None

    def _parse_grid(values_list_elem) -> np.ndarray:
        rows = []
        for row_elem in values_list_elem.findall("VALUES"):
            rows.append([float(v) for v in row_elem.text.split()])
        return np.array(rows, dtype=np.float64)

    # --- Solar angles --------------------------------------------------------
    sun_grid_elem = root.find(".//Sun_Angles_Grid")
    if sun_grid_elem is None:
        logger.warning("granule_angles: no Sun_Angles_Grid in %s", item_id)
        with _xml_cache_lock:
            _xml_cache[item_id] = None
        return None

    sun_zen = _parse_grid(sun_grid_elem.find("Zenith/Values_List"))
    sun_az  = _parse_grid(sun_grid_elem.find("Azimuth/Values_List"))

    # --- Viewing angles (per band, per detector — average across detectors) --
    view_zen_accum: dict[str, list[np.ndarray]] = {}
    view_az_accum:  dict[str, list[np.ndarray]] = {}

    for vag in root.findall(".//Viewing_Incidence_Angles_Grids"):
        band_id = vag.get("bandId")
        zen_elem = vag.find("Zenith/Values_List")
        az_elem  = vag.find("Azimuth/Values_List")
        if zen_elem is None or az_elem is None:
            continue
        view_zen_accum.setdefault(band_id, []).append(_parse_grid(zen_elem))
        view_az_accum.setdefault(band_id, []).append(_parse_grid(az_elem))

    view_zen: dict[str, np.ndarray] = {}
    view_az:  dict[str, np.ndarray] = {}
    for band_id in view_zen_accum:
        stacked_z = np.stack(view_zen_accum[band_id], axis=0)
        stacked_a = np.stack(view_az_accum[band_id],  axis=0)
        # NaN-aware mean across detector dimension; all-NaN cells (grid nodes
        # not covered by any detector) produce NaN — handled by interpolation.
        view_zen[band_id] = np.nanmean(stacked_z, axis=0)
        view_az[band_id]  = np.nanmean(stacked_a, axis=0)

    # --- Grid UL corner (UTM) ------------------------------------------------
    # The angle grid origin is the UL corner of the granule tile in UTM.
    # It is encoded in the TILE_GEOCODING element.
    ul_utm: tuple[float, float] | None = None
    geocoding = root.find(".//Tile_Geocoding")
    if geocoding is not None:
        ul_e_elem = geocoding.find("ULX")
        ul_n_elem = geocoding.find("ULY")
        if ul_e_elem is not None and ul_n_elem is not None:
            ul_utm = (float(ul_e_elem.text), float(ul_n_elem.text))

    if ul_utm is None:
        # Fallback: try geoposition element (older metadata format)
        geopos = root.find(".//Geoposition[@resolution='10']")
        if geopos is not None:
            ulx = geopos.find("ULX")
            uly = geopos.find("ULY")
            if ulx is not None and uly is not None:
                ul_utm = (float(ulx.text), float(uly.text))

    if ul_utm is None:
        logger.warning("granule_angles: cannot determine UL corner for %s", item_id)
        with _xml_cache_lock:
            _xml_cache[item_id] = None
        return None

    result = {
        "sun_zen": sun_zen,
        "sun_az":  sun_az,
        "view_zen": view_zen,
        "view_az":  view_az,
        "ul_utm": ul_utm,
    }
    with _xml_cache_lock:
        _xml_cache[item_id] = result
    return result


# S2 band index → bandId string used in the XML
_BAND_ID: dict[str, str] = {
    "B02": "1",
    "B03": "2",
    "B04": "3",
    "B05": "4",
    "B06": "5",
    "B07": "6",
    "B08": "7",
    "B8A": "8",
    "B11": "11",
    "B12": "12",
}


def get_item_angles(
    item,
    lons: np.ndarray,
    lats: np.ndarray,
    utm_crs: str,
    bands: list[str],
) -> dict[str, dict[str, np.ndarray]] | None:
    """Return per-band angle arrays interpolated to (lon, lat) pixel positions.

    Parameters
    ----------
    item      : pystac.Item
    lons, lats: WGS84 coordinates, shape (N,)
    utm_crs   : UTM CRS of the tile, e.g. "EPSG:32754"
    bands     : list of S2 band names, e.g. ["B04", "B05", "B07"]

    Returns
    -------
    dict  {band: {'sza': (N,), 'vza': (N,), 'saa': (N,), 'vaa': (N,)}}
    or None on fetch/parse failure.
    """
    # Locate the granule_metadata asset
    asset = item.assets.get("granule_metadata")
    if asset is None:
        logger.warning("granule_angles: no granule_metadata asset for %s", item.id)
        return None

    parsed = _fetch_and_parse_xml(item.id, asset.href)
    if parsed is None:
        return None

    ul_e, ul_n = parsed["ul_utm"]

    # Build UTM coordinates for the angle grid nodes (row-major, UL origin)
    # The grid is defined at the centres of 5 km cells starting from UL.
    grid_eastings  = ul_e + np.arange(_GRID_SIZE) * _GRID_STEP_M
    grid_northings = ul_n - np.arange(_GRID_SIZE) * _GRID_STEP_M  # N decreases southward

    # Convert pixel lon/lat → UTM
    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    px_e, px_n = to_utm.transform(lons, lats)

    # Clip query points to the grid extent (avoids extrapolation errors on edge pixels)
    px_e_clipped = np.clip(px_e, grid_eastings[0],  grid_eastings[-1])
    px_n_clipped = np.clip(px_n, grid_northings[-1], grid_northings[0])

    def _interp(grid_2d: np.ndarray) -> np.ndarray:
        # RegularGridInterpolator expects axes in ascending order.
        # northings are descending, so flip rows and the axis.
        interp = RegularGridInterpolator(
            (grid_northings[::-1], grid_eastings),
            grid_2d[::-1, :],
            method="linear",
            bounds_error=False,
            fill_value=None,  # nearest extrapolation outside bounds
        )
        return interp(np.column_stack([px_n_clipped, px_e_clipped])).astype(np.float32)

    # Solar angles are granule-wide (single grid)
    sza_all = _interp(parsed["sun_zen"])
    saa_all = _interp(parsed["sun_az"])

    result: dict[str, dict[str, np.ndarray]] = {}
    for band in bands:
        band_id = _BAND_ID.get(band)
        if band_id is None or band_id not in parsed["view_zen"]:
            # Fall back to solar-only (vza=0, vaa=0 — minimal correction)
            result[band] = {
                "sza": sza_all,
                "vza": np.zeros_like(sza_all),
                "saa": saa_all,
                "vaa": np.zeros_like(sza_all),
            }
            continue

        vza = _interp(parsed["view_zen"][band_id])
        vaa = _interp(parsed["view_az"][band_id])
        result[band] = {"sza": sza_all, "vza": vza, "saa": saa_all, "vaa": vaa}

    return result
