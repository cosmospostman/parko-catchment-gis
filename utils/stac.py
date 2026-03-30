"""utils/stac.py — STAC search and data loading helpers.

All functions are designed to be mockable in tests — no I/O at import time.
"""

import logging
from typing import Any, Dict, List, Optional

import xarray as xr

logger = logging.getLogger(__name__)


def search_sentinel2(
    bbox: List[float],
    start: str,
    end: str,
    cloud_cover_max: int,
    endpoint: str,
    collection: str,
) -> List[Any]:
    """Search a STAC endpoint for Sentinel-2 items."""
    import pystac_client

    catalog = pystac_client.Client.open(endpoint)
    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": cloud_cover_max}},
    )
    items = list(search.items())
    logger.info("S2 STAC search: %d items found", len(items))
    return items


def search_sentinel1(
    bbox: List[float],
    start: str,
    end: str,
    endpoint: str,
    collection: str,
) -> List[Any]:
    """Search a STAC endpoint for Sentinel-1 GRD items."""
    import pystac_client

    catalog = pystac_client.Client.open(endpoint)
    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=f"{start}/{end}",
    )
    items = list(search.items())
    logger.info("S1 STAC search: %d items found", len(items))
    return items


def load_stackstac(
    items: List[Any],
    bands: List[str],
    resolution: int,
    bbox: List[float],
    crs: str,
    chunk_spatial: int = 2048,
) -> xr.DataArray:
    """Stack STAC items into a lazy dask DataArray using stackstac."""
    import stackstac

    da = stackstac.stack(
        items,
        assets=bands,
        resolution=resolution,
        bounds_latlon=bbox,
        epsg=int(crs.split(":")[1]),
        chunksize=chunk_spatial,
    )
    logger.info("Stacked %d items, shape: %s", len(items), da.shape)
    return da


def load_dea_landsat(
    bbox: List[float],
    start: str,
    end: str,
    bands: List[str],
    collection: str,
    resolution: int,
    crs: str,
) -> xr.Dataset:
    """Load DEA Landsat ARD data via odc-stac."""
    import odc.stac
    import pystac_client

    catalog = pystac_client.Client.open("https://explorer.dea.ga.gov.au/stac")
    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=f"{start}/{end}",
        max_items=1000,
    )
    items = list(search.items())
    logger.info("DEA Landsat search: %d items found", len(items))
    ds = odc.stac.load(
        items,
        bands=bands,
        resolution=resolution,
        bbox=bbox,
        crs=crs,
        chunks={"x": 2048, "y": 2048},
    )
    return ds
