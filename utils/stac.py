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
    max_items: Optional[int] = None,
) -> List[Any]:
    """Search a STAC endpoint for Sentinel-2 items."""
    import pystac_client

    catalog = pystac_client.Client.open(endpoint)
    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": cloud_cover_max}},
        max_items=max_items,
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
    modifier=None,
) -> List[Any]:
    """Search for Sentinel-1 items via MPC GeoParquet on Azure Blob Storage.

    Queries the monthly-partitioned STAC GeoParquet directly via DuckDB + adlfs,
    bypassing the pystac-client API endpoint entirely. This avoids the server-side
    timeouts that affect long date-range STAC searches on Planetary Computer.

    Falls back to the pystac-client API if the GeoParquet approach fails.
    """
    try:
        return _search_sentinel1_geoparquet(bbox, start, end, modifier=modifier)
    except Exception as exc:
        logger.warning(
            "S1 GeoParquet search failed (%s), falling back to STAC API", exc,
        )
        return _search_sentinel1_api(bbox, start, end, endpoint, collection, modifier=modifier)


def _search_sentinel1_geoparquet(
    bbox: List[float],
    start: str,
    end: str,
    modifier=None,
) -> List[Any]:
    """Query S1-RTC STAC items from MPC GeoParquet, selecting only relevant monthly partitions."""
    import json
    import requests
    import adlfs
    import duckdb
    import pystac
    import planetary_computer

    # Get a fresh SAS token via the PC SAS token API directly (avoids STAC catalog endpoint)
    import requests
    resp = requests.get(
        "https://planetarycomputer.microsoft.com/api/sas/v1/token/pcstacitems/items",
        timeout=30,
    )
    resp.raise_for_status()
    sas_token = resp.json()["token"]
    storage_opts = {"account_name": "pcstacitems", "credential": sas_token}

    import logging as _logging
    _logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(_logging.WARNING)

    fs = adlfs.AzureBlobFileSystem(**storage_opts)

    # Select only the monthly partition files that overlap the requested date range
    all_partitions = fs.ls("items/sentinel-1-rtc.parquet")
    start_dt = start[:7]  # "YYYY-MM"
    end_dt   = end[:7]
    selected = [
        f"abfs://{p}" for p in all_partitions
        if _partition_overlaps(p, start_dt, end_dt)
    ]
    if not selected:
        logger.info("S1 GeoParquet: no partitions found for %s/%s", start, end)
        return []

    logger.info(
        "S1 GeoParquet: querying %d monthly partition(s) for bbox=%s %s→%s",
        len(selected), bbox, start, end,
    )

    con = duckdb.connect()
    con.register_filesystem(fs)
    con.execute("INSTALL spatial; LOAD spatial;")

    minx, miny, maxx, maxy = bbox
    files_expr = ", ".join(f"'{f}'" for f in selected)

    rows = con.execute(f"""
        SELECT id, datetime, start_datetime, geometry, assets,
               "sat:orbit_state", "sar:instrument_mode", "sar:polarizations",
               "proj:epsg", "proj:bbox", "proj:shape", "proj:transform",
               "sat:relative_orbit", platform,
               ST_XMin(geometry) AS wgs84_minx,
               ST_YMin(geometry) AS wgs84_miny,
               ST_XMax(geometry) AS wgs84_maxx,
               ST_YMax(geometry) AS wgs84_maxy
        FROM read_parquet([{files_expr}])
        WHERE datetime >= '{start}'
          AND datetime <= '{end}'
          AND ST_Intersects(
                geometry,
                ST_MakeEnvelope({minx}, {miny}, {maxx}, {maxy})
              )
    """).df()

    logger.info("S1 GeoParquet: %d items found", len(rows))

    # Reconstruct minimal pystac Item objects so the rest of the pipeline is unchanged
    items = []
    for _, row in rows.iterrows():
        assets_dict = json.loads(row["assets"]) if isinstance(row["assets"], str) else row["assets"]
        # Sign asset hrefs so COG reads are authenticated
        signed_assets = {}
        for band, asset_data in assets_dict.items():
            if not asset_data:
                continue
            href = asset_data.get("href", "")
            signed_assets[band] = pystac.Asset(href=href)

        item = pystac.Item(
            id=row["id"],
            geometry=None,
            bbox=[row["wgs84_minx"], row["wgs84_miny"], row["wgs84_maxx"], row["wgs84_maxy"]],
            datetime=row["datetime"],
            properties={
                "datetime": str(row["datetime"]),
                "sat:orbit_state":      row["sat:orbit_state"],
                "sar:instrument_mode":  row["sar:instrument_mode"],
                "sar:polarizations":    row["sar:polarizations"],
                "proj:epsg":            row["proj:epsg"],
                "proj:shape":           row["proj:shape"],
                "proj:transform":       row["proj:transform"],
                "sat:relative_orbit":   row["sat:relative_orbit"],
                "platform":             row["platform"],
            },
        )
        item.assets = signed_assets
        # Sign the item so asset hrefs get SAS tokens for COG reads
        planetary_computer.sign_inplace(item)
        items.append(item)

    return items


def _partition_overlaps(partition_path: str, start_ym: str, end_ym: str) -> bool:
    """Return True if a monthly partition filename overlaps [start_ym, end_ym] (YYYY-MM)."""
    # Partition names look like: items/sentinel-1-rtc.parquet/part-0042_2021-06-...parquet
    import re
    m = re.search(r"(\d{4}-\d{2})-\d{2}T", partition_path)
    if not m:
        return False
    part_ym = m.group(1)
    return start_ym <= part_ym <= end_ym


def _search_sentinel1_api(
    bbox: List[float],
    start: str,
    end: str,
    endpoint: str,
    collection: str,
    modifier=None,
) -> List[Any]:
    """Fallback: search S1 via pystac-client API, chunked by year with retries."""
    import time as _time
    import pystac_client

    start_year = int(start[:4])
    end_year   = int(end[:4])

    catalog = pystac_client.Client.open(endpoint, modifier=modifier)
    seen: set = set()
    items: List[Any] = []

    max_retries = 3

    def _fetch_chunk(chunk_start: str, chunk_end: str) -> int:
        for attempt in range(1, max_retries + 1):
            logger.info(
                "S1 STAC API (attempt %d/%d): bbox=%s datetime=%s/%s",
                attempt, max_retries, bbox, chunk_start, chunk_end,
            )
            try:
                search = catalog.search(
                    collections=[collection],
                    bbox=bbox,
                    datetime=f"{chunk_start}/{chunk_end}",
                )
                n_before = len(items)
                for item in search.items():
                    if item.id not in seen:
                        seen.add(item.id)
                        items.append(item)
                return len(items) - n_before
            except Exception as exc:
                if attempt == max_retries:
                    raise
                wait = 30 * attempt
                logger.warning(
                    "S1 STAC API %s/%s failed (attempt %d/%d): %s — retrying in %ds",
                    chunk_start, chunk_end, attempt, max_retries, exc, wait,
                )
                _time.sleep(wait)

    for year in range(start_year, end_year + 1):
        chunk_start = max(start, f"{year}-01-01")
        chunk_end   = min(end,   f"{year}-12-31")
        try:
            n_new = _fetch_chunk(chunk_start, chunk_end)
            logger.info("S1 STAC API year %d: +%d items (total %d)", year, n_new, len(items))
        except Exception as exc:
            logger.warning(
                "S1 STAC API year %d failed, splitting into 2x 6-month chunks: %s", year, exc,
            )
            n_new = 0
            for cs, ce in [
                (chunk_start, min(chunk_end, f"{year}-06-30")),
                (max(chunk_start, f"{year}-07-01"), chunk_end),
            ]:
                if cs > ce:
                    continue
                n = _fetch_chunk(cs, ce)
                n_new += n
            logger.info("S1 STAC API year %d (split): +%d items (total %d)", year, n_new, len(items))

    logger.info("S1 STAC API: %d items found (%d years)", len(items), end_year - start_year + 1)
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


def filter_items_by_bbox(items: List[Any], bbox: List[float]) -> List[Any]:
    """Return only those items whose bbox intersects the given bbox.

    Both item.bbox and bbox are [minx, miny, maxx, maxy] in EPSG:4326.
    Filters out scenes that don't overlap the tile, eliminating the COG
    header round-trip overhead for non-intersecting scenes.
    """
    minx, miny, maxx, maxy = bbox
    result = []
    for item in items:
        ib = item.bbox  # [minx, miny, maxx, maxy]
        if ib[0] <= maxx and ib[2] >= minx and ib[1] <= maxy and ib[3] >= miny:
            result.append(item)
    return result


def rewrite_hrefs_to_local(items, local_root, bands=None):
    """Replace S3/HTTPS asset hrefs with local file paths where files exist.

    If bands is None, all assets are considered for rewriting.
    """
    import copy
    import urllib.parse
    from pathlib import Path

    patched = []
    for item in items:
        item = copy.deepcopy(item)
        for band, asset in item.assets.items():
            if bands is not None and band not in bands:
                continue
            parsed = urllib.parse.urlparse(asset.href)
            if parsed.scheme == "s3":
                # s3://sentinel-cogs/a/b/c.tif → /mnt/s2cache/sentinel-cogs/a/b/c.tif
                local_path = Path(local_root) / parsed.netloc / parsed.path.lstrip("/")
            elif "sentinel-cogs" in parsed.netloc:
                # https://sentinel-cogs.s3.us-west-2.amazonaws.com/a/b/c.tif
                #   → /mnt/s2cache/sentinel-cogs/a/b/c.tif
                local_path = Path(local_root) / "sentinel-cogs" / parsed.path.lstrip("/")
            else:
                continue
            if local_path.exists():
                asset.href = str(local_path)
        patched.append(item)
    return patched


def rewrite_dea_hrefs_to_s3(items, bands):
    """Rewrite DEA HTTPS asset hrefs to s3:// URIs for same-region S3 access.

    DEA STAC items return hrefs as:
        https://data.dea.ga.gov.au/<path>
    which maps directly to:
        s3://dea-public-data/<path>

    On an ap-southeast-2 instance this eliminates cross-internet latency and
    uses free same-region S3 transfer instead of HTTPS egress.
    """
    import copy
    import urllib.parse

    patched = []
    for item in items:
        item = copy.deepcopy(item)
        for band, asset in item.assets.items():
            if band not in bands:
                continue
            parsed = urllib.parse.urlparse(asset.href)
            if parsed.netloc == "data.dea.ga.gov.au":
                asset.href = f"s3://dea-public-data{parsed.path}"
        patched.append(item)
    return patched


DEA_LANDSAT_COLLECTIONS = [
    "ga_ls5t_ard_3",
    "ga_ls7e_ard_3",
    "ga_ls8c_ard_3",
    "ga_ls9c_ard_3",
]


def load_dea_landsat(
    bbox: List[float],
    start: str,
    end: str,
    bands: List[str],
    collection: str,
    resolution: int,
    crs: str,
) -> xr.Dataset:
    """Load DEA Landsat ARD data via odc-stac.

    The `collection` parameter is ignored — DEA split the former `ga_ls_ard_3`
    collection into per-sensor collections (ls5/ls7/ls8/ls9). All four are
    searched and combined.
    """
    import odc.stac
    import pystac_client

    catalog = pystac_client.Client.open("https://explorer.dea.ga.gov.au/stac")
    search = catalog.search(
        collections=DEA_LANDSAT_COLLECTIONS,
        bbox=bbox,
        datetime=f"{start}/{end}",
        max_items=1000,
    )
    items = list(search.items())
    logger.info("DEA Landsat search: %d items found (collections: %s)", len(items), DEA_LANDSAT_COLLECTIONS)
    ds = odc.stac.load(
        items,
        bands=bands,
        resolution=resolution,
        bbox=bbox,
        crs=crs,
        chunks={"x": 2048, "y": 2048},
    )
    return ds
