"""
fetch_ala_occurrences.py — pre-cache ALA occurrence records for Stage 5.

Fetches all Parkinsonia aculeata records within the catchment bounding box
from the ALA biocache API and saves them as a GeoPackage to the cache directory.

Stage 5 will use the cache file if present, skipping the API call entirely.

Usage:
    python analysis/fetch_ala_occurrences.py

Output:
    {CACHE_DIR}/ala_occurrences.gpkg
"""
import logging
import sys
from datetime import date
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

ALA_BIOCACHE_URL = "https://biocache.ala.org.au/ws/occurrences/search"
PAGE_SIZE = 1000
MAX_RECORDS = 10000

FIELDS = ",".join([
    "decimalLongitude",
    "decimalLatitude",
    "eventDate",
    "year",
    "month",
    "recordedBy",
    "dataResourceName",
])

logger = logging.getLogger(__name__)


def fetch_all(species: str) -> gpd.GeoDataFrame:
    """Fetch all occurrence records nationwide, paginating through results."""
    params = {
        "q": f'taxon_name:"{species}"',
        "pageSize": PAGE_SIZE,
        "fl": FIELDS,
        "startIndex": 0,
    }

    # Get total count first
    probe = requests.get(ALA_BIOCACHE_URL, params={**params, "pageSize": 0}, timeout=30)
    probe.raise_for_status()
    total = probe.json().get("totalRecords", 0)
    logger.info("Total ALA records for '%s' in catchment bbox: %d", species, total)

    if total == 0:
        logger.warning("No records found — check species name and catchment bbox")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    to_fetch = min(total, MAX_RECORDS)
    logger.info("Fetching %d records in pages of %d...", to_fetch, PAGE_SIZE)

    records = []
    start = 0
    while start < to_fetch:
        params["startIndex"] = start
        params["pageSize"] = min(PAGE_SIZE, to_fetch - start)
        resp = requests.get(ALA_BIOCACHE_URL, params=params, timeout=30)
        resp.raise_for_status()
        page = resp.json().get("occurrences", [])
        if not page:
            break
        records.extend(page)
        start += params["pageSize"]
        logger.info("  fetched %d / %d", min(len(records), to_fetch), to_fetch)

    df = pd.DataFrame(records)
    df = df.dropna(subset=["decimalLongitude", "decimalLatitude"])
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(r["decimalLongitude"], r["decimalLatitude"]) for _, r in df.iterrows()],
        crs="EPSG:4326",
    )
    if "eventDate" in gdf.columns:
        gdf["eventDate"] = pd.to_datetime(gdf["eventDate"], unit="ms", errors="coerce").dt.strftime("%Y-%m-%d")
    gdf["fetched_date"] = date.today().isoformat()
    logger.info("Records with valid coordinates: %d", len(gdf))
    return gdf


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        stream=sys.stdout,
    )

    import config

    Path(config.CACHE_DIR).mkdir(parents=True, exist_ok=True)

    gdf = fetch_all(config.ALA_SPECIES_QUERY)

    out_path = Path(config.CACHE_DIR) / "ala_occurrences.gpkg"
    gdf.to_file(out_path, driver="GPKG")
    logger.info("Written: %s", out_path)


if __name__ == "__main__":
    main()
