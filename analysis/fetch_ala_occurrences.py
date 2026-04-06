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
from datetime import date, datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

ALA_BIOCACHE_URL = "https://biocache.ala.org.au/ws/occurrences/search"
PAGE_SIZE = 1000
MAX_RECORDS = 100_000

FIELDS = ",".join([
    "decimalLongitude",
    "decimalLatitude",
    "coordinateUncertaintyInMeters",
    "eventDate",
    "year",
    "month",
    "stateProvince",
    "recordedBy",
    "dataResourceName",
    "basisOfRecord",
    "spatiallyValid",
    "geospatialKosher",
])

logger = logging.getLogger(__name__)


def _parse_event_date(value) -> str | None:
    """Normalise ALA eventDate to YYYY-MM-DD string.

    The API returns either a Unix timestamp in milliseconds (int) or a
    ISO-8601 string.  Returns None if the value is absent or unparseable.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(value, str):
        return value[:10] if len(value) >= 10 else value
    return None


def _fetch_bbox_chunk(species: str, lon_min: float, lon_max: float) -> list[dict]:
    """Fetch all records in a lon strip, paginating up to the API's 5k hard cap.

    Returns raw record dicts. Caller is responsible for deduplication.
    """
    params = {
        "q": f'taxon_name:"{species}"',
        "fq": f"longitude:[{lon_min} TO {lon_max}]",
        "pageSize": PAGE_SIZE,
        "fl": FIELDS,
        "startIndex": 0,
    }

    chunk: list[dict] = []
    total = None
    while True:
        resp = requests.get(ALA_BIOCACHE_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        if total is None:
            total = data.get("totalRecords", 0)

        occurrences = data.get("occurrences", [])
        if not occurrences:
            break

        for occ in occurrences:
            lat = occ.get("decimalLatitude")
            lon = occ.get("decimalLongitude")
            if lat is None or lon is None:
                continue
            try:
                lat = float(lat)
                lon = float(lon)
            except (TypeError, ValueError):
                continue
            chunk.append({
                "decimalLongitude":              lon,
                "decimalLatitude":               lat,
                "coordinateUncertaintyInMeters": occ.get("coordinateUncertaintyInMeters"),
                "eventDate":                     _parse_event_date(occ.get("eventDate")),
                "year":                          occ.get("year"),
                "month":                         occ.get("month"),
                "stateProvince":                 occ.get("stateProvince"),
                "recordedBy":                    occ.get("recordedBy"),
                "dataResourceName":              occ.get("dataResourceName"),
                "basisOfRecord":                 occ.get("basisOfRecord"),
                "spatiallyValid":                occ.get("spatiallyValid"),
                "geospatialKosher":              occ.get("geospatialKosher"),
            })

        params["startIndex"] += PAGE_SIZE
        if params["startIndex"] >= total:
            break

    return chunk


def _count_bbox(species: str, lon_min: float, lon_max: float) -> int:
    resp = requests.get(ALA_BIOCACHE_URL, params={
        "q": f'taxon_name:"{species}"',
        "fq": f"longitude:[{lon_min} TO {lon_max}]",
        "pageSize": 0,
    }, timeout=30)
    resp.raise_for_status()
    return resp.json().get("totalRecords", 0)


def fetch_all(species: str) -> gpd.GeoDataFrame:
    """Fetch all occurrence records nationwide by bbox-chunking to stay under the
    ALA biocache 5,000-record pagination cap.

    Splits Australia (lon 113–154) into 1-degree strips, subdividing any strip
    that still exceeds 4,500 records into 0.5-degree sub-strips.
    """
    AUS_LON_MIN, AUS_LON_MAX = 113.0, 154.0
    STRIP_DEG = 1.0
    SOFT_CAP = 4500  # subdivide if a strip exceeds this

    # Build initial strip boundaries
    import math
    n_strips = math.ceil((AUS_LON_MAX - AUS_LON_MIN) / STRIP_DEG)
    strips = [
        (AUS_LON_MIN + i * STRIP_DEG, min(AUS_LON_MIN + (i + 1) * STRIP_DEG, AUS_LON_MAX))
        for i in range(n_strips)
    ]

    # Subdivide any strip with too many records
    final_strips: list[tuple[float, float]] = []
    for lo, hi in strips:
        n = _count_bbox(species, lo, hi)
        if n > SOFT_CAP:
            mid = (lo + hi) / 2
            logger.info("Strip [%.1f, %.1f] has %d records — subdividing", lo, hi, n)
            final_strips.append((lo, mid))
            final_strips.append((mid, hi))
        else:
            final_strips.append((lo, hi))

    records: list[dict] = []
    seen: set[tuple[float, float]] = set()  # dedup by (lon, lat) — crude but effective

    for i, (lo, hi) in enumerate(final_strips):
        n = _count_bbox(species, lo, hi)
        logger.info("Strip %d/%d  lon [%.2f, %.2f]  ~%d records",
                    i + 1, len(final_strips), lo, hi, n)
        chunk = _fetch_bbox_chunk(species, lo, hi)
        for rec in chunk:
            key = (rec["decimalLongitude"], rec["decimalLatitude"])
            if key not in seen:
                seen.add(key)
                records.append(rec)

    logger.info("Fetched %d records with valid coordinates", len(records))

    if not records:
        return gpd.GeoDataFrame(
            columns=["decimalLongitude", "decimalLatitude",
                     "coordinateUncertaintyInMeters", "eventDate", "year", "month",
                     "stateProvince", "recordedBy", "dataResourceName",
                     "basisOfRecord", "spatiallyValid", "geospatialKosher", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    df = pd.DataFrame(records)
    geometry = [Point(row.decimalLongitude, row.decimalLatitude) for row in df.itertuples()]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        stream=sys.stdout,
    )

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    cache_dir = PROJECT_ROOT / "outputs" / "ala_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "ala_occurrences.gpkg"

    if out_path.exists():
        logger.info("Overwriting existing cache: %s", out_path)
        out_path.unlink()

    species = "Parkinsonia aculeata"
    logger.info("Fetching ALA occurrences for '%s' ...", species)
    gdf = fetch_all(species)
    gdf.to_file(out_path, driver="GPKG")
    logger.info("Written %d records to %s", len(gdf), out_path)


if __name__ == "__main__":
    main()
