"""
fetch_ala_australia.py — fetch all Australian Parkinsonia aculeata records from ALA
and produce a distribution map.

Does NOT depend on config.py / env vars — runs standalone.

Usage:
    python analysis/fetch_ala_australia.py

Outputs:
    ala_australia_occurrences.gpkg   — point layer (WGS84)
    ala_australia_distribution.png   — map figure
"""
import logging
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import requests
from shapely.geometry import Point

ALA_BIOCACHE_URL = "https://biocache.ala.org.au/ws/occurrences/search"
SPECIES = "Parkinsonia aculeata"
PAGE_SIZE = 1000
MAX_RECORDS = 50_000          # raise if needed; ALA has ~3-4k verified Aus records

# Australia bounding box (WGS84)
AUS_BBOX = [112.0, -44.0, 154.0, -10.0]

OUT_DIR = Path(__file__).parent.parent / "outputs" / "australia_occurrences"

logger = logging.getLogger(__name__)


def fetch_all(species: str, bbox: list[float]) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = bbox
    params = {
        "q": f'taxon_name:"{species}"',
        "fq": (
            f"decimalLongitude:[{minx} TO {maxx}] "
            f"AND decimalLatitude:[{miny} TO {maxy}]"
        ),
        "pageSize": 0,
        "fl": "decimalLongitude,decimalLatitude,stateProvince,coordinateUncertaintyInMeters",
        "startIndex": 0,
    }

    probe = requests.get(ALA_BIOCACHE_URL, params=params, timeout=30)
    probe.raise_for_status()
    total = probe.json().get("totalRecords", 0)
    logger.info("Total ALA records for '%s' in Australia bbox: %d", species, total)

    if total == 0:
        logger.warning("No records found.")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    to_fetch = min(total, MAX_RECORDS)
    logger.info("Fetching %d records in pages of %d...", to_fetch, PAGE_SIZE)
    params["pageSize"] = PAGE_SIZE

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
        start += len(page)
        logger.info("  fetched %d / %d", len(records), to_fetch)

    df = pd.DataFrame(records)
    df = df.dropna(subset=["decimalLongitude", "decimalLatitude"])
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(r.decimalLongitude, r.decimalLatitude) for r in df.itertuples()],
        crs="EPSG:4326",
    )
    logger.info("Records with valid coordinates: %d", len(gdf))
    return gdf


def make_map(gdf: gpd.GeoDataFrame, out_path: Path) -> None:
    # Load Australia states outline from naturalearth (bundled with geopandas)
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    aus = world[world["name"] == "Australia"]

    fig, ax = plt.subplots(figsize=(12, 9))
    aus.plot(ax=ax, color="#f5f0eb", edgecolor="#aaa", linewidth=0.8)

    # Hexbin density layer
    if len(gdf) > 0:
        x = gdf.geometry.x.values
        y = gdf.geometry.y.values
        hb = ax.hexbin(x, y, gridsize=60, cmap="YlOrRd", mincnt=1, alpha=0.85,
                       extent=[112, 154, -44, -10])
        cb = fig.colorbar(hb, ax=ax, shrink=0.6, label="Sightings per hex cell")

    ax.set_xlim(112, 154)
    ax.set_ylim(-44, -10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"Parkinsonia aculeata — ALA occurrence records (n={len(gdf):,})\n"
        "Source: Atlas of Living Australia biocache API",
        fontsize=13,
    )
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Map saved: %s", out_path)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        stream=sys.stdout,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gdf = fetch_all(SPECIES, AUS_BBOX)

    gpkg_path = OUT_DIR / "ala_australia_occurrences.gpkg"
    gdf.to_file(gpkg_path, driver="GPKG")
    logger.info("GeoPackage written: %s", gpkg_path)

    map_path = OUT_DIR / "ala_australia_distribution.png"
    make_map(gdf, map_path)


if __name__ == "__main__":
    main()
