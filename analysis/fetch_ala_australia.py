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
import pandas as pd
import requests
from shapely.geometry import Point

ALA_BIOCACHE_URL = "https://biocache.ala.org.au/ws/occurrences/search"
SPECIES = "Parkinsonia aculeata"
PAGE_SIZE = 1000
ALA_MAX_START_INDEX = 5000   # ALA biocache hard cap: returns empty page beyond this

# Australia bounding box (WGS84)
AUS_BBOX = [112.0, -44.0, 154.0, -10.0]

# Mitchell catchment bounding box — used for reporting only
MITCHELL_BBOX = [141.3453505, -18.23350524, 145.51819104, -14.92188926]

# Tile grid dimensions — each tile must stay under ALA_MAX_START_INDEX records.
# 8 lon x 6 lat = 48 tiles across Australia.
TILE_LON_STEPS = 8
TILE_LAT_STEPS = 6

OUT_DIR = Path(__file__).parent.parent / "outputs" / "australia_occurrences"

logger = logging.getLogger(__name__)


def fetch_bbox(species: str, minx: float, miny: float, maxx: float, maxy: float,
               depth: int = 0) -> list[dict]:
    """Fetch all records within a bbox, recursively splitting if total exceeds ALA's cap."""
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

    if total == 0:
        return []

    if total > ALA_MAX_START_INDEX:
        logger.debug(
            "%sTile [%.2f,%.2f,%.2f,%.2f] has %d records — splitting (depth %d)",
            "  " * depth, minx, miny, maxx, maxy, total, depth,
        )
        midx = (minx + maxx) / 2
        midy = (miny + maxy) / 2
        return (
            fetch_bbox(species, minx, miny, midx, midy, depth + 1)
            + fetch_bbox(species, midx, miny, maxx, midy, depth + 1)
            + fetch_bbox(species, minx, midy, midx, maxy, depth + 1)
            + fetch_bbox(species, midx, midy, maxx, maxy, depth + 1)
        )

    params["pageSize"] = PAGE_SIZE
    records = []
    start = 0
    while start < total:
        params["startIndex"] = start
        params["pageSize"] = min(PAGE_SIZE, total - start)
        resp = requests.get(ALA_BIOCACHE_URL, params=params, timeout=30)
        resp.raise_for_status()
        page = resp.json().get("occurrences", [])
        if not page:
            break
        records.extend(page)
        start += len(page)

    return records


def fetch_all(species: str, bbox: list[float]) -> gpd.GeoDataFrame:
    """Tile the bbox into a grid and fetch each tile, recursively splitting dense tiles."""
    minx, miny, maxx, maxy = bbox
    lon_edges = [minx + (maxx - minx) * i / TILE_LON_STEPS for i in range(TILE_LON_STEPS + 1)]
    lat_edges = [miny + (maxy - miny) * i / TILE_LAT_STEPS for i in range(TILE_LAT_STEPS + 1)]

    tiles = [
        (lon_edges[i], lat_edges[j], lon_edges[i + 1], lat_edges[j + 1])
        for i in range(TILE_LON_STEPS)
        for j in range(TILE_LAT_STEPS)
    ]
    logger.info("Fetching '%s' across %d initial tiles...", species, len(tiles))

    all_records = []
    for n, (tx0, ty0, tx1, ty1) in enumerate(tiles, 1):
        recs = fetch_bbox(species, tx0, ty0, tx1, ty1)
        all_records.extend(recs)
        logger.info("  tile %d/%d → %d records (total so far: %d)", n, len(tiles), len(recs), len(all_records))

    logger.info("Total raw records: %d", len(all_records))

    df = pd.DataFrame(all_records)
    df = df.dropna(subset=["decimalLongitude", "decimalLatitude"])
    df = df.drop_duplicates(subset=["decimalLongitude", "decimalLatitude"])
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(r.decimalLongitude, r.decimalLatitude) for r in df.itertuples()],
        crs="EPSG:4326",
    )
    logger.info("Records with valid coordinates (deduped): %d", len(gdf))
    return gdf


def make_map(gdf: gpd.GeoDataFrame, out_path: Path) -> None:
    # Load Australia outline from naturalearth via cartopy or a simple bbox polygon
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent([112, 154, -44, -10], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#f5f0eb")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="#aaa")
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="#ccc")
        ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="#ccc")
        if len(gdf) > 0:
            x = gdf.geometry.x.values
            y = gdf.geometry.y.values
            hb = ax.hexbin(x, y, gridsize=60, cmap="YlOrRd", mincnt=1, alpha=0.85,
                           extent=[112, 154, -44, -10], transform=ccrs.PlateCarree())
            fig.colorbar(hb, ax=ax, shrink=0.6, label="Sightings per hex cell")
        ax.set_title(
            f"Parkinsonia aculeata — ALA occurrence records (n={len(gdf):,})\n"
            "Source: Atlas of Living Australia biocache API",
            fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Map saved: %s", out_path)
        plt.close(fig)
        return
    except ImportError:
        pass

    # Fallback: download naturalearth shapefile on the fly, extract to temp dir
    import io, tempfile, zipfile, urllib.request
    NE_URL = (
        "https://naturalearth.s3.amazonaws.com/110m_cultural/"
        "ne_110m_admin_0_countries.zip"
    )
    logger.info("Downloading naturalearth countries shapefile for basemap...")
    with urllib.request.urlopen(NE_URL, timeout=30) as resp:
        zf = zipfile.ZipFile(io.BytesIO(resp.read()))
    with tempfile.TemporaryDirectory() as tmpdir:
        zf.extractall(tmpdir)
        shp_path = next(Path(tmpdir).glob("*.shp"))
        world = gpd.read_file(shp_path)
    aus = world[world["NAME"] == "Australia"]

    fig, ax = plt.subplots(figsize=(12, 9))
    aus.plot(ax=ax, color="#f5f0eb", edgecolor="#aaa", linewidth=0.8)

    if len(gdf) > 0:
        x = gdf.geometry.x.values
        y = gdf.geometry.y.values
        hb = ax.hexbin(x, y, gridsize=60, cmap="YlOrRd", mincnt=1, alpha=0.85,
                       extent=[112, 154, -44, -10])
        fig.colorbar(hb, ax=ax, shrink=0.6, label="Sightings per hex cell")

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

    minx, miny, maxx, maxy = MITCHELL_BBOX
    in_mitchell = gdf.cx[minx:maxx, miny:maxy]
    logger.info("Sightings within Mitchell catchment bbox: %d", len(in_mitchell))

    gpkg_path = OUT_DIR / "ala_australia_occurrences.gpkg"
    gdf.to_file(gpkg_path, driver="GPKG")
    logger.info("GeoPackage written: %s", gpkg_path)

    map_path = OUT_DIR / "ala_australia_distribution.png"
    make_map(gdf, map_path)


if __name__ == "__main__":
    main()
