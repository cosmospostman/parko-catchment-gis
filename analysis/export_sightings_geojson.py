"""
export_sightings_geojson.py — convert ALA occurrences cache to GeoJSON for UI.

Reads outputs/ala_cache/ala_occurrences.gpkg and writes a trimmed GeoJSON file
suitable for client-side rendering in the map UI.

Usage:
    python analysis/export_sightings_geojson.py

Output:
    outputs/ala_cache/ala_sightings.geojson
"""
import sys
from pathlib import Path

import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "outputs" / "ala_cache" / "ala_occurrences.gpkg"
DST = PROJECT_ROOT / "outputs" / "ala_cache" / "ala_sightings.geojson"

DROP_COLS = {"decimalLongitude", "decimalLatitude", "geospatialKosher", "stateProvince"}


def main() -> None:
    if not SRC.exists():
        print(f"ERROR: source file not found: {SRC}", file=sys.stderr)
        sys.exit(1)

    gdf = gpd.read_file(SRC)
    keep = [c for c in gdf.columns if c not in DROP_COLS]
    gdf = gdf[keep]

    # Normalise recordedBy: strip surrounding list notation e.g. "['alice']" -> "alice"
    if "recordedBy" in gdf.columns:
        gdf["recordedBy"] = (
            gdf["recordedBy"]
            .astype(str)
            .str.strip()
            .str.removeprefix("['")
            .str.removesuffix("']")
            .replace("None", None)
        )

    # Ensure year/month are int-compatible (drop float .0 suffix)
    if "year" in gdf.columns:
        gdf["year"] = gdf["year"].astype("Int64")
    if "month" in gdf.columns:
        gdf["month"] = gdf["month"].astype(str).replace("None", None).replace("<NA>", None)

    gdf.to_file(DST, driver="GeoJSON")
    print(f"Written {len(gdf):,} features to {DST}")


if __name__ == "__main__":
    main()
