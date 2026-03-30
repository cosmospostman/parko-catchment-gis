"""
Step 06 — Treatment priority patch delineation.

Produces: priority_patches_{year}.gpkg  (GeoPackage, EPSG:7844)
"""
import logging
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import shape
from shapely.ops import unary_union

# Script-level constants
TIER_THRESHOLDS: Dict[str, float] = {"A": 0.85, "B": 0.75, "C": 0.60}
SEED_FLUX_STREAM_ORDER_WEIGHTS: Dict[int, float] = {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0, 6: 1.0}
KOWANYAMA_COORDS = (-141.7400, -15.4833)   # lon, lat — downstream reference point
OUTPUT_ATTRIBUTES = [
    "tier", "area_ha", "prob_mean", "prob_max",
    "dist_to_kowanyama_km", "seed_flux_score", "stream_order",
]

logger = logging.getLogger(__name__)


def _stream_order_for_patch(patch_geom, drainage: gpd.GeoDataFrame) -> int:
    """Return max Strahler stream order intersecting the patch (1 if none)."""
    if drainage is None or drainage.empty:
        return 1
    intersecting = drainage[drainage.intersects(patch_geom)]
    if intersecting.empty:
        return 1
    if "stream_order" in intersecting.columns:
        return int(intersecting["stream_order"].max())
    return 1


def main() -> None:
    import config
    from utils.io import configure_logging, ensure_output_dirs, read_raster
    from utils.quicklook import save_quicklook

    configure_logging()

    ensure_output_dirs(config.YEAR)

    catchment = gpd.read_file(config.CATCHMENT_GEOJSON).to_crs(config.TARGET_CRS)

    prob_path = config.probability_raster_path(config.YEAR)
    if not prob_path.exists():
        raise FileNotFoundError(f"Step 05 output not found: {prob_path}")

    prob_da = read_raster(prob_path).squeeze()
    prob_arr = prob_da.values.astype(np.float32)

    # Load drainage network if available
    drain_path = Path(config.BASE_DIR) / "data" / "drainage_network.gpkg"
    drainage = None
    if drain_path.exists():
        drainage = gpd.read_file(str(drain_path)).to_crs(config.TARGET_CRS)

    # Kowanyama reference point for distance calculation
    from shapely.geometry import Point
    kowanyama = gpd.GeoDataFrame(
        geometry=[Point(KOWANYAMA_COORDS)], crs="EPSG:4326"
    ).to_crs(config.TARGET_CRS).geometry[0]

    all_patches = []

    for tier, threshold in TIER_THRESHOLDS.items():
        binary = (prob_arr >= threshold).astype(np.uint8)

        # Vectorise
        import rasterio.features
        transform = prob_da.rio.transform()
        shapes = list(rasterio.features.shapes(binary, mask=binary, transform=transform))

        for geom_dict, val in shapes:
            if val != 1:
                continue
            geom = shape(geom_dict)
            area_m2 = geom.area
            area_ha = area_m2 / 10_000

            if area_ha < config.MIN_PATCH_AREA_HA:
                continue

            # Sample probability stats
            patch_mask = rasterio.features.rasterize(
                [(geom_dict, 1)],
                out_shape=prob_arr.shape,
                transform=transform,
                fill=0,
                dtype=np.uint8,
            )
            patch_probs = prob_arr[patch_mask == 1]
            valid_probs = patch_probs[~np.isnan(patch_probs)]
            if len(valid_probs) == 0:
                continue

            prob_mean = float(np.mean(valid_probs))
            prob_max  = float(np.max(valid_probs))

            # Distance to Kowanyama
            dist_km = geom.centroid.distance(kowanyama) / 1000.0

            # Stream order
            so = _stream_order_for_patch(geom, drainage)

            # Seed flux score
            seed_flux = SEED_FLUX_STREAM_ORDER_WEIGHTS.get(so, 1.0) * prob_mean

            all_patches.append({
                "geometry": geom,
                "tier": tier,
                "area_ha": round(area_ha, 4),
                "prob_mean": round(prob_mean, 4),
                "prob_max": round(prob_max, 4),
                "dist_to_kowanyama_km": round(dist_km, 2),
                "seed_flux_score": round(seed_flux, 4),
                "stream_order": so,
            })

    if not all_patches:
        logger.warning("No patches above threshold — writing empty GeoDataFrame")
        gdf = gpd.GeoDataFrame(
            {attr: [] for attr in OUTPUT_ATTRIBUTES}, geometry=[], crs=config.TARGET_CRS
        )
    else:
        gdf = gpd.GeoDataFrame(all_patches, crs=config.TARGET_CRS)
        # Deduplicate: keep highest tier for each geometry
        gdf = gdf.drop_duplicates(subset=["geometry"])
        gdf = gdf.sort_values("tier").reset_index(drop=True)
        # Clip to catchment
        gdf = gpd.clip(gdf, catchment)

    out_path = config.priority_patches_path(config.YEAR)
    gdf.to_file(str(out_path), driver="GPKG")
    logger.info("Written: %s  (%d patches)", out_path, len(gdf))


if __name__ == "__main__":
    main()
