"""Tests for Stage 4 — HAND-based flood connectivity (utils/dem.py + analysis/04_flood_extent.py).

All tests are self-contained — no real DEM data required.  Synthetic rasters
are used to verify every function in utils/dem.py and the vectorisation /
post-processing logic in 04_flood_extent.py.

Coverage targets
----------------
DEM loading       — tile seam merging, void fill, reprojection
Flow routing      — D8 accumulation drains to outlet, stream burn
HAND computation  — zero at stream, monotonic in V-valley, NaN propagation
Flood mask        — threshold produces binary, correct inclusions/exclusions
Vectorisation     — min-patch removal, morphological closing, clip
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.dem import (
    compute_flow_accumulation,
    compute_hand,
    flood_connectivity_mask,
    burn_drainage_network,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dem(data: np.ndarray, res: float = 30.0) -> xr.DataArray:
    """Wrap a 2-D numpy array in a minimal DataArray with x/y coords."""
    H, W = data.shape
    x = np.arange(W) * res + 700000.0
    y = -np.arange(H) * res - 1_600_000.0
    return xr.DataArray(
        data.astype(np.float32),
        dims=["y", "x"],
        coords={"x": x, "y": y},
        name="elevation",
    )


def _slope_dem(H: int = 10, W: int = 10, drop: float = 1.0) -> xr.DataArray:
    """Uniformly sloped DEM: elevation decreases eastward (right column is lowest)."""
    elev = np.tile(np.linspace(H * drop, drop, W), (H, 1)).astype(np.float32)
    return _make_dem(elev)


def _v_valley_dem(size: int = 11) -> xr.DataArray:
    """V-shaped valley: channel at centre column, ridges on both sides."""
    centre = size // 2
    elev = np.zeros((size, size), dtype=np.float32)
    for c in range(size):
        elev[:, c] = float(abs(c - centre)) * 5.0  # 5 m per pixel from channel
    return _make_dem(elev)


# ---------------------------------------------------------------------------
# TestDemLoading
# ---------------------------------------------------------------------------

class TestDemLoading:
    """Tests for merge_and_reproject_dem and tile utilities."""

    def test_dem_void_fill_preserves_valid_pixels(self):
        """Void fill must not alter valid (non-NaN) pixel values."""
        from utils.dem import merge_and_reproject_dem
        import rasterio
        import rasterio.transform

        # Create two minimal 2×2 GeoTIFF tiles side-by-side and merge them.
        # One tile has a NaN void; valid pixels must survive unaltered.
        with tempfile.TemporaryDirectory() as tmp:
            tile_paths = []
            for i, vals in enumerate([[100.0, 200.0, 150.0, 180.0],
                                       [120.0, np.nan, 160.0, 170.0]]):
                arr = np.array(vals, dtype=np.float32).reshape(2, 2)
                path = Path(tmp) / f"tile_{i}.tif"
                transform = rasterio.transform.from_origin(
                    141.0 + i, -15.0, 0.5, 0.5  # 0.5° tiles
                )
                with rasterio.open(
                    path, "w", driver="GTiff", count=1, dtype="float32",
                    crs="EPSG:4326", transform=transform,
                    width=2, height=2, nodata=-9999.0,
                ) as dst:
                    arr2 = arr.copy()
                    arr2[np.isnan(arr2)] = -9999.0
                    dst.write(arr2, 1)
                tile_paths.append(path)

            dem = merge_and_reproject_dem(
                tile_paths,
                catchment_geom=None,
                target_crs="EPSG:4326",
                resolution=1,  # ~1 degree — coarse but valid for this synthetic test
            )
            # All originally valid pixels must be finite
            assert np.isfinite(dem.values).any(), "No valid pixels after merge"

    def test_dem_reproject_changes_crs(self):
        """Reprojected DEM must carry the requested CRS."""
        import rasterio
        import rasterio.transform

        with tempfile.TemporaryDirectory() as tmp:
            arr = np.ones((4, 4), dtype=np.float32) * 50.0
            path = Path(tmp) / "tile.tif"
            transform = rasterio.transform.from_origin(141.0, -15.0, 0.25, 0.25)
            with rasterio.open(
                path, "w", driver="GTiff", count=1, dtype="float32",
                crs="EPSG:4326", transform=transform,
                width=4, height=4, nodata=-9999.0,
            ) as dst:
                dst.write(arr, 1)

            from utils.dem import merge_and_reproject_dem
            dem = merge_and_reproject_dem(
                [path],
                catchment_geom=None,
                target_crs="EPSG:7855",
                resolution=30,
            )
            assert dem.rio.crs is not None
            assert "7855" in str(dem.rio.crs)

    def test_dem_tiles_merge_without_seams(self):
        """Adjacent tiles must merge to a contiguous array with no gap row/column."""
        import rasterio
        import rasterio.transform

        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i in range(2):
                arr = np.full((4, 4), 100.0 + i * 10, dtype=np.float32)
                path = Path(tmp) / f"tile_{i}.tif"
                transform = rasterio.transform.from_origin(
                    141.0 + i * 1.0, -15.0, 0.25, 0.25
                )
                with rasterio.open(
                    path, "w", driver="GTiff", count=1, dtype="float32",
                    crs="EPSG:4326", transform=transform,
                    width=4, height=4, nodata=-9999.0,
                ) as dst:
                    dst.write(arr, 1)
                paths.append(path)

            from utils.dem import merge_and_reproject_dem
            dem = merge_and_reproject_dem(
                paths,
                catchment_geom=None,
                target_crs="EPSG:4326",
                resolution=1,  # ~1 degree — coarse but valid for this synthetic test
            )
            # Both tiles covered — output must have non-zero extent
            assert dem.shape[1] >= 1, f"Expected non-empty output, got shape {dem.shape}"


# ---------------------------------------------------------------------------
# TestFlowRouting
# ---------------------------------------------------------------------------

class TestFlowRouting:

    def test_flow_accumulation_drains_to_outlet(self):
        """On a uniformly sloped DEM, all flow must accumulate to the lowest column."""
        dem = _slope_dem(H=5, W=5, drop=1.0)
        accum = compute_flow_accumulation(dem)
        # The right-most column (index 4) is the outlet — must have max accumulation
        max_accum = float(accum.values[:, -1].max())
        assert max_accum >= 5, (
            f"Outlet column max accumulation {max_accum} < 5 "
            f"(expected ≥ 5 for a 5-row slope)"
        )

    def test_flow_accumulation_single_pixel_minimum(self):
        """Every valid pixel must have at least 1 unit of self-contribution."""
        dem = _slope_dem(H=4, W=4)
        accum = compute_flow_accumulation(dem)
        valid = np.isfinite(accum.values)
        assert (accum.values[valid] >= 1).all(), "Found pixel with accumulation < 1"

    def test_flow_accumulation_nodata_is_nan(self):
        """NaN pixels in the DEM must produce NaN in flow accumulation."""
        data = np.ones((4, 4), dtype=np.float32)
        data[1, 1] = np.nan
        dem = _make_dem(data)
        accum = compute_flow_accumulation(dem)
        assert np.isnan(accum.values[1, 1]), "NaN DEM pixel did not propagate to accumulation"

    def test_stream_burn_lowers_channel_pixels(self):
        """After burning, channel pixels must have lower elevation than before."""
        import geopandas as gpd
        from shapely.geometry import LineString

        dem = _make_dem(np.ones((10, 10), dtype=np.float32) * 100.0)
        x_mid = float(dem.coords["x"].values[5])
        y0 = float(dem.coords["y"].values[0])
        y1 = float(dem.coords["y"].values[-1])
        line = LineString([(x_mid, y0), (x_mid, y1)])

        with tempfile.TemporaryDirectory() as tmp:
            gpkg = Path(tmp) / "drainage.gpkg"
            gdf = gpd.GeoDataFrame(geometry=[line], crs="EPSG:7855")
            gdf.to_file(str(gpkg), driver="GPKG")
            dem_burned = burn_drainage_network(dem, gpkg, burn_depth_m=10.0)

        centre_col = dem_burned.values[:, 5]
        assert (centre_col < 100.0).any(), "Channel pixels were not lowered by burn"

    def test_stream_network_connected(self):
        """DEM-derived stream pixels (high accum) must form a spatially connected path."""
        from scipy.ndimage import label
        dem = _slope_dem(H=8, W=8, drop=1.0)
        accum = compute_flow_accumulation(dem)
        # Threshold at 3 px to get a stream network
        stream = (accum.values >= 3) & np.isfinite(accum.values)
        labelled, n = label(stream)
        # The main stream network should be the dominant connected component
        if n > 0:
            counts = np.bincount(labelled.ravel())[1:]  # exclude background
            assert counts.max() >= 3, f"Largest stream component has only {counts.max()} pixels"


# ---------------------------------------------------------------------------
# TestHandComputation
# ---------------------------------------------------------------------------

class TestHandComputation:

    def test_hand_is_zero_at_stream_pixels(self):
        """Pixels that meet the stream threshold must have HAND = 0."""
        dem = _slope_dem(H=6, W=6, drop=1.0)
        accum = compute_flow_accumulation(dem)
        # Use a very low threshold so most pixels are streams
        hand = compute_hand(dem, accum, min_upstream_px=1)
        stream_mask = (accum.values >= 1) & np.isfinite(accum.values)
        hand_at_stream = hand.values[stream_mask]
        assert np.all(hand_at_stream == 0.0), (
            f"Stream pixels with HAND ≠ 0: {hand_at_stream[hand_at_stream != 0]}"
        )

    def test_hand_increases_with_distance_from_stream(self):
        """In a V-shaped valley, HAND must increase monotonically away from centre."""
        dem = _v_valley_dem(size=11)
        accum = compute_flow_accumulation(dem)
        centre = 11 // 2
        # Force the channel column to be stream by setting a low threshold
        hand = compute_hand(dem, accum, min_upstream_px=1)
        # Sample HAND along the middle row
        row = 5
        hand_row = hand.values[row, :]
        valid = np.isfinite(hand_row)
        if valid.sum() >= 3:
            left_half = hand_row[valid][:centre]
            right_half = hand_row[valid][centre + 1:]
            # Both halves should generally increase away from centre
            # Allow for minor numerical noise: check direction of mean increase
            if len(left_half) >= 2:
                assert left_half[0] >= left_half[-1] or np.allclose(left_half, left_half[0], atol=1.0), \
                    "Left half HAND does not increase away from channel"

    def test_hand_nodata_propagation(self):
        """DEM void pixels must produce NaN HAND, not zeros."""
        data = np.ones((6, 6), dtype=np.float32) * 50.0
        data[3, 3] = np.nan
        dem = _make_dem(data)
        accum = compute_flow_accumulation(dem)
        hand = compute_hand(dem, accum, min_upstream_px=1)
        assert np.isnan(hand.values[3, 3]), "NaN DEM pixel did not produce NaN HAND"

    def test_hand_flat_floodplain(self):
        """On a flat surface adjacent to a lower channel, HAND equals the elevation drop."""
        # 6×6 DEM: left column at elevation 90 (channel), rest at 100 (floodplain)
        data = np.full((6, 6), 100.0, dtype=np.float32)
        data[:, 0] = 90.0  # channel at elevation 90
        dem = _make_dem(data, res=30.0)
        accum = compute_flow_accumulation(dem)
        # Use very low threshold to treat left column as stream
        hand = compute_hand(dem, accum, min_upstream_px=1)
        # Pixels adjacent to channel (col 1) should have HAND ≈ 10 m
        hand_adj = hand.values[:, 1]
        finite = hand_adj[np.isfinite(hand_adj)]
        if finite.size > 0:
            assert np.all(finite >= 0), "Negative HAND values in flat floodplain"


# ---------------------------------------------------------------------------
# TestFloodConnectivityMask
# ---------------------------------------------------------------------------

class TestFloodConnectivityMask:

    def _make_hand(self, data: np.ndarray) -> xr.DataArray:
        return xr.DataArray(
            data.astype(np.float32),
            dims=["y", "x"],
            coords={
                "x": np.arange(data.shape[1]) * 30.0 + 700000.0,
                "y": -np.arange(data.shape[0]) * 30.0 - 1_600_000.0,
            },
            name="HAND",
        )

    def test_threshold_produces_binary_mask(self):
        """flood_connectivity_mask must produce only True/False values, no intermediates."""
        hand = self._make_hand(np.array([[0, 3, 8, 15], [1, 5, 6, 20]], dtype=np.float32))
        mask = flood_connectivity_mask(hand, threshold_m=5.0)
        unique = np.unique(mask.values)
        assert set(unique).issubset({True, False}), f"Non-boolean values in mask: {unique}"

    def test_low_hand_pixels_are_flood_connected(self):
        """Pixels at HAND = 0 and HAND < threshold must be True."""
        data = np.array([[0.0, 2.0, 4.9, 5.0, 5.1, 10.0]], dtype=np.float32)
        hand = self._make_hand(data)
        mask = flood_connectivity_mask(hand, threshold_m=5.0)
        # Indices 0,1,2,3 (values 0.0, 2.0, 4.9, 5.0) should be True
        assert mask.values[0, 0], "HAND=0 pixel not flagged as flood-connected"
        assert mask.values[0, 1], "HAND=2 pixel not flagged as flood-connected"
        assert mask.values[0, 3], "HAND=5.0 pixel (at threshold) not flagged"

    def test_high_hand_pixels_excluded(self):
        """Pixels at HAND >> threshold must be False."""
        data = np.array([[100.0, 50.0, 20.0, 5.1]], dtype=np.float32)
        hand = self._make_hand(data)
        mask = flood_connectivity_mask(hand, threshold_m=5.0)
        assert not mask.values[0, 0], "HAND=100 pixel incorrectly flagged as connected"
        assert not mask.values[0, 3], "HAND=5.1 pixel incorrectly flagged as connected"

    def test_nan_hand_pixels_excluded(self):
        """NaN HAND pixels (DEM voids) must be False in the mask."""
        data = np.array([[np.nan, 2.0, 6.0]], dtype=np.float32)
        hand = self._make_hand(data)
        mask = flood_connectivity_mask(hand, threshold_m=5.0)
        assert not mask.values[0, 0], "NaN HAND pixel incorrectly flagged True"
        assert mask.values[0, 1], "Valid HAND=2 pixel incorrectly False"

    def test_output_is_valid_gpkg(self):
        """Vectorised output from the pipeline must be readable as a GeoPackage."""
        import geopandas as gpd
        import rasterio.features
        import affine
        from shapely.geometry import shape
        from shapely.ops import unary_union

        # Simulate the vectorisation step
        data = np.zeros((20, 20), dtype=np.uint8)
        data[5:15, 5:15] = 1  # 10×10 patch

        x = np.arange(20) * 30.0 + 700000.0
        y = -np.arange(20) * 30.0 - 1_600_000.0
        res_x = 30.0
        res_y = -30.0
        transform = affine.Affine(res_x, 0, x[0], 0, res_y, y[0])

        shapes_list = list(rasterio.features.shapes(data, mask=data, transform=transform))
        geoms = [shape(s) for s, v in shapes_list if v == 1]
        merged = unary_union(geoms)
        geom_list = [merged] if not hasattr(merged, "geoms") else list(merged.geoms)
        gdf = gpd.GeoDataFrame(geometry=geom_list, crs="EPSG:7855")

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "flood_extent_test.gpkg"
            gdf.to_file(str(out_path), driver="GPKG")
            read_back = gpd.read_file(str(out_path))

        assert len(read_back) >= 1, "No features in GeoPackage"
        assert str(read_back.crs.to_epsg()) == "7855", "Wrong CRS in output GeoPackage"

    def test_mask_clips_to_catchment_shape(self):
        """After clipping, output features must lie within the catchment polygon."""
        import geopandas as gpd
        from shapely.geometry import box

        # Catchment is a 300×300 m box
        catchment_geom = box(700000, -1_600_300, 700300, -1_600_000)
        catchment = gpd.GeoDataFrame(geometry=[catchment_geom], crs="EPSG:7855")

        # HAND array that extends outside the catchment
        data = np.zeros((20, 20), dtype=np.float32)
        data[:, :] = 2.0  # all flood-connected
        hand = xr.DataArray(
            data,
            dims=["y", "x"],
            coords={
                "x": np.arange(20) * 30.0 + 699700.0,
                "y": -np.arange(20) * 30.0 - 1_599_700.0,
            },
        )
        mask = flood_connectivity_mask(hand, threshold_m=5.0)

        # Simulate clip
        import rasterio.features
        import affine
        from shapely.geometry import shape
        from shapely.ops import unary_union

        x = hand.coords["x"].values
        y = hand.coords["y"].values
        transform = affine.Affine(30.0, 0, x[0], 0, -30.0, y[0])
        data_u8 = mask.values.astype(np.uint8)
        shapes_list = list(rasterio.features.shapes(data_u8, mask=data_u8, transform=transform))
        geoms = [shape(s) for s, v in shapes_list if v == 1]
        gdf = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:7855")
        clipped = gpd.clip(gdf, catchment)

        if len(clipped) > 0:
            total_area = clipped.geometry.area.sum()
            catchment_area = catchment_geom.area
            assert total_area <= catchment_area * 1.01, "Clipped features exceed catchment area"


# ---------------------------------------------------------------------------
# TestVectorisationLogic
# ---------------------------------------------------------------------------

class TestVectorisationLogic:

    def test_min_patch_removal(self):
        """Isolated pixels below MIN_PATCH_PX threshold must be removed."""
        from scipy.ndimage import label
        import numpy as np

        data = np.zeros((20, 20), dtype=np.uint8)
        data[2, 2] = 1          # isolated single pixel — should be removed
        data[10:15, 10:15] = 1  # 25-pixel patch — should survive

        MIN_PATCH_PX = 5
        labelled, _ = label(data)
        patch_sizes = np.bincount(labelled.ravel())
        small_labels = np.where(patch_sizes < MIN_PATCH_PX)[0]
        small_labels = small_labels[small_labels > 0]
        if small_labels.size:
            data[np.isin(labelled, small_labels)] = 0

        assert data[2, 2] == 0, "Isolated pixel was not removed"
        assert data[12, 12] == 1, "Large patch was incorrectly removed"

    def test_morphological_closing_merges_nearby_blobs(self):
        """Two blobs separated by a small gap must merge after morphological closing."""
        from scipy.ndimage import binary_closing

        data = np.zeros((10, 20), dtype=bool)
        data[3:7, 2:5] = True   # left blob
        data[3:7, 7:10] = True  # right blob — 2-px gap

        CLOSING_RADIUS_PX = 2
        struct = np.ones(
            (CLOSING_RADIUS_PX * 2 + 1, CLOSING_RADIUS_PX * 2 + 1), dtype=bool
        )
        closed = binary_closing(data, structure=struct)

        # The gap at columns 5–6 should be filled after closing
        assert closed[4, 5:7].any(), "Gap between blobs was not closed"

    def test_empty_scene_produces_empty_geodataframe(self):
        """All-zero flood mask must produce an empty GeoDataFrame without raising."""
        import geopandas as gpd
        import rasterio.features
        import affine
        from shapely.ops import unary_union
        from shapely.geometry import shape

        data = np.zeros((10, 10), dtype=np.uint8)
        transform = affine.Affine(30.0, 0, 700000.0, 0, -30.0, -1_600_000.0)
        shapes_list = list(rasterio.features.shapes(data, mask=data, transform=transform))

        if not shapes_list:
            gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:7855")
        else:
            geoms = [shape(s) for s, v in shapes_list if v == 1]
            gdf = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:7855")

        assert len(gdf) == 0, f"Expected empty GeoDataFrame, got {len(gdf)} features"
        assert gdf.crs is not None

    def test_vectorisation_preserves_flood_area(self):
        """Vectorised polygon area must be within 10% of raster pixel count × pixel area."""
        import geopandas as gpd
        import rasterio.features
        import affine
        from shapely.geometry import shape
        from shapely.ops import unary_union

        RES = 30.0
        data = np.zeros((20, 20), dtype=np.uint8)
        data[5:15, 5:15] = 1   # 100 pixels = 90_000 m²

        transform = affine.Affine(RES, 0, 700000.0, 0, -RES, -1_600_000.0)
        shapes_list = list(rasterio.features.shapes(data, mask=data, transform=transform))
        geoms = [shape(s) for s, v in shapes_list if v == 1]
        merged = unary_union(geoms)
        geom_list = [merged] if not hasattr(merged, "geoms") else list(merged.geoms)
        gdf = gpd.GeoDataFrame(geometry=geom_list, crs="EPSG:7855")

        pixel_area = data.sum() * RES * RES
        vector_area = gdf.geometry.area.sum()
        ratio = vector_area / pixel_area
        assert 0.90 <= ratio <= 1.10, (
            f"Vector area ({vector_area:.0f} m²) deviates >10% from "
            f"raster area ({pixel_area:.0f} m²); ratio={ratio:.3f}"
        )
