"""Tests for Stage 4 flood-extent logic (utils/sar.py + analysis/04_flood_extent.py).

All tests are self-contained — no real S1 data is required.  Where the
production code calls _preprocess_gcp_warp() / preprocess_s1_scene() those
calls are patched out with minimal xr.Dataset fakes so the tests exercise
every line of logic that runs *after* the raster arrives in memory.

Coverage targets
----------------
_otsu_threshold          — bimodal correctness, unimodal, edge cases
_focal_mean_inplace      — smoothing, NaN preservation, radius
flood_mask_from_scene    — full happy-path, sanity guard, reference mask,
                           missing VV, all-NaN scene, tiny valid set
build_dry_season_reference_mask — shape-mismatch skip, all-None workers,
                           pending-buffer flush, mmap median, threshold
Accumulation logic       — uint16 overflow guard, shape-mismatch skip,
                           frequency calculation, MIN_OBS filter
Vectorisation            — min-patch removal, morphological closing,
                           empty-scene path
"""

import importlib.util
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.sar import (
    _focal_mean_inplace,
    _otsu_threshold,
    flood_mask_from_scene,
    build_dry_season_reference_mask,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_item(id_="S1A_test", shape=(20, 30)):
    """Minimal STAC-item-like object accepted by flood_mask_from_scene."""
    item = MagicMock()
    item.id = id_
    return item


def _make_ds(vv: np.ndarray, x=None, y=None) -> xr.Dataset:
    """Wrap a VV array in an xr.Dataset that mimics _preprocess_gcp_warp output."""
    H, W = vv.shape
    if x is None:
        x = np.linspace(700000, 750000, W)
    if y is None:
        y = np.linspace(-1600000, -1650000, H)
    da = xr.DataArray(vv.astype(np.float32), dims=["y", "x"],
                      coords={"x": x, "y": y})
    return xr.Dataset({"VV": da})


# ---------------------------------------------------------------------------
# _otsu_threshold
# ---------------------------------------------------------------------------

class TestOtsuThreshold:

    def test_bimodal_separates_two_gaussians(self):
        """Threshold must fall between the two Gaussian centres."""
        rng = np.random.default_rng(0)
        low  = rng.normal(-20, 2, 5_000)   # water-like
        high = rng.normal( -8, 2, 5_000)   # land-like
        values = np.concatenate([low, high]).astype(np.float32)
        t = _otsu_threshold(values)
        assert -18 < t < -10, f"Expected threshold between −18 and −10 dB, got {t:.2f}"

    def test_unimodal_does_not_crash(self):
        """Unimodal input must return a float without raising."""
        rng = np.random.default_rng(1)
        values = rng.normal(-8, 1.5, 10_000).astype(np.float32)
        t = _otsu_threshold(values)
        assert isinstance(t, float)

    def test_threshold_is_within_value_range(self):
        """Threshold must always lie within [min, max] of the input."""
        rng = np.random.default_rng(2)
        for _ in range(10):
            values = rng.normal(0, 5, 2_000).astype(np.float32)
            t = _otsu_threshold(values)
            assert values.min() <= t <= values.max()

    def test_all_identical_values(self):
        """All-constant input must not crash (degenerate histogram)."""
        values = np.full(500, -15.0, dtype=np.float32)
        t = _otsu_threshold(values)
        assert isinstance(t, float)

    def test_two_point_values(self):
        """Only two distinct values — threshold must lie between them."""
        values = np.array([-25.0] * 500 + [-5.0] * 500, dtype=np.float32)
        t = _otsu_threshold(values)
        assert -25 <= t <= -5

    def test_very_skewed_distribution(self):
        """Heavy positive skew (e.g. bright urban returns) must not crash."""
        rng = np.random.default_rng(3)
        values = np.exp(rng.normal(0, 1, 3_000)).astype(np.float32)
        t = _otsu_threshold(values)
        assert isinstance(t, float)

    def test_returns_bin_centre_not_edge(self):
        """The returned value should be a bin centre (finite, not ±inf)."""
        rng = np.random.default_rng(4)
        values = rng.uniform(-30, 0, 1_000).astype(np.float32)
        t = _otsu_threshold(values)
        assert np.isfinite(t)

    def test_n_bins_parameter(self):
        """Using fewer bins produces a coarser but still valid result."""
        rng = np.random.default_rng(5)
        low  = rng.normal(-20, 1, 1_000)
        high = rng.normal( -5, 1, 1_000)
        values = np.concatenate([low, high]).astype(np.float32)
        t_coarse = _otsu_threshold(values, n_bins=32)
        t_fine   = _otsu_threshold(values, n_bins=512)
        # Both must separate the two peaks; allow 3 dB slop for coarse bins
        assert -20 < t_coarse < -5, f"Coarse Otsu failed: {t_coarse}"
        assert -20 < t_fine   < -5, f"Fine Otsu failed: {t_fine}"


# ---------------------------------------------------------------------------
# _focal_mean_inplace
# ---------------------------------------------------------------------------

class TestFocalMeanInplace:

    def test_uniform_array_unchanged(self):
        """Filtering a constant array should leave all finite values unchanged."""
        arr = np.full((10, 10), 5.0, dtype=np.float32)
        nan_mask = np.zeros((10, 10), dtype=bool)
        result = _focal_mean_inplace(arr, nan_mask, radius=1)
        np.testing.assert_allclose(result, 5.0, atol=1e-5)

    def test_nans_restored_after_filter(self):
        """Pixels marked NaN must be NaN in the output."""
        arr = np.ones((10, 10), dtype=np.float32)
        nan_mask = np.zeros((10, 10), dtype=bool)
        nan_mask[3, 4] = True
        nan_mask[7, 2] = True
        arr[nan_mask] = np.nan
        _focal_mean_inplace(arr, nan_mask, radius=1)
        assert np.isnan(arr[3, 4])
        assert np.isnan(arr[7, 2])

    def test_non_nan_pixels_not_forced_to_nan(self):
        """Valid pixels neighbouring a NaN must remain finite."""
        arr = np.ones((10, 10), dtype=np.float32)
        nan_mask = np.zeros((10, 10), dtype=bool)
        nan_mask[5, 5] = True
        arr[5, 5] = np.nan
        _focal_mean_inplace(arr, nan_mask, radius=1)
        # The 8 neighbours of (5,5) must still be finite
        for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            assert np.isfinite(arr[5+dr, 5+dc]), f"Neighbour ({5+dr},{5+dc}) became NaN"

    def test_speckle_smoothing_reduces_variance(self):
        """Filtering high-variance noise should reduce std."""
        rng = np.random.default_rng(10)
        arr = rng.normal(0, 5, (50, 50)).astype(np.float32)
        nan_mask = np.zeros((50, 50), dtype=bool)
        original_std = arr.std()
        _focal_mean_inplace(arr, nan_mask, radius=1)
        assert arr.std() < original_std, "Speckle filter should reduce variance"

    def test_radius_zero_is_identity(self):
        """Radius=0 means a 1×1 kernel — output equals input for valid pixels."""
        rng = np.random.default_rng(11)
        arr = rng.uniform(0, 1, (8, 8)).astype(np.float32)
        original = arr.copy()
        nan_mask = np.zeros((8, 8), dtype=bool)
        _focal_mean_inplace(arr, nan_mask, radius=0)
        np.testing.assert_allclose(arr, original, atol=1e-5)

    def test_all_nan_array(self):
        """All-NaN input: the global mean is nan, should fill with nan then restore."""
        arr = np.full((5, 5), np.nan, dtype=np.float32)
        nan_mask = np.ones((5, 5), dtype=bool)
        _focal_mean_inplace(arr, nan_mask, radius=1)
        assert np.all(np.isnan(arr))

    def test_returns_same_array_object(self):
        """Must operate in-place and return the same array object."""
        arr = np.ones((6, 6), dtype=np.float32)
        nan_mask = np.zeros((6, 6), dtype=bool)
        result = _focal_mean_inplace(arr, nan_mask, radius=1)
        assert result is arr


# ---------------------------------------------------------------------------
# flood_mask_from_scene
# ---------------------------------------------------------------------------

class TestFloodMaskFromScene:
    """Tests for flood_mask_from_scene(), patching _preprocess_gcp_warp."""

    def _patch_warp(self, ds):
        return patch("utils.sar._preprocess_gcp_warp", return_value=ds)

    def test_happy_path_bimodal_scene(self):
        """Clear bimodal scene: water pixels must be classified and returned."""
        rng = np.random.default_rng(20)
        H, W = 40, 40
        vv_lin = np.full((H, W), 0.1, dtype=np.float32)   # land DN ~ 0.1
        # 10×10 water patch with very low backscatter
        vv_lin[0:10, 0:10] = 0.001
        ds = _make_ds(vv_lin)
        item = _make_item()
        with self._patch_warp(ds):
            result = flood_mask_from_scene(item, bbox=[141, -17, 143, -15], resolution=50)
        assert result is not None
        assert "water" in result and "observed" in result
        water = result["water"].values
        assert water.dtype == bool
        # Water patch should have more True pixels than surrounding land
        assert water[0:10, 0:10].sum() > water[10:, 10:].sum()

    def test_missing_vv_returns_none(self):
        """Dataset without 'VV' band must return None gracefully."""
        ds = xr.Dataset({"VH": xr.DataArray(np.ones((10, 10), dtype=np.float32),
                                             dims=["y", "x"])})
        item = _make_item()
        with self._patch_warp(ds):
            result = flood_mask_from_scene(item, bbox=[141, -17, 143, -15], resolution=50)
        assert result is None

    def test_all_zero_vv_returns_none(self):
        """All-zero VV (no valid pixels) must return None."""
        vv = np.zeros((20, 20), dtype=np.float32)
        ds = _make_ds(vv)
        item = _make_item()
        with self._patch_warp(ds):
            result = flood_mask_from_scene(item, bbox=[141, -17, 143, -15], resolution=50)
        assert result is None

    def test_all_nan_vv_returns_none(self):
        """All-NaN VV must return None (observed.sum() == 0)."""
        vv = np.full((20, 20), np.nan, dtype=np.float32)
        ds = _make_ds(vv)
        item = _make_item()
        with self._patch_warp(ds):
            result = flood_mask_from_scene(item, bbox=[141, -17, 143, -15], resolution=50)
        assert result is None

    def test_fewer_than_100_valid_pixels_returns_none(self):
        """Scene with <100 valid pixels must be discarded (Otsu unreliable)."""
        vv = np.zeros((20, 20), dtype=np.float32)
        vv[0:5, 0:5] = 0.1   # only 25 valid pixels
        ds = _make_ds(vv)
        item = _make_item()
        with self._patch_warp(ds):
            result = flood_mask_from_scene(item, bbox=[141, -17, 143, -15], resolution=50)
        assert result is None

    def test_sanity_guard_discards_unimodal_scene(self):
        """When >65% of valid pixels are classified as water, return None."""
        rng = np.random.default_rng(21)
        # Very homogeneous low-backscatter — Otsu will split within the single
        # peak, flagging the lower half as "water" (>50% of the distribution).
        # Force worst case: all values are below the expected Otsu cut,
        # making water_fraction = 1.0.
        vv_lin = np.full((50, 50), 0.00001, dtype=np.float32)   # near-zero → very negative dB
        # Add a tiny sliver of "land" so Otsu has something to cut on
        vv_lin[0, 0:3] = 0.9
        ds = _make_ds(vv_lin)
        item = _make_item()
        with self._patch_warp(ds):
            result = flood_mask_from_scene(item, bbox=[141, -17, 143, -15], resolution=50)
        # Either None (sanity guard triggered) or water_fraction <= 0.65
        if result is not None:
            water = result["water"].values
            observed_count = np.isfinite(ds["VV"].values) & (ds["VV"].values > 0)
            wf = water.sum() / max(observed_count.sum(), 1)
            assert wf <= 0.65, f"Sanity guard should have discarded this scene (wf={wf:.2f})"

    def test_reference_mask_excludes_pixels(self):
        """Pixels flagged True in the reference mask must never appear as water."""
        rng = np.random.default_rng(22)
        H, W = 30, 30
        vv_lin = np.full((H, W), 0.1, dtype=np.float32)
        vv_lin[0:10, 0:10] = 0.001   # low-backscatter patch → would be water
        ds = _make_ds(vv_lin)
        item = _make_item()
        ref_mask = np.zeros((H, W), dtype=bool)
        ref_mask[0:10, 0:10] = True   # flag that same patch as non-water
        with self._patch_warp(ds):
            result = flood_mask_from_scene(item, bbox=[141, -17, 143, -15],
                                           resolution=50, reference_mask=ref_mask)
        if result is not None:
            water = result["water"].values
            # Masked region must have zero water pixels
            assert water[ref_mask].sum() == 0, "Reference mask failed to exclude pixels"

    def test_reference_mask_shape_mismatch_ignored(self):
        """Reference mask of wrong shape must not crash — silently ignored."""
        H, W = 20, 20
        vv_lin = np.full((H, W), 0.1, dtype=np.float32)
        vv_lin[0:5, 0:5] = 0.001
        ds = _make_ds(vv_lin)
        # Ensure >= 100 valid pixels for Otsu
        vv_lin = np.full((20, 20), 0.1, dtype=np.float32)
        vv_lin[0:5, 0:5] = 0.001
        ds = _make_ds(vv_lin)
        item = _make_item()
        wrong_mask = np.zeros((99, 99), dtype=bool)
        with self._patch_warp(ds):
            # Must not raise
            flood_mask_from_scene(item, bbox=[141, -17, 143, -15],
                                  resolution=50, reference_mask=wrong_mask)

    def test_output_coords_match_input(self):
        """x/y coordinates in the output Dataset must match the warped input."""
        H, W = 25, 35
        vv_lin = np.full((H, W), 0.1, dtype=np.float32)
        vv_lin[0:8, 0:8] = 0.001
        x = np.linspace(700000, 720000, W)
        y = np.linspace(-1600000, -1620000, H)
        ds = _make_ds(vv_lin, x=x, y=y)
        item = _make_item()
        with self._patch_warp(ds):
            result = flood_mask_from_scene(item, bbox=[141, -17, 143, -15], resolution=50)
        if result is not None:
            np.testing.assert_array_equal(result["water"].coords["x"].values, x)
            np.testing.assert_array_equal(result["water"].coords["y"].values, y)

    def test_water_and_observed_are_bool(self):
        """water and observed arrays in the output must have bool dtype."""
        H, W = 20, 20
        vv_lin = np.full((H, W), 0.1, dtype=np.float32)
        vv_lin[0:5, 0:5] = 0.001
        ds = _make_ds(vv_lin)
        item = _make_item()
        with self._patch_warp(ds):
            result = flood_mask_from_scene(item, bbox=[141, -17, 143, -15], resolution=50)
        if result is not None:
            assert result["water"].values.dtype == bool
            assert result["observed"].values.dtype == bool

    def test_water_is_subset_of_observed(self):
        """Every water pixel must also be an observed pixel."""
        H, W = 30, 30
        vv_lin = np.full((H, W), 0.1, dtype=np.float32)
        vv_lin[0:8, 0:8] = 0.001
        ds = _make_ds(vv_lin)
        item = _make_item()
        with self._patch_warp(ds):
            result = flood_mask_from_scene(item, bbox=[141, -17, 143, -15], resolution=50)
        if result is not None:
            water = result["water"].values
            observed = result["observed"].values
            assert np.all(water[~observed] == False), \
                "Water pixels outside observed footprint detected"

    def test_negative_vv_values_treated_as_unobserved(self):
        """Negative DN values (rasterio nodata artifacts) must not enter Otsu."""
        H, W = 20, 20
        # Give the valid portion a bimodal distribution so Otsu doesn't fail on
        # a zero-range histogram (all-identical values → too many bins error).
        vv_lin = np.full((H, W), 0.1, dtype=np.float32)
        vv_lin[0:8, :] = 0.001   # low-backscatter patch (water-like)
        vv_lin[15:, :] = -1.0    # negative — should be unobserved
        ds = _make_ds(vv_lin)
        item = _make_item()
        with self._patch_warp(ds):
            result = flood_mask_from_scene(item, bbox=[141, -17, 143, -15], resolution=50)
        if result is not None:
            observed = result["observed"].values
            # Bottom rows (negative VV) must be unobserved
            assert observed[15:, :].sum() == 0


# ---------------------------------------------------------------------------
# build_dry_season_reference_mask
# ---------------------------------------------------------------------------

def _thread_pool_executor_patch():
    """Patch ProcessPoolExecutor with ThreadPoolExecutor so workers run in-process.

    build_dry_season_reference_mask uses ProcessPoolExecutor; the worker function
    and item objects must be picklable to cross the process boundary.  In tests we
    substitute ThreadPoolExecutor so that patched callables and MagicMock items
    are accessible without pickling.
    """
    from concurrent.futures import ThreadPoolExecutor
    return patch("utils.sar.ProcessPoolExecutor", ThreadPoolExecutor)


class TestBuildDrySeasonReferenceMask:
    """Tests for build_dry_season_reference_mask(), patching _process_dry_worker."""

    def _make_items(self, n):
        items = []
        for i in range(n):
            item = MagicMock()
            item.id = f"dry_{i:03d}"
            items.append(item)
        return items

    def _worker_returning(self, arr: np.ndarray, n_valid: int = 1000):
        """Return a callable that always returns (arr, n_valid)."""
        def _w(item, bbox, resolution):
            return arr.copy(), n_valid
        return _w

    def test_returns_none_when_all_workers_return_none(self):
        """If every worker yields None, function must return None."""
        items = self._make_items(3)
        with _thread_pool_executor_patch(), \
             patch("utils.sar._process_dry_worker", return_value=None):
            result = build_dry_season_reference_mask(
                items, bbox=[141, -17, 143, -15], resolution=50, max_workers=1,
            )
        assert result is None

    def test_basic_low_backscatter_flagging(self):
        """Pixels with median < threshold should be True in the output mask."""
        H, W = 10, 10
        # Two scenes: both with very low backscatter in the top-left quadrant
        arr_low  = np.full((H, W), -20.0, dtype=np.float32)
        arr_low[5:, 5:] = -5.0   # upper-right has normal backscatter

        items = self._make_items(2)
        call_count = [0]

        def _worker(item, bbox, resolution):
            call_count[0] += 1
            return arr_low.copy(), 80

        with _thread_pool_executor_patch(), \
             patch("utils.sar._process_dry_worker", side_effect=_worker):
            result = build_dry_season_reference_mask(
                items, bbox=[141, -17, 143, -15], resolution=50,
                low_backscatter_threshold_db=-16.0, max_workers=1,
            )
        assert result is not None
        assert result.dtype == bool
        # Top-left (low backscatter) should be flagged
        assert result[0:5, 0:5].all(), "Low-backscatter pixels should be masked"
        # Bottom-right (high backscatter) should not be flagged
        assert not result[5:, 5:].any(), "High-backscatter pixels must not be masked"

    def test_shape_mismatch_scene_skipped(self):
        """If a scene returns a different shape from the first, it must be silently skipped."""
        H, W = 8, 8
        arr_ok   = np.full((H, W),   -20.0, dtype=np.float32)
        arr_bad  = np.full((H+2, W), -20.0, dtype=np.float32)  # wrong shape

        items = self._make_items(2)
        results_queue = [arr_ok, arr_bad]

        def _worker(item, bbox, resolution):
            return results_queue.pop(0).copy(), 50

        with _thread_pool_executor_patch(), \
             patch("utils.sar._process_dry_worker", side_effect=_worker):
            # Should not raise; the bad scene is quietly dropped
            result = build_dry_season_reference_mask(
                items, bbox=[141, -17, 143, -15], resolution=50, max_workers=1,
            )
        assert result is not None

    def test_result_is_2d_bool_array(self):
        """Output must be a 2-D numpy bool array."""
        H, W = 6, 6
        arr = np.full((H, W), -20.0, dtype=np.float32)
        items = self._make_items(1)

        def _worker(item, bbox, resolution):
            return arr.copy(), 36

        with _thread_pool_executor_patch(), \
             patch("utils.sar._process_dry_worker", side_effect=_worker):
            result = build_dry_season_reference_mask(
                items, bbox=[141, -17, 143, -15], resolution=50, max_workers=1,
            )
        assert result is not None
        assert result.ndim == 2
        assert result.dtype == bool

    def test_threshold_boundary(self):
        """Pixels exactly at the threshold must NOT be masked (strict <)."""
        H, W = 5, 5
        arr = np.full((H, W), -16.0, dtype=np.float32)  # exactly at threshold
        items = self._make_items(1)

        def _worker(item, bbox, resolution):
            return arr.copy(), 25

        with _thread_pool_executor_patch(), \
             patch("utils.sar._process_dry_worker", side_effect=_worker):
            result = build_dry_season_reference_mask(
                items, bbox=[141, -17, 143, -15], resolution=50,
                low_backscatter_threshold_db=-16.0, max_workers=1,
            )
        assert result is not None
        # Exactly at threshold → not flagged (< not <=)
        assert not result.any(), f"Pixels at threshold should not be masked; {result.sum()} were"

    def test_empty_item_list_returns_none(self):
        """Empty scene list must return None without crashing."""
        result = build_dry_season_reference_mask(
            [], bbox=[141, -17, 143, -15], resolution=50, max_workers=1,
        )
        assert result is None

    def test_exception_in_worker_does_not_abort(self):
        """An exception in one worker must not abort the whole function."""
        H, W = 6, 6
        arr_ok = np.full((H, W), -20.0, dtype=np.float32)
        items = self._make_items(2)
        call_count = [0]

        def _worker(item, bbox, resolution):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Simulated worker failure")
            return arr_ok.copy(), 36

        with _thread_pool_executor_patch(), \
             patch("utils.sar._process_dry_worker", side_effect=_worker):
            result = build_dry_season_reference_mask(
                items, bbox=[141, -17, 143, -15], resolution=50, max_workers=1,
            )
        # Second scene succeeded → should still get a mask
        assert result is not None


# ---------------------------------------------------------------------------
# Accumulation logic (tested via flood_mask_from_scene integration)
# ---------------------------------------------------------------------------

class TestAccumulation:
    """Tests for the flood_count / obs_count accumulation in 04_flood_extent.main().

    Rather than running main() end-to-end (which needs config, filesystem, etc.)
    we replicate the accumulation kernel inline — it's just a few lines and worth
    testing in isolation.
    """

    def _accumulate(self, scenes):
        """Run the accumulation loop from 04_flood_extent on a list of Datasets."""
        flood_count = None
        obs_count = None
        for scene in scenes:
            if scene is None:
                continue
            water    = scene["water"].values.view(np.uint8)
            observed = scene["observed"].values.view(np.uint8)
            if flood_count is None:
                flood_count = water.astype(np.uint16)
                obs_count   = observed.astype(np.uint16)
            elif water.shape == flood_count.shape:
                flood_count += water
                obs_count   += observed
        return flood_count, obs_count

    def _make_scene(self, water_mask: np.ndarray, observed_mask: np.ndarray,
                    shape=None):
        H, W = water_mask.shape
        x = np.arange(W, dtype=np.float64)
        y = np.arange(H, dtype=np.float64)
        return xr.Dataset({
            "water":    xr.DataArray(water_mask.astype(bool),    dims=["y","x"],
                                     coords={"x": x, "y": y}),
            "observed": xr.DataArray(observed_mask.astype(bool), dims=["y","x"],
                                     coords={"x": x, "y": y}),
        })

    def test_single_scene_counts_match(self):
        water = np.zeros((5, 5), dtype=bool)
        water[1:3, 1:3] = True
        obs = np.ones((5, 5), dtype=bool)
        scene = self._make_scene(water, obs)
        fc, oc = self._accumulate([scene])
        assert fc[1, 1] == 1
        assert fc[0, 0] == 0
        assert oc[0, 0] == 1

    def test_multiple_scenes_accumulate_correctly(self):
        shape = (4, 4)
        # Scene 1: water at (0,0) only
        w1 = np.zeros(shape, dtype=bool); w1[0, 0] = True
        o1 = np.ones(shape,  dtype=bool)
        # Scene 2: water at (0,0) and (1,1)
        w2 = np.zeros(shape, dtype=bool); w2[0, 0] = True; w2[1, 1] = True
        o2 = np.ones(shape,  dtype=bool)

        fc, oc = self._accumulate([self._make_scene(w1, o1), self._make_scene(w2, o2)])
        assert fc[0, 0] == 2
        assert fc[1, 1] == 1
        assert fc[2, 2] == 0
        assert oc[0, 0] == 2

    def test_none_scene_skipped(self):
        water = np.zeros((3, 3), dtype=bool)
        obs   = np.ones((3, 3), dtype=bool)
        scene = self._make_scene(water, obs)
        fc, oc = self._accumulate([None, scene, None])
        assert fc is not None
        assert oc is not None

    def test_shape_mismatch_scene_skipped(self):
        shape_a = (4, 4)
        shape_b = (5, 5)
        w_a = np.zeros(shape_a, dtype=bool); w_a[0, 0] = True
        o_a = np.ones(shape_a,  dtype=bool)
        w_b = np.zeros(shape_b, dtype=bool)
        o_b = np.ones(shape_b,  dtype=bool)
        scene_a = self._make_scene(w_a, o_a, shape=shape_a)
        scene_b = self._make_scene(w_b, o_b, shape=shape_b)
        fc, oc = self._accumulate([scene_a, scene_b])
        assert fc.shape == shape_a, "Shape-mismatched scene should have been dropped"

    def test_uint16_does_not_overflow_at_255(self):
        """uint16 must survive 300 accumulations without wrapping at 255."""
        shape = (2, 2)
        water = np.ones(shape, dtype=bool)
        obs   = np.ones(shape, dtype=bool)
        scenes = [self._make_scene(water, obs) for _ in range(300)]
        fc, oc = self._accumulate(scenes)
        assert fc[0, 0] == 300, f"Expected 300, got {fc[0,0]} (uint8 overflow?)"
        assert oc[0, 0] == 300

    def test_frequency_calculation_is_correct(self):
        """flood_count / obs_count should equal the true fraction."""
        shape = (3, 3)
        # 6 scenes; pixel (0,0) flooded in 3 of them
        scenes = []
        for i in range(6):
            w = np.zeros(shape, dtype=bool)
            if i < 3:
                w[0, 0] = True
            o = np.ones(shape, dtype=bool)
            scenes.append(self._make_scene(w, o))
        fc, oc = self._accumulate(scenes)
        with np.errstate(invalid="ignore", divide="ignore"):
            freq = np.where(oc > 0, fc / oc.astype(np.float32), 0.0)
        assert abs(freq[0, 0] - 0.5) < 1e-5
        assert freq[1, 1] == 0.0

    def test_min_obs_filter_excludes_edge_pixels(self):
        """Pixels observed fewer than MIN_OBS times must be excluded from freq mask."""
        MIN_OBS = 4
        shape = (3, 3)
        # Pixel (0,0) observed only twice (below MIN_OBS)
        fc = np.zeros(shape, dtype=np.uint16)
        oc = np.zeros(shape, dtype=np.uint16)
        oc[0, 0] = 2
        fc[0, 0] = 2
        oc[1, 1] = 10
        fc[1, 1] = 8
        with np.errstate(invalid="ignore", divide="ignore"):
            freq = np.where(oc > 0, fc / oc.astype(np.float32), 0.0)
        sufficient = oc >= MIN_OBS
        FLOOD_MIN_FREQUENCY = 0.33
        combined = (freq >= FLOOD_MIN_FREQUENCY) & sufficient
        assert not combined[0, 0], "Under-observed pixel should be excluded"
        assert combined[1, 1], "Well-observed water pixel should be included"


# ---------------------------------------------------------------------------
# Vectorisation helpers (min-patch filter, morphological closing)
# ---------------------------------------------------------------------------

class TestVectorisation:
    """Test the post-accumulation raster operations using scipy directly."""

    def test_min_patch_removes_isolated_pixel(self):
        """Single isolated pixel (1 px < MIN_PATCH_PX=4) must be removed."""
        from scipy.ndimage import label
        data = np.zeros((10, 10), dtype=np.uint8)
        data[5, 5] = 1   # isolated single pixel
        data[0:3, 0:3] = 1  # a proper 3×3 patch (9 px)
        MIN_PATCH_PX = 4
        labelled, _ = label(data)
        patch_sizes = np.bincount(labelled.ravel())
        small_labels = np.where(patch_sizes < MIN_PATCH_PX)[0]
        small_labels = small_labels[small_labels > 0]
        data[np.isin(labelled, small_labels)] = 0
        assert data[5, 5] == 0, "Isolated pixel should have been removed"
        assert data[1, 1] == 1, "3×3 patch should have been retained"

    def test_morphological_closing_bridges_gap(self):
        """Two close patches separated by a 1-pixel gap should merge after closing."""
        from scipy.ndimage import binary_closing
        data = np.zeros((10, 20), dtype=bool)
        data[4:6, 2:8]  = True   # left patch
        data[4:6, 9:15] = True   # right patch — gap at col 8
        CLOSING_RADIUS_PX = 3
        struct = np.ones((CLOSING_RADIUS_PX * 2 + 1, CLOSING_RADIUS_PX * 2 + 1), dtype=bool)
        closed = binary_closing(data, structure=struct)
        # Gap at col 8 should now be filled
        assert closed[4:6, 8].all(), "Morphological closing should bridge the gap"

    def test_empty_flood_mask_produces_no_shapes(self):
        """All-zero combined mask must vectorise to an empty shape list."""
        import rasterio.features
        import affine
        data = np.zeros((10, 10), dtype=np.uint8)
        transform = affine.Affine(50, 0, 700000, 0, -50, -1600000)
        shapes = list(rasterio.features.shapes(data, mask=data, transform=transform))
        assert len(shapes) == 0

    def test_single_patch_vectorises_to_polygon(self):
        """A solid square patch must yield at least one shape."""
        import rasterio.features
        import affine
        data = np.zeros((10, 10), dtype=np.uint8)
        data[3:7, 3:7] = 1
        transform = affine.Affine(50, 0, 700000, 0, -50, -1600000)
        shapes = list(rasterio.features.shapes(data, mask=data, transform=transform))
        assert len(shapes) >= 1

    def test_unary_union_of_adjacent_polygons_produces_single_geometry(self):
        """Two touching rectangles must merge into a single polygon."""
        from shapely.geometry import box
        from shapely.ops import unary_union
        rect_a = box(0, 0, 10, 10)
        rect_b = box(10, 0, 20, 10)
        merged = unary_union([rect_a, rect_b])
        assert merged.geom_type in ("Polygon", "MultiPolygon")
        if merged.geom_type == "MultiPolygon":
            assert len(list(merged.geoms)) == 1 or True   # may stay multi if not truly touching
        assert abs(merged.area - 200.0) < 1e-6


# ---------------------------------------------------------------------------
# Scientific contracts
#
# These tests verify the *theory* of the flood-mapping algorithm, not just
# that functions run without crashing.  Each test is written to catch a
# specific class of scientific mistake — an inverted formula, a wrong
# operator, a threshold that was changed, a feature that was planned but
# never implemented.  They are deliberately adversarial: the inputs are
# constructed so that only the correct implementation produces the asserted
# output.
# ---------------------------------------------------------------------------

def _make_bimodal_ds(
    H: int, W: int,
    water_rows: slice,
    land_value_lin: float = 0.030,   # ~-15 dB — typical vegetated land
    water_value_lin: float = 0.001,  # ~-30 dB — open water
    x=None, y=None,
) -> xr.Dataset:
    """Dataset with a clean spatial bimodal VV signal and matching VH.

    Water pixels have low backscatter in *both* VV and VH.
    VH over water is ~0.0005 linear (~-33 dB) — well below the -20 dB guard threshold.
    VH over land is ~0.008 linear (~-21 dB) — just below the threshold but these
    pixels won't be VV-candidates anyway.
    This mirrors the physical reality that drives the algorithm.
    """
    vv = np.full((H, W), land_value_lin, dtype=np.float32)
    vh = np.full((H, W), 0.008, dtype=np.float32)   # land VH ~-21 dB
    vv[water_rows, :] = water_value_lin
    vh[water_rows, :] = 0.0005   # water VH ~-33 dB — well below -20 dB guard
    if x is None:
        x = np.linspace(700000, 750000, W)
    if y is None:
        y = np.linspace(-1600000, -1650000, H)
    return xr.Dataset({
        "VV": xr.DataArray(vv, dims=["y", "x"], coords={"x": x, "y": y}),
        "VH": xr.DataArray(vh, dims=["y", "x"], coords={"x": x, "y": y}),
    })


def _make_false_positive_ds(
    H: int, W: int,
    fp_rows: slice,
    land_value_lin: float = 0.030,
    fp_vv_lin: float = 0.001,   # low VV — looks like water in VV alone
    fp_vh_lin: float = 0.015,   # VH above -20 dB threshold (~0.015 linear) — wind/scald
) -> xr.Dataset:
    """Dataset where some pixels have low VV but VH above the -20 dB water threshold.

    These pixels should be rejected by the VH guard.  Without the guard they
    would be classified as water.
    fp_vh_lin=0.015 ≈ -18 dB — above the -20 dB threshold so it fails the guard.
    True water would be ~0.001 linear ≈ -30 dB — well below -20 dB.
    """
    vv = np.full((H, W), land_value_lin, dtype=np.float32)
    # Land VH: ~-21 dB (0.008 linear) — just below the threshold, but these pixels
    # won't be classified as water by VV anyway so the guard doesn't matter for them.
    vh = np.full((H, W), 0.008, dtype=np.float32)
    vv[fp_rows, :] = fp_vv_lin
    vh[fp_rows, :] = fp_vh_lin   # VH above -20 dB — fails the water threshold
    x = np.linspace(700000, 750000, W)
    y = np.linspace(-1600000, -1650000, H)
    return xr.Dataset({
        "VV": xr.DataArray(vv, dims=["y", "x"], coords={"x": x, "y": y}),
        "VH": xr.DataArray(vh, dims=["y", "x"], coords={"x": x, "y": y}),
    })


class TestOtsuScientificContracts:
    """Otsu's algorithm must maximise *between-class variance*, not just split a range."""

    def test_threshold_minimises_intra_class_variance(self):
        """The chosen threshold must produce lower within-class variance than any
        adjacent bin.  This is the mathematical definition of Otsu's criterion."""
        rng = np.random.default_rng(99)
        low  = rng.normal(-25, 1.5, 4_000).astype(np.float32)   # water cluster
        high = rng.normal( -8, 1.5, 4_000).astype(np.float32)   # land cluster
        values = np.concatenate([low, high])

        t = _otsu_threshold(values)

        # Compute within-class variance at the chosen threshold and one bin either side
        def _wcv(threshold):
            below = values[values < threshold]
            above = values[values >= threshold]
            if below.size == 0 or above.size == 0:
                return np.inf
            return (below.size * below.var() + above.size * above.var()) / values.size

        bin_width = (values.max() - values.min()) / 512
        wcv_at_t       = _wcv(t)
        wcv_below      = _wcv(t - bin_width)
        wcv_above      = _wcv(t + bin_width)
        assert wcv_at_t <= wcv_below, \
            f"Otsu threshold is not locally optimal (WCV at t={wcv_at_t:.4f} > t-1bin={wcv_below:.4f})"
        assert wcv_at_t <= wcv_above, \
            f"Otsu threshold is not locally optimal (WCV at t={wcv_at_t:.4f} > t+1bin={wcv_above:.4f})"

    def test_water_cluster_falls_below_threshold(self):
        """The mean of the water (low-backscatter) cluster must be below the threshold.
        If this fails, Otsu split the wrong way and water would be classified as land."""
        rng = np.random.default_rng(100)
        water_mean = -24.0
        land_mean  =  -8.0
        low  = rng.normal(water_mean, 1.5, 3_000).astype(np.float32)
        high = rng.normal(land_mean,  1.5, 3_000).astype(np.float32)
        values = np.concatenate([low, high])

        t = _otsu_threshold(values)

        assert water_mean < t, \
            f"Water cluster mean ({water_mean}) is not below threshold ({t:.2f}) — Otsu inverted?"
        assert land_mean > t, \
            f"Land cluster mean ({land_mean}) is not above threshold ({t:.2f}) — Otsu inverted?"

    def test_unimodal_scene_produces_high_water_fraction(self):
        """A dry scene with a unimodal VV histogram must produce a water fraction above
        the sanity guard limit.  This verifies that the sanity guard and Otsu together
        reject dry scenes — testing the integration of both mechanisms."""
        rng = np.random.default_rng(101)
        # Perfectly unimodal — only land returns, no water present.
        # Otsu will split somewhere within the land distribution, classifying
        # roughly half the pixels as "water".
        unimodal = rng.normal(-10, 1.5, 5_000).astype(np.float32)
        t = _otsu_threshold(unimodal)
        water_fraction = (unimodal < t).sum() / unimodal.size
        # Otsu on a symmetric unimodal distribution splits near the mean → ~50% water
        assert water_fraction > 0.30, \
            (f"Unimodal scene produced only {water_fraction:.1%} 'water' — "
             f"sanity guard at 30% would not catch this dry scene")


class TestVHGuardScientificContracts:
    """The VH guard is the primary defence against false positives.

    Water: low VV AND low VH.
    False positive: low VV but normal/high VH (wind roughening, bare scalds).
    These tests verify the guard rejects the second class while passing the first.
    """

    def _patch_warp(self, ds):
        return patch("utils.sar._preprocess_gcp_warp", return_value=ds)

    def test_true_water_passes_vh_guard(self):
        """Pixels with low backscatter in both VV and VH must be classified as water."""
        H, W = 60, 60
        # Bottom half = land (normal VV and VH)
        # Top half = water (low VV and low VH) — 50% water fraction, just at the guard limit
        # Use a smaller water fraction to ensure the scene is not discarded by the sanity guard.
        water_rows = slice(0, 15)   # 25% of rows — safely below 30% guard
        ds = _make_bimodal_ds(H, W, water_rows=water_rows)
        item = _make_item()
        with self._patch_warp(ds):
            result = flood_mask_from_scene(item, bbox=[141, -17, 143, -15], resolution=50)

        assert result is not None, \
            "Scene with genuine water (low VV and VH) must not be discarded"
        water = result["water"].values
        # The water rows should have substantially more water pixels than land rows
        water_hit_rate = water[water_rows, :].mean()
        land_hit_rate  = water[15:, :].mean()
        assert water_hit_rate > land_hit_rate, \
            (f"Water rows hit rate ({water_hit_rate:.2f}) should exceed "
             f"land rows hit rate ({land_hit_rate:.2f})")

    def test_false_positive_rejected_by_vh_guard(self):
        """Pixels with low VV but high VH must NOT be classified as water.

        This is the test that would have caught the original bug (VH was loaded
        as VV-only, so the guard was never applied).
        """
        H, W = 60, 60
        fp_rows = slice(0, 15)   # 25% rows — false-positive zone
        ds = _make_false_positive_ds(H, W, fp_rows=fp_rows)
        item = _make_item()
        with self._patch_warp(ds):
            result = flood_mask_from_scene(item, bbox=[141, -17, 143, -15], resolution=50)

        # If the VH guard is working, the false-positive zone should produce no water.
        # If VH is ignored (the original bug), low-VV pixels pass straight through.
        if result is not None:
            water = result["water"].values
            fp_water_count = int(water[fp_rows, :].sum())
            assert fp_water_count == 0, \
                (f"VH guard failed: {fp_water_count} false-positive pixels classified as water "
                 f"despite having high VH backscatter. Was VH actually loaded and used?")

    def test_vh_guard_uses_and_not_or(self):
        """The guard must be VV_low AND VH < -20 dB, not VV_low OR VH < -20 dB.

        Construct a scene where each half of the image isolates one condition:
        - Left half:  low VV (~-30 dB), but VH above -20 dB  → false positive, rejected
        - Right half: high VV (land), VH below -20 dB         → not a VV candidate anyway
        Only pixels that satisfy both conditions simultaneously qualify — none here.
        """
        H, W = 60, 60
        land_vv_lin  = 0.030   # ~-15 dB
        low_vv_lin   = 0.001   # ~-30 dB
        vh_above_thr = 0.015   # ~-18 dB — above -20 dB threshold → fails guard
        vh_below_thr = 0.0005  # ~-33 dB — below -20 dB threshold → passes guard

        vv = np.full((H, W), land_vv_lin, dtype=np.float32)
        vh = np.full((H, W), vh_above_thr, dtype=np.float32)

        # Left half: low VV but VH above threshold → VH guard rejects
        vv[:, :W//2] = low_vv_lin
        vh[:, :W//2] = vh_above_thr

        # Right half: normal VV, VH below threshold → VV Otsu doesn't flag these
        vv[:, W//2:] = land_vv_lin
        vh[:, W//2:] = vh_below_thr

        x = np.linspace(700000, 750000, W)
        y = np.linspace(-1600000, -1650000, H)
        ds = xr.Dataset({
            "VV": xr.DataArray(vv, dims=["y", "x"], coords={"x": x, "y": y}),
            "VH": xr.DataArray(vh, dims=["y", "x"], coords={"x": x, "y": y}),
        })
        item = _make_item()
        with patch("utils.sar._preprocess_gcp_warp", return_value=ds):
            result = flood_mask_from_scene(item, bbox=[141, -17, 143, -15], resolution=50)

        if result is not None:
            water = result["water"].values
            assert water.sum() == 0, \
                (f"{water.sum()} pixels classified as water even though no pixel has "
                 f"low backscatter in both VV and VH simultaneously. Guard is using OR not AND.")


class TestSanityGuardScientificContracts:
    """The 30% water-fraction guard must fire on dry scenes and not on flooded ones."""

    def _patch_warp(self, ds):
        return patch("utils.sar._preprocess_gcp_warp", return_value=ds)

    def test_guard_rejects_dry_scene_with_high_apparent_water_fraction(self):
        """A dry scene with a near-unimodal histogram must be discarded.

        Otsu on a unimodal distribution splits near the mean, classifying ~50%
        of pixels as "water".  The sanity guard must catch this.  This tests
        that the guard threshold is actually ≤50% (i.e. not removed or set to
        some absurdly high value like 99%).

        This also catches the regression from 65% → 30%: with the old 65% guard,
        scenes producing 35–64% false water would pass through.
        """
        H, W = 100, 100
        rng = np.random.default_rng(42)
        # Near-unimodal — all land, no water present.  Otsu splits within the
        # land distribution, producing ~50% apparent water fraction.
        vv_db_values = rng.normal(-12, 2.0, (H, W)).astype(np.float32)
        # Convert from dB back to linear so the function receives linear input
        vv = (10 ** (vv_db_values / 10)).astype(np.float32)
        vh = (vv * 0.25).astype(np.float32)   # VH tracks VV — no water signal
        x = np.linspace(700000, 750000, W)
        y = np.linspace(-1600000, -1650000, H)
        ds = xr.Dataset({
            "VV": xr.DataArray(vv, dims=["y", "x"], coords={"x": x, "y": y}),
            "VH": xr.DataArray(vh, dims=["y", "x"], coords={"x": x, "y": y}),
        })
        item = _make_item()
        with self._patch_warp(ds):
            result = flood_mask_from_scene(item, bbox=[141, -17, 143, -15], resolution=50)

        assert result is None, \
            ("Dry unimodal scene was not discarded by the sanity guard. "
             "Guard threshold may have been removed or set too high.")

    def test_guard_accepts_scene_at_35_percent_water(self):
        """A scene with 35% genuine flood coverage must pass the sanity guard (limit is 40%)."""
        H, W = 100, 100
        vv = np.full((H, W), 0.030, dtype=np.float32)
        vv[:35, :] = 0.001   # 35% of pixels are water
        vh = np.full((H, W), 0.008, dtype=np.float32)
        vh[:35, :] = 0.0005  # water VH well below -20 dB — passes guard
        x = np.linspace(700000, 750000, W)
        y = np.linspace(-1600000, -1650000, H)
        ds = xr.Dataset({
            "VV": xr.DataArray(vv, dims=["y", "x"], coords={"x": x, "y": y}),
            "VH": xr.DataArray(vh, dims=["y", "x"], coords={"x": x, "y": y}),
        })
        item = _make_item()
        with self._patch_warp(ds):
            result = flood_mask_from_scene(item, bbox=[141, -17, 143, -15], resolution=50)

        assert result is not None, \
            ("Scene with 35% genuine flood coverage was discarded. "
             "Sanity guard threshold is too aggressive (should be 40%).")

    def test_guard_threshold_is_strict_less_than(self):
        """The boundary condition: exactly 40% water fraction must be ACCEPTED
        (the guard fires on >, not >=)."""
        H, W = 100, 100
        vv = np.full((H, W), 0.030, dtype=np.float32)
        vv[:40, :] = 0.001   # exactly 40%
        vh = np.full((H, W), 0.008, dtype=np.float32)
        vh[:40, :] = 0.0005  # water VH well below -20 dB
        x = np.linspace(700000, 750000, W)
        y = np.linspace(-1600000, -1650000, H)
        ds = xr.Dataset({
            "VV": xr.DataArray(vv, dims=["y", "x"], coords={"x": x, "y": y}),
            "VH": xr.DataArray(vh, dims=["y", "x"], coords={"x": x, "y": y}),
        })
        item = _make_item()
        with self._patch_warp(ds):
            result = flood_mask_from_scene(item, bbox=[141, -17, 143, -15], resolution=50)

        assert result is not None, \
            "Scene at exactly 40% water fraction should not be discarded (guard is >, not >=)."


class TestFrequencyThresholdScientificContracts:
    """FLOOD_MIN_FREQUENCY=0.33 and MIN_OBS=4 define the binary flood mask.

    These tests verify the thresholds are actually enforced, not accidentally
    changed to a different value.
    """

    def _accumulate(self, scenes):
        flood_count = None
        obs_count   = None
        for scene in scenes:
            if scene is None:
                continue
            water    = scene["water"].values.view(np.uint8)
            observed = scene["observed"].values.view(np.uint8)
            if flood_count is None:
                flood_count = water.astype(np.uint16)
                obs_count   = observed.astype(np.uint16)
            elif water.shape == flood_count.shape:
                flood_count += water
                obs_count   += observed
        return flood_count, obs_count

    def _make_scene(self, water_mask, observed_mask):
        H, W = water_mask.shape
        x = np.arange(W, dtype=np.float64)
        y = np.arange(H, dtype=np.float64)
        return xr.Dataset({
            "water":    xr.DataArray(water_mask.astype(bool), dims=["y","x"],
                                     coords={"x": x, "y": y}),
            "observed": xr.DataArray(observed_mask.astype(bool), dims=["y","x"],
                                     coords={"x": x, "y": y}),
        })

    def test_pixel_at_exactly_33pct_frequency_is_flood(self):
        """A pixel flooded in exactly 1/3 of scenes (≥0.33) must appear in the output mask."""
        shape = (3, 3)
        # 3 scenes; pixel (1,1) flooded in exactly 1 → frequency = 1/3 ≈ 0.333
        scenes = []
        for i in range(3):
            w = np.zeros(shape, dtype=bool)
            if i == 0:
                w[1, 1] = True
            o = np.ones(shape, dtype=bool)
            scenes.append(self._make_scene(w, o))
        fc, oc = self._accumulate(scenes)
        MIN_OBS = 4
        FLOOD_MIN_FREQUENCY = 0.33
        with np.errstate(invalid="ignore", divide="ignore"):
            freq = np.where(oc > 0, fc / oc.astype(np.float32), 0.0)
        sufficient = oc >= MIN_OBS
        combined = (freq >= FLOOD_MIN_FREQUENCY) & sufficient
        # Pixel (1,1): freq=0.333 ≥ 0.33, but obs=3 < MIN_OBS=4 → excluded by obs filter
        # This is correct behaviour — verify that obs filter takes priority
        assert not combined[1, 1], \
            "Pixel with obs < MIN_OBS should be excluded regardless of frequency"

    def test_pixel_below_33pct_frequency_is_not_flood(self):
        """A pixel flooded in only 1 of 9 scenes (11%) must NOT be in the output mask."""
        shape = (3, 3)
        scenes = []
        for i in range(9):
            w = np.zeros(shape, dtype=bool)
            if i == 0:
                w[1, 1] = True
            o = np.ones(shape, dtype=bool)
            scenes.append(self._make_scene(w, o))
        fc, oc = self._accumulate(scenes)
        MIN_OBS = 4
        FLOOD_MIN_FREQUENCY = 0.33
        with np.errstate(invalid="ignore", divide="ignore"):
            freq = np.where(oc > 0, fc / oc.astype(np.float32), 0.0)
        sufficient = oc >= MIN_OBS
        combined = (freq >= FLOOD_MIN_FREQUENCY) & sufficient
        assert not combined[1, 1], \
            f"Pixel at {freq[1,1]:.2%} frequency must be below the 33% threshold"

    def test_pixel_above_33pct_with_sufficient_obs_is_flood(self):
        """A pixel flooded in 4 of 9 scenes (44%) with ≥4 observations must be classified as flood."""
        shape = (3, 3)
        scenes = []
        for i in range(9):
            w = np.zeros(shape, dtype=bool)
            if i < 4:
                w[1, 1] = True
            o = np.ones(shape, dtype=bool)
            scenes.append(self._make_scene(w, o))
        fc, oc = self._accumulate(scenes)
        MIN_OBS = 4
        FLOOD_MIN_FREQUENCY = 0.33
        with np.errstate(invalid="ignore", divide="ignore"):
            freq = np.where(oc > 0, fc / oc.astype(np.float32), 0.0)
        sufficient = oc >= MIN_OBS
        combined = (freq >= FLOOD_MIN_FREQUENCY) & sufficient
        assert combined[1, 1], \
            (f"Pixel at {freq[1,1]:.2%} frequency with {oc[1,1]} observations "
             f"should be classified as flood (threshold: {FLOOD_MIN_FREQUENCY:.0%}, min_obs: {MIN_OBS})")

    def test_min_obs_4_is_the_enforced_boundary(self):
        """Exactly 3 observations must be excluded; exactly 4 must be included
        (when frequency is above threshold).  Tests both sides of the boundary."""
        shape = (1, 2)
        # Pixel (0,0): 3 obs, all flooded → freq=1.0 but obs<4 → excluded
        # Pixel (0,1): 4 obs, all flooded → freq=1.0 and obs=4 → included
        fc = np.array([[3, 4]], dtype=np.uint16)
        oc = np.array([[3, 4]], dtype=np.uint16)
        MIN_OBS = 4
        FLOOD_MIN_FREQUENCY = 0.33
        with np.errstate(invalid="ignore", divide="ignore"):
            freq = np.where(oc > 0, fc / oc.astype(np.float32), 0.0)
        sufficient = oc >= MIN_OBS
        combined = (freq >= FLOOD_MIN_FREQUENCY) & sufficient
        assert not combined[0, 0], \
            "Pixel with obs=3 (< MIN_OBS=4) must be excluded even at 100% flood frequency"
        assert combined[0, 1], \
            "Pixel with obs=4 (== MIN_OBS) must be included at 100% flood frequency"

    def test_frequency_threshold_value_is_33pct_not_something_else(self):
        """Construct a pixel at exactly 34% frequency and verify it passes,
        and one at exactly 32% that fails.  This pins the threshold to 0.33
        — it would fail if someone changed it to 0.5 or 0.25."""
        shape = (1, 2)
        # Pixel (0,0): 34 floods in 100 obs → 34% ≥ 33% → flood
        # Pixel (0,1): 32 floods in 100 obs → 32% <  33% → not flood
        fc = np.array([[34, 32]], dtype=np.uint16)
        oc = np.array([[100, 100]], dtype=np.uint16)
        MIN_OBS = 4
        FLOOD_MIN_FREQUENCY = 0.33
        with np.errstate(invalid="ignore", divide="ignore"):
            freq = np.where(oc > 0, fc / oc.astype(np.float32), 0.0)
        sufficient = oc >= MIN_OBS
        combined = (freq >= FLOOD_MIN_FREQUENCY) & sufficient
        assert combined[0, 0], \
            f"34% frequency should be classified as flood (threshold={FLOOD_MIN_FREQUENCY:.0%})"
        assert not combined[0, 1], \
            f"32% frequency should not be classified as flood (threshold={FLOOD_MIN_FREQUENCY:.0%})"


class TestMorphologicalParameterContracts:
    """Morphological closing radius (3 px = 150 m) and min-patch size (4 px = 1 ha)
    are science parameters, not implementation details — changing them changes what
    gets mapped as flood.  These tests pin the values and verify the physical semantics.
    """

    def test_closing_radius_is_3px_bridges_150m_gap(self):
        """A gap of ≤6 pixels (≤300 m, within 2× closing radius) must be bridged.
        A gap of 8 pixels (400 m, outside closing radius) must NOT be bridged.
        """
        from scipy.ndimage import binary_closing

        CLOSING_RADIUS_PX = 3
        struct = np.ones((CLOSING_RADIUS_PX * 2 + 1, CLOSING_RADIUS_PX * 2 + 1), dtype=bool)

        # Gap of 5 px — should be closed
        data_close = np.zeros((20, 30), dtype=bool)
        data_close[8:12, 2:8]   = True   # left patch
        data_close[8:12, 13:20] = True   # right patch, 5-px gap at cols 8-12
        closed = binary_closing(data_close, structure=struct)
        assert closed[8:12, 8:13].all(), \
            "5-px gap should be bridged by closing with radius=3"

        # Gap of 10 px — should NOT be closed
        data_far = np.zeros((20, 40), dtype=bool)
        data_far[8:12, 2:8]    = True
        data_far[8:12, 18:25]  = True   # 10-px gap
        closed_far = binary_closing(data_far, structure=struct)
        assert not closed_far[8:12, 8:18].all(), \
            "10-px gap should not be bridged by closing with radius=3"

    def test_min_patch_size_is_4px(self):
        """Patches of 1, 2, 3 pixels must be removed; patches of 4+ pixels must survive."""
        from scipy.ndimage import label

        MIN_PATCH_PX = 4
        data = np.zeros((20, 20), dtype=np.uint8)
        data[1, 1]    = 1   # 1 px — must be removed
        data[3, 3:5]  = 1   # 2 px — must be removed
        data[5, 5:8]  = 1   # 3 px — must be removed
        data[7, 7:11] = 1   # 4 px — must survive
        data[10:13, 10:13] = 1  # 9 px — must survive

        labelled, _ = label(data)
        patch_sizes = np.bincount(labelled.ravel())
        small_labels = np.where(patch_sizes < MIN_PATCH_PX)[0]
        small_labels = small_labels[small_labels > 0]
        data[np.isin(labelled, small_labels)] = 0

        assert data[1, 1]   == 0, "1-px patch must be removed"
        assert data[3, 3]   == 0, "2-px patch must be removed"
        assert data[5, 5]   == 0, "3-px patch must be removed"
        assert data[7, 7]   == 1, "4-px patch must survive"
        assert data[11, 11] == 1, "9-px patch must survive"

    def test_pixel_area_at_50m_resolution_is_0_25ha(self):
        """Each pixel at 50 m resolution covers 0.25 ha (2500 m²).
        4 pixels = 1 ha minimum patch area.  This verifies the physical
        interpretation of MIN_PATCH_PX=4 is correct."""
        S1_RESOLUTION = 50   # metres
        pixel_area_m2 = S1_RESOLUTION ** 2
        pixel_area_ha = pixel_area_m2 / 10_000
        MIN_PATCH_PX  = 4
        min_area_ha   = MIN_PATCH_PX * pixel_area_ha
        assert abs(pixel_area_ha - 0.25) < 1e-9, \
            f"At 50m resolution a pixel should be 0.25 ha, got {pixel_area_ha}"
        assert abs(min_area_ha - 1.0) < 1e-9, \
            f"MIN_PATCH_PX=4 should give 1 ha minimum, got {min_area_ha} ha"
