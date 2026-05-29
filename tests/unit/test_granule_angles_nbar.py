"""Tests for granule_angles bilinear interpolation and nbar c_factor_rad."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from utils.granule_angles import _GRID_SIZE, _GRID_STEP_M
import utils.granule_angles as ga
import utils.nbar as nbar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid(ul_e: float = 300_000.0, ul_n: float = 6_900_000.0):
    """Return (grid_northings_asc, grid_eastings, i0, j0, di, dj) for a standard
    23×23 grid anchored at ul_e, ul_n."""
    grid_e = ul_e + np.arange(_GRID_SIZE) * _GRID_STEP_M
    grid_n = ul_n - np.arange(_GRID_SIZE) * _GRID_STEP_M  # descending
    return grid_e, grid_n


def _reference_interp(grid_2d, grid_n, grid_e, px_n, px_e):
    """Reference implementation using scipy RegularGridInterpolator."""
    interp = RegularGridInterpolator(
        (grid_n[::-1], grid_e),
        grid_2d[::-1, :],
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    return interp(np.column_stack([px_n, px_e])).astype(np.float32)


def _fast_interp(grid_2d, grid_n, grid_e, px_n, px_e):
    """Our new implementation extracted for testing in isolation."""
    n_n = len(grid_n)
    n_e = len(grid_e)
    grid_n_asc = grid_n[::-1]

    i_f = np.interp(px_n, grid_n_asc, np.arange(n_n, dtype=np.float64))
    j_f = np.interp(px_e, grid_e,     np.arange(n_e, dtype=np.float64))

    i0 = np.clip(np.floor(i_f).astype(np.intp), 0, n_n - 2)
    j0 = np.clip(np.floor(j_f).astype(np.intp), 0, n_e - 2)
    di  = (i_f - i0).astype(np.float32)
    dj  = (j_f - j0).astype(np.float32)
    di1 = 1.0 - di
    dj1 = 1.0 - dj

    g = grid_2d[::-1, :].astype(np.float32)
    return (
        g[i0,   j0  ] * di1 * dj1 +
        g[i0+1, j0  ] * di  * dj1 +
        g[i0,   j0+1] * di1 * dj  +
        g[i0+1, j0+1] * di  * dj
    )


# ---------------------------------------------------------------------------
# Bilinear interpolation — correctness vs scipy reference
# ---------------------------------------------------------------------------

class TestBilinearInterp:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.grid_2d = rng.uniform(20.0, 70.0, (_GRID_SIZE, _GRID_SIZE))
        self.grid_e, self.grid_n = _make_grid()

    def _interior_points(self, n=500):
        rng = np.random.default_rng(0)
        px_e = rng.uniform(self.grid_e[1], self.grid_e[-2], n)
        px_n = rng.uniform(self.grid_n[-2], self.grid_n[1], n)
        return px_e, px_n

    def test_interior_matches_scipy(self):
        px_e, px_n = self._interior_points()
        ref  = _reference_interp(self.grid_2d, self.grid_n, self.grid_e, px_n, px_e)
        fast = _fast_interp(    self.grid_2d, self.grid_n, self.grid_e, px_n, px_e)
        np.testing.assert_allclose(fast, ref, rtol=1e-5, atol=1e-4)

    def test_at_grid_nodes_exact(self):
        """Values exactly at grid nodes should equal the grid values."""
        # Pick a few interior nodes
        for i, j in [(0, 0), (5, 3), (11, 11), (22, 22)]:
            px_e = np.array([self.grid_e[j]])
            px_n = np.array([self.grid_n[i]])
            fast = _fast_interp(self.grid_2d, self.grid_n, self.grid_e, px_n, px_e)
            expected = float(self.grid_2d[i, j])
            assert abs(float(fast[0]) - expected) < 1e-3, (
                f"node ({i},{j}): got {fast[0]:.4f}, expected {expected:.4f}"
            )

    def test_out_of_bounds_clamps_not_nan(self):
        """Points outside the grid extent must return a finite value (nearest clamp)."""
        px_e = np.array([self.grid_e[0] - 10_000, self.grid_e[-1] + 10_000])
        px_n = np.array([self.grid_n[0] + 10_000, self.grid_n[-1] - 10_000])
        fast = _fast_interp(self.grid_2d, self.grid_n, self.grid_e, px_n, px_e)
        assert np.all(np.isfinite(fast)), "out-of-bounds points must not produce NaN"

    def test_large_n_matches_scipy(self):
        """6 M points — the real strip size — must still match scipy."""
        rng = np.random.default_rng(99)
        n = 6_000_000
        px_e = rng.uniform(self.grid_e[0], self.grid_e[-1], n)
        px_n = rng.uniform(self.grid_n[-1], self.grid_n[0], n)
        ref  = _reference_interp(self.grid_2d, self.grid_n, self.grid_e, px_n, px_e)
        fast = _fast_interp(    self.grid_2d, self.grid_n, self.grid_e, px_n, px_e)
        np.testing.assert_allclose(fast, ref, rtol=1e-5, atol=1e-4)

    def test_grid_indices_reused_across_calls(self):
        """Calling _fast_interp twice with the same points gives identical results."""
        px_e, px_n = self._interior_points(100)
        r1 = _fast_interp(self.grid_2d, self.grid_n, self.grid_e, px_n, px_e)
        r2 = _fast_interp(self.grid_2d, self.grid_n, self.grid_e, px_n, px_e)
        np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# nbar.c_factor_rad — matches c_factor (deg) and is faster
# ---------------------------------------------------------------------------

class TestCFactorRad:
    def setup_method(self):
        rng = np.random.default_rng(7)
        n = 10_000
        self.sza_deg = rng.uniform(10.0, 70.0, n).astype(np.float32)
        self.vza_deg = rng.uniform(0.0,  30.0, n).astype(np.float32)
        self.raa_deg = rng.uniform(0.0, 180.0, n).astype(np.float32)

    def test_c_factor_rad_matches_c_factor_deg(self):
        for band in nbar.BRDF_COEFFICIENTS:
            ref  = nbar.c_factor(    self.sza_deg, self.vza_deg, self.raa_deg, band)
            fast = nbar.c_factor_rad(
                np.deg2rad(self.sza_deg),
                np.deg2rad(self.vza_deg),
                np.deg2rad(self.raa_deg),
                band,
            )
            np.testing.assert_allclose(fast, ref, rtol=1e-6, atol=1e-6,
                                       err_msg=f"band={band}")

    def test_c_factor_rad_clamped(self):
        for band in nbar.BRDF_COEFFICIENTS:
            cf = nbar.c_factor_rad(
                np.deg2rad(self.sza_deg),
                np.deg2rad(self.vza_deg),
                np.deg2rad(self.raa_deg),
                band,
            )
            assert cf.min() >= 0.5 - 1e-6
            assert cf.max() <= 2.0 + 1e-6

    def test_single_deg2rad_equiv_per_band(self):
        """Shared sza_rad across bands gives same result as per-band conversion."""
        sza_rad = np.deg2rad(self.sza_deg)
        for band in nbar.BRDF_COEFFICIENTS:
            vza_rad = np.deg2rad(self.vza_deg)
            raa_rad = np.deg2rad(self.raa_deg)
            r1 = nbar.c_factor_rad(sza_rad, vza_rad, raa_rad, band)
            r2 = nbar.c_factor(self.sza_deg, self.vza_deg, self.raa_deg, band)
            np.testing.assert_allclose(r1, r2, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# get_item_angles — unit test using a synthetic parsed dict (no network)
# ---------------------------------------------------------------------------

class TestGetItemAnglesUnit:
    """Test get_item_angles with a mocked _fetch_and_parse_xml."""

    def _make_parsed(self, ul_e=300_000.0, ul_n=6_900_000.0):
        rng = np.random.default_rng(1)
        shape = (_GRID_SIZE, _GRID_SIZE)
        view_zen = {str(i): rng.uniform(0.0, 20.0, shape) for i in range(13)}
        view_az  = {str(i): rng.uniform(0.0, 360.0, shape) for i in range(13)}
        return {
            "sun_zen":  rng.uniform(20.0, 60.0, shape),
            "sun_az":   rng.uniform(100.0, 200.0, shape),
            "view_zen": view_zen,
            "view_az":  view_az,
            "ul_utm":   (ul_e, ul_n),
        }

    def _make_item(self, item_id="S2A_54LWH_20251206_0_L2A"):
        from unittest.mock import MagicMock
        item = MagicMock()
        item.id = item_id
        item.assets = {"granule_metadata": MagicMock(href="http://fake/metadata.xml")}
        return item

    def test_returns_dict_for_all_bands(self, monkeypatch):
        from utils.pixel_collector import BANDS
        parsed = self._make_parsed()
        monkeypatch.setattr(ga, "_fetch_and_parse_xml", lambda *_: parsed)

        rng = np.random.default_rng(2)
        n = 1000
        lons = rng.uniform(148.0, 149.0, n)
        lats = rng.uniform(-21.0, -20.0, n)

        result = ga.get_item_angles(
            self._make_item(), lons, lats, "EPSG:32754", list(BANDS)
        )
        assert result is not None
        for band in BANDS:
            assert band in result
            for key in ("sza", "vza", "saa", "vaa"):
                assert result[band][key].shape == (n,)
                assert np.all(np.isfinite(result[band][key]))

    def test_matches_scipy_reference(self, monkeypatch):
        """Fast interpolation must match scipy for the same query points."""
        from utils.pixel_collector import BANDS
        parsed = self._make_parsed()

        # Capture what _fast_interp produces by running get_item_angles
        monkeypatch.setattr(ga, "_fetch_and_parse_xml", lambda *_: parsed)

        rng = np.random.default_rng(3)
        n = 2000
        ul_e, ul_n = parsed["ul_utm"]
        grid_e = ul_e + np.arange(_GRID_SIZE) * _GRID_STEP_M
        grid_n = ul_n - np.arange(_GRID_SIZE) * _GRID_STEP_M

        from pyproj import Transformer
        to_wgs = Transformer.from_crs("EPSG:32754", "EPSG:4326", always_xy=True)
        px_e = rng.uniform(grid_e[1], grid_e[-2], n)
        px_n = rng.uniform(grid_n[-2], grid_n[1], n)
        lons_arr, lats_arr = to_wgs.transform(px_e, px_n)

        result = ga.get_item_angles(
            self._make_item(), np.array(lons_arr), np.array(lats_arr),
            "EPSG:32754", list(BANDS),
            utm_xy=(px_e, px_n),
        )
        assert result is not None

        # Compare one band's sza against scipy reference
        band = BANDS[0]
        ref_sza = _reference_interp(parsed["sun_zen"], grid_n, grid_e, px_n, px_e)
        np.testing.assert_allclose(result[band]["sza"], ref_sza, rtol=1e-5, atol=1e-4)
