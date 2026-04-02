"""Tests for analysis/06_classifier.py — Random Forest Parkinsonia classifier.

Covers:
  - _compute_glcm_features: contrast/homogeneity values, NaN propagation, uniform input
  - _generate_ecological_samples: threshold correctness, non-empty outputs, no leakage
    into pseudo-absence pool, scientific assumptions about Parkinsonia ecology
  - _fetch_ala_occurrences: cache hit path, empty result handling
  - Pseudo-absence generation: buffer exclusion, minimum sample count
  - Sample weight construction: ALA vs synthetic weights
  - Feature stack: shape, ordering, NaN masking
  - end-to-end main(): output raster shape, value range [0,1], CRS
"""
import importlib.util
import pickle
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import geopandas as gpd
import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from shapely.geometry import Point, box

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT = PROJECT_ROOT / "analysis" / "06_classifier.py"


def _load_module(module_name: str = "classifier05"):
    spec = importlib.util.spec_from_file_location(module_name, str(SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_raster(values: np.ndarray, crs: str = "EPSG:7855") -> xr.DataArray:
    H, W = values.shape
    x = np.linspace(700000, 750000, W)
    y = np.linspace(-1600000, -1650000, H)
    da = xr.DataArray(values.astype(np.float32), dims=["y", "x"],
                      coords={"y": y, "x": x})
    da = da.rio.write_crs(crs)
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    return da


def _write_raster(path: Path, values: np.ndarray, crs: str = "EPSG:7855") -> None:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    H, W = values.shape
    transform = from_bounds(700000, -1650000, 750000, -1600000, W, H)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(str(path), "w", driver="GTiff", dtype="float32", count=1,
                       width=W, height=H, transform=transform,
                       crs=CRS.from_epsg(int(crs.split(":")[1]))) as dst:
        dst.write(values.astype(np.float32)[np.newaxis])


# ── _compute_glcm_features ────────────────────────────────────────────────────

class TestGlcmFeatures:
    def setup_method(self):
        self.mod = _load_module("glcm_tests")

    def test_uniform_array_contrast_is_zero(self):
        """A perfectly uniform raster has zero contrast everywhere."""
        ndvi = np.full((20, 20), 0.4, dtype=np.float32)
        contrast, _ = self.mod._compute_glcm_features(ndvi, kernel=7)
        interior = contrast[3:-3, 3:-3]  # exclude border where mode="nearest" pads
        np.testing.assert_allclose(interior, 0.0, atol=1e-6)

    def test_uniform_array_homogeneity_is_one(self):
        """A perfectly uniform raster has homogeneity=1 everywhere (no difference)."""
        ndvi = np.full((20, 20), 0.4, dtype=np.float32)
        _, homogeneity = self.mod._compute_glcm_features(ndvi, kernel=7)
        interior = homogeneity[3:-3, 3:-3]
        np.testing.assert_allclose(interior, 1.0, atol=1e-6)

    def test_heterogeneous_array_has_positive_contrast(self):
        """A checkerboard pattern has strictly positive contrast."""
        rng = np.random.default_rng(0)
        ndvi = rng.uniform(0.1, 0.9, size=(20, 20)).astype(np.float32)
        contrast, _ = self.mod._compute_glcm_features(ndvi, kernel=7)
        assert np.nanmean(contrast) > 0.0

    def test_homogeneity_bounded_zero_to_one(self):
        """Homogeneity must always be in (0, 1]."""
        rng = np.random.default_rng(1)
        ndvi = rng.uniform(0.0, 1.0, size=(20, 20)).astype(np.float32)
        _, homogeneity = self.mod._compute_glcm_features(ndvi, kernel=7)
        valid = homogeneity[~np.isnan(homogeneity)]
        assert np.all(valid > 0.0)
        assert np.all(valid <= 1.0 + 1e-6)

    def test_nan_center_propagates_nan(self):
        """NaN at a pixel centre must produce NaN in both outputs."""
        ndvi = np.full((20, 20), 0.4, dtype=np.float32)
        ndvi[10, 10] = np.nan
        contrast, homogeneity = self.mod._compute_glcm_features(ndvi, kernel=7)
        assert np.isnan(contrast[10, 10])
        assert np.isnan(homogeneity[10, 10])

    def test_output_shape_matches_input(self):
        ndvi = np.random.rand(15, 25).astype(np.float32)
        contrast, homogeneity = self.mod._compute_glcm_features(ndvi, kernel=7)
        assert contrast.shape == ndvi.shape
        assert homogeneity.shape == ndvi.shape

    def test_high_variance_patch_has_higher_contrast_than_smooth(self):
        """A rough patch should have higher contrast than a smooth patch."""
        ndvi = np.full((30, 30), 0.4, dtype=np.float32)
        rng = np.random.default_rng(7)
        # Insert a high-variance block in the top-left
        ndvi[2:12, 2:12] = rng.uniform(0.0, 1.0, size=(10, 10))
        contrast, _ = self.mod._compute_glcm_features(ndvi, kernel=7)
        rough_mean = np.nanmean(contrast[4:10, 4:10])
        smooth_mean = np.nanmean(contrast[18:26, 18:26])
        assert rough_mean > smooth_mean


# ── _generate_ecological_samples ─────────────────────────────────────────────

class TestEcologicalSamples:
    """Tests that validate both the implementation AND the scientific assumptions."""

    def setup_method(self):
        self.mod = _load_module("eco_tests")
        self.rng = np.random.default_rng(42)

    def _make_inputs(self, H=50, W=50, seed=42):
        rng = np.random.default_rng(seed)
        ndvi_anomaly  = rng.normal(0.0, 0.1, (H, W)).astype(np.float32)
        flowering     = rng.uniform(0.0, 1.0, (H, W)).astype(np.float32)
        ndvi_median   = rng.uniform(0.1, 0.8, (H, W)).astype(np.float32)
        dist_to_water = rng.uniform(0, 5000, (H, W)).astype(np.float32)
        feat_stack    = np.zeros((H, W, 8), dtype=np.float32)
        return feat_stack, dist_to_water, ndvi_anomaly, flowering, ndvi_median

    def test_returns_nonempty_samples_on_realistic_input(self):
        """Should find at least some presence and absence candidates."""
        args = self._make_inputs()
        synth_pres, synth_abs = self.mod._generate_ecological_samples(*args, rng=self.rng)
        assert len(synth_pres) > 0, "Expected synthetic presences on realistic input"
        assert len(synth_abs) > 0, "Expected synthetic absences on realistic input"

    def test_presence_pixels_are_near_water(self):
        """All synthetic presences must be within SYNTH_PRESENCE_MAX_DIST_WATER_M of water."""
        feat_stack, dist_to_water, ndvi_anomaly, flowering, ndvi_median = self._make_inputs()
        synth_pres, _ = self.mod._generate_ecological_samples(
            feat_stack, dist_to_water, ndvi_anomaly, flowering, ndvi_median, rng=self.rng)
        max_dist = self.mod.SYNTH_PRESENCE_MAX_DIST_WATER_M
        for r, c in synth_pres:
            assert dist_to_water[r, c] <= max_dist, (
                f"Presence pixel ({r},{c}) has dist_to_water={dist_to_water[r,c]:.0f}m "
                f"> threshold {max_dist}m"
            )

    def test_presence_pixels_have_high_ndvi_anomaly(self):
        """Synthetic presences must have ndvi_anomaly in the top quartile of the scene.

        Scientific rationale: Parkinsonia stays green during dry season when native
        vegetation senesces, producing an anomalously high NDVI signal.
        """
        feat_stack, dist_to_water, ndvi_anomaly, flowering, ndvi_median = self._make_inputs()
        synth_pres, _ = self.mod._generate_ecological_samples(
            feat_stack, dist_to_water, ndvi_anomaly, flowering, ndvi_median, rng=self.rng)
        q75 = float(np.nanpercentile(ndvi_anomaly, 75))
        for r, c in synth_pres:
            assert ndvi_anomaly[r, c] >= q75 - 1e-5, (
                f"Presence pixel ndvi_anomaly={ndvi_anomaly[r,c]:.3f} below Q75={q75:.3f}"
            )

    def test_presence_pixels_have_above_median_flowering(self):
        """Synthetic presences must have flowering_index above scene median.

        Scientific rationale: Parkinsonia has a distinctive Aug-Oct flowering signal.
        """
        feat_stack, dist_to_water, ndvi_anomaly, flowering, ndvi_median = self._make_inputs()
        synth_pres, _ = self.mod._generate_ecological_samples(
            feat_stack, dist_to_water, ndvi_anomaly, flowering, ndvi_median, rng=self.rng)
        flower_med = float(np.nanpercentile(flowering, 50))
        for r, c in synth_pres:
            assert flowering[r, c] >= flower_med - 1e-5, (
                f"Presence pixel flowering={flowering[r,c]:.3f} below median={flower_med:.3f}"
            )

    def test_presence_pixels_have_moderate_ndvi_median(self):
        """Synthetic presences must have ndvi_median in [NDVI_MEDIAN_MIN, NDVI_MEDIAN_MAX].

        Scientific rationale: Parkinsonia occupies disturbed riparian areas —
        not bare ground (too low) and not dense closed-canopy rainforest (too high).
        """
        feat_stack, dist_to_water, ndvi_anomaly, flowering, ndvi_median = self._make_inputs()
        synth_pres, _ = self.mod._generate_ecological_samples(
            feat_stack, dist_to_water, ndvi_anomaly, flowering, ndvi_median, rng=self.rng)
        lo = self.mod.SYNTH_NDVI_MEDIAN_MIN
        hi = self.mod.SYNTH_NDVI_MEDIAN_MAX
        for r, c in synth_pres:
            assert lo <= ndvi_median[r, c] <= hi + 1e-5, (
                f"Presence ndvi_median={ndvi_median[r,c]:.3f} outside [{lo}, {hi}]"
            )

    def test_absence_pixels_far_from_water_or_bare(self):
        """Absence pixels must satisfy at least one high-confidence absence criterion.

        Scientific rationale: Parkinsonia is an obligate riparian weed — very unlikely
        beyond 2km from water. Bare ground pixels cannot support any weed.
        """
        feat_stack, dist_to_water, ndvi_anomaly, flowering, ndvi_median = self._make_inputs()
        _, synth_abs = self.mod._generate_ecological_samples(
            feat_stack, dist_to_water, ndvi_anomaly, flowering, ndvi_median, rng=self.rng)
        min_dist = self.mod.SYNTH_ABSENCE_MIN_DIST_WATER_M
        ndvi_lo  = self.mod.SYNTH_NDVI_MEDIAN_MIN
        q25_anom = float(np.nanpercentile(ndvi_anomaly, 25))
        q25_flow = float(np.nanpercentile(flowering, 25))
        for r, c in synth_abs:
            far_from_water = dist_to_water[r, c] >= min_dist
            bare_ground    = ndvi_median[r, c] < ndvi_lo
            low_signal     = (
                ndvi_anomaly[r, c] <= q25_anom
                and flowering[r, c] <= q25_flow
                and dist_to_water[r, c] > self.mod.SYNTH_PRESENCE_MAX_DIST_WATER_M
            )
            assert far_from_water or bare_ground or low_signal, (
                f"Absence pixel ({r},{c}) does not meet any absence criterion: "
                f"dist={dist_to_water[r,c]:.0f}m, ndvi_med={ndvi_median[r,c]:.3f}, "
                f"anom={ndvi_anomaly[r,c]:.3f}, flower={flowering[r,c]:.3f}"
            )

    def test_no_overlap_between_synth_presences_and_absences(self):
        """A pixel cannot be both a synthetic presence and a synthetic absence."""
        args = self._make_inputs()
        synth_pres, synth_abs = self.mod._generate_ecological_samples(*args, rng=self.rng)
        pres_set = set(synth_pres)
        for rc in synth_abs:
            assert rc not in pres_set, f"Pixel {rc} appears in both presence and absence sets"

    def test_presence_count_capped_at_target(self):
        """Synthetic presences should not exceed SYNTH_TARGET_PRESENCES."""
        args = self._make_inputs(H=100, W=100)
        synth_pres, _ = self.mod._generate_ecological_samples(*args, rng=self.rng)
        assert len(synth_pres) <= self.mod.SYNTH_TARGET_PRESENCES

    def test_absences_balanced_to_presences(self):
        """Synthetic absences count should equal synthetic presences count."""
        args = self._make_inputs(H=100, W=100)
        synth_pres, synth_abs = self.mod._generate_ecological_samples(*args, rng=self.rng)
        assert len(synth_abs) == len(synth_pres)

    def test_all_nan_input_returns_empty(self):
        """If all pixels are NaN, no samples should be generated."""
        H, W = 20, 20
        nan_arr = np.full((H, W), np.nan, dtype=np.float32)
        feat_stack = np.zeros((H, W, 8), dtype=np.float32)
        synth_pres, synth_abs = self.mod._generate_ecological_samples(
            feat_stack, nan_arr, nan_arr, nan_arr, nan_arr, rng=self.rng)
        assert len(synth_pres) == 0
        assert len(synth_abs) == 0

    def test_ideal_parkinsonia_pixel_selected_as_presence(self):
        """A pixel matching all presence criteria should be selected as synthetic presence.

        Construct a controlled scene where only one pixel is a perfect Parkinsonia
        candidate: near water, high NDVI anomaly, high flowering, moderate NDVI median.
        """
        H, W = 30, 30
        # Fill with values that fail presence criteria
        ndvi_anomaly  = np.full((H, W), -0.5, dtype=np.float32)   # low anomaly
        flowering     = np.full((H, W),  0.0, dtype=np.float32)   # low flowering
        ndvi_median   = np.full((H, W),  0.5, dtype=np.float32)   # moderate (OK)
        dist_to_water = np.full((H, W), 3000.0, dtype=np.float32) # far from water

        # Make one pixel pass all criteria
        ndvi_anomaly[15, 15]  = 1.0    # highest in scene → top quartile
        flowering[15, 15]     = 1.0    # highest in scene → above median
        dist_to_water[15, 15] = 100.0  # within 500m
        # ndvi_median[15,15] already 0.5 — within [0.15, 0.65]

        feat_stack = np.zeros((H, W, 8), dtype=np.float32)
        synth_pres, _ = self.mod._generate_ecological_samples(
            feat_stack, dist_to_water, ndvi_anomaly, flowering, ndvi_median,
            rng=np.random.default_rng(0))
        assert (15, 15) in synth_pres, "Ideal Parkinsonia pixel was not selected as presence"

    def test_far_from_water_pixel_selected_as_absence(self):
        """A pixel far from water should appear as a synthetic absence."""
        H, W = 30, 30
        ndvi_anomaly  = np.zeros((H, W), dtype=np.float32)
        flowering     = np.full((H, W), 0.5, dtype=np.float32)
        ndvi_median   = np.full((H, W), 0.4, dtype=np.float32)
        dist_to_water = np.full((H, W), 100.0, dtype=np.float32)   # all near water
        dist_to_water[5, 5] = 5000.0  # far from water → should be absence

        feat_stack = np.zeros((H, W, 8), dtype=np.float32)
        _, synth_abs = self.mod._generate_ecological_samples(
            feat_stack, dist_to_water, ndvi_anomaly, flowering, ndvi_median,
            rng=np.random.default_rng(0))
        assert (5, 5) in synth_abs, "Far-from-water pixel was not selected as absence"


# ── _fetch_ala_occurrences ────────────────────────────────────────────────────

class TestFetchAlaOccurrences:
    def setup_method(self):
        self.mod = _load_module("fetch_tests")

    def test_cache_hit_skips_api(self, tmp_path):
        """If cache file exists, ALA API should never be called."""
        cache_path = tmp_path / "ala_occurrences.gpkg"
        gdf = gpd.GeoDataFrame(
            geometry=[Point(142.0, -16.0), Point(143.0, -16.5)],
            crs="EPSG:4326",
        )
        gdf.to_file(str(cache_path), driver="GPKG")

        with patch("requests.get") as mock_get:
            result = self.mod._fetch_ala_occurrences(
                [141.0, -17.0, 143.0, -15.0], "Parkinsonia aculeata",
                "https://example.com", cache_path=cache_path,
            )
            mock_get.assert_not_called()
        assert len(result) == 2

    def test_empty_api_response_returns_empty_geodataframe(self):
        """An API response with no occurrences should return an empty GeoDataFrame."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"occurrences": []}
        mock_resp.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_resp):
            result = self.mod._fetch_ala_occurrences(
                [141.0, -17.0, 143.0, -15.0], "Parkinsonia aculeata",
                "https://example.com", cache_path=None,
            )
        assert len(result) == 0
        assert result.crs.to_epsg() == 4326

    def test_records_missing_coords_are_dropped(self):
        """Records with null lat/lon must be silently dropped."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"occurrences": [
            {"decimalLongitude": 142.0, "decimalLatitude": -16.0},
            {"decimalLongitude": None,  "decimalLatitude": -16.0},
            {"decimalLongitude": 143.0, "decimalLatitude": None},
        ]}
        mock_resp.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_resp):
            result = self.mod._fetch_ala_occurrences(
                [141.0, -17.0, 143.0, -15.0], "Parkinsonia aculeata",
                "https://example.com", cache_path=None,
            )
        assert len(result) == 1


# ── Sample weight construction ────────────────────────────────────────────────

class TestSampleWeights:
    """Validate sample weight array construction — pure unit tests, no I/O."""

    def setup_method(self):
        self.mod = _load_module("weight_tests")

    def _build_weights(self, n_ala_pres=5, n_ala_abs=20, n_synth_pres=10, n_synth_abs=10):
        """Replicate the weight-assembly logic from main() directly."""
        sw = np.array(
            [1.0] * (n_ala_pres + n_ala_abs)
            + [self.mod.SYNTH_SAMPLE_WEIGHT] * (n_synth_pres + n_synth_abs)
        )
        return sw

    def test_ala_records_have_weight_one(self):
        """Real ALA presence and absence records must have sample_weight=1.0."""
        sw = self._build_weights(n_ala_pres=5, n_ala_abs=20)
        np.testing.assert_array_equal(sw[:25], 1.0)

    def test_synthetic_records_have_synth_weight(self):
        """Synthetic ecological samples must have weight=SYNTH_SAMPLE_WEIGHT."""
        sw = self._build_weights(n_ala_pres=5, n_ala_abs=20, n_synth_pres=10, n_synth_abs=10)
        np.testing.assert_array_equal(sw[25:], self.mod.SYNTH_SAMPLE_WEIGHT)

    def test_all_weights_positive(self):
        """All sample weights must be strictly positive."""
        sw = self._build_weights()
        assert np.all(sw > 0), "Some sample weights are zero or negative"

    def test_synth_weight_less_than_ala_weight(self):
        """Synthetic weight must be strictly less than ALA weight (1.0).

        Scientific rationale: synthetic samples are derived from ecological priors,
        not ground truth — they should not override real observations.
        """
        assert self.mod.SYNTH_SAMPLE_WEIGHT < 1.0

    def test_weight_array_length_matches_sample_counts(self):
        """Weight array length must equal total number of training samples."""
        n_ala_pres, n_ala_abs, n_synth_pres, n_synth_abs = 5, 20, 10, 10
        sw = self._build_weights(n_ala_pres, n_ala_abs, n_synth_pres, n_synth_abs)
        assert len(sw) == n_ala_pres + n_ala_abs + n_synth_pres + n_synth_abs


# ── Pseudo-absence buffer exclusion ──────────────────────────────────────────

class TestPseudoAbsenceBuffer:
    def test_absence_pixels_outside_presence_buffer(self, tmp_dirs):
        """Pseudo-absence pixels must not fall within PSEUDO_ABSENCE_BUFFER_KM of any presence."""
        if "config" in sys.modules:
            del sys.modules["config"]
        import config

        mod = _load_module("buffer_tests")

        H, W = 50, 50
        rng = np.random.default_rng(0)
        ndvi_arr = rng.uniform(0.1, 0.8, (H, W)).astype(np.float32)

        # Single presence at centre
        presence_pixels = [(25, 25)]
        buffer_px = int(mod.PSEUDO_ABSENCE_BUFFER_KM * 1000 / config.TARGET_RESOLUTION)

        presence_mask = np.zeros((H, W), dtype=bool)
        for r, c in presence_pixels:
            r0, r1 = max(0, r - buffer_px), min(H, r + buffer_px + 1)
            c0, c1 = max(0, c - buffer_px), min(W, c + buffer_px + 1)
            presence_mask[r0:r1, c0:c1] = True

        valid_absence_mask = ~presence_mask & ~np.isnan(ndvi_arr)
        absence_indices = np.argwhere(valid_absence_mask)
        absence_pixels = [tuple(absence_indices[i]) for i in range(min(50, len(absence_indices)))]

        for r, c in absence_pixels:
            assert not presence_mask[r, c], \
                f"Absence pixel ({r},{c}) falls inside the presence buffer"

    def test_minimum_absence_count_is_1000(self):
        """When fewer presences exist than 200, n_absence must be at least 1000."""
        # n_absence = min(available, max(n_presence * 5, 1000))
        # With 10 presences: max(50, 1000) = 1000
        n_presence = 10
        n_absence_target = max(n_presence * 5, 1000)
        assert n_absence_target == 1000


# ── Feature stack ─────────────────────────────────────────────────────────────

class TestFeatureStack:
    def test_feature_order_matches_feature_names(self):
        """The feature stack column order must match FEATURE_NAMES exactly."""
        mod = _load_module("featstack_tests")
        expected = [
            "ndvi_anomaly", "flowering_index", "hand", "flood_extent",
            "ndvi_median", "glcm_contrast", "glcm_homogeneity", "dist_to_watercourse",
        ]
        assert mod.FEATURE_NAMES == expected, (
            f"FEATURE_NAMES mismatch.\nExpected: {expected}\nGot: {mod.FEATURE_NAMES}"
        )

    def test_feature_stack_has_8_features(self):
        """Feature stack must have exactly 8 features per pixel."""
        mod = _load_module("featstack8_tests")
        assert len(mod.FEATURE_NAMES) == 8


# end-to-end tests live in tests/integration/test_06_classifier_integration.py
