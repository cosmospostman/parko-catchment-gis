"""TD-* tests for TAMDataset, TAMSample, collate_fn."""

from __future__ import annotations

import datetime

import numpy as np
import polars as pl
import pytest
import torch

from tam.core.dataset import (
    BAND_COLS,
    MAX_SEQ_LEN,
    MIN_OBS_PER_YEAR,
    N_BANDS,
    S1_FEATURE_COLS,
    TAMDataset,
    TAMSample,
    collate_fn,
    despeckle_s1,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _date_range(start: str, periods: int, freq_days: int) -> list[datetime.date]:
    d0 = datetime.date.fromisoformat(start)
    return [d0 + datetime.timedelta(days=freq_days * i) for i in range(periods)]


def _make_pixel_df(band_cols, pids, n_obs=30, scl_purity=1.0, year=2023, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    start = datetime.date(2023, 1, 15)
    rows = []
    for pid in pids:
        dates = [start + datetime.timedelta(days=12 * i) for i in range(n_obs)]
        for d in dates:
            rows.append({
                "point_id": pid,
                "date": str(d),
                "scl_purity": scl_purity,
                "year": year,
                **{b: float(rng.uniform(0.01, 0.5)) for b in band_cols},
            })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# TD-1
# ---------------------------------------------------------------------------

class TestTD1BasicConstruction:
    def test_length(self, pixel_df, labels, band_cols):
        ds = TAMDataset(pixel_df, labels)
        assert len(ds) == 2


# ---------------------------------------------------------------------------
# TD-2
# ---------------------------------------------------------------------------

class TestTD2BandNormalisationTraining:
    def test_band_mean_shape_and_finite(self, pixel_df, labels, band_cols):
        ds = TAMDataset(pixel_df, labels)
        assert ds.band_mean.shape == (N_BANDS,)
        assert np.all(np.isfinite(ds.band_mean))

    def test_band_std_no_zeros(self, pixel_df, labels, band_cols):
        ds = TAMDataset(pixel_df, labels)
        assert np.all(ds.band_std >= 1e-6)

    def test_band_std_is_clamped_not_raw(self, band_cols, labels):
        rng = np.random.default_rng(0)
        rows = []
        for pid in ["px_pres", "px_abs"]:
            dates = _date_range("2023-01-01", 20, 15)
            for d in dates:
                row = {
                    "point_id": pid, "date": str(d),
                    "scl_purity": 1.0, "year": 2023,
                    **{b: float(rng.uniform(0.01, 0.5)) for b in band_cols},
                }
                row["B02"] = 0.1
                rows.append(row)
        df = pl.DataFrame(rows)
        ds = TAMDataset(df, labels)
        assert ds.band_std[0] >= 1e-6


# ---------------------------------------------------------------------------
# TD-3
# ---------------------------------------------------------------------------

class TestTD3BandNormalisationInference:
    def test_supplied_stats_not_recomputed(self, pixel_df, labels, band_cols):
        supplied_mean = np.ones(N_BANDS, dtype=np.float32) * 0.25
        supplied_std  = np.ones(N_BANDS, dtype=np.float32) * 0.05
        ds = TAMDataset(pixel_df, labels, band_mean=supplied_mean, band_std=supplied_std)
        np.testing.assert_array_equal(ds.band_mean, supplied_mean)
        np.testing.assert_array_equal(ds.band_std, supplied_std)


# ---------------------------------------------------------------------------
# TD-4
# ---------------------------------------------------------------------------

class TestTD4NormalisedBandsApproximatelyStandardised:
    def test_mean_approx_zero_std_approx_one(self, band_cols):
        rng = np.random.default_rng(7)
        pids = [f"p{i}" for i in range(20)]
        labels = {pid: float(i % 2) for i, pid in enumerate(pids)}
        df = _make_pixel_df(band_cols, pids, n_obs=40, rng=rng)
        ds = TAMDataset(df, labels)
        all_normed = []
        for i in range(len(ds)):
            s = ds[i]
            n = int((~s.mask).sum())
            all_normed.append(s.bands[:n].numpy())
        arr = np.concatenate(all_normed, axis=0)
        np.testing.assert_allclose(arr.mean(axis=0), 0.0, atol=0.1)
        np.testing.assert_allclose(arr.std(axis=0), 1.0, atol=0.1)


# ---------------------------------------------------------------------------
# TD-5
# ---------------------------------------------------------------------------

class TestTD5PaddingMaskCorrectness:
    def test_mask_shape_and_values(self, pixel_df, labels):
        ds = TAMDataset(pixel_df, labels)
        s = ds[0]
        n = int((~s.mask).sum())
        assert s.mask.shape == (MAX_SEQ_LEN,)
        assert not s.mask[:n].any()
        assert s.mask[n:].all()


# ---------------------------------------------------------------------------
# TD-6
# ---------------------------------------------------------------------------

class TestTD6PaddingPositionsZero:
    def test_bands_zero_at_padding(self, pixel_df, labels):
        ds = TAMDataset(pixel_df, labels)
        s = ds[0]
        n = int((~s.mask).sum())
        assert (s.bands[n:] == 0).all()

    def test_doy_zero_at_padding(self, pixel_df, labels):
        ds = TAMDataset(pixel_df, labels)
        s = ds[0]
        n = int((~s.mask).sum())
        assert (s.doy[n:] == 0).all()


# ---------------------------------------------------------------------------
# TD-7
# ---------------------------------------------------------------------------

class TestTD7DOYValidRange:
    def test_doy_in_1_to_365(self, pixel_df, labels):
        ds = TAMDataset(pixel_df, labels)
        for i in range(len(ds)):
            s = ds[i]
            n = int((~s.mask).sum())
            doy_valid = s.doy[:n]
            assert (doy_valid >= 1).all() and (doy_valid <= 365).all()


# ---------------------------------------------------------------------------
# TD-8
# ---------------------------------------------------------------------------

class TestTD8DOYMonotonicallyNonDecreasing:
    def test_doy_sorted(self, pixel_df, labels):
        ds = TAMDataset(pixel_df, labels)
        for i in range(len(ds)):
            s = ds[i]
            n = int((~s.mask).sum())
            diffs = np.diff(s.doy[:n].numpy())
            assert (diffs >= 0).all()


# ---------------------------------------------------------------------------
# TD-9
# ---------------------------------------------------------------------------

class TestTD9SCLFilterRemovesLowPurity:
    def test_fewer_obs_after_scl_filter(self, band_cols, labels):
        rng = np.random.default_rng(42)
        rows = []
        for pid in ["px_pres", "px_abs"]:
            dates = _date_range("2023-01-01", 20, 15)
            for i, d in enumerate(dates):
                rows.append({
                    "point_id": pid, "date": str(d),
                    "scl_purity": 0.1 if i % 2 == 0 else 1.0,
                    "year": 2023,
                    **{b: float(rng.uniform(0.01, 0.5)) for b in band_cols},
                })
        df_mixed = pl.DataFrame(rows)
        df_clean = df_mixed.with_columns(pl.lit(1.0).alias("scl_purity"))

        ds_mixed = TAMDataset(df_mixed, labels, scl_purity_min=0.5, min_obs_per_year=1)
        ds_clean = TAMDataset(df_clean, labels, scl_purity_min=0.5, min_obs_per_year=1)

        def total_valid(ds):
            return sum(int((~ds[i].mask).sum()) for i in range(len(ds)))

        assert total_valid(ds_mixed) < total_valid(ds_clean)


# ---------------------------------------------------------------------------
# TD-10
# ---------------------------------------------------------------------------

class TestTD10MinObsPerYearFilter:
    def test_pixel_with_few_obs_excluded(self, band_cols, labels):
        rng = np.random.default_rng(42)
        rows = []
        for pid, n_obs in [("px_pres", 30), ("px_abs", 30), ("px_sparse", 5)]:
            dates = _date_range("2023-01-01", n_obs, 15)
            for d in dates:
                rows.append({
                    "point_id": pid, "date": str(d),
                    "scl_purity": 1.0, "year": 2023,
                    **{b: float(rng.uniform(0.01, 0.5)) for b in band_cols},
                })
        df = pl.DataFrame(rows)
        all_labels = {"px_pres": 1.0, "px_abs": 0.0, "px_sparse": 0.0}
        ds = TAMDataset(df, all_labels, min_obs_per_year=8)
        assert "px_sparse" not in ds.unique_pixels()

    def test_filter_is_post_scl(self, band_cols):
        rng = np.random.default_rng(0)
        rows = []
        dates = _date_range("2023-01-01", 10, 30)
        for i, d in enumerate(dates):
            rows.append({
                "point_id": "px_tricky", "date": str(d),
                "scl_purity": 1.0 if i < 4 else 0.1,
                "year": 2023,
                **{b: float(rng.uniform(0.01, 0.5)) for b in band_cols},
            })
        df = pl.DataFrame(rows)
        labels = {"px_tricky": 1.0}
        ds = TAMDataset(df, labels, scl_purity_min=0.5, min_obs_per_year=8)
        assert "px_tricky" not in ds.unique_pixels()


# ---------------------------------------------------------------------------
# TD-11
# ---------------------------------------------------------------------------

class TestTD11LabelsRestrictPixels:
    def test_unlabelled_pixel_excluded(self, band_cols, labels):
        rng = np.random.default_rng(42)
        extra_rows = []
        dates = _date_range("2023-01-01", 20, 15)
        for d in dates:
            extra_rows.append({
                "point_id": "px_extra", "date": str(d),
                "scl_purity": 1.0, "year": 2023,
                **{b: float(rng.uniform(0.01, 0.5)) for b in band_cols},
            })
        df_base = _make_pixel_df(band_cols, ["px_pres", "px_abs"])
        df = pl.concat([df_base, pl.DataFrame(extra_rows)])
        ds = TAMDataset(df, labels)
        assert "px_extra" not in ds.unique_pixels()


# ---------------------------------------------------------------------------
# TD-12
# ---------------------------------------------------------------------------

class TestTD12InferenceModeIncludesAll:
    def test_labels_none_includes_all_pixels(self, pixel_df, band_cols):
        ds = TAMDataset(pixel_df, labels=None)
        assert set(ds.unique_pixels()) == {"px_pres", "px_abs"}

    def test_label_is_zero_in_inference(self, pixel_df, band_cols):
        ds = TAMDataset(pixel_df, labels=None)
        for i in range(len(ds)):
            assert ds[i].label.item() == 0.0


# ---------------------------------------------------------------------------
# TD-13
# ---------------------------------------------------------------------------

class TestTD13DOYJitterTrainVaries:
    def test_jitter_produces_variation(self, pixel_df, labels):
        ds = TAMDataset(pixel_df, labels, doy_jitter=7)
        doy_arrays = [ds[0].doy.numpy().copy() for _ in range(20)]
        unique = {arr.tobytes() for arr in doy_arrays}
        assert len(unique) >= 2


# ---------------------------------------------------------------------------
# TD-14
# ---------------------------------------------------------------------------

class TestTD14DOYJitterOrderPreserved:
    def test_order_preserved_after_jitter(self, pixel_df, labels):
        ds = TAMDataset(pixel_df, labels, doy_jitter=7)
        for _ in range(500):
            s = ds[0]
            n = int((~s.mask).sum())
            diffs = np.diff(s.doy[:n].numpy())
            assert (diffs >= 0).all(), "DOY order violated after jitter"


# ---------------------------------------------------------------------------
# TD-15
# ---------------------------------------------------------------------------

class TestTD15DOYJitterValuesInRange:
    def test_doy_stays_in_1_to_365(self, pixel_df, labels):
        ds = TAMDataset(pixel_df, labels, doy_jitter=7)
        for _ in range(500):
            s = ds[0]
            n = int((~s.mask).sum())
            doy_valid = s.doy[:n].numpy()
            assert doy_valid.min() >= 1
            assert doy_valid.max() <= 365


# ---------------------------------------------------------------------------
# TD-16
# ---------------------------------------------------------------------------

class TestTD16DOYJitterZeroDeterministic:
    def test_zero_jitter_is_deterministic(self, pixel_df, labels):
        ds = TAMDataset(pixel_df, labels, doy_jitter=0)
        doys = [ds[0].doy.numpy().copy() for _ in range(10)]
        for arr in doys[1:]:
            np.testing.assert_array_equal(arr, doys[0])


# ---------------------------------------------------------------------------
# TD-17
# ---------------------------------------------------------------------------

class TestTD17MultiYearPixelMultipleWindows:
    def test_multi_year_produces_extra_windows(self, band_cols, labels):
        rng = np.random.default_rng(42)
        rows = []
        for year in [2023, 2024]:
            dates = _date_range(f"{year}-01-15", 20, 15)
            for d in dates:
                rows.append({
                    "point_id": "px_pres", "date": str(d),
                    "scl_purity": 1.0, "year": year,
                    **{b: float(rng.uniform(0.01, 0.5)) for b in band_cols},
                })
        dates_abs = _date_range("2023-01-15", 20, 15)
        for d in dates_abs:
            rows.append({
                "point_id": "px_abs", "date": str(d),
                "scl_purity": 1.0, "year": 2023,
                **{b: float(rng.uniform(0.01, 0.5)) for b in band_cols},
            })
        df = pl.DataFrame(rows)
        ds = TAMDataset(df, labels)
        assert len(ds) == 3


# ---------------------------------------------------------------------------
# TD-18
# ---------------------------------------------------------------------------

class TestTD18CollateFnShapes:
    def test_batch_shapes(self, pixel_df, labels):
        ds = TAMDataset(pixel_df, labels)
        samples = [ds[i % len(ds)] for i in range(4)]
        batch = collate_fn(samples)

        assert batch["bands"].shape == (4, MAX_SEQ_LEN, N_BANDS)
        assert batch["bands"].dtype == torch.float32
        assert batch["doy"].shape == (4, MAX_SEQ_LEN)
        assert batch["doy"].dtype == torch.int64
        assert batch["mask"].shape == (4, MAX_SEQ_LEN)
        assert batch["mask"].dtype == torch.bool
        assert batch["label"].shape == (4,)
        assert batch["label"].dtype == torch.float32
        assert batch["weight"].shape == (4,)
        assert batch["weight"].dtype == torch.float32
        assert len(batch["point_id"]) == 4
        assert len(batch["year"]) == 4


# ---------------------------------------------------------------------------
# TD-19
# ---------------------------------------------------------------------------

class TestTD19BandStatsCopies:
    def test_mutating_returned_stats_does_not_affect_dataset(self, pixel_df, labels):
        ds = TAMDataset(pixel_df, labels)
        mean, std = ds.band_stats
        original_mean = ds.band_mean.copy()
        original_std  = ds.band_std.copy()
        mean[:] = 99.0
        std[:] = 99.0
        np.testing.assert_array_equal(ds.band_mean, original_mean)
        np.testing.assert_array_equal(ds.band_std, original_std)


# ---------------------------------------------------------------------------
# TD-20
# ---------------------------------------------------------------------------

class TestTD20BandNoiseZeroIsDeterministic:
    def test_no_noise_same_bands_each_call(self, pixel_df, labels):
        ds = TAMDataset(pixel_df, labels, band_noise_std=0.0)
        bands = [ds[0].bands.numpy().copy() for _ in range(10)]
        for arr in bands[1:]:
            np.testing.assert_array_equal(arr, bands[0])


# ---------------------------------------------------------------------------
# TD-21
# ---------------------------------------------------------------------------

class TestTD21BandNoiseVariesAcrossCalls:
    def test_nonzero_noise_produces_variation(self, pixel_df, labels):
        ds = TAMDataset(pixel_df, labels, band_noise_std=0.5)
        bands = [ds[0].bands.numpy().copy() for _ in range(20)]
        unique = {arr.tobytes() for arr in bands}
        assert len(unique) >= 2


# ---------------------------------------------------------------------------
# TD-22
# ---------------------------------------------------------------------------

class TestTD22BandNoiseIsConstantWithinWindow:
    def test_offset_is_uniform_across_observations(self, pixel_df, labels):
        ds_zero  = TAMDataset(pixel_df, labels, band_noise_std=0.0)
        ds_noisy = TAMDataset(pixel_df, labels, band_noise_std=0.5)

        zero_sample = ds_zero[0]
        n = int((~zero_sample.mask).sum())
        expected_diff = (zero_sample.bands[0] - zero_sample.bands[1]).numpy()

        mismatches = 0
        for _ in range(200):
            noisy = ds_noisy[0]
            diff  = (noisy.bands[0] - noisy.bands[1]).numpy()
            if not np.allclose(diff, expected_diff, atol=1e-5):
                mismatches += 1

        assert mismatches == 0, (
            f"{mismatches}/200 draws had within-window variation — "
            "offset is not constant across observations"
        )


# ---------------------------------------------------------------------------
# TD-23
# ---------------------------------------------------------------------------

class TestTD23BandNoisePaddingStaysZero:
    def test_padding_positions_remain_zero(self, pixel_df, labels):
        ds = TAMDataset(pixel_df, labels, band_noise_std=0.5)
        for _ in range(50):
            s = ds[0]
            n = int((~s.mask).sum())
            assert (s.bands[n:] == 0).all(), "Band jitter leaked into padding positions"


# ---------------------------------------------------------------------------
# Helpers for S1 despeckle tests
# ---------------------------------------------------------------------------

def _make_s1_df(pids: list[str], n_obs: int = 20, rng=None) -> pl.DataFrame:
    """Make a minimal S1 DataFrame with vh/vv linear power columns."""
    if rng is None:
        rng = np.random.default_rng(0)
    rows = []
    for pid in pids:
        dates = _date_range("2023-01-01", n_obs, 6)
        for d in dates:
            rows.append({
                "point_id": pid,
                "date": str(d),
                "source": "S1",
                "vh": float(rng.uniform(1e-4, 1e-2)),
                "vv": float(rng.uniform(1e-4, 1e-2)),
                "orbit": "ascending",
            })
    return pl.DataFrame(rows)


def _make_combined_df(band_cols, pids, n_s2=20, n_s1=20, rng=None) -> pl.DataFrame:
    """Make a combined S2+S1 DataFrame for TAMDataset tests."""
    if rng is None:
        rng = np.random.default_rng(1)
    rows = []
    for pid in pids:
        s2_dates = _date_range("2023-01-10", n_s2, 15)
        for d in s2_dates:
            rows.append({
                "point_id": pid,
                "date": str(d),
                "source": "S2",
                "scl_purity": 1.0,
                "year": 2023,
                "vh": None, "vv": None, "orbit": None,
                **{b: float(rng.uniform(0.01, 0.5)) for b in band_cols},
            })
        s1_dates = _date_range("2023-01-08", n_s1, 6)
        for d in s1_dates:
            rows.append({
                "point_id": pid,
                "date": str(d),
                "source": "S1",
                "scl_purity": None,
                "year": 2023,
                "vh": float(rng.uniform(1e-4, 1e-2)),
                "vv": float(rng.uniform(1e-4, 1e-2)),
                "orbit": "ascending",
                **{b: None for b in band_cols},
            })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# TD-24  despeckle_s1 — core function
# ---------------------------------------------------------------------------

class TestTD24DespeckleS1Core:
    def test_noop_when_window_less_than_2(self):
        df = _make_s1_df(["px1"])
        original_vh = df["vh"].to_numpy().copy()
        result = despeckle_s1(df, window=0)
        np.testing.assert_array_equal(result["vh"].to_numpy(), original_vh)

    def test_noop_when_window_1(self):
        df = _make_s1_df(["px1"])
        original_vh = df["vh"].to_numpy().copy()
        result = despeckle_s1(df, window=1)
        np.testing.assert_array_equal(result["vh"].to_numpy(), original_vh)

    def test_output_shape_unchanged(self):
        df = _make_s1_df(["px1", "px2"], n_obs=15)
        result = despeckle_s1(df, window=3)
        assert result.shape == df.shape

    def test_smoothed_values_differ_from_original(self):
        rng = np.random.default_rng(42)
        df = _make_s1_df(["px1"], n_obs=20, rng=rng)
        result = despeckle_s1(df, window=5)
        assert not np.allclose(result["vh"].to_numpy(), df["vh"].to_numpy())

    def test_smoothing_is_per_pixel_not_across_pixels(self):
        rows = []
        for d in _date_range("2023-01-01", 15, 6):
            rows.append({"point_id": "px_low",  "date": str(d), "source": "S1",
                         "vh": 1e-5, "vv": 1e-5, "orbit": "ascending"})
            rows.append({"point_id": "px_high", "date": str(d), "source": "S1",
                         "vh": 1e-1, "vv": 1e-1, "orbit": "ascending"})
        df = pl.DataFrame(rows)
        result = despeckle_s1(df, window=3)
        low_vh  = result.filter(pl.col("point_id") == "px_low")["vh"].to_numpy()
        high_vh = result.filter(pl.col("point_id") == "px_high")["vh"].to_numpy()
        assert (low_vh  < 1e-3).all(), "px_low smoothed values unexpectedly large"
        assert (high_vh > 1e-3).all(), "px_high smoothed values unexpectedly small"

    def test_wider_window_produces_smoother_result(self):
        rng = np.random.default_rng(3)
        df = _make_s1_df(["px1"], n_obs=30, rng=rng)
        r3 = despeckle_s1(df, window=3)
        r7 = despeckle_s1(df, window=7)
        std3 = r3["vh"].std()
        std7 = r7["vh"].std()
        assert std7 <= std3, "Wider window should produce equal or lower variance"

    def test_does_not_modify_input(self):
        df = _make_s1_df(["px1"])
        original = df["vh"].to_numpy().copy()
        despeckle_s1(df, window=5)
        np.testing.assert_array_equal(df["vh"].to_numpy(), original)


# ---------------------------------------------------------------------------
# TD-26  TAMDataset with s1_only + despeckle
# ---------------------------------------------------------------------------

class TestTD26TAMDatasetS1OnlyDespeckle:
    def _make_s1_pixel_df(self, pids, n_obs=20):
        rng = np.random.default_rng(5)
        rows = []
        for pid in pids:
            for d in _date_range("2023-01-01", n_obs, 6):
                rows.append({
                    "point_id": pid,
                    "date": str(d),
                    "source": "S1",
                    "year": 2023,
                    "vh": float(rng.uniform(1e-4, 1e-2)),
                    "vv": float(rng.uniform(1e-4, 1e-2)),
                    "orbit": "ascending",
                })
        return pl.DataFrame(rows)

    def test_constructs_without_error(self):
        df = self._make_s1_pixel_df(["px_pres", "px_abs"])
        labels = {"px_pres": 1.0, "px_abs": 0.0}
        ds = TAMDataset(df, labels, use_s1="s1_only", s1_despeckle_window=3)
        assert len(ds) == 2

    def test_despeckle_changes_band_values(self):
        df = self._make_s1_pixel_df(["px_pres", "px_abs"])
        labels = {"px_pres": 1.0, "px_abs": 0.0}
        ds_raw   = TAMDataset(df, labels, use_s1="s1_only", s1_despeckle_window=0)
        ds_clean = TAMDataset(df, labels, use_s1="s1_only", s1_despeckle_window=5)
        raw_bands   = ds_raw[0].bands.numpy()
        clean_bands = ds_clean[0].bands.numpy()
        assert not np.allclose(raw_bands, clean_bands), \
            "Despeckle window=5 should change band values vs window=0"

    def test_sample_shape_unchanged(self):
        df = self._make_s1_pixel_df(["px_pres"])
        labels = {"px_pres": 1.0}
        ds = TAMDataset(df, labels, use_s1="s1_only", s1_despeckle_window=3)
        s = ds[0]
        assert s.bands.shape == (MAX_SEQ_LEN, len(S1_FEATURE_COLS))
