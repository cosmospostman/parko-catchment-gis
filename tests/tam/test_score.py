"""TS-* tests for score_pixels_chunked and its scoring flags.

Each test builds a minimal in-memory parquet (one row group per pixel-year),
trains or stubs a tiny TAMClassifier, then runs score_pixels_chunked and
asserts on the output.  All tests are CPU-only and complete in seconds.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from tam.core.dataset import BAND_COLS
from tam.core.model import TAMClassifier
from tam.core.score import score_pixels_chunked
from tam.config import TAMConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_OBS_PER_YEAR = 15  # enough to clear the min_obs_per_year=8 threshold


def _band_row(rng: np.random.Generator) -> dict:
    return {b: float(rng.uniform(0.01, 0.5)) for b in BAND_COLS}


def _make_parquet(
    tmp_path: Path,
    pixels: list[str],
    years: list[int],
    scl_purity: float = 1.0,
    tile_tag: str | None = None,
    filename: str = "pixels.parquet",
) -> Path:
    """Write a parquet with one row group per (pixel, year) combination.

    pixels × years are the cross product of observations to generate.
    Each combination gets N_OBS_PER_YEAR rows spread uniformly through the year.
    """
    rng = np.random.default_rng(0)
    rows = []
    for pid in pixels:
        for yr in years:
            dates = pd.date_range(f"{yr}-01-15", periods=N_OBS_PER_YEAR, freq="23D")
            for d in dates:
                row = {
                    "point_id": pid,
                    "date": d,
                    "scl_purity": scl_purity,
                }
                if tile_tag is not None:
                    row["item_id"] = f"S2_{tile_tag}_foo"
                row.update(_band_row(rng))
                rows.append(row)

    df = pd.DataFrame(rows)
    # Store date as timestamp[us] — matches what the real collector writes
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[us]")

    path = tmp_path / filename
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, row_group_size=N_OBS_PER_YEAR)
    return path


def _stub_model() -> tuple[TAMClassifier, np.ndarray, np.ndarray]:
    """Tiny untrained model + zero-mean unit-std stats — enough for shape checks."""
    cfg = TAMConfig(d_model=16, n_heads=2, n_layers=1, d_ff=32)
    model = TAMClassifier.from_config(cfg)
    model.eval()
    n_features = 13  # ALL_FEATURE_COLS length
    band_mean = np.zeros(n_features, dtype=np.float32)
    band_std  = np.ones(n_features, dtype=np.float32)
    return model, band_mean, band_std


# ---------------------------------------------------------------------------
# TS-1  end_year filters future observations
# ---------------------------------------------------------------------------

class TestTS1EndYearFiltersObservations:
    """Scoring with end_year=2021 must not use 2022+ observations.

    We give two pixels identical band values in 2021 but very different
    values in 2022.  With end_year=2021 both pixels see the same data,
    so their scores should be close.  Without the filter they diverge.
    """

    def test_scores_identical_when_only_shared_year_visible(self, tmp_path):
        rng = np.random.default_rng(1)

        rows = []
        dates_2021 = pd.date_range("2021-01-15", periods=N_OBS_PER_YEAR, freq="23D")
        dates_2022 = pd.date_range("2022-01-15", periods=N_OBS_PER_YEAR, freq="23D")

        shared_bands = _band_row(rng)  # identical for both pixels

        for pid in ["px_a", "px_b"]:
            for d in dates_2021:
                rows.append({"point_id": pid, "date": d, "scl_purity": 1.0, **shared_bands})
            # 2022: px_a gets very high signal, px_b gets very low — should be invisible at end_year=2021
            unique_val = 0.9 if pid == "px_a" else 0.01
            for d in dates_2022:
                rows.append({"point_id": pid, "date": d, "scl_purity": 1.0,
                             **{b: unique_val for b in BAND_COLS}})

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[us]")
        path = tmp_path / "pixels.parquet"
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)

        model, band_mean, band_std = _stub_model()
        result = score_pixels_chunked(
            path, model, band_mean, band_std,
            end_year=2021, decay=0.0, device="cpu",
        )

        assert set(result["point_id"]) == {"px_a", "px_b"}
        score_a = result.loc[result["point_id"] == "px_a", "prob_tam"].iloc[0]
        score_b = result.loc[result["point_id"] == "px_b", "prob_tam"].iloc[0]
        # With end_year=2021 both pixels saw the same data — scores must be identical
        assert score_a == pytest.approx(score_b, abs=1e-5)

    def test_scores_differ_without_end_year_filter(self, tmp_path):
        """Sanity-check: without end_year the divergent 2022 data is visible."""
        rng = np.random.default_rng(1)

        rows = []
        dates_2021 = pd.date_range("2021-01-15", periods=N_OBS_PER_YEAR, freq="23D")
        dates_2022 = pd.date_range("2022-01-15", periods=N_OBS_PER_YEAR, freq="23D")
        shared_bands = _band_row(rng)

        for pid in ["px_a", "px_b"]:
            for d in dates_2021:
                rows.append({"point_id": pid, "date": d, "scl_purity": 1.0, **shared_bands})
            unique_val = 0.9 if pid == "px_a" else 0.01
            for d in dates_2022:
                rows.append({"point_id": pid, "date": d, "scl_purity": 1.0,
                             **{b: unique_val for b in BAND_COLS}})

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[us]")
        path = tmp_path / "pixels.parquet"
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)

        model, band_mean, band_std = _stub_model()
        result = score_pixels_chunked(
            path, model, band_mean, band_std,
            end_year=2022, decay=0.0, device="cpu",
        )

        score_a = result.loc[result["point_id"] == "px_a", "prob_tam"].iloc[0]
        score_b = result.loc[result["point_id"] == "px_b", "prob_tam"].iloc[0]
        assert score_a != pytest.approx(score_b, abs=1e-3)

    def test_no_end_year_uses_all_data(self, tmp_path):
        """end_year=None must return all pixels, not an empty result."""
        path = _make_parquet(tmp_path, pixels=["px1", "px2"], years=[2020, 2023])
        model, band_mean, band_std = _stub_model()
        result = score_pixels_chunked(
            path, model, band_mean, band_std,
            end_year=None, device="cpu",
        )
        assert set(result["point_id"]) == {"px1", "px2"}


# ---------------------------------------------------------------------------
# TS-2  scl_purity threshold
# ---------------------------------------------------------------------------

class TestTS2SclPurityThreshold:
    """Observations below scl_purity_min must be dropped before inference."""

    def test_cloudy_pixel_excluded(self, tmp_path):
        """A pixel whose only observations are below the threshold must not appear."""
        rng = np.random.default_rng(2)
        rows = []
        dates = pd.date_range("2023-01-15", periods=N_OBS_PER_YEAR, freq="23D")

        # px_clean: all observations at purity 1.0 — should be scored
        for d in dates:
            rows.append({"point_id": "px_clean", "date": d, "scl_purity": 1.0, **_band_row(rng)})

        # px_cloudy: all observations below threshold — should be absent from output
        for d in dates:
            rows.append({"point_id": "px_cloudy", "date": d, "scl_purity": 0.1, **_band_row(rng)})

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[us]")
        path = tmp_path / "pixels.parquet"
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)

        model, band_mean, band_std = _stub_model()
        result = score_pixels_chunked(
            path, model, band_mean, band_std,
            scl_purity_min=0.5, device="cpu",
        )

        assert "px_clean" in result["point_id"].values
        assert "px_cloudy" not in result["point_id"].values

    def test_purity_zero_includes_all(self, tmp_path):
        """scl_purity_min=0.0 passes every observation."""
        path = _make_parquet(tmp_path, pixels=["px1"], years=[2023], scl_purity=0.01)
        model, band_mean, band_std = _stub_model()
        result = score_pixels_chunked(
            path, model, band_mean, band_std,
            scl_purity_min=0.0, device="cpu",
        )
        assert "px1" in result["point_id"].values


# ---------------------------------------------------------------------------
# TS-3  decay weighting  (tested via aggregate_year_probs directly)
# ---------------------------------------------------------------------------

class TestTS3DecayWeighting:
    """Exponential decay up-weights recent years.

    Tested directly against aggregate_year_probs rather than through
    score_pixels_chunked: the model is untrained so its per-year scores are
    effectively random, making end-to-end decay assertions brittle.
    aggregate_year_probs owns the decay math and is the right unit to test.
    """

    def test_high_decay_emphasises_recent_year(self):
        """Pixel with high score only in end_year outranks one with high score
        only three years back, under strong decay."""
        from tam.core.score import aggregate_year_probs

        decay = 2.0
        end_year = 2024

        pids  = [np.array(["px_recent"]), np.array(["px_recent"]),
                 np.array(["px_old"]),    np.array(["px_old"])]
        years = [np.array([2024], dtype=np.int32), np.array([2021], dtype=np.int32),
                 np.array([2024], dtype=np.int32), np.array([2021], dtype=np.int32)]
        probs = [np.array([0.9], dtype=np.float32), np.array([0.1], dtype=np.float32),
                 np.array([0.1], dtype=np.float32), np.array([0.9], dtype=np.float32)]

        df = aggregate_year_probs(pids, years, probs, end_year=end_year, decay=decay)
        recent = df.loc[df["point_id"] == "px_recent", "prob_tam"].iloc[0]
        old    = df.loc[df["point_id"] == "px_old",    "prob_tam"].iloc[0]
        assert recent > old

    def test_zero_decay_equalises_years(self):
        """decay=0 makes all years equally weighted — mirrored pixels score equally."""
        from tam.core.score import aggregate_year_probs

        pids  = [np.array(["px_a"]), np.array(["px_a"]),
                 np.array(["px_b"]), np.array(["px_b"])]
        years = [np.array([2024], np.int32), np.array([2021], np.int32),
                 np.array([2024], np.int32), np.array([2021], np.int32)]
        probs = [np.array([0.9], np.float32), np.array([0.1], np.float32),
                 np.array([0.1], np.float32), np.array([0.9], np.float32)]

        df = aggregate_year_probs(pids, years, probs, end_year=2024, decay=0.0)
        score_a = df.loc[df["point_id"] == "px_a", "prob_tam"].iloc[0]
        score_b = df.loc[df["point_id"] == "px_b", "prob_tam"].iloc[0]
        assert score_a == pytest.approx(score_b, abs=1e-5)

    def test_decay_flag_passed_through_to_aggregation(self, tmp_path):
        """score_pixels_chunked passes decay to aggregate_year_probs.

        With a two-year parquet and decay=0 vs decay=5, the resulting scores
        must differ for a pixel that has data in both years (different weighting).
        """
        rng = np.random.default_rng(3)
        rows = []
        for yr in [2020, 2024]:
            for d in pd.date_range(f"{yr}-01-15", periods=N_OBS_PER_YEAR, freq="23D"):
                rows.append({"point_id": "px1", "date": d, "scl_purity": 1.0,
                             **_band_row(rng)})
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[us]")
        path = tmp_path / "pixels.parquet"
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)

        model, band_mean, band_std = _stub_model()
        r0 = score_pixels_chunked(path, model, band_mean, band_std,
                                  end_year=2024, decay=0.0, device="cpu")
        r5 = score_pixels_chunked(path, model, band_mean, band_std,
                                  end_year=2024, decay=5.0, device="cpu")

        s0 = r0.loc[r0["point_id"] == "px1", "prob_tam"].iloc[0]
        s5 = r5.loc[r5["point_id"] == "px1", "prob_tam"].iloc[0]
        # The two runs used different decay — scores must differ
        assert s0 != pytest.approx(s5, abs=1e-4)


# ---------------------------------------------------------------------------
# TS-4  tile_id filter
# ---------------------------------------------------------------------------

class TestTS4TileIdFilter:
    """tile_id restricts inference to observations from a single S2 tile."""

    def test_only_matching_tile_scored(self, tmp_path):
        """Pixels that appear only in tile B should be absent when filtering for tile A."""
        rng = np.random.default_rng(4)
        rows = []
        dates = pd.date_range("2023-01-15", periods=N_OBS_PER_YEAR, freq="23D")

        for d in dates:
            rows.append({"point_id": "px_tile_a", "date": d, "scl_purity": 1.0,
                         "item_id": "S2_55HBU_2023", **_band_row(rng)})
        for d in dates:
            rows.append({"point_id": "px_tile_b", "date": d, "scl_purity": 1.0,
                         "item_id": "S2_54HWE_2023", **_band_row(rng)})

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[us]")
        path = tmp_path / "pixels.parquet"
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)

        model, band_mean, band_std = _stub_model()
        result = score_pixels_chunked(
            path, model, band_mean, band_std,
            tile_id="55HBU", device="cpu",
        )

        assert "px_tile_a" in result["point_id"].values
        assert "px_tile_b" not in result["point_id"].values

    def test_no_tile_filter_returns_all(self, tmp_path):
        """tile_id=None must return all pixels regardless of item_id."""
        rng = np.random.default_rng(4)
        rows = []
        dates = pd.date_range("2023-01-15", periods=N_OBS_PER_YEAR, freq="23D")

        for pid, tile in [("px_a", "55HBU"), ("px_b", "54HWE")]:
            for d in dates:
                rows.append({"point_id": pid, "date": d, "scl_purity": 1.0,
                             "item_id": f"S2_{tile}_2023", **_band_row(rng)})

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[us]")
        path = tmp_path / "pixels.parquet"
        # tile_id=None means item_id column is not read — write without it
        df_notile = df.drop(columns=["item_id"])
        pq.write_table(pa.Table.from_pandas(df_notile, preserve_index=False), path)

        model, band_mean, band_std = _stub_model()
        result = score_pixels_chunked(
            path, model, band_mean, band_std,
            tile_id=None, device="cpu",
        )
        assert set(result["point_id"]) == {"px_a", "px_b"}


# ---------------------------------------------------------------------------
# TS-5  output schema
# ---------------------------------------------------------------------------

class TestTS5OutputSchema:
    """score_pixels_chunked must always return a DataFrame with the expected columns."""

    def test_columns_present(self, tmp_path):
        path = _make_parquet(tmp_path, pixels=["px1"], years=[2023])
        model, band_mean, band_std = _stub_model()
        result = score_pixels_chunked(path, model, band_mean, band_std, device="cpu")
        assert "point_id" in result.columns
        assert "prob_tam" in result.columns

    def test_probs_in_unit_interval(self, tmp_path):
        path = _make_parquet(tmp_path, pixels=["px1", "px2", "px3"], years=[2022, 2023])
        model, band_mean, band_std = _stub_model()
        result = score_pixels_chunked(path, model, band_mean, band_std, device="cpu")
        assert (result["prob_tam"] >= 0.0).all()
        assert (result["prob_tam"] <= 1.0).all()

    def test_one_row_per_pixel(self, tmp_path):
        pixels = ["px1", "px2", "px3"]
        path = _make_parquet(tmp_path, pixels=pixels, years=[2021, 2022, 2023])
        model, band_mean, band_std = _stub_model()
        result = score_pixels_chunked(path, model, band_mean, band_std, device="cpu")
        assert len(result) == len(pixels)
        assert result["point_id"].nunique() == len(pixels)
