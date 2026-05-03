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

from tam.core.dataset import BAND_COLS, S1_FEATURE_COLS
from tam.core.model import TAMClassifier
from tam.core.score import (
    score_pixels_chunked, score_location_years, score_tiles_chunked,
    _compute_s1_despeckle_lookup,
)
from tam.core.config import TAMConfig


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
    """Tiny seeded model + zero-mean unit-std stats — enough for shape checks."""
    from tam.core.dataset import ALL_FEATURE_COLS
    n_features = len(ALL_FEATURE_COLS)  # 13: S2-only features
    cfg = TAMConfig(d_model=16, n_heads=2, n_layers=1, d_ff=32,
                    n_bands=n_features, use_s1=False, n_global_features=0)
    torch.manual_seed(42)
    model = TAMClassifier.from_config(cfg)
    model._use_s1 = False
    model._pixel_zscore = False
    model.eval()
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


# ---------------------------------------------------------------------------
# TS-MT  Multi-tile scoring via score_location_years
# ---------------------------------------------------------------------------

class TestMultiTileScoring:
    """score_location_years with per-tile parquets produces identical results
    to a single merged parquet (regression guard for tile-partitioned layout)."""

    def test_multi_tile_produces_one_row_per_pixel(self, tmp_path):
        """Passing two tile parquets for the same year yields one score per pixel."""
        pixels_a = ["px_0000_0000", "px_0001_0000"]
        pixels_b = ["px_0002_0000", "px_0003_0000"]
        path_a = _make_parquet(tmp_path, pixels=pixels_a, years=[2022], filename="tileA.parquet")
        path_b = _make_parquet(tmp_path, pixels=pixels_b, years=[2022], filename="tileB.parquet")
        model, band_mean, band_std = _stub_model()

        result = score_location_years(
            year_parquets=[(2022, path_a), (2022, path_b)],
            model=model,
            band_mean=band_mean,
            band_std=band_std,
            device="cpu",
            end_year=2022,
        )
        all_pixels = pixels_a + pixels_b
        assert set(result["point_id"]) == set(all_pixels)
        assert result["point_id"].nunique() == len(all_pixels)

    def test_multi_tile_matches_merged_parquet(self, tmp_path):
        """Scores from per-tile parquets match scores from a single merged parquet."""
        pixels_a = ["px_0000_0000", "px_0001_0000"]
        pixels_b = ["px_0002_0000", "px_0003_0000"]
        years = [2022, 2023]

        path_a = _make_parquet(tmp_path, pixels=pixels_a, years=years, filename="tileA.parquet")
        path_b = _make_parquet(tmp_path, pixels=pixels_b, years=years, filename="tileB.parquet")

        # Build merged parquet from both tiles
        df_a = pd.read_parquet(path_a)
        df_b = pd.read_parquet(path_b)
        merged = pd.concat([df_a, df_b], ignore_index=True).sort_values(["point_id", "date"])
        merged_path = tmp_path / "merged.parquet"
        pq.write_table(pa.Table.from_pandas(merged, preserve_index=False), merged_path, row_group_size=N_OBS_PER_YEAR)

        model, band_mean, band_std = _stub_model()

        result_tiled = score_location_years(
            year_parquets=[(y, p) for y in years for p in [path_a, path_b]],
            model=model, band_mean=band_mean, band_std=band_std,
            device="cpu", end_year=max(years),
        ).sort_values("point_id").reset_index(drop=True)

        result_merged = score_location_years(
            year_parquets=[(y, merged_path) for y in years],
            model=model, band_mean=band_mean, band_std=band_std,
            device="cpu", end_year=max(years),
        ).sort_values("point_id").reset_index(drop=True)

        pd.testing.assert_frame_equal(result_tiled, result_merged, check_like=True)


# ---------------------------------------------------------------------------
# TestTileShardedOutput — score_tiles_chunked
# ---------------------------------------------------------------------------

class TestTileShardedOutput:
    """score_tiles_chunked writes one parquet per tile with uint8 prob_tam."""

    def test_one_file_per_tile(self, tmp_path):
        pixels_a = ["px_0000_0000", "px_0001_0000"]
        pixels_b = ["px_0002_0000", "px_0003_0000"]
        path_a = _make_parquet(tmp_path, pixels=pixels_a, years=[2022, 2023],
                               tile_tag="54LWH", filename="54LWH.parquet")
        path_b = _make_parquet(tmp_path, pixels=pixels_b, years=[2022, 2023],
                               tile_tag="54LWJ", filename="54LWJ.parquet")
        model, band_mean, band_std = _stub_model()

        out_dir = tmp_path / "scores"
        final_paths = score_tiles_chunked(
            tile_year_parquets={
                "54LWH": [(2022, path_a), (2023, path_a)],
                "54LWJ": [(2022, path_b), (2023, path_b)],
            },
            model=model,
            band_mean=band_mean,
            band_std=band_std,
            out_dir=out_dir,
            device="cpu",
            end_year=2023,
        )

        assert len(final_paths) == 2
        names = {p.name for p in final_paths}
        assert "54LWH.scores.parquet" in names
        assert "54LWJ.scores.parquet" in names

    def test_schema_uint8_in_range(self, tmp_path):
        path = _make_parquet(tmp_path, pixels=["px_0000_0000", "px_0001_0000"],
                             years=[2022], tile_tag="54LWH", filename="54LWH.parquet")
        model, band_mean, band_std = _stub_model()

        out_dir = tmp_path / "scores"
        final_paths = score_tiles_chunked(
            tile_year_parquets={"54LWH": [(2022, path)]},
            model=model, band_mean=band_mean, band_std=band_std,
            out_dir=out_dir, device="cpu", end_year=2022,
        )

        import pyarrow.parquet as pq
        tbl = pq.read_table(final_paths[0])
        assert tbl.schema.field("point_id").type == pa.string()
        assert tbl.schema.field("prob_tam").type == pa.uint8()
        scores = tbl.column("prob_tam").to_pylist()
        assert all(0 <= v <= 100 for v in scores)

    def test_pixels_partitioned_by_tile(self, tmp_path):
        pixels_a = ["px_0000_0000", "px_0001_0000"]
        pixels_b = ["px_0002_0000", "px_0003_0000"]
        path_a = _make_parquet(tmp_path, pixels=pixels_a, years=[2022],
                               tile_tag="54LWH", filename="54LWH.parquet")
        path_b = _make_parquet(tmp_path, pixels=pixels_b, years=[2022],
                               tile_tag="54LWJ", filename="54LWJ.parquet")
        model, band_mean, band_std = _stub_model()

        out_dir = tmp_path / "scores"
        final_paths = score_tiles_chunked(
            tile_year_parquets={
                "54LWH": [(2022, path_a)],
                "54LWJ": [(2022, path_b)],
            },
            model=model, band_mean=band_mean, band_std=band_std,
            out_dir=out_dir, device="cpu", end_year=2022,
        )

        by_name = {p.name: pd.read_parquet(p) for p in final_paths}
        pids_a = set(by_name["54LWH.scores.parquet"]["point_id"])
        pids_b = set(by_name["54LWJ.scores.parquet"]["point_id"])
        assert pids_a == set(pixels_a)
        assert pids_b == set(pixels_b)
        assert pids_a.isdisjoint(pids_b)

    def test_staging_cleaned_up(self, tmp_path):
        path = _make_parquet(tmp_path, pixels=["px_0000_0000"], years=[2022],
                             tile_tag="54LWH", filename="54LWH.parquet")
        model, band_mean, band_std = _stub_model()

        out_dir = tmp_path / "scores"
        score_tiles_chunked(
            tile_year_parquets={"54LWH": [(2022, path)]},
            model=model, band_mean=band_mean, band_std=band_std,
            out_dir=out_dir, device="cpu", end_year=2022,
        )
        assert not (out_dir / "staging").exists()

    def test_idempotent_staging_skip(self, tmp_path):
        """A pre-existing staging file is reused and produces the same final result."""
        pixels = ["px_0000_0000", "px_0001_0000"]
        path = _make_parquet(tmp_path, pixels=pixels, years=[2022],
                             tile_tag="54LWH", filename="54LWH.parquet")
        model, band_mean, band_std = _stub_model()

        out_dir = tmp_path / "scores"
        kwargs = dict(
            tile_year_parquets={"54LWH": [(2022, path)]},
            model=model, band_mean=band_mean, band_std=band_std,
            out_dir=out_dir, device="cpu", end_year=2022,
        )

        paths1 = score_tiles_chunked(**kwargs)
        df1 = pd.read_parquet(paths1[0]).sort_values("point_id").reset_index(drop=True)

        # Remove final output so phase 2 re-runs, but leave no staging (already cleaned)
        paths1[0].unlink()

        paths2 = score_tiles_chunked(**kwargs)
        df2 = pd.read_parquet(paths2[0]).sort_values("point_id").reset_index(drop=True)

        pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# TS-BUF  Reader buffer-split correctness
#
# The _reader stage in score_pixels_chunked accumulates row groups into a
# buffer of size buffer_row_groups, then calls _emit().  _emit() holds back
# the trailing pixel's rows as a "leftover" so that pixel-year windows are
# never split across two emitted chunks.
#
# These tests verify that scores are identical regardless of how row groups
# are partitioned into buffers — i.e. the leftover logic is transparent to
# the inference result.
# ---------------------------------------------------------------------------

def _make_split_parquet(
    tmp_path: Path,
    pixels: list[str],
    obs_per_rg: int,
    filename: str = "pixels.parquet",
) -> Path:
    """Write a parquet where each pixel's observations are split across
    multiple small row groups of exactly obs_per_rg rows.

    This maximises the chance that a pixel-year window straddles a buffer
    boundary when buffer_row_groups=1.
    """
    rng = np.random.default_rng(99)
    n_total = N_OBS_PER_YEAR
    rows = []
    for pid in pixels:
        dates = pd.date_range("2023-01-15", periods=n_total, freq="23D")
        for d in dates:
            rows.append({
                "point_id": pid,
                "date": d,
                "scl_purity": 1.0,
                **_band_row(rng),
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[us]")
    path = tmp_path / filename
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        path,
        row_group_size=obs_per_rg,
    )
    return path


class TestReaderBufferSplit:
    """buffer_row_groups=1 forces a flush after every row group, maximising
    the frequency with which pixel-year windows straddle buffer boundaries.
    Scores must match those produced with a large buffer (no splitting)."""

    def test_split_scores_match_no_split_single_pixel(self, tmp_path):
        """Single pixel whose obs span many row groups: split vs no-split must agree."""
        path = _make_split_parquet(tmp_path, pixels=["px1"], obs_per_rg=3)
        model, band_mean, band_std = _stub_model()

        r_split = score_pixels_chunked(
            path, model, band_mean, band_std,
            device="cpu", buffer_row_groups=1,
        )
        r_bulk = score_pixels_chunked(
            path, model, band_mean, band_std,
            device="cpu", buffer_row_groups=999,
        )

        assert set(r_split["point_id"]) == {"px1"}
        s_split = r_split.loc[r_split["point_id"] == "px1", "prob_tam"].iloc[0]
        s_bulk  = r_bulk.loc[r_bulk["point_id"] == "px1",  "prob_tam"].iloc[0]
        assert s_split == pytest.approx(s_bulk, abs=1e-5)

    def test_split_scores_match_no_split_multiple_pixels(self, tmp_path):
        """Multiple pixels interleaved across many small row groups."""
        pixels = ["px1", "px2", "px3"]
        path = _make_split_parquet(tmp_path, pixels=pixels, obs_per_rg=4)
        model, band_mean, band_std = _stub_model()

        r_split = score_pixels_chunked(
            path, model, band_mean, band_std,
            device="cpu", buffer_row_groups=1,
        ).sort_values("point_id").reset_index(drop=True)

        r_bulk = score_pixels_chunked(
            path, model, band_mean, band_std,
            device="cpu", buffer_row_groups=999,
        ).sort_values("point_id").reset_index(drop=True)

        assert set(r_split["point_id"]) == set(pixels)
        pd.testing.assert_frame_equal(r_split, r_bulk, check_like=True)

    def test_split_produces_one_row_per_pixel(self, tmp_path):
        """buffer_row_groups=1 must not create duplicate rows for the same pixel."""
        pixels = ["px1", "px2", "px3", "px4"]
        path = _make_split_parquet(tmp_path, pixels=pixels, obs_per_rg=3)
        model, band_mean, band_std = _stub_model()

        result = score_pixels_chunked(
            path, model, band_mean, band_std,
            device="cpu", buffer_row_groups=1,
        )

        assert result["point_id"].nunique() == len(pixels)
        assert len(result) == len(pixels)

    def test_split_boundary_pixel_not_dropped(self, tmp_path):
        """The last pixel in a flushed buffer must not be silently dropped.

        _emit strips the trailing pixel into leftover.  If that leftover is
        never re-emitted (bug: leftover only emitted when buf is non-empty),
        the last pixel in the file disappears.
        """
        # Write exactly one pixel whose obs fill exactly one row group.
        # With buffer_row_groups=1 this pixel is always the boundary pixel
        # at the final flush — it ends up in leftover, then must be emitted
        # by the post-loop "leftover = _emit(buffer, leftover, is_last=True)"
        # followed by "if not leftover.empty: raw_q.put(leftover)".
        rng = np.random.default_rng(42)
        rows = []
        for d in pd.date_range("2023-01-15", periods=N_OBS_PER_YEAR, freq="23D"):
            rows.append({"point_id": "px_last", "date": d, "scl_purity": 1.0, **_band_row(rng)})
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[us]")
        path = tmp_path / "last.parquet"
        # Single row group — pixel is always the boundary/leftover candidate.
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path,
                       row_group_size=N_OBS_PER_YEAR)

        model, band_mean, band_std = _stub_model()
        result = score_pixels_chunked(
            path, model, band_mean, band_std,
            device="cpu", buffer_row_groups=1,
        )

        assert "px_last" in result["point_id"].values

    def test_split_leftover_when_entire_buffer_is_one_pixel(self, tmp_path):
        """When every row in the buffer belongs to the boundary pixel, the chunk
        after stripping is empty — _emit must not enqueue an empty DataFrame,
        and the pixel must still be scored correctly on the next flush."""
        rng = np.random.default_rng(7)
        # Two pixels, each with exactly obs_per_rg observations, so each row
        # group is entirely one pixel.  With buffer_row_groups=1, every flush
        # sees a chunk that is all one pixel — the boundary strip leaves an
        # empty chunk (not emitted) and the whole thing goes to leftover.
        # The leftover should then be prepended to the next buffer.
        rows = []
        for pid in ["px_a", "px_b"]:
            for d in pd.date_range("2023-01-15", periods=N_OBS_PER_YEAR, freq="23D"):
                rows.append({"point_id": pid, "date": d, "scl_purity": 1.0, **_band_row(rng)})
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[us]")
        path = tmp_path / "twopix.parquet"
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path,
                       row_group_size=N_OBS_PER_YEAR)

        model, band_mean, band_std = _stub_model()
        result = score_pixels_chunked(
            path, model, band_mean, band_std,
            device="cpu", buffer_row_groups=1,
        )

        assert set(result["point_id"]) == {"px_a", "px_b"}
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Helpers for S1 despeckle tests
# ---------------------------------------------------------------------------

def _make_s1_parquet(tmp_path: Path, pixels: list[str], years: list[int],
                     filename: str = "s1.parquet") -> Path:
    """Write a minimal S1-only parquet with vh/vv linear power columns."""
    rng = np.random.default_rng(99)
    rows = []
    for pid in pixels:
        for yr in years:
            dates = pd.date_range(f"{yr}-01-01", periods=N_OBS_PER_YEAR, freq="6D")
            for d in dates:
                rows.append({
                    "point_id": pid,
                    "date": d,
                    "source": "S1",
                    "vh": float(rng.uniform(1e-4, 1e-2)),
                    "vv": float(rng.uniform(1e-4, 1e-2)),
                    "orbit": "ascending",
                })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[us]")
    path = tmp_path / filename
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path,
                   row_group_size=N_OBS_PER_YEAR)
    return path


def _stub_s1_model(n_features: int = 4) -> tuple[TAMClassifier, np.ndarray, np.ndarray]:
    """Tiny S1-only model stub."""
    cfg = TAMConfig(d_model=16, n_heads=2, n_layers=1, d_ff=32,
                    n_bands=n_features, use_s1="s1_only", n_global_features=0)
    torch.manual_seed(7)
    model = TAMClassifier.from_config(cfg)
    model._use_s1 = "s1_only"
    model._pixel_zscore = False
    model.eval()
    band_mean = np.zeros(n_features, dtype=np.float32)
    band_std  = np.ones(n_features, dtype=np.float32)
    return model, band_mean, band_std


# ---------------------------------------------------------------------------
# TS-DS-1  _compute_s1_despeckle_lookup
# ---------------------------------------------------------------------------

class TestTSDS1DespeckleLookup:
    def test_returns_dataframe_with_expected_columns(self, tmp_path):
        path = _make_s1_parquet(tmp_path, pixels=["px1", "px2"], years=[2023])
        result = _compute_s1_despeckle_lookup(path, window=3)
        assert isinstance(result, pd.DataFrame)
        for col in ("point_id", "date", "vh", "vv"):
            assert col in result.columns, f"Missing column: {col}"

    def test_row_count_matches_input_s1_rows(self, tmp_path):
        pixels = ["px1", "px2"]
        path = _make_s1_parquet(tmp_path, pixels=pixels, years=[2023])
        result = _compute_s1_despeckle_lookup(path, window=3)
        assert len(result) == len(pixels) * N_OBS_PER_YEAR

    def test_smoothed_values_differ_from_raw(self, tmp_path):
        path = _make_s1_parquet(tmp_path, pixels=["px1"], years=[2023])
        raw = _compute_s1_despeckle_lookup(path, window=1)   # no-op
        smoothed = _compute_s1_despeckle_lookup(path, window=5)
        # window=5 should change at least some values vs window=1 (no-op)
        assert not raw["vh"].reset_index(drop=True).equals(
            smoothed["vh"].reset_index(drop=True)
        )

    def test_empty_parquet_returns_empty_dataframe(self, tmp_path):
        # Parquet with no S1 rows (S2 only — no vh/vv columns at all)
        rng = np.random.default_rng(0)
        rows = [{"point_id": "px1", "date": pd.Timestamp("2023-01-01"),
                 "scl_purity": 1.0, **{b: 0.1 for b in BAND_COLS}}]
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[us]")
        path = tmp_path / "s2only.parquet"
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
        result = _compute_s1_despeckle_lookup(path, window=3)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# TS-DS-2  score_pixels_chunked with s1_only + despeckle
# ---------------------------------------------------------------------------

class TestTSDS2ScorePixelsChunkedS1Despeckle:
    def test_scores_produced_with_despeckle_window(self, tmp_path):
        path = _make_s1_parquet(tmp_path, pixels=["px1", "px2"], years=[2023])
        model, band_mean, band_std = _stub_s1_model()
        result = score_pixels_chunked(
            path, model, band_mean, band_std,
            device="cpu", s1_only=True, s1_despeckle_window=3,
        )
        assert set(result["point_id"]) == {"px1", "px2"}
        assert result["prob_tam"].between(0, 1).all()

    def test_despeckle_changes_scores_vs_no_despeckle(self, tmp_path):
        path = _make_s1_parquet(tmp_path, pixels=["px1", "px2"], years=[2023])
        model, band_mean, band_std = _stub_s1_model()
        r_raw   = score_pixels_chunked(
            path, model, band_mean, band_std,
            device="cpu", s1_only=True, s1_despeckle_window=0,
        ).set_index("point_id")
        r_clean = score_pixels_chunked(
            path, model, band_mean, band_std,
            device="cpu", s1_only=True, s1_despeckle_window=5,
        ).set_index("point_id")
        # At least one pixel's score should differ after smoothing
        diffs = (r_raw["prob_tam"] - r_clean["prob_tam"]).abs()
        assert diffs.max() > 1e-4, "Expected scores to differ after despeckle"

    def test_output_schema_unchanged_with_despeckle(self, tmp_path):
        path = _make_s1_parquet(tmp_path, pixels=["px1"], years=[2023])
        model, band_mean, band_std = _stub_s1_model()
        result = score_pixels_chunked(
            path, model, band_mean, band_std,
            device="cpu", s1_only=True, s1_despeckle_window=3,
        )
        assert list(result.columns) == ["point_id", "prob_tam"]

    def test_despeckle_window_0_matches_no_flag(self, tmp_path):
        path = _make_s1_parquet(tmp_path, pixels=["px1", "px2"], years=[2023])
        model, band_mean, band_std = _stub_s1_model()
        r_default = score_pixels_chunked(
            path, model, band_mean, band_std,
            device="cpu", s1_only=True,
        ).set_index("point_id")
        r_zero = score_pixels_chunked(
            path, model, band_mean, band_std,
            device="cpu", s1_only=True, s1_despeckle_window=0,
        ).set_index("point_id")
        pd.testing.assert_frame_equal(r_default, r_zero)
