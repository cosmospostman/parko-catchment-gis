"""TS-* tests for score_pixels_chunked and its scoring flags.

Each test builds a minimal in-memory parquet (one row group per pixel-year),
trains or stubs a tiny TAMClassifier, then runs score_pixels_chunked and
asserts on the output.  All tests are CPU-only and complete in seconds.
"""

from __future__ import annotations

import io
from pathlib import Path

import datetime

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from tam.core.dataset import ALL_FEATURE_COLS, BAND_COLS, S1_FEATURE_COLS, MIN_S1_OBS_PER_YEAR
from tam.core.model import TAMClassifier
from tam.core.score import (
    score_pixels_chunked, score_location_years, score_tiles_chunked,
    _compute_s1_despeckle_lookup, _extract_mixed_pa, _PASlice,
)
from tam.core.config import TAMConfig
from tam.core._preprocess_numba import extract_features
from analysis.constants import UINT16_BAND_SCALE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_OBS_PER_YEAR = 15  # enough to clear the min_obs_per_year=8 threshold


def _band_row(rng: np.random.Generator) -> dict:
    return {b: float(rng.uniform(0.01, 0.5)) for b in BAND_COLS}


def _band_row_uint16(rng: np.random.Generator) -> dict:
    """Raw-cache-shaped band row: uint16 DN values (reflectance × UINT16_BAND_SCALE).

    Matches the real chunkstore schema (e.g. /mnt/external/chunkstore/<year>/<tile>/
    *.parquet stores B02..B12 as UInt16 ×10000) — as opposed to _band_row's
    already-reflectance-scaled floats, which is what every other mixed-mode
    fixture here writes and why the missing /UINT16_BAND_SCALE conversion in
    _extract_mixed_pa went undetected (see [[project_annual_feature_parity_bugs]]
    follow-on: an all-zero scoring run traced to raw DN values, ~10,000x the
    expected reflectance scale, being fed straight to the model).
    """
    return {b: int(rng.integers(50, 4000)) for b in BAND_COLS}


def _make_parquet(
    tmp_path: Path,
    pixels: list[str],
    years: list[int],
    scl_purity: float = 1.0,
    tile_tag: str | None = None,
    filename: str = "pixels.parquet",
) -> Path:
    """Write a parquet with one row group per (pixel, year) combination."""
    rng = np.random.default_rng(0)
    rows = []
    for pid in pixels:
        for yr in years:
            start = datetime.date(yr, 1, 15)
            dates = [start + datetime.timedelta(days=23 * i) for i in range(N_OBS_PER_YEAR)]
            for d in dates:
                row = {
                    "point_id": pid,
                    "date": datetime.datetime(d.year, d.month, d.day),
                    "scl_purity": scl_purity,
                }
                if tile_tag is not None:
                    row["item_id"] = f"S2_{tile_tag}_foo"
                row.update(_band_row(rng))
                rows.append(row)

    df = pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Datetime("us"))
    )
    path = tmp_path / filename
    pq.write_table(df.to_arrow(), path, row_group_size=N_OBS_PER_YEAR)
    return path


def _stub_model() -> tuple[TAMClassifier, np.ndarray, np.ndarray]:
    """Tiny seeded model + zero-mean unit-std stats — enough for shape checks."""
    from tam.core.dataset import ALL_FEATURE_COLS
    n_features = len(ALL_FEATURE_COLS)  # 13: S2-only features
    cfg = TAMConfig(d_model=16, n_heads=2, n_layers=1, d_ff=32,
                    n_bands=n_features, use_s1=False, n_annual_features=0)
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
        dates_2021 = [datetime.date(2021,1,15)+datetime.timedelta(days=23*i) for i in range(N_OBS_PER_YEAR)]
        dates_2022 = [datetime.date(2022,1,15)+datetime.timedelta(days=23*i) for i in range(N_OBS_PER_YEAR)]

        shared_bands = _band_row(rng)  # identical for both pixels

        for pid in ["px_a", "px_b"]:
            for d in dates_2021:
                rows.append({"point_id": pid, "date": d, "scl_purity": 1.0, **shared_bands})
            # 2022: px_a gets very high signal, px_b gets very low — should be invisible at end_year=2021
            unique_val = 0.9 if pid == "px_a" else 0.01
            for d in dates_2022:
                rows.append({"point_id": pid, "date": d, "scl_purity": 1.0,
                             **{b: unique_val for b in BAND_COLS}})

        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("us")))
        path = tmp_path / "pixels.parquet"
        pq.write_table(df.to_arrow(), path)

        model, band_mean, band_std = _stub_model()
        result = score_pixels_chunked(
            path, model, band_mean, band_std,
            end_year=2021, decay=0.0, device="cpu", mixed=False,
        )

        assert set(result["point_id"]) == {"px_a", "px_b"}
        score_a = result.filter(pl.col("point_id") == "px_a")["prob_tam"][0]
        score_b = result.filter(pl.col("point_id") == "px_b")["prob_tam"][0]
        # With end_year=2021 both pixels saw the same data — scores must be identical
        assert score_a == pytest.approx(score_b, abs=1e-5)

    def test_scores_differ_without_end_year_filter(self, tmp_path):
        """Sanity-check: without end_year the divergent 2022 data is visible."""
        rng = np.random.default_rng(1)

        rows = []
        dates_2021 = [datetime.date(2021,1,15)+datetime.timedelta(days=23*i) for i in range(N_OBS_PER_YEAR)]
        dates_2022 = [datetime.date(2022,1,15)+datetime.timedelta(days=23*i) for i in range(N_OBS_PER_YEAR)]
        shared_bands = _band_row(rng)

        for pid in ["px_a", "px_b"]:
            for d in dates_2021:
                rows.append({"point_id": pid, "date": d, "scl_purity": 1.0, **shared_bands})
            unique_val = 0.9 if pid == "px_a" else 0.01
            for d in dates_2022:
                rows.append({"point_id": pid, "date": d, "scl_purity": 1.0,
                             **{b: unique_val for b in BAND_COLS}})

        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("us")))
        path = tmp_path / "pixels.parquet"
        pq.write_table(df.to_arrow(), path)

        model, band_mean, band_std = _stub_model()
        result = score_pixels_chunked(
            path, model, band_mean, band_std,
            end_year=2022, decay=0.0, device="cpu", mixed=False,
        )

        score_a = result.filter(pl.col("point_id") == "px_a")["prob_tam"][0]
        score_b = result.filter(pl.col("point_id") == "px_b")["prob_tam"][0]
        assert score_a != pytest.approx(score_b, abs=1e-3)

    def test_no_end_year_uses_all_data(self, tmp_path):
        """end_year=None must return all pixels, not an empty result."""
        path = _make_parquet(tmp_path, pixels=["px1", "px2"], years=[2020, 2023])
        model, band_mean, band_std = _stub_model()
        result = score_pixels_chunked(
            path, model, band_mean, band_std,
            end_year=None, device="cpu", mixed=False,
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
        dates = [datetime.date(2023,1,15)+datetime.timedelta(days=23*i) for i in range(N_OBS_PER_YEAR)]

        # px_clean: all observations at purity 1.0 — should be scored
        for d in dates:
            rows.append({"point_id": "px_clean", "date": d, "scl_purity": 1.0, **_band_row(rng)})

        # px_cloudy: all observations below threshold — should be absent from output
        for d in dates:
            rows.append({"point_id": "px_cloudy", "date": d, "scl_purity": 0.1, **_band_row(rng)})

        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("us")))
        path = tmp_path / "pixels.parquet"
        pq.write_table(df.to_arrow(), path)

        model, band_mean, band_std = _stub_model()
        result = score_pixels_chunked(
            path, model, band_mean, band_std,
            scl_purity_min=0.5, device="cpu", mixed=False,
        )

        assert "px_clean" in result["point_id"].to_list()
        assert "px_cloudy" not in result["point_id"].to_list()

    def test_purity_zero_includes_all(self, tmp_path):
        """scl_purity_min=0.0 passes every observation."""
        path = _make_parquet(tmp_path, pixels=["px1"], years=[2023], scl_purity=0.01)
        model, band_mean, band_std = _stub_model()
        result = score_pixels_chunked(
            path, model, band_mean, band_std,
            scl_purity_min=0.0, device="cpu", mixed=False,
        )
        assert "px1" in result["point_id"].to_list()


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
        recent = df.filter(pl.col("point_id") == "px_recent")["prob_tam"][0]
        old    = df.filter(pl.col("point_id") == "px_old")["prob_tam"][0]
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
        score_a = df.filter(pl.col("point_id") == "px_a")["prob_tam"][0]
        score_b = df.filter(pl.col("point_id") == "px_b")["prob_tam"][0]
        assert score_a == pytest.approx(score_b, abs=1e-5)

    def test_decay_flag_passed_through_to_aggregation(self, tmp_path):
        """score_pixels_chunked passes decay to aggregate_year_probs.

        With a two-year parquet and decay=0 vs decay=5, the resulting scores
        must differ for a pixel that has data in both years (different weighting).
        """
        rng = np.random.default_rng(3)
        rows = []
        for yr in [2020, 2024]:
            for d in [datetime.date(yr,1,15)+datetime.timedelta(days=23*i) for i in range(N_OBS_PER_YEAR)]:
                rows.append({"point_id": "px1", "date": d, "scl_purity": 1.0,
                             **_band_row(rng)})
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("us")))
        path = tmp_path / "pixels.parquet"
        pq.write_table(df.to_arrow(), path)

        model, band_mean, band_std = _stub_model()
        r0 = score_pixels_chunked(path, model, band_mean, band_std,
                                  end_year=2024, decay=0.0, device="cpu", mixed=False)
        r5 = score_pixels_chunked(path, model, band_mean, band_std,
                                  end_year=2024, decay=5.0, device="cpu", mixed=False)

        s0 = r0.filter(pl.col("point_id") == "px1")["prob_tam"][0]
        s5 = r5.filter(pl.col("point_id") == "px1")["prob_tam"][0]
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
        dates = [datetime.date(2023,1,15)+datetime.timedelta(days=23*i) for i in range(N_OBS_PER_YEAR)]

        for d in dates:
            rows.append({"point_id": "px_tile_a", "date": d, "scl_purity": 1.0,
                         "item_id": "S2_55HBU_2023", **_band_row(rng)})
        for d in dates:
            rows.append({"point_id": "px_tile_b", "date": d, "scl_purity": 1.0,
                         "item_id": "S2_54HWE_2023", **_band_row(rng)})

        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("us")))
        path = tmp_path / "pixels.parquet"
        pq.write_table(df.to_arrow(), path)

        model, band_mean, band_std = _stub_model()
        result = score_pixels_chunked(
            path, model, band_mean, band_std,
            tile_id="55HBU", device="cpu", mixed=False,
        )

        assert "px_tile_a" in result["point_id"].to_list()
        assert "px_tile_b" not in result["point_id"].to_list()

    def test_no_tile_filter_returns_all(self, tmp_path):
        """tile_id=None must return all pixels regardless of item_id."""
        rng = np.random.default_rng(4)
        rows = []
        dates = [datetime.date(2023,1,15)+datetime.timedelta(days=23*i) for i in range(N_OBS_PER_YEAR)]

        for pid, tile in [("px_a", "55HBU"), ("px_b", "54HWE")]:
            for d in dates:
                rows.append({"point_id": pid, "date": d, "scl_purity": 1.0,
                             "item_id": f"S2_{tile}_2023", **_band_row(rng)})

        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("us")))
        path = tmp_path / "pixels.parquet"
        # tile_id=None means item_id column is not read — write without it
        pq.write_table(df.drop("item_id").to_arrow(), path)

        model, band_mean, band_std = _stub_model()
        result = score_pixels_chunked(
            path, model, band_mean, band_std,
            tile_id=None, device="cpu", mixed=False,
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
        result = score_pixels_chunked(path, model, band_mean, band_std, device="cpu", mixed=False)
        assert "point_id" in result.columns
        assert "prob_tam" in result.columns

    def test_probs_in_unit_interval(self, tmp_path):
        path = _make_parquet(tmp_path, pixels=["px1", "px2", "px3"], years=[2022, 2023])
        model, band_mean, band_std = _stub_model()
        result = score_pixels_chunked(path, model, band_mean, band_std, device="cpu", mixed=False)
        assert (result["prob_tam"] >= 0.0).all()
        assert (result["prob_tam"] <= 1.0).all()

    def test_one_row_per_pixel(self, tmp_path):
        pixels = ["px1", "px2", "px3"]
        path = _make_parquet(tmp_path, pixels=pixels, years=[2021, 2022, 2023])
        model, band_mean, band_std = _stub_model()
        result = score_pixels_chunked(path, model, band_mean, band_std, device="cpu", mixed=False)
        assert len(result) == len(pixels)
        assert result["point_id"].n_unique() == len(pixels)


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
            mixed=False,
        )
        all_pixels = pixels_a + pixels_b
        assert set(result["point_id"]) == set(all_pixels)
        assert result["point_id"].n_unique() == len(all_pixels)

    def test_multi_tile_matches_merged_parquet(self, tmp_path):
        """Scores from per-tile parquets match scores from a single merged parquet."""
        pixels_a = ["px_0000_0000", "px_0001_0000"]
        pixels_b = ["px_0002_0000", "px_0003_0000"]
        years = [2022, 2023]

        path_a = _make_parquet(tmp_path, pixels=pixels_a, years=years, filename="tileA.parquet")
        path_b = _make_parquet(tmp_path, pixels=pixels_b, years=years, filename="tileB.parquet")

        # Build merged parquet from both tiles
        merged = pl.concat([pl.read_parquet(path_a), pl.read_parquet(path_b)]).sort(["point_id", "date"])
        merged_path = tmp_path / "merged.parquet"
        pq.write_table(merged.to_arrow(), merged_path, row_group_size=N_OBS_PER_YEAR)

        model, band_mean, band_std = _stub_model()

        result_tiled = score_location_years(
            year_parquets=[(y, p) for y in years for p in [path_a, path_b]],
            model=model, band_mean=band_mean, band_std=band_std,
            device="cpu", end_year=max(years), mixed=False,
        ).sort("point_id")

        result_merged = score_location_years(
            year_parquets=[(y, merged_path) for y in years],
            model=model, band_mean=band_mean, band_std=band_std,
            device="cpu", end_year=max(years), mixed=False,
        ).sort("point_id")

        assert result_tiled["point_id"].to_list() == result_merged["point_id"].to_list()
        for a, b in zip(result_tiled["prob_tam"].to_list(), result_merged["prob_tam"].to_list()):
            assert a == pytest.approx(b, abs=1e-5)


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
            mixed=False,
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
            out_dir=out_dir, device="cpu", end_year=2022, mixed=False,
        )

        import pyarrow.parquet as pq
        tbl = pq.read_table(final_paths[0])
        assert tbl.schema.field("point_id").type in (pa.string(), pa.large_string())
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
            out_dir=out_dir, device="cpu", end_year=2022, mixed=False,
        )

        by_name = {p.name: pl.read_parquet(p) for p in final_paths}
        pids_a = set(by_name["54LWH.scores.parquet"]["point_id"].to_list())
        pids_b = set(by_name["54LWJ.scores.parquet"]["point_id"].to_list())
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
            out_dir=out_dir, device="cpu", end_year=2022, mixed=False,
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
            out_dir=out_dir, device="cpu", end_year=2022, mixed=False,
        )

        paths1 = score_tiles_chunked(**kwargs)
        df1 = pl.read_parquet(paths1[0]).sort("point_id")

        # Remove final output so phase 2 re-runs, but leave no staging (already cleaned)
        paths1[0].unlink()

        paths2 = score_tiles_chunked(**kwargs)
        df2 = pl.read_parquet(paths2[0]).sort("point_id")

        assert df1["point_id"].to_list() == df2["point_id"].to_list()
        assert df1["prob_tam"].to_list() == df2["prob_tam"].to_list()


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
        dates = [datetime.date(2023,1,15)+datetime.timedelta(days=23*i) for i in range(n_total)]
        for d in dates:
            rows.append({
                "point_id": pid,
                "date": d,
                "scl_purity": 1.0,
                **_band_row(rng),
            })
    df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("us")))
    path = tmp_path / filename
    pq.write_table(df.to_arrow(), path, row_group_size=obs_per_rg)
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
            device="cpu", buffer_row_groups=1, mixed=False,
        )
        r_bulk = score_pixels_chunked(
            path, model, band_mean, band_std,
            device="cpu", buffer_row_groups=999, mixed=False,
        )

        assert set(r_split["point_id"]) == {"px1"}
        s_split = r_split.filter(pl.col("point_id") == "px1")["prob_tam"][0]
        s_bulk  = r_bulk.filter(pl.col("point_id") == "px1")["prob_tam"][0]
        assert s_split == pytest.approx(s_bulk, abs=1e-5)

    def test_split_scores_match_no_split_multiple_pixels(self, tmp_path):
        """Multiple pixels interleaved across many small row groups."""
        pixels = ["px1", "px2", "px3"]
        path = _make_split_parquet(tmp_path, pixels=pixels, obs_per_rg=4)
        model, band_mean, band_std = _stub_model()

        r_split = score_pixels_chunked(
            path, model, band_mean, band_std,
            device="cpu", buffer_row_groups=1, mixed=False,
        ).sort("point_id")

        r_bulk = score_pixels_chunked(
            path, model, band_mean, band_std,
            device="cpu", buffer_row_groups=999, mixed=False,
        ).sort("point_id")

        assert set(r_split["point_id"]) == set(pixels)
        assert r_split["point_id"].to_list() == r_bulk["point_id"].to_list()
        for a, b in zip(r_split["prob_tam"].to_list(), r_bulk["prob_tam"].to_list()):
            assert a == pytest.approx(b, abs=1e-5)

    def test_split_produces_one_row_per_pixel(self, tmp_path):
        """buffer_row_groups=1 must not create duplicate rows for the same pixel."""
        pixels = ["px1", "px2", "px3", "px4"]
        path = _make_split_parquet(tmp_path, pixels=pixels, obs_per_rg=3)
        model, band_mean, band_std = _stub_model()

        result = score_pixels_chunked(
            path, model, band_mean, band_std,
            device="cpu", buffer_row_groups=1, mixed=False,
        )

        assert result["point_id"].n_unique() == len(pixels)
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
        for d in [datetime.date(2023,1,15)+datetime.timedelta(days=23*i) for i in range(N_OBS_PER_YEAR)]:
            rows.append({"point_id": "px_last", "date": d, "scl_purity": 1.0, **_band_row(rng)})
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("us")))
        path = tmp_path / "last.parquet"
        # Single row group — pixel is always the boundary/leftover candidate.
        pq.write_table(df.to_arrow(), path, row_group_size=N_OBS_PER_YEAR)

        model, band_mean, band_std = _stub_model()
        result = score_pixels_chunked(
            path, model, band_mean, band_std,
            device="cpu", buffer_row_groups=1, mixed=False,
        )

        assert "px_last" in result["point_id"].to_list()

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
            for d in [datetime.date(2023,1,15)+datetime.timedelta(days=23*i) for i in range(N_OBS_PER_YEAR)]:
                rows.append({"point_id": pid, "date": d, "scl_purity": 1.0, **_band_row(rng)})
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("us")))
        path = tmp_path / "twopix.parquet"
        pq.write_table(df.to_arrow(), path, row_group_size=N_OBS_PER_YEAR)

        model, band_mean, band_std = _stub_model()
        result = score_pixels_chunked(
            path, model, band_mean, band_std,
            device="cpu", buffer_row_groups=1, mixed=False,
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
            dates = [datetime.date(yr,1,1)+datetime.timedelta(days=6*i) for i in range(N_OBS_PER_YEAR)]
            for d in dates:
                rows.append({
                    "point_id": pid,
                    "date": d,
                    "source": "S1",
                    "vh": float(rng.uniform(1e-4, 1e-2)),
                    "vv": float(rng.uniform(1e-4, 1e-2)),
                    "orbit": "ascending",
                })
    df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("us")))
    path = tmp_path / filename
    pq.write_table(df.to_arrow(), path, row_group_size=N_OBS_PER_YEAR)
    return path


def _stub_s1_model(n_features: int = 4) -> tuple[TAMClassifier, np.ndarray, np.ndarray]:
    """Tiny S1-only model stub."""
    cfg = TAMConfig(d_model=16, n_heads=2, n_layers=1, d_ff=32,
                    n_bands=n_features, use_s1="s1_only", n_annual_features=0)
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
        assert isinstance(result, pl.DataFrame)
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
        assert not raw["vh"].equals(smoothed["vh"])

    def test_empty_parquet_returns_empty_dataframe(self, tmp_path):
        # Parquet with no S1 rows (S2 only — no vh/vv columns at all)
        rng = np.random.default_rng(0)
        rows = [{"point_id": "px1", "date": datetime.datetime(2023, 1, 1),
                 "scl_purity": 1.0, **{b: 0.1 for b in BAND_COLS}}]
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("us")))
        path = tmp_path / "s2only.parquet"
        pq.write_table(df.to_arrow(), path)
        result = _compute_s1_despeckle_lookup(path, window=3)
        assert isinstance(result, pl.DataFrame)
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
            device="cpu", s1_only=True, mixed=False, s1_despeckle_window=3,
        )
        assert set(result["point_id"]) == {"px1", "px2"}
        assert result["prob_tam"].is_between(0, 1).all()

    def test_despeckle_changes_scores_vs_no_despeckle(self, tmp_path):
        path = _make_s1_parquet(tmp_path, pixels=["px1", "px2"], years=[2023])
        model, band_mean, band_std = _stub_s1_model()
        r_raw   = score_pixels_chunked(
            path, model, band_mean, band_std,
            device="cpu", s1_only=True, mixed=False, s1_despeckle_window=0,
        ).sort("point_id")
        r_clean = score_pixels_chunked(
            path, model, band_mean, band_std,
            device="cpu", s1_only=True, mixed=False, s1_despeckle_window=5,
        ).sort("point_id")
        # At least one pixel's score should differ after smoothing
        diffs = (r_raw["prob_tam"] - r_clean["prob_tam"]).abs()
        assert diffs.max() > 1e-4, "Expected scores to differ after despeckle"

    def test_output_schema_unchanged_with_despeckle(self, tmp_path):
        path = _make_s1_parquet(tmp_path, pixels=["px1"], years=[2023])
        model, band_mean, band_std = _stub_s1_model()
        result = score_pixels_chunked(
            path, model, band_mean, band_std,
            device="cpu", s1_only=True, mixed=False, s1_despeckle_window=3,
        )
        assert list(result.columns) == ["point_id", "prob_tam"]

    def test_despeckle_window_0_matches_no_flag(self, tmp_path):
        path = _make_s1_parquet(tmp_path, pixels=["px1", "px2"], years=[2023])
        model, band_mean, band_std = _stub_s1_model()
        r_default = score_pixels_chunked(
            path, model, band_mean, band_std,
            device="cpu", s1_only=True, mixed=False,
        ).sort("point_id")
        r_zero = score_pixels_chunked(
            path, model, band_mean, band_std,
            device="cpu", s1_only=True, mixed=False, s1_despeckle_window=0,
        ).sort("point_id")
        assert r_default["point_id"].to_list() == r_zero["point_id"].to_list()
        for a, b in zip(r_default["prob_tam"].to_list(), r_zero["prob_tam"].to_list()):
            assert a == pytest.approx(b, abs=1e-6)


# ---------------------------------------------------------------------------
# Helpers for mixed-mode tests
# ---------------------------------------------------------------------------

N_S1_OBS = max(MIN_S1_OBS_PER_YEAR, 6)  # enough S1 obs to clear the threshold


def _make_mixed_parquet(
    tmp_path: Path,
    pixels: list[str],
    years: list[int],
    include_s1: bool = True,
    n_s1_obs: int = N_S1_OBS,
    filename: str = "mixed.parquet",
    band_dtype: "pl.DataType" = pl.Float32,
) -> Path:
    """Write a parquet with interleaved S2 and S1 rows (source column).

    S2 rows carry spectral bands + scl_purity.
    S1 rows carry raw linear vh/vv; all spectral band columns are absent/null.

    band_dtype: pl.Float32 (default) writes already-reflectance-scaled band
    values via _band_row — the shape every other mixed-mode fixture here uses.
    pl.UInt16 instead writes raw chunkstore-shaped DN values via
    _band_row_uint16 (×UINT16_BAND_SCALE), matching the real cache schema that
    _extract_mixed_pa must convert back to reflectance before computing
    spectral indices — see _band_row_uint16's docstring for why this distinction
    matters.
    """
    rng = np.random.default_rng(55)
    _row_fn = _band_row_uint16 if band_dtype == pl.UInt16 else _band_row
    rows = []
    for pid in pixels:
        for yr in years:
            # S2 observations
            s2_dates = [datetime.date(yr, 1, 15) + datetime.timedelta(days=23 * i)
                        for i in range(N_OBS_PER_YEAR)]
            for d in s2_dates:
                row = {
                    "point_id": pid,
                    "date": datetime.datetime(d.year, d.month, d.day),
                    "source": "S2",
                    "scl_purity": 1.0,
                    "vh": None,
                    "vv": None,
                }
                row.update(_row_fn(rng))
                rows.append(row)

            if include_s1:
                # S1 observations (sparser cadence — ~12-day repeat)
                s1_dates = [datetime.date(yr, 1, 1) + datetime.timedelta(days=12 * i)
                            for i in range(n_s1_obs)]
                for d in s1_dates:
                    rows.append({
                        "point_id": pid,
                        "date": datetime.datetime(d.year, d.month, d.day),
                        "source": "S1",
                        "scl_purity": None,
                        "vh": float(rng.uniform(1e-4, 1e-2)),
                        "vv": float(rng.uniform(1e-4, 1e-2)),
                        **{b: None for b in BAND_COLS},
                    })

    df = pl.DataFrame(rows).with_columns(
        [pl.col("date").cast(pl.Datetime("us"))]
        + [pl.col(b).cast(band_dtype) for b in BAND_COLS]
    ).sort(["point_id", "date"])
    path = tmp_path / filename
    pq.write_table(df.to_arrow(), path, row_group_size=N_OBS_PER_YEAR + n_s1_obs)
    return path


def _stub_mixed_model(
    n_s2: int = 3,
    n_s1: int = 2,
) -> tuple[TAMClassifier, np.ndarray, np.ndarray]:
    """Tiny mixed S2+S1 model stub. n_bands = n_s2 + n_s1."""
    n_bands = n_s2 + n_s1
    cfg = TAMConfig(d_model=16, n_heads=2, n_layers=1, d_ff=32,
                    n_bands=n_bands, use_s1="mixed", n_annual_features=0)
    torch.manual_seed(13)
    model = TAMClassifier.from_config(cfg)
    model._use_s1 = "mixed"
    model._pixel_zscore = False
    model.eval()
    band_mean = np.zeros(n_bands, dtype=np.float32)
    band_std  = np.ones(n_bands,  dtype=np.float32)
    return model, band_mean, band_std


def _stub_mixed_model_with_annual_features(
    n_s2: int = 3,
    n_s1: int = 2,
) -> tuple[TAMClassifier, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mixed S2+S1 model stub with a non-trivial annual-feature head.

    n_annual_features = (n_s2 + n_s1) * 3 (p5/p95/std per S2 column then per S1
    column), matching the real "annual feature" convention — S2 summaries first,
    then S1 summaries appended after (see score.py's compute_band_summaries call
    site). annual_feat_mean/std are set to plausible reflectance-scale band-summary
    statistics (~0.01-0.5, std ~0.01-0.1).

    This is the piece _stub_mixed_model lacks (n_annual_features=0): without it,
    a model can't exhibit the actual production failure mode, where ~10,000x
    out-of-scale band summaries fed through a reflectance-calibrated
    normalisation saturate the head and collapse every prediction toward the
    same extreme value.
    """
    n_bands = n_s2 + n_s1
    n_annual = (n_s2 + n_s1) * 3
    cfg = TAMConfig(d_model=16, n_heads=2, n_layers=1, d_ff=32,
                    n_bands=n_bands, use_s1="mixed", n_annual_features=n_annual)
    torch.manual_seed(13)
    model = TAMClassifier.from_config(cfg)
    model._use_s1 = "mixed"
    model._pixel_zscore = False
    model.eval()
    band_mean = np.zeros(n_bands, dtype=np.float32)
    band_std  = np.ones(n_bands,  dtype=np.float32)

    rng = np.random.default_rng(91)
    annual_feat_mean = rng.uniform(0.01, 0.5, size=n_annual).astype(np.float32)
    annual_feat_std  = rng.uniform(0.01, 0.1, size=n_annual).astype(np.float32)
    return model, band_mean, band_std, annual_feat_mean, annual_feat_std


# ---------------------------------------------------------------------------
# TS-MX  Mixed S2+S1 mode
# ---------------------------------------------------------------------------

class TestMixedModeScoring:
    """score_pixels_chunked with mixed=True requires both S2 and S1 rows."""

    S2_COLS = ["B02", "B03", "B04"]   # minimal S2 feature set for stub model
    S1_COLS = ["s1_vh", "s1_vv"]

    def _score(self, tmp_path, parquet_path, **kwargs):
        model, band_mean, band_std = _stub_mixed_model(n_s2=len(self.S2_COLS),
                                                        n_s1=len(self.S1_COLS))
        return score_pixels_chunked(
            parquet_path, model, band_mean, band_std,
            device="cpu", mixed=True,
            s2_feature_cols=self.S2_COLS,
            s1_feature_cols=self.S1_COLS,
            **kwargs,
        )

    def test_scores_produced_when_both_sources_present(self, tmp_path):
        """Pixels with sufficient S2 and S1 obs must appear in output."""
        path = _make_mixed_parquet(tmp_path, pixels=["px1", "px2"], years=[2023])
        result = self._score(tmp_path, path)
        assert set(result["point_id"]) == {"px1", "px2"}
        assert result["prob_tam"].is_between(0, 1).all()

    def test_pixels_without_s1_rows_are_excluded(self, tmp_path):
        """A parquet with no S1 rows must produce zero scored pixels in mixed mode."""
        path = _make_mixed_parquet(tmp_path, pixels=["px1", "px2"], years=[2023],
                                   include_s1=False)
        result = self._score(tmp_path, path)
        assert len(result) == 0, (
            "mixed mode must drop pixels with no S1 observations, got: "
            f"{result['point_id'].to_list()}"
        )

    def test_pixels_with_too_few_s1_obs_are_excluded(self, tmp_path):
        """Pixels with fewer than MIN_S1_OBS_PER_YEAR S1 obs must be dropped."""
        path = _make_mixed_parquet(tmp_path, pixels=["px_sparse_s1"], years=[2023],
                                   n_s1_obs=MIN_S1_OBS_PER_YEAR - 1)
        result = self._score(tmp_path, path)
        assert len(result) == 0, (
            f"Expected no scores with only {MIN_S1_OBS_PER_YEAR - 1} S1 obs "
            f"(threshold={MIN_S1_OBS_PER_YEAR}), got: {result['point_id'].to_list()}"
        )

    def test_minimum_s1_obs_threshold_exactly_met(self, tmp_path):
        """Exactly MIN_S1_OBS_PER_YEAR S1 obs must be sufficient to score."""
        path = _make_mixed_parquet(tmp_path, pixels=["px_exact"], years=[2023],
                                   n_s1_obs=MIN_S1_OBS_PER_YEAR)
        result = self._score(tmp_path, path)
        assert "px_exact" in result["point_id"].to_list(), (
            f"Pixel with exactly {MIN_S1_OBS_PER_YEAR} S1 obs should be scored"
        )

    def test_s2_scl_filter_does_not_drop_s1_rows(self, tmp_path):
        """SCL purity filtering must not eliminate S1 rows (they have null scl_purity)."""
        path = _make_mixed_parquet(tmp_path, pixels=["px1"], years=[2023])
        # High scl_purity_min would drop S1 rows if filter incorrectly applied to them
        result = self._score(tmp_path, path, scl_purity_min=0.99)
        assert "px1" in result["point_id"].to_list(), (
            "S1 rows must survive SCL purity filter applied to S2 rows"
        )

    def test_mixed_scores_differ_from_s2_only(self, tmp_path):
        """A mixed model fed mixed data should score differently than if S1 rows are absent.

        This guards against S1 obs being silently dropped — if they were, the two
        runs would produce the same scores (same S2 data, same model).
        """
        path_mixed  = _make_mixed_parquet(tmp_path, pixels=["px1"], years=[2023],
                                          include_s1=True,  filename="mixed.parquet")
        path_s2only = _make_mixed_parquet(tmp_path, pixels=["px1"], years=[2023],
                                          include_s1=False, filename="s2only.parquet")

        model, band_mean, band_std = _stub_mixed_model(n_s2=len(self.S2_COLS),
                                                        n_s1=len(self.S1_COLS))

        def _run(path):
            return score_pixels_chunked(
                path, model, band_mean, band_std,
                device="cpu", mixed=True,
                s2_feature_cols=self.S2_COLS,
                s1_feature_cols=self.S1_COLS,
            )

        r_mixed  = _run(path_mixed)
        r_s2only = _run(path_s2only)

        assert len(r_mixed) == 1, "mixed path must score px1"
        assert len(r_s2only) == 0, "s2-only path must score nothing (no S1 obs)"


# ---------------------------------------------------------------------------
# TS-RCV  Raw-chunk → model "pixel view" contract
#
# _extract_mixed_pa is the GIL-free PyArrow path that turns raw chunkstore rows
# into the `feat` matrix fed to the model in mixed (use_s1=True) mode — the path
# every production scoring run with an S1-aware checkpoint (e.g. tam-v10) takes.
#
# The real chunkstore stores S2 bands as uint16 DN values (×UINT16_BAND_SCALE,
# see analysis.constants.ensure_float32_bands), and never stores spectral index
# columns (NDVI, NDWI, ...) — those are derived. Training computes its
# "annual feature" statistics by running ensure_float32_bands + the
# `extract_features` numba kernel over that same raw cache shape
# (tam/_prep_worker.py calls ensure_float32_bands before _compute_band_summaries,
# which itself now delegates to extract_features — see _compute_band_summaries'
# docstring and [[project_annual_feature_parity_bugs]]).
#
# _extract_mixed_pa must therefore present the model the *same reflectance-scale,
# index-augmented view* of each pixel, or the model's annual-feature normalisation
# (calibrated against reflectance-scale stats) silently saturates. That's exactly
# what happened in production: raw DN values ~10,000x the expected scale were fed
# straight through, collapsing every prediction to prob_tam_raw == 0.0 — see the
# Mitchell tile 54LWH/2025 all-zero scoring investigation that prompted this test.
#
# These tests build a chunkstore-shaped (uint16-band) parquet and assert the S2
# columns of `feat` match the oracle: ensure_float32_bands-equivalent scaling
# (raw / UINT16_BAND_SCALE) run through the same `extract_features` kernel,
# selected by ALL_FEATURE_COLS index — never a hand-rolled formula, to avoid
# reintroducing the "two independent implementations silently drift" risk this
# whole bug class stems from.
# ---------------------------------------------------------------------------

_RCV_S2_COLS = ["B02", "B03", "B04", "NDVI", "NDWI", "MAVI", "NDRE", "CI_RE"]
_RCV_S1_COLS = ["s1_vh", "s1_vv"]


def _pa_slices_from_parquet(path: Path, s2_cols: list[str], s1_cols: list[str],
                            scl_purity_min: float = 0.5) -> list["_PASlice"]:
    """Read a parquet into _PASlice objects exactly as the mixed-mode reader does.

    Mirrors the year/doy derivation in score.py's _reader (pyarrow.compute.year /
    day_of_year, both cast to int32, appended as columns) so the resulting table
    has the shape _extract_mixed_pa expects — one slice per row group, which is
    enough for these single-pixel, single-row-group fixtures.
    """
    import pyarrow.compute as pac
    tbl = pq.read_table(path)
    date_col = tbl.column("date")
    year_col = pac.year(date_col).cast("int32")
    doy_col  = pac.day_of_year(date_col).cast("int32")
    tbl = tbl.append_column(pa.field("year", pa.int32()), year_col)
    tbl = tbl.append_column(pa.field("doy",  pa.int32()), doy_col)
    return [_PASlice(tbl, s2_cols, s1_cols, scl_purity_min)]


def _expected_s2_features(raw_uint16: dict[str, np.ndarray], s2_cols: list[str]) -> np.ndarray:
    """Oracle: scale raw uint16 DN bands to reflectance, run extract_features,
    select the requested columns by ALL_FEATURE_COLS index — the same sequence
    ensure_float32_bands + extract_features performs on the train-prep side."""
    band_arrs = [raw_uint16[b].astype(np.float32) / UINT16_BAND_SCALE for b in BAND_COLS]
    n = len(band_arrs[0])
    full = np.empty((n, len(ALL_FEATURE_COLS)), dtype=np.float32)
    extract_features(*band_arrs, full)
    col_idx = [ALL_FEATURE_COLS.index(c) for c in s2_cols]
    return full[:, col_idx]


class TestRawChunkPixelView:
    """_extract_mixed_pa must convert raw chunkstore rows to the model's expected view."""

    def test_uint16_bands_scaled_and_indices_computed(self, tmp_path):
        """S2 feature columns extracted from a uint16-band chunk must equal the
        ensure_float32_bands + extract_features oracle — both the raw-band scale
        (reflectance, not DN) and the derived spectral indices."""
        path = _make_mixed_parquet(tmp_path, pixels=["px1"], years=[2023],
                                   band_dtype=pl.UInt16, filename="rcv_uint16.parquet")
        raw_df = pl.read_parquet(path).filter(pl.col("source") == "S2").sort("date")
        raw_uint16 = {b: raw_df[b].to_numpy() for b in BAND_COLS}

        slices = _pa_slices_from_parquet(path, _RCV_S2_COLS, _RCV_S1_COLS)
        chunk = _extract_mixed_pa(slices[0])
        assert chunk is not None
        s2_feat = chunk.feat[~chunk.is_s1][:, :chunk.n_s2]

        expected = _expected_s2_features(raw_uint16, _RCV_S2_COLS)
        np.testing.assert_allclose(s2_feat, expected, rtol=1e-5, atol=1e-6)

    def test_raw_dn_values_are_rescaled_not_passed_through(self, tmp_path):
        """Regression guard for the actual failure mode: raw DN values (tens to
        thousands) must not appear verbatim in `feat` — they must be divided down
        to reflectance (~0-1). A test that only checks index *formulas* could
        still pass with the scaling bug present, since ratio-based indices
        (NDVI etc.) are scale-invariant — which is exactly why this bug was
        subtle and the model's sequence features looked superficially fine while
        its annual features (which depend on absolute band scale) saturated."""
        path = _make_mixed_parquet(tmp_path, pixels=["px1"], years=[2023],
                                   band_dtype=pl.UInt16, filename="rcv_scale.parquet")
        raw_df = pl.read_parquet(path).filter(pl.col("source") == "S2").sort("date")
        raw_b02 = raw_df["B02"].to_numpy().astype(np.float32)
        assert raw_b02.max() > 10, "fixture sanity check: DN values should be in the hundreds-thousands"

        slices = _pa_slices_from_parquet(path, _RCV_S2_COLS, _RCV_S1_COLS)
        chunk = _extract_mixed_pa(slices[0])
        assert chunk is not None
        b02_idx = _RCV_S2_COLS.index("B02")
        b02_feat = chunk.feat[~chunk.is_s1][:, b02_idx]

        np.testing.assert_allclose(b02_feat, raw_b02 / UINT16_BAND_SCALE, rtol=1e-6, atol=1e-7)
        assert b02_feat.max() < 1.0, (
            f"B02 feature values must be reflectance-scale (<1.0), got max={b02_feat.max()} "
            "— raw DN values are being passed through without /UINT16_BAND_SCALE conversion"
        )

    def test_float_band_fixture_unaffected(self, tmp_path):
        """Sanity check: the default (Float32) fixture shape — used by every other
        mixed-mode test — must pass through _extract_mixed_pa unscaled, confirming
        the uint16 path is additive and doesn't regress the already-reflectance
        case (e.g. older cache formats / pre-scaled test data)."""
        path = _make_mixed_parquet(tmp_path, pixels=["px1"], years=[2023],
                                   filename="rcv_float.parquet")  # band_dtype=Float32 default
        raw_df = pl.read_parquet(path).filter(pl.col("source") == "S2").sort("date")
        raw_b02 = raw_df["B02"].to_numpy().astype(np.float32)

        slices = _pa_slices_from_parquet(path, _RCV_S2_COLS, _RCV_S1_COLS)
        chunk = _extract_mixed_pa(slices[0])
        assert chunk is not None
        b02_idx = _RCV_S2_COLS.index("B02")
        b02_feat = chunk.feat[~chunk.is_s1][:, b02_idx]

        np.testing.assert_allclose(b02_feat, raw_b02, rtol=1e-6, atol=1e-7)


class TestMixedModeSaturationGuard:
    """End-to-end guard: a wrong-scale `feat` must not silently saturate the model."""

    S2_COLS = _RCV_S2_COLS
    S1_COLS = _RCV_S1_COLS

    def test_uint16_chunkstore_input_produces_varied_scores(self, tmp_path):
        """score_pixels_chunked on chunkstore-shaped (uint16-band) input must
        produce prob_tam values with real spread across pixels, not the
        all-identical/all-saturated output a ~10,000x band-scale error produces
        (every pixel's annual features pinned to the same extreme normalised
        value regardless of its actual spectral signature).

        Uses _stub_mixed_model_with_annual_features — a plain n_annual_features=0
        stub can't exhibit this failure mode, since the per-window band z-scoring
        on the sequence path is scale-invariant; only the annual-feature head
        (normalised against fixed, reflectance-scale annual_feat_mean/std) is
        sensitive to absolute band scale, and that's where production saturated."""
        path = _make_mixed_parquet(tmp_path, pixels=["px1", "px2", "px3", "px4"], years=[2023],
                                   band_dtype=pl.UInt16, filename="rcv_e2e.parquet")
        model, band_mean, band_std, afm, afs = _stub_mixed_model_with_annual_features(
            n_s2=len(self.S2_COLS), n_s1=len(self.S1_COLS))
        result = score_pixels_chunked(
            path, model, band_mean, band_std,
            device="cpu", mixed=True,
            s2_feature_cols=self.S2_COLS,
            s1_feature_cols=self.S1_COLS,
            annual_feat_mean=afm,
            annual_feat_std=afs,
        )
        assert len(result) == 4
        assert result["prob_tam"].is_between(0, 1).all()
        vals = result["prob_tam"].to_list()
        assert result["prob_tam"].n_unique() > 1, (
            "expected varied scores across pixels with distinct random band values; "
            f"got: {vals} — looks like saturated/collapsed output"
        )
        # The untrained stub's absolute output magnitude is small and arbitrary
        # (depends on random init), so check *relative* spread rather than an
        # absolute threshold — the failure mode this guards against is every
        # pixel being pinned to the *same* extreme value (e.g. all 1.0, as the
        # ~10,000x-out-of-scale annual features saturate the head identically
        # regardless of each pixel's actual spectral signature), not merely
        # "small" outputs.
        lo, hi = min(vals), max(vals)
        rel_spread = (hi - lo) / hi if hi > 0 else 0.0
        assert rel_spread > 0.01, (
            f"expected meaningfully varied scores (relative spread={rel_spread:.4g}); "
            f"got: {vals} — annual-feature head looks saturated"
        )


# ---------------------------------------------------------------------------
# TS-CPS  ChunkPixelSource cross-file boundary correctness
#
# The parser thread peels off the trailing pixel of each row group as a
# "leftover" to handle pixels that straddle row-group boundaries.  It finds
# the split point by scanning backward from the end of the combined
# (leftover + current rg) array.
#
# Bug repro: the original code used np.searchsorted, which requires a sorted
# array.  ChunkPixelSource presents files in spatial (row-major) order, so
# point_id strings are NOT globally lexicographically sorted across files —
# the last pixel of file N can have a higher string value than the first pixel
# of file N+1.  searchsorted then returns a wrong split point, causing the
# accumulated leftover to grow to millions of rows and swallow entire chunks
# without scoring them.
#
# These tests construct a two-file ChunkPixelSource where the boundary pixel
# of file 0 sorts AFTER the first pixel of file 1 (i.e. the real-world
# failure condition), then assert that every pixel is scored.
# ---------------------------------------------------------------------------

_CPS_S2_COLS = ["B02", "B03", "B04"]
_CPS_S1_COLS = ["s1_vh", "s1_vv"]


def _make_chunk_parquet(
    tmp_path: Path,
    pixels: list[str],
    year: int = 2023,
    filename: str = "chunk.parquet",
    obs_per_rg: int | None = None,
) -> Path:
    """Write a mixed S2+S1 chunk parquet in the supplied pixel order (not sorted).

    Rows are written in the order given by `pixels` to reproduce spatial
    (non-lex) ordering, matching real chunk parquets that are sorted by
    pixel position rather than point_id string.  The mixed format is required
    because the searchsorted bug lives in the mixed-mode parser path.
    """
    from tam.core.dataset import BAND_COLS as _BC
    rng = np.random.default_rng(77)
    rows = []
    start_s2 = datetime.date(year, 1, 15)
    start_s1 = datetime.date(year, 1, 1)
    for pid in pixels:
        for i in range(N_OBS_PER_YEAR):
            d = start_s2 + datetime.timedelta(days=23 * i)
            row = {
                "point_id": pid,
                "date": datetime.datetime(d.year, d.month, d.day),
                "source": "S2",
                "scl_purity": 1.0,
                "vh": None,
                "vv": None,
            }
            row.update({b: float(rng.uniform(0.01, 0.5)) for b in _CPS_S2_COLS})
            rows.append(row)
        for i in range(N_S1_OBS):
            d = start_s1 + datetime.timedelta(days=12 * i)
            rows.append({
                "point_id": pid,
                "date": datetime.datetime(d.year, d.month, d.day),
                "source": "S1",
                "scl_purity": None,
                "vh": float(rng.uniform(1e-4, 1e-2)),
                "vv": float(rng.uniform(1e-4, 1e-2)),
                **{b: None for b in _CPS_S2_COLS},
            })
    df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("us")))
    path = tmp_path / filename
    rg_size = obs_per_rg if obs_per_rg is not None else (N_OBS_PER_YEAR + N_S1_OBS)
    pq.write_table(df.to_arrow(), path, row_group_size=rg_size)
    return path


class TestChunkPixelSourceBoundary:
    """Parser thread leftover logic must not lose pixels at chunk file boundaries
    when point_id strings are not lexicographically sorted across files.

    The bug lives in the mixed-mode parser path (``mixed=True``) which uses
    PyArrow tables and the searchsorted-based tail-finding logic.  All tests
    here use mixed=True to exercise that path.
    """

    def _score_chunk_source(self, tmp_path, paths, **kwargs):
        from tam.core.score import score_pixels_chunked
        from tam.core.pixel_source import ChunkPixelSource
        from tam.core.config import TAMConfig

        n_bands = len(_CPS_S2_COLS) + len(_CPS_S1_COLS)
        cfg = TAMConfig(d_model=16, n_heads=2, n_layers=1, d_ff=32,
                        n_bands=n_bands, use_s1="mixed", n_annual_features=0)
        torch.manual_seed(0)
        model = TAMClassifier.from_config(cfg)
        model._use_s1 = "mixed"
        model._pixel_zscore = False
        model.eval()
        band_mean = np.zeros(n_bands, dtype=np.float32)
        band_std  = np.ones(n_bands,  dtype=np.float32)

        src = ChunkPixelSource(paths)
        return score_pixels_chunked(
            source=src, model=model, band_mean=band_mean, band_std=band_std,
            device="cpu", mixed=True,
            s2_feature_cols=_CPS_S2_COLS,
            s1_feature_cols=_CPS_S1_COLS,
            decay=0.0, **kwargs,
        )

    def test_all_pixels_scored_when_boundary_is_lex_descending(self, tmp_path):
        """File 0 ends with a lex-high point_id; file 1 ends with a lex-low one.

        This is the exact real-world topology that caused the searchsorted bug:
        spatial row-major order means the last pixel of file N has a higher
        lex value than the last pixel of file N+1 (column index resets between
        rows).  The leftover from file N is prepended to file N+1's first row
        group, making the combined array unsorted.  searchsorted then returns
        the wrong split point (often 0), swallowing entire chunks as leftover.
        """
        # File 0: pixels with high string values (simulate end of spatial row N)
        # File 1: pixels with low string values (simulate spatial row N+1, col 0)
        # Crucially: last pid of file1 ALSO < leftover pid from file0, so
        # searchsorted(combined, pid_last='px_0003_0001') returns 0 on the
        # unsorted [px_9003_0000 * N, px_0000_0001 * N, ..., px_0003_0001 * N] array.
        pids_file0 = [f"px_9{i:03d}_0000" for i in range(4)]
        pids_file1 = [f"px_0{i:03d}_0001" for i in range(4)]

        assert pids_file0[-1] > pids_file1[-1], (
            "Test setup: last pid of file0 must lex-sort AFTER last pid of file1"
        )

        p0 = _make_chunk_parquet(tmp_path, pids_file0, filename="chunk_00.parquet")
        p1 = _make_chunk_parquet(tmp_path, pids_file1, filename="chunk_01.parquet")

        result = self._score_chunk_source(tmp_path, [p0, p1])
        scored = set(result["point_id"].to_list())
        missing = (set(pids_file0) | set(pids_file1)) - scored
        assert not missing, f"Pixels not scored across lex-descending boundary: {missing}"

    def test_pixel_count_matches_across_lex_descending_boundary(self, tmp_path):
        """Score count must equal total pixel count regardless of lex order."""
        pids_file0 = [f"px_8{i:03d}_0010" for i in range(6)]
        pids_file1 = [f"px_1{i:03d}_0020" for i in range(6)]
        assert pids_file0[-1] > pids_file1[-1]

        p0 = _make_chunk_parquet(tmp_path, pids_file0, filename="c0.parquet")
        p1 = _make_chunk_parquet(tmp_path, pids_file1, filename="c1.parquet")

        result = self._score_chunk_source(tmp_path, [p0, p1])
        assert len(result) == len(pids_file0) + len(pids_file1)
        assert result["point_id"].n_unique() == len(pids_file0) + len(pids_file1)

    def test_scores_match_single_merged_parquet(self, tmp_path):
        """Scores via ChunkPixelSource must match a single merged parquet.

        This is the definitive regression test: if the leftover logic is
        correct, chunked and merged scoring produce identical results.
        """
        # These pids ensure: last pid of file1 ('px_0102_0001') < last pid of
        # file0 ('px_7002_0000'), so searchsorted on the combined unsorted array
        # returns 0 instead of the correct tail position.
        pids_file0 = ["px_7000_0000", "px_7001_0000", "px_7002_0000"]
        pids_file1 = ["px_0100_0001", "px_0101_0001", "px_0102_0001"]
        assert pids_file0[-1] > pids_file1[-1]

        p0 = _make_chunk_parquet(tmp_path, pids_file0, filename="c0.parquet")
        p1 = _make_chunk_parquet(tmp_path, pids_file1, filename="c1.parquet")

        # Build a merged single parquet (all pixels, sorted by point_id then date)
        merged = pl.concat([pl.read_parquet(p0), pl.read_parquet(p1)]).sort(["point_id", "date"])
        merged_path = tmp_path / "merged.parquet"
        pq.write_table(merged.to_arrow(), merged_path, row_group_size=N_OBS_PER_YEAR)

        result_chunked = self._score_chunk_source(tmp_path, [p0, p1]).sort("point_id")
        result_merged  = self._score_chunk_source(tmp_path, [merged_path]).sort("point_id")

        assert result_chunked["point_id"].to_list() == result_merged["point_id"].to_list()
        for a, b in zip(result_chunked["prob_tam"].to_list(), result_merged["prob_tam"].to_list()):
            assert a == pytest.approx(b, abs=1e-5), (
                f"Score mismatch for pixel — chunked={a:.6f} merged={b:.6f}"
            )

    def test_three_file_chain_with_multiple_lex_drops(self, tmp_path):
        """Multiple lex-descending boundaries across three files must all be handled."""
        # Simulate a 3×3 spatial grid: each row resets the column index.
        # row0→row1 and row1→row2 are both lex-descending drops.
        pids_row0 = ["px_9000_0000", "px_9001_0000"]
        pids_row1 = ["px_0500_0001", "px_0501_0001"]
        pids_row2 = ["px_0100_0002", "px_0101_0002"]
        assert pids_row0[-1] > pids_row1[-1], "row0→row1 boundary must be lex-descending"
        assert pids_row1[-1] > pids_row2[-1], "row1→row2 boundary must be lex-descending"

        p0 = _make_chunk_parquet(tmp_path, pids_row0, filename="r0.parquet")
        p1 = _make_chunk_parquet(tmp_path, pids_row1, filename="r1.parquet")
        p2 = _make_chunk_parquet(tmp_path, pids_row2, filename="r2.parquet")

        result = self._score_chunk_source(tmp_path, [p0, p1, p2])
        all_pids = set(pids_row0) | set(pids_row1) | set(pids_row2)
        assert set(result["point_id"].to_list()) == all_pids

    def test_lex_ascending_boundary_still_correct(self, tmp_path):
        """When file boundaries happen to be lex-ascending, scoring must still be correct."""
        pids_file0 = ["px_0100_0000", "px_0101_0000"]
        pids_file1 = ["px_0200_0001", "px_0201_0001"]
        assert pids_file0[-1] < pids_file1[-1], "must be lex-ascending for this test"

        p0 = _make_chunk_parquet(tmp_path, pids_file0, filename="asc0.parquet")
        p1 = _make_chunk_parquet(tmp_path, pids_file1, filename="asc1.parquet")

        result = self._score_chunk_source(tmp_path, [p0, p1])
        assert set(result["point_id"].to_list()) == set(pids_file0) | set(pids_file1)

    def test_spatially_ordered_within_file_boundary(self, tmp_path):
        """Reproduces the exact failure mode: pids are spatially ordered (lex-
        descending) within each file, mirroring real chunk parquets where
        pixels are processed west→east causing XXXX to decrease within a row.

        File 0 ends with a high-lex pid (px_7290_*).  File 1 contains pixels
        with decreasing pids so pid_last of file 1's rg is lex-lower than the
        leftover from file 0.  With the old np.searchsorted code this returns
        tail_start=0, swallowing the entire combined table into leftover and
        splitting each pixel's observations across emitted chunks.  Fragmented
        obs fall below min_obs_per_year, so affected pixels are silently dropped
        from the output.
        """
        # File 0: px_7289_0000 → px_7270_0000, decreasing (spatial west-to-east)
        pids_file0 = [f"px_{7289 - i:04d}_0000" for i in range(20)]
        # File 1: px_7199_0001 → px_7180_0001, also decreasing
        # pid_last of file1 rg = px_7180_0001 < leftover px_7270_0000
        pids_file1 = [f"px_{7199 - i:04d}_0001" for i in range(20)]

        assert pids_file0[-1] > pids_file1[-1], (
            "last pid of file0 must lex-sort after last pid of file1"
        )

        p0 = _make_chunk_parquet(tmp_path, pids_file0, filename="sp0.parquet")
        p1 = _make_chunk_parquet(tmp_path, pids_file1, filename="sp1.parquet")

        result = self._score_chunk_source(tmp_path, [p0, p1])
        all_pids = set(pids_file0) | set(pids_file1)
        scored   = set(result["point_id"].to_list())
        missing  = all_pids - scored
        assert not missing, (
            f"{len(missing)} pixels not scored (searchsorted boundary bug): "
            f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
        )


# ---------------------------------------------------------------------------
# TS-CPS-UNIT  Direct unit test of the tail-finding logic
#
# The end-to-end tests above verify correct scoring results but cannot
# reliably trigger the searchsorted misfire at small array sizes (the binary
# search happens to work when the combined array is short).  This unit test
# directly verifies that the backward scan correctly locates the tail start
# on an unsorted combined array, exercising the exact array topology that
# caused the production bug.
# ---------------------------------------------------------------------------

class TestTailFindingLogic:
    """The tail-finding backward scan must return the correct split point on
    unsorted point_id arrays produced by spatial (non-lex) ordering."""

    @staticmethod
    def _tail_start_new(pid_arr: np.ndarray) -> int:
        """Replicate the fixed backward-scan logic from score.py."""
        pid_last = pid_arr[-1]
        tail_start = len(pid_arr) - 1
        while tail_start > 0 and pid_arr[tail_start - 1] == pid_last:
            tail_start -= 1
        return tail_start

    @staticmethod
    def _tail_start_old(pid_arr: np.ndarray) -> int:
        """Replicate the buggy searchsorted logic (for comparison)."""
        return int(np.searchsorted(pid_arr, pid_arr[-1], side="left"))

    def _make_combined(self, leftover_pid: str, leftover_n: int,
                       body_pids: list[str], obs_per_pid: int) -> np.ndarray:
        """Build a combined [leftover + body] array as the parser sees it."""
        arr = [leftover_pid] * leftover_n
        for pid in body_pids:
            arr.extend([pid] * obs_per_pid)
        return np.array(arr)

    def test_lex_descending_boundary_correct_tail(self):
        """Leftover pid > pid_last: backward scan returns correct tail, searchsorted fails."""
        # leftover from file0: high-lex px_7290_6124
        # file1 body: descending px_7289, px_7288, ..., ending at px_7200
        body_pids = [f"px_{7289 - i:04d}_6124" for i in range(90)]
        combined = self._make_combined("px_7290_6124", 39, body_pids, obs_per_pid=15)

        pid_last = combined[-1]  # px_7200_6124
        correct_tail = len(combined) - 15  # last 15 rows belong to pid_last

        new_tail = self._tail_start_new(combined)
        old_tail = self._tail_start_old(combined)

        assert new_tail == correct_tail, (
            f"backward scan: expected {correct_tail}, got {new_tail}"
        )
        # The old searchsorted code must give a wrong answer on this input to
        # justify the fix.  If this assertion fails the test topology is wrong.
        assert old_tail != correct_tail, (
            "searchsorted gave the right answer — test topology does not "
            "reproduce the failure condition; adjust leftover/body_pids"
        )

    def test_lex_ascending_boundary_both_correct(self):
        """When combined is lex-sorted, both methods agree."""
        body_pids = [f"px_0{i:03d}_0001" for i in range(10)]
        combined = self._make_combined("px_0000_0000", 5, body_pids, obs_per_pid=10)

        new_tail = self._tail_start_new(combined)
        old_tail = self._tail_start_old(combined)
        correct  = len(combined) - 10

        assert new_tail == correct
        assert old_tail == correct  # both should agree on sorted data

    def test_single_pixel_entire_rg(self):
        """All rows belong to one pixel → tail_start=0 (carry forward)."""
        combined = np.array(["px_1234_5678"] * 100)
        assert self._tail_start_new(combined) == 0

    def test_two_pixels_equal_split(self):
        """Exactly half leftover, half body — tail starts at midpoint."""
        combined = np.array(["px_A"] * 50 + ["px_B"] * 50)
        assert self._tail_start_new(combined) == 50

    def test_large_unsorted_array_correct_tail(self):
        """At production-scale array sizes the backward scan must still be correct.

        The real failure topology: the body of the combined array is also lex-
        descending (spatial ordering within a file decreases the XXXX index as
        pixels are processed east-to-west), so the full combined array is
        completely unsorted.  searchsorted returns 0 instead of the correct tail.
        """
        # Simulate the real within-file sort: rg0 of r03_c10 has pids like
        # px_7239_6124, px_7238_6124, ..., px_6924_6160 (XXXX decreasing).
        # The leftover (px_7239_6124, 16 rows) is prepended to rg1 which
        # continues the descent.  pid_last ends at px_6924_6160.
        obs_per_pid = 15
        leftover_pid = "px_7239_6124"
        # Build a lex-descending body: 7238 down to 6925 (313 pids × 15 obs = 4695 rows)
        body_pids = [f"px_{7238 - i:04d}_6124" for i in range(313)]
        body_last_pid = f"px_{7238 - 312:04d}_6124"  # px_6926_6124
        combined = self._make_combined(leftover_pid, 16, body_pids, obs_per_pid)

        correct_tail = len(combined) - obs_per_pid
        new_tail = self._tail_start_new(combined)
        old_tail = self._tail_start_old(combined)

        assert new_tail == correct_tail, (
            f"backward scan wrong at production scale: {new_tail} != {correct_tail}"
        )
        assert old_tail != correct_tail, (
            "searchsorted correct on descending body — test topology is wrong"
        )
