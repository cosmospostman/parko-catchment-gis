"""TT-* tests for spatial_split and train_tam."""

from __future__ import annotations

import json
from pathlib import Path

import datetime

import numpy as np
import polars as pl
import pytest
import torch

from tam.core.config import TAMConfig
from tam.core.dataset import ALL_FEATURE_COLS, BAND_COLS, TAMDataset, collate_fn
from tam.core.model import TAMClassifier
from tam.core.train import load_tam, spatial_split, train_tam


# ---------------------------------------------------------------------------
# Shared fixtures (supplement conftest)
# ---------------------------------------------------------------------------

@pytest.fixture
def many_labels() -> dict[str, float]:
    """8 presence + 8 absence pixels for spatial split tests."""
    return {f"p{i}": 1.0 for i in range(8)} | {f"a{i}": 0.0 for i in range(8)}


@pytest.fixture
def many_coords(many_labels) -> pl.DataFrame:
    """Unique latitudes so sort order is deterministic."""
    rows = []
    for i, pid in enumerate(many_labels.keys()):
        rows.append({"point_id": pid, "lon": 144.0, "lat": -20.0 - i * 0.5})
    return pl.DataFrame(rows)


@pytest.fixture
def smoke_cfg() -> TAMConfig:
    return TAMConfig(
        d_model=16, n_heads=2, n_layers=1, d_ff=32,
        n_bands=len(ALL_FEATURE_COLS),
        use_s1=False,
        n_global_features=0,
        n_epochs=5, patience=5, batch_size=4,
        doy_jitter=3, val_frac=0.3,
    )


@pytest.fixture
def smoke_pixel_df(band_cols) -> pl.DataFrame:
    """10 presence + 10 absence pixels; 3 of each land in val so AUC is always computable."""
    rng = np.random.default_rng(42)
    rows = []
    for pid in [f"p{i}" for i in range(10)] + [f"a{i}" for i in range(10)]:
        start = datetime.date(2023, 1, 15)
        dates = [start + datetime.timedelta(days=15 * i) for i in range(20)]
        for d in dates:
            rows.append({
                "point_id": pid, "date": str(d),
                "scl_purity": 1.0, "year": 2023,
                **{b: float(rng.uniform(0.01, 0.5)) for b in band_cols},
            })
    return pl.DataFrame(rows)


@pytest.fixture
def smoke_labels() -> dict[str, float]:
    return {f"p{i}": 1.0 for i in range(10)} | {f"a{i}": 0.0 for i in range(10)}


@pytest.fixture
def smoke_coords(smoke_labels) -> pl.DataFrame:
    rows = []
    for i, pid in enumerate(smoke_labels.keys()):
        rows.append({"point_id": pid, "lon": 144.0, "lat": -20.0 - i * 0.5})
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# TT-1
# ---------------------------------------------------------------------------

class TestTT1BothClassesInTrainAndVal:
    def test_both_classes_in_both_splits(self, many_labels, many_coords):
        train, val = spatial_split(many_labels, many_coords, val_frac=0.2)
        assert set(train.values()) == {0.0, 1.0}
        assert set(val.values()) == {0.0, 1.0}


# ---------------------------------------------------------------------------
# TT-2
# ---------------------------------------------------------------------------

class TestTT2ValFractionApproximatelyCorrect:
    def test_val_fraction(self, many_labels, many_coords):
        val_frac = 0.2
        train, val = spatial_split(many_labels, many_coords, val_frac=val_frac)
        total = len(train) + len(val)
        actual_frac = len(val) / total
        assert abs(actual_frac - val_frac) <= 2 / total + 1e-9


# ---------------------------------------------------------------------------
# TT-3
# ---------------------------------------------------------------------------

class TestTT3NoOverlapBetweenSplits:
    def test_disjoint_indices(self, many_labels, many_coords):
        train, val = spatial_split(many_labels, many_coords, val_frac=0.2)
        assert set(train.keys()) & set(val.keys()) == set()


# ---------------------------------------------------------------------------
# TT-4
# ---------------------------------------------------------------------------

class TestTT4ValPixelsSouthernmost:
    def test_val_pixels_have_lower_lat(self, many_labels, many_coords):
        train, val = spatial_split(many_labels, many_coords, val_frac=0.25)
        pid_to_lat = dict(zip(many_coords["point_id"].to_list(), many_coords["lat"].to_list()))
        for cls_val in [0.0, 1.0]:
            train_cls = {k for k, v in train.items() if v == cls_val}
            val_cls   = {k for k, v in val.items()   if v == cls_val}
            if not train_cls or not val_cls:
                continue
            max_val_lat   = max(pid_to_lat[pid] for pid in val_cls)
            min_train_lat = min(pid_to_lat[pid] for pid in train_cls)
            assert max_val_lat <= min_train_lat, (
                f"Class {cls_val}: val pixel lat {max_val_lat} > train min lat {min_train_lat}"
            )


# ---------------------------------------------------------------------------
# TT-5
# ---------------------------------------------------------------------------

class TestTT5SinglePixelPerClassDoesNotCrash:
    def test_single_pixel_per_class(self):
        labels = {"p0": 1.0, "a0": 0.0}
        coords = pl.DataFrame({"point_id": ["p0", "a0"], "lon": [144.0, 144.0], "lat": [-20.0, -21.0]})
        train, val = spatial_split(labels, coords, val_frac=0.5)
        assert len(train) + len(val) == 2


# ---------------------------------------------------------------------------
# TT-6
# ---------------------------------------------------------------------------

class TestTT6TrainTamSmokeCheckpoints:
    def test_checkpoints_written_and_model_finite(self, tmp_path, smoke_pixel_df, smoke_labels, smoke_coords, smoke_cfg):
        model, best_val_auc = train_tam(
            smoke_pixel_df, smoke_labels, smoke_coords,
            out_dir=tmp_path, cfg=smoke_cfg, device="cpu",
        )
        assert (tmp_path / "tam_model.pt").exists()
        assert (tmp_path / "tam_config.json").exists()
        assert (tmp_path / "tam_band_stats.npz").exists()

        with open(tmp_path / "tam_config.json") as fh:
            cfg_dict = json.load(fh)
        assert "best_val_auc" in cfg_dict
        assert cfg_dict["best_val_auc"] == pytest.approx(best_val_auc, abs=1e-5)

        stats = np.load(tmp_path / "tam_band_stats.npz")
        assert "mean" in stats
        assert "std" in stats

        ds = TAMDataset(
            smoke_pixel_df, smoke_labels,
            band_mean=stats["mean"], band_std=stats["std"],
        )
        samples = [ds[i] for i in range(len(ds))]
        batch = collate_fn(samples)
        model.eval()
        with torch.no_grad():
            prob, _ = model(batch["bands"], batch["doy"], batch["mask"], batch["n_obs"])
        assert torch.isfinite(prob).all()


# ---------------------------------------------------------------------------
# TT-7
# ---------------------------------------------------------------------------

class TestTT7LoadTamReconstructsArchitecture:
    def test_config_matches(self, tmp_path, smoke_pixel_df, smoke_labels, smoke_coords, smoke_cfg):
        trained, _ = train_tam(
            smoke_pixel_df, smoke_labels, smoke_coords,
            out_dir=tmp_path, cfg=smoke_cfg, device="cpu",
        )
        loaded, *_ = load_tam(tmp_path, device="cpu")
        assert loaded.config() == trained.config()


# ---------------------------------------------------------------------------
# TT-8
# ---------------------------------------------------------------------------

class TestTT8LoadTamIdenticalPredictions:
    def test_predictions_match(self, tmp_path, smoke_pixel_df, smoke_labels, smoke_coords, smoke_cfg):
        trained, _ = train_tam(
            smoke_pixel_df, smoke_labels, smoke_coords,
            out_dir=tmp_path, cfg=smoke_cfg, device="cpu",
        )
        loaded, band_mean, band_std, *_ = load_tam(tmp_path, device="cpu")

        ds = TAMDataset(
            smoke_pixel_df, smoke_labels,
            band_mean=band_mean, band_std=band_std,
        )
        samples = [ds[i] for i in range(len(ds))]
        batch = collate_fn(samples)

        trained.eval()
        loaded.eval()
        with torch.no_grad():
            prob_trained, _ = trained(batch["bands"], batch["doy"], batch["mask"], batch["n_obs"])
            prob_loaded,  _ = loaded(batch["bands"],  batch["doy"], batch["mask"], batch["n_obs"])

        torch.testing.assert_close(prob_trained, prob_loaded)


# ---------------------------------------------------------------------------
# TT-9: best_val_auc persisted to tam_config.json
# ---------------------------------------------------------------------------

class TestTT9BestValAucPersisted:
    def test_best_val_auc_returned_and_saved(self, tmp_path, smoke_pixel_df, smoke_labels, smoke_coords, smoke_cfg):
        _, best_val_auc = train_tam(
            smoke_pixel_df, smoke_labels, smoke_coords,
            out_dir=tmp_path, cfg=smoke_cfg, device="cpu",
        )
        assert isinstance(best_val_auc, float)
        cfg_dict = json.loads((tmp_path / "tam_config.json").read_text())
        assert "best_val_auc" in cfg_dict
        assert cfg_dict["best_val_auc"] == pytest.approx(best_val_auc, abs=1e-5)


# ---------------------------------------------------------------------------
# TT-10: S2 columns loaded for noise filter in S1-only experiments
# ---------------------------------------------------------------------------

class TestTT10S2ColsLoadedForNoiseFilter:
    def _make_multiyear_df(self) -> pl.DataFrame:
        rng = np.random.default_rng(0)
        rows = []
        for pid in ["px_pres", "px_abs"]:
            for year in [2021, 2022, 2023]:
                start = datetime.date(year, 1, 1)
                end   = datetime.date(year, 12, 31)
                dates = []
                d = start
                while d <= end:
                    dates.append(d)
                    d += datetime.timedelta(days=15)
                for d in dates:
                    rows.append({
                        "point_id": pid,
                        "date": str(d),
                        "year": year,
                        "doy": d.timetuple().tm_yday,
                        "scl_purity": 1.0,
                        "B08": float(rng.uniform(0.2, 0.5)),
                        "B04": float(rng.uniform(0.05, 0.15)),
                    })
        return pl.DataFrame(rows)

    def test_s2_global_features_not_nan_when_b08_b04_present(self):
        from tam.core.global_features import compute_global_features

        df = self._make_multiyear_df()
        gf = compute_global_features(df)

        assert "dry_ndvi" in gf.columns
        assert "rec_p" in gf.columns
        assert "nir_cv" in gf.columns
        assert gf["dry_ndvi"].is_not_null().any(), "dry_ndvi is all NaN — B08/B04 missing from pixel_df"
        assert gf["rec_p"].is_not_null().any(),    "rec_p is all NaN — B08/B04 missing from pixel_df"
        assert gf["nir_cv"].is_not_null().any(),   "nir_cv is all NaN — B08/B04 missing from pixel_df"

    def test_s2_global_features_all_nan_when_b08_b04_absent(self):
        from tam.core.global_features import compute_global_features

        df = self._make_multiyear_df().drop(["B08", "B04"])
        gf = compute_global_features(df)

        assert gf["dry_ndvi"].is_null().all(), "dry_ndvi should be NaN when B08/B04 absent"
        assert gf["rec_p"].is_null().all(),    "rec_p should be NaN when B08/B04 absent"
        assert gf["nir_cv"].is_null().all(),   "nir_cv should be NaN when B08/B04 absent"
