"""TT-* tests for spatial_split and train_tam."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
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
def many_labels() -> pd.Series:
    """8 presence + 8 absence pixels for spatial split tests."""
    return pd.Series(
        {f"p{i}": 1.0 for i in range(8)} |
        {f"a{i}": 0.0 for i in range(8)}
    )


@pytest.fixture
def many_coords(many_labels) -> pd.DataFrame:
    """Unique latitudes so sort order is deterministic."""
    rows = []
    for i, pid in enumerate(many_labels.index):
        rows.append({"point_id": pid, "lon": 144.0, "lat": -20.0 - i * 0.5})
    return pd.DataFrame(rows)


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
def smoke_pixel_df(band_cols) -> pd.DataFrame:
    """10 presence + 10 absence pixels; 3 of each land in val so AUC is always computable."""
    rng = np.random.default_rng(42)
    rows = []
    for pid in [f"p{i}" for i in range(10)] + [f"a{i}" for i in range(10)]:
        dates = pd.date_range("2023-01-15", periods=20, freq="15D")
        for d in dates:
            rows.append({
                "point_id": pid, "date": str(d.date()),
                "scl_purity": 1.0, "year": 2023,
                **{b: float(rng.uniform(0.01, 0.5)) for b in band_cols},
            })
    return pd.DataFrame(rows)


@pytest.fixture
def smoke_labels() -> pd.Series:
    return pd.Series(
        {f"p{i}": 1.0 for i in range(10)} |
        {f"a{i}": 0.0 for i in range(10)}
    )


@pytest.fixture
def smoke_coords(smoke_labels) -> pd.DataFrame:
    rows = []
    for i, pid in enumerate(smoke_labels.index):
        rows.append({"point_id": pid, "lon": 144.0, "lat": -20.0 - i * 0.5})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TT-1
# ---------------------------------------------------------------------------

class TestTT1BothClassesInTrainAndVal:
    def test_both_classes_in_both_splits(self, many_labels, many_coords):
        train, val = spatial_split(many_labels, many_coords, val_frac=0.2)
        assert set(train.unique()) == {0.0, 1.0}
        assert set(val.unique()) == {0.0, 1.0}


# ---------------------------------------------------------------------------
# TT-2
# ---------------------------------------------------------------------------

class TestTT2ValFractionApproximatelyCorrect:
    def test_val_fraction(self, many_labels, many_coords):
        val_frac = 0.2
        train, val = spatial_split(many_labels, many_coords, val_frac=val_frac)
        total = len(train) + len(val)
        actual_frac = len(val) / total
        # split rounds per-class via max(1, int(...)), so allow up to 2-pixel slack
        assert abs(actual_frac - val_frac) <= 2 / total + 1e-9


# ---------------------------------------------------------------------------
# TT-3
# ---------------------------------------------------------------------------

class TestTT3NoOverlapBetweenSplits:
    def test_disjoint_indices(self, many_labels, many_coords):
        train, val = spatial_split(many_labels, many_coords, val_frac=0.2)
        assert set(train.index) & set(val.index) == set()


# ---------------------------------------------------------------------------
# TT-4
# ---------------------------------------------------------------------------

class TestTT4ValPixelsSouthernmost:
    def test_val_pixels_have_lower_lat(self, many_labels, many_coords):
        train, val = spatial_split(many_labels, many_coords, val_frac=0.25)
        coords = many_coords.set_index("point_id")
        for cls_val in [0.0, 1.0]:
            train_cls = train[train == cls_val]
            val_cls   = val[val == cls_val]
            if len(train_cls) == 0 or len(val_cls) == 0:
                continue
            max_val_lat   = coords.loc[val_cls.index, "lat"].max()
            min_train_lat = coords.loc[train_cls.index, "lat"].min()
            assert max_val_lat <= min_train_lat, (
                f"Class {cls_val}: val pixel lat {max_val_lat} > train min lat {min_train_lat}"
            )


# ---------------------------------------------------------------------------
# TT-5
# ---------------------------------------------------------------------------

class TestTT5SinglePixelPerClassDoesNotCrash:
    def test_single_pixel_per_class(self):
        labels = pd.Series({"p0": 1.0, "a0": 0.0})
        coords = pd.DataFrame({"point_id": ["p0", "a0"], "lon": [144.0, 144.0], "lat": [-20.0, -21.0]})
        # Should not raise
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

        # tam_config.json must be valid JSON and contain best_val_auc
        with open(tmp_path / "tam_config.json") as fh:
            cfg_dict = json.load(fh)
        assert "best_val_auc" in cfg_dict
        assert cfg_dict["best_val_auc"] == pytest.approx(best_val_auc, abs=1e-5)

        # tam_band_stats.npz must have mean and std keys
        stats = np.load(tmp_path / "tam_band_stats.npz")
        assert "mean" in stats
        assert "std" in stats

        # Model produces finite probabilities on training data
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
        loaded, _, _ = load_tam(tmp_path, device="cpu")
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
        loaded, band_mean, band_std = load_tam(tmp_path, device="cpu")

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
    """train_tam must return (model, best_val_auc) and write it to tam_config.json."""

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
    """B08 and B04 must be present in pixel_df when compute_global_features is called,
    even for S1-only experiments where they are not model input features.
    Without them nir_cv/rec_p/dry_ndvi are all NaN and the noise filter is silently
    disabled — a regression that caused NR presence pixels to pass through unfiltered.
    """

    def _make_multiyear_df(self) -> pd.DataFrame:
        """Two pixels, 3 years of observations spanning the full calendar year.
        nir_cv requires ≥2 years of dry-season (DOY 121–304) data to be non-NaN.
        """
        rng = np.random.default_rng(0)
        rows = []
        for pid in ["px_pres", "px_abs"]:
            for year in [2021, 2022, 2023]:
                dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="15D")
                for d in dates:
                    rows.append({
                        "point_id": pid,
                        "date": str(d.date()),
                        "year": year,
                        "doy": d.day_of_year,
                        "scl_purity": 1.0,
                        "B08": float(rng.uniform(0.2, 0.5)),
                        "B04": float(rng.uniform(0.05, 0.15)),
                    })
        return pd.DataFrame(rows)

    def test_s2_global_features_not_nan_when_b08_b04_present(self):
        from tam.core.global_features import compute_global_features

        df = self._make_multiyear_df()
        gf = compute_global_features(df)

        assert "dry_ndvi" in gf.columns
        assert "rec_p" in gf.columns
        assert "nir_cv" in gf.columns
        assert gf["dry_ndvi"].notna().any(), "dry_ndvi is all NaN — B08/B04 missing from pixel_df"
        assert gf["rec_p"].notna().any(),    "rec_p is all NaN — B08/B04 missing from pixel_df"
        assert gf["nir_cv"].notna().any(),   "nir_cv is all NaN — B08/B04 missing from pixel_df"

    def test_s2_global_features_all_nan_when_b08_b04_absent(self):
        from tam.core.global_features import compute_global_features

        # Drop B08/B04 — simulates the S1-only loading bug.
        df = self._make_multiyear_df().drop(columns=["B08", "B04"])
        gf = compute_global_features(df)

        # Without B08/B04 all S2-derived features should be NaN.
        assert gf["dry_ndvi"].isna().all(), "dry_ndvi should be NaN when B08/B04 absent"
        assert gf["rec_p"].isna().all(),    "rec_p should be NaN when B08/B04 absent"
        assert gf["nir_cv"].isna().all(),   "nir_cv should be NaN when B08/B04 absent"
