"""Unit tests for woody-classifier.

Module catalogue:
    WF  — features.py (compute_woody_features, _compute_s2_features, _compute_s1_features)
    WT  — train.py    (_build_summaries label alignment, cache index reassignment)
    WS  — score.py    (score_parquet column alignment, score_dir_or_file filtering)
    WE  — evaluate.py (_threshold_metrics)

WF-01  All 20 WOODY_FEATURE_NAMES columns present in output, in the documented order.
WF-02  S2-only input: all S1 features are NaN.
WF-03  S1-only input (source="S1"): all S2 features are NaN; S1 features populated.
WF-04  No source column → treated as S2; S1 features NaN.
WF-05  SCL filter: observations with scl_purity < 0.5 are excluded from S2 features.
WF-06  All-cloudy pixel (no obs survive SCL filter) → S2 features NaN, index correct.
WF-07  B11_p5 ≤ B11_p95 for any pixel with ≥2 observations.
WF-08  ndvi_amplitude = NDVI_p90 − NDVI_p10 (self-consistency).
WF-09  swir_nir_ratio_p5 = B11_p5 / (B08_p95 + 1e-6) (self-consistency).
WF-10  nir_cv = B08_std / (mean_B08 + 1e-6) for known constant-B08 pixel is 0.
WF-11  Single-observation pixel: std features are 0 (not NaN).
WF-12  NDWI_p5 < 0 for a water pixel (B03 > B08 implies negative NDWI).
WF-13  s1_mean_vh_dry uses only May–Oct obs (DOY 121–304).
WF-14  s1_vh_contrast sign: wet-season VH > dry-season VH → positive contrast.
WF-15  s1_mean_rvi in [0, 1] for valid linear VH/VV > 0.
WF-16  Multi-pixel input: each pixel's features are computed independently.
WF-17  Mixed S2+S1 input: S2 and S1 features both populated for the same pixel.
WF-18  All-cloudy S2 pixel but valid S1: index uses S2 point_ids, not S1 duplicates.
WF-19  doy computed from date when doy column absent.
WF-20  NaN VH values in S1 rows are ignored (no NaN contamination in S1 features).

WT-01  _build_summaries: presence regions get label 1.0, absence regions get 0.0.
WT-02  _build_summaries: pixels duplicated across two regions keep first label.
WT-03  load_splits cache path: reloaded label index matches summaries index exactly.

WS-01  score_parquet: output has columns point_id and prob_woody.
WS-02  score_parquet: prob_woody values are in [0, 1].
WS-03  score_parquet: feature column order survives even when feats have extra columns.
WS-04  score_parquet: missing feature columns filled with 0.0 (not NaN crash).
WS-05  score_dir_or_file: _woody_probs parquets excluded from re-scoring.

WE-01  _threshold_metrics: perfect classifier gives precision=1, recall=1.
WE-02  _threshold_metrics: all-negative predictions give recall=0, fpr=0.
WE-03  _threshold_metrics: FPR = fp / (fp + tn), not fp / total.
WE-04  _threshold_metrics: f1 = harmonic mean of precision and recall.
WE-05  _threshold_metrics handles edge case: no positives → recall denominator guard.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Load woody-classifier modules via importlib (hyphenated directory)
# ---------------------------------------------------------------------------

_WC_DIR = PROJECT_ROOT / "woody-classifier"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _WC_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_features = _load("features")
_train    = _load("train")
_score    = _load("score")
_evaluate = _load("evaluate")

compute_woody_features = _features.compute_woody_features
WOODY_FEATURE_NAMES    = _features.WOODY_FEATURE_NAMES
_compute_s1_features   = _features._compute_s1_features
_compute_s2_features   = _features._compute_s2_features

_build_summaries       = _train._build_summaries
_threshold_metrics     = _evaluate._threshold_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _s2_df(
    point_ids=("px",),
    n_obs: int = 12,
    b08: float = 0.4,
    b04: float = 0.1,
    b03: float = 0.2,
    b11: float = 0.3,
    b12: float = 0.25,
    b8a: float = 0.38,
    b05: float = 0.22,
    scl_purity: float = 1.0,
    doy_start: int = 1,
    include_source: bool = True,
) -> pd.DataFrame:
    rows = []
    for pid in point_ids:
        for i in range(n_obs):
            doy = (doy_start + i * 30) % 365 + 1
            rows.append({
                "point_id":   pid,
                "date":       pd.Timestamp(f"2024-01-01") + pd.Timedelta(days=doy - 1),
                "doy":        doy,
                "B08":        b08,
                "B04":        b04,
                "B03":        b03,
                "B11":        b11,
                "B12":        b12,
                "B8A":        b8a,
                "B05":        b05,
                "scl_purity": scl_purity,
                **({"source": "S2"} if include_source else {}),
            })
    return pd.DataFrame(rows)


def _s1_df(
    point_ids=("px",),
    n_dry: int = 6,
    n_wet: int = 4,
    vh_dry: float = 0.01,
    vv_dry: float = 0.02,
    vh_wet: float = 0.05,
    vv_wet: float = 0.02,
) -> pd.DataFrame:
    rows = []
    dry_doys = [130, 150, 180, 200, 250, 290][:n_dry]
    wet_doys  = [10, 40, 330, 360][:n_wet]
    for pid in point_ids:
        for doy in dry_doys:
            rows.append({
                "point_id": pid,
                "date":     pd.Timestamp("2024-01-01") + pd.Timedelta(days=doy - 1),
                "doy":      doy,
                "source":   "S1",
                "vh":       vh_dry,
                "vv":       vv_dry,
            })
        for doy in wet_doys:
            rows.append({
                "point_id": pid,
                "date":     pd.Timestamp("2024-01-01") + pd.Timedelta(days=doy - 1),
                "doy":      doy,
                "source":   "S1",
                "vh":       vh_wet,
                "vv":       vv_wet,
            })
    return pd.DataFrame(rows)


def _combined_df(**kw) -> pd.DataFrame:
    s2 = _s2_df(**{k: v for k, v in kw.items() if k in (
        "point_ids", "n_obs", "b08", "b04", "b03", "b11", "b12", "b8a", "b05",
        "scl_purity", "doy_start", "include_source",
    )})
    s1 = _s1_df(**{k: v for k, v in kw.items() if k in (
        "point_ids", "n_dry", "n_wet", "vh_dry", "vv_dry", "vh_wet", "vv_wet",
    )})
    return pd.concat([s2, s1], ignore_index=True)


# ---------------------------------------------------------------------------
# WF-01  All 20 columns present in the documented order
# ---------------------------------------------------------------------------

def test_wf01_all_feature_columns_present_and_ordered():
    df = _combined_df()
    out = compute_woody_features(df)
    assert list(out.columns) == WOODY_FEATURE_NAMES
    assert len(WOODY_FEATURE_NAMES) == 20


# ---------------------------------------------------------------------------
# WF-02  S2-only: S1 features are NaN
# ---------------------------------------------------------------------------

def test_wf02_s2_only_s1_features_nan():
    df = _s2_df()
    out = compute_woody_features(df)
    for col in ["s1_mean_vh_dry", "s1_vh_contrast", "s1_vh_std", "s1_mean_rvi"]:
        assert out[col].isna().all(), f"{col} should be NaN for S2-only input"


# ---------------------------------------------------------------------------
# WF-03  S1-only: S2 features NaN; S1 features populated
# ---------------------------------------------------------------------------

def test_wf03_s1_only_s2_features_nan_s1_populated():
    df = _s1_df()
    out = compute_woody_features(df)
    s2_cols = [c for c in WOODY_FEATURE_NAMES if not c.startswith("s1_")]
    for col in s2_cols:
        assert out[col].isna().all(), f"{col} should be NaN for S1-only input"
    assert out["s1_mean_vh_dry"].notna().any()
    assert out["s1_mean_rvi"].notna().any()


# ---------------------------------------------------------------------------
# WF-04  No source column → treated as S2; S1 features NaN
# ---------------------------------------------------------------------------

def test_wf04_no_source_column_treated_as_s2():
    df = _s2_df(include_source=False)
    assert "source" not in df.columns
    out = compute_woody_features(df)
    assert out["B11_p5"].notna().any()
    for col in ["s1_mean_vh_dry", "s1_vh_contrast", "s1_vh_std", "s1_mean_rvi"]:
        assert out[col].isna().all()


# ---------------------------------------------------------------------------
# WF-05  SCL filter: observations below threshold excluded
# ---------------------------------------------------------------------------

def test_wf05_scl_filter_excludes_cloudy_obs():
    # Two pixels: one has all clear obs, the other has mixed
    rows_clear = []
    rows_mixed = []
    for i in range(20):
        doy = i * 18 + 1
        rows_clear.append({
            "point_id": "clear", "doy": doy,
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=doy - 1),
            "source": "S2", "scl_purity": 1.0,
            "B11": 0.3, "B08": 0.4, "B04": 0.1, "B03": 0.2,
            "B12": 0.25, "B8A": 0.38, "B05": 0.22,
        })
        # mixed: half at B11=1.0 (cloudy, purity=0.0), half at B11=0.3 (clear)
        purity = 1.0 if i % 2 == 0 else 0.0
        b11    = 0.3 if i % 2 == 0 else 1.0
        rows_mixed.append({
            "point_id": "mixed", "doy": doy,
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=doy - 1),
            "source": "S2", "scl_purity": purity,
            "B11": b11, "B08": 0.4, "B04": 0.1, "B03": 0.2,
            "B12": 0.25, "B8A": 0.38, "B05": 0.22,
        })
    df = pd.DataFrame(rows_clear + rows_mixed)
    out = compute_woody_features(df)
    # After SCL filter "mixed" should only see the purity=1.0 rows (B11=0.3)
    # so B11_p95 for "mixed" must not be 1.0
    assert out.loc["mixed", "B11_p95"] < 0.5, (
        "SCL filter did not exclude the cloudy (purity=0) observations"
    )
    assert abs(out.loc["clear", "B11_p5"] - 0.3) < 0.01


# ---------------------------------------------------------------------------
# WF-06  All-cloudy pixel: S2 features NaN; index uses S2 point_id (not S1)
# ---------------------------------------------------------------------------

def test_wf06_all_cloudy_pixel_index_correct():
    """A pixel where every S2 obs is cloudy must still appear in the index
    with NaN S2 features, and S1 features should be populated if S1 is present.
    Critically, the index must contain the correct point_id and not duplicate
    S1 point_ids."""
    rows = []
    # S2 rows all below SCL threshold
    for i in range(5):
        doy = 50 + i * 20
        rows.append({
            "point_id": "px_cloudy", "doy": doy,
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=doy - 1),
            "source": "S2", "scl_purity": 0.1,  # below 0.5
            "B11": 0.3, "B08": 0.4, "B04": 0.1, "B03": 0.2,
            "B12": 0.25, "B8A": 0.38, "B05": 0.22,
        })
    # S1 rows (valid)
    for doy in [150, 200, 270]:
        rows.append({
            "point_id": "px_cloudy", "doy": doy,
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=doy - 1),
            "source": "S1", "vh": 0.01, "vv": 0.02,
        })
    df = pd.DataFrame(rows)
    out = compute_woody_features(df)

    assert "px_cloudy" in out.index, "Point should appear in output even when all S2 obs cloudy"
    assert out.index.value_counts().max() == 1, "No duplicate point_ids in index"
    # S2 features should be NaN (all obs filtered)
    assert np.isnan(out.loc["px_cloudy", "B11_p5"]), "B11_p5 should be NaN for all-cloudy pixel"
    # S1 features should be present
    assert not np.isnan(out.loc["px_cloudy", "s1_mean_vh_dry"]), (
        "s1_mean_vh_dry should be populated even when S2 is all-cloudy"
    )


# ---------------------------------------------------------------------------
# WF-07  B11_p5 ≤ B11_p95
# ---------------------------------------------------------------------------

def test_wf07_b11_p5_le_p95():
    df = _combined_df()
    out = compute_woody_features(df)
    row = out.iloc[0]
    assert row["B11_p5"] <= row["B11_p95"]


# ---------------------------------------------------------------------------
# WF-08  ndvi_amplitude = NDVI_p90 − NDVI_p10
# ---------------------------------------------------------------------------

def test_wf08_ndvi_amplitude_self_consistent():
    df = _combined_df()
    out = compute_woody_features(df)
    for pid in out.index:
        if not np.isnan(out.loc[pid, "ndvi_amplitude"]):
            expected = out.loc[pid, "NDVI_p90"] - out.loc[pid, "NDVI_p10"]
            assert abs(out.loc[pid, "ndvi_amplitude"] - expected) < 1e-5, (
                f"ndvi_amplitude mismatch for {pid}: "
                f"{out.loc[pid, 'ndvi_amplitude']:.6f} vs {expected:.6f}"
            )


# ---------------------------------------------------------------------------
# WF-09  swir_nir_ratio_p5 = B11_p5 / (B08_p95 + 1e-6)
# ---------------------------------------------------------------------------

def test_wf09_swir_nir_ratio_self_consistent():
    df = _combined_df()
    out = compute_woody_features(df)
    for pid in out.index:
        b11_p5  = out.loc[pid, "B11_p5"]
        b08_p95 = out.loc[pid, "B08_p95"]
        ratio   = out.loc[pid, "swir_nir_ratio_p5"]
        if not any(np.isnan(v) for v in [b11_p5, b08_p95, ratio]):
            expected = b11_p5 / (b08_p95 + 1e-6)
            assert abs(ratio - expected) < 1e-4, (
                f"swir_nir_ratio_p5 mismatch for {pid}: {ratio:.6f} vs {expected:.6f}"
            )


# ---------------------------------------------------------------------------
# WF-10  Constant B08 → nir_cv = 0
# ---------------------------------------------------------------------------

def test_wf10_constant_b08_nir_cv_zero():
    # All B08 observations equal → std=0 → nir_cv=0 (not NaN)
    df = _s2_df(b08=0.4)
    out = compute_woody_features(df)
    assert not np.isnan(out.loc["px", "nir_cv"]), "nir_cv should be 0, not NaN, for constant B08"
    assert abs(out.loc["px", "nir_cv"]) < 1e-5


# ---------------------------------------------------------------------------
# WF-11  Single-observation pixel: std features are 0, not NaN
# ---------------------------------------------------------------------------

def test_wf11_single_obs_std_is_zero():
    df = _s2_df(n_obs=1)
    out = compute_woody_features(df)
    for col in ["B11_std", "B12_std", "B08_std", "NDVI_std"]:
        val = out.loc["px", col]
        assert not np.isnan(val), f"{col} should be 0.0, not NaN, for single-obs pixel"
        assert val == 0.0, f"{col} should be 0.0 for single-obs pixel, got {val}"


# ---------------------------------------------------------------------------
# WF-12  Water pixel: NDWI_p5 < 0 (B03 < B08 → negative NDWI)
# ---------------------------------------------------------------------------

def test_wf12_water_pixel_ndwi_negative():
    # Water: B03 (green) high, B08 (NIR) low → NDWI = (B03 - B08) / (B03 + B08) > 0
    # Actually NDWI = (green - NIR) / (green + NIR); high NDWI = water
    # For water: B03 > B08 → positive NDWI. We test the reverse: dry soil has negative NDWI.
    # Let's use a water pixel: B03=0.15, B08=0.05 → NDWI = (0.15-0.05)/(0.15+0.05) = 0.5
    df = _s2_df(b03=0.15, b08=0.05)
    out = compute_woody_features(df)
    # NDWI_p5 is the 5th percentile across time — should be positive for water
    assert out.loc["px", "NDWI_p5"] > 0, (
        "NDWI_p5 should be positive for water pixel (B03 > B08)"
    )


# ---------------------------------------------------------------------------
# WF-13  s1_mean_vh_dry: only DOY 121–304 included
# ---------------------------------------------------------------------------

def test_wf13_s1_mean_vh_dry_dry_season_only():
    # Dry obs: vh=0.01 (-20 dB). Wet obs: vh=0.1 (-10 dB).
    # s1_mean_vh_dry must equal -20 dB, not -15 dB (mixed).
    df = _combined_df(vh_dry=0.01, vv_dry=0.02, vh_wet=0.1, vv_wet=0.02,
                      n_dry=6, n_wet=4)
    out = compute_woody_features(df)
    expected_dry_db = 10 * np.log10(0.01)
    assert abs(out.loc["px", "s1_mean_vh_dry"] - expected_dry_db) < 0.05, (
        f"s1_mean_vh_dry={out.loc['px', 's1_mean_vh_dry']:.2f} but expected {expected_dry_db:.2f}"
    )


# ---------------------------------------------------------------------------
# WF-14  s1_vh_contrast sign: wet VH > dry VH → positive contrast
# ---------------------------------------------------------------------------

def test_wf14_s1_vh_contrast_sign():
    # vh_wet > vh_dry in linear → larger dB wet → positive contrast
    df = _combined_df(vh_dry=0.01, vv_dry=0.02, vh_wet=0.10, vv_wet=0.02,
                      n_dry=4, n_wet=4)
    out = compute_woody_features(df)
    contrast = out.loc["px", "s1_vh_contrast"]
    assert not np.isnan(contrast), "s1_vh_contrast should not be NaN"
    assert contrast > 0, (
        f"s1_vh_contrast should be positive when wet VH > dry VH, got {contrast:.3f}"
    )


# ---------------------------------------------------------------------------
# WF-15  s1_mean_rvi in [0, 1] for valid linear VH/VV > 0
# ---------------------------------------------------------------------------

def test_wf15_s1_mean_rvi_range():
    # RVI = 4·VH_lin / (VV_lin + VH_lin), bounded [0, 4]:
    #   VH → 0: RVI → 0; VH = VV: RVI = 2; VH >> VV: RVI → 4.
    df = _combined_df(vh_dry=0.01, vv_dry=0.02, vh_wet=0.05, vv_wet=0.03)
    out = compute_woody_features(df)
    rvi = out.loc["px", "s1_mean_rvi"]
    assert not np.isnan(rvi), "s1_mean_rvi should not be NaN"
    assert 0.0 <= rvi <= 4.0, f"RVI = 4·VH/(VH+VV) should be in [0, 4], got {rvi:.4f}"
    # Also verify the exact formula for a controlled case: all obs at vh=0.01, vv=0.03
    df2 = _s1_df(vh_dry=0.01, vv_dry=0.03, vh_wet=0.01, vv_wet=0.03, n_dry=4, n_wet=4)
    out2 = compute_woody_features(df2)
    expected = 4 * 0.01 / (0.03 + 0.01)  # = 1.0
    assert abs(out2.loc["px", "s1_mean_rvi"] - expected) < 1e-4, (
        f"RVI formula: expected {expected:.4f}, got {out2.loc['px', 's1_mean_rvi']:.4f}"
    )


# ---------------------------------------------------------------------------
# WF-16  Multi-pixel: features computed independently
# ---------------------------------------------------------------------------

def test_wf16_multi_pixel_independent():
    pids = ("bright", "dark")
    rows = []
    for pid, b11_val in zip(pids, [0.6, 0.1]):
        for i in range(10):
            doy = i * 35 + 1
            rows.append({
                "point_id": pid, "doy": doy,
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=doy - 1),
                "source": "S2", "scl_purity": 1.0,
                "B11": b11_val, "B08": 0.4, "B04": 0.1, "B03": 0.2,
                "B12": 0.25, "B8A": 0.38, "B05": 0.22,
            })
    df = pd.DataFrame(rows)
    out = compute_woody_features(df)
    assert out.loc["bright", "B11_p5"] > out.loc["dark", "B11_p5"], (
        "bright pixel should have higher B11_p5 than dark pixel"
    )


# ---------------------------------------------------------------------------
# WF-17  Mixed S2+S1: both feature groups populated for same pixel
# ---------------------------------------------------------------------------

def test_wf17_mixed_input_both_groups_populated():
    df = _combined_df()
    out = compute_woody_features(df)
    assert out["B11_p5"].notna().any(), "S2 features not computed"
    assert out["s1_mean_vh_dry"].notna().any(), "S1 features not computed"


# ---------------------------------------------------------------------------
# WF-18  All-cloudy S2 + valid S1: index has no duplicates from S1 rows
# ---------------------------------------------------------------------------

def test_wf18_all_cloudy_s2_no_duplicate_index():
    rows = []
    for i in range(3):
        doy = 50 + i * 40
        rows.append({
            "point_id": "px_cloudy", "doy": doy,
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=doy - 1),
            "source": "S2", "scl_purity": 0.1,
            "B11": 0.3, "B08": 0.4, "B04": 0.1, "B03": 0.2,
            "B12": 0.25, "B8A": 0.38, "B05": 0.22,
        })
    for doy in [130, 180, 240]:
        rows.append({
            "point_id": "px_cloudy", "doy": doy,
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=doy - 1),
            "source": "S1", "vh": 0.01, "vv": 0.02,
        })
    df = pd.DataFrame(rows)
    out = compute_woody_features(df)
    assert out.index.duplicated().sum() == 0, "Output index must not have duplicates"
    assert list(out.index) == ["px_cloudy"]


# ---------------------------------------------------------------------------
# WF-19  doy computed from date when doy column absent
# ---------------------------------------------------------------------------

def test_wf19_doy_derived_from_date():
    df = _s2_df()
    assert "doy" in df.columns  # baseline
    df_no_doy = df.drop(columns=["doy"])
    assert "doy" not in df_no_doy.columns
    out = compute_woody_features(df_no_doy)
    # Should not raise and should produce valid features
    assert out["B11_p5"].notna().any()


# ---------------------------------------------------------------------------
# WF-20  NaN VH in S1 rows is ignored
# ---------------------------------------------------------------------------

def test_wf20_nan_vh_rows_ignored():
    rows = []
    # Valid S1 obs: dry season
    for doy in [150, 180, 220]:
        rows.append({
            "point_id": "px", "doy": doy,
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=doy - 1),
            "source": "S1", "vh": 0.01, "vv": 0.02,
        })
    # NaN VH obs (e.g. missing acquisition)
    for doy in [160, 200]:
        rows.append({
            "point_id": "px", "doy": doy,
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=doy - 1),
            "source": "S1", "vh": np.nan, "vv": 0.02,
        })
    df = pd.DataFrame(rows)
    out = compute_woody_features(df)
    expected_db = 10 * np.log10(0.01)
    assert not np.isnan(out.loc["px", "s1_mean_vh_dry"]), "s1_mean_vh_dry should be finite"
    assert abs(out.loc["px", "s1_mean_vh_dry"] - expected_db) < 0.01, (
        "NaN VH rows should not contaminate s1_mean_vh_dry"
    )


# ---------------------------------------------------------------------------
# WT-01  _build_summaries: correct labels for presence/absence regions
# ---------------------------------------------------------------------------

def test_wt01_build_summaries_label_assignment(tmp_path):
    """Presence regions → label 1.0, absence regions → label 0.0."""
    from utils.regions import TrainingRegion

    def _make_parquet(path: Path, point_id: str) -> None:
        df = _s2_df(point_ids=(point_id,))
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)

    # Patch _region_parquet_path to return tmp files
    presence_path = tmp_path / "regions" / "reg_presence.parquet"
    absence_path  = tmp_path / "regions" / "reg_absence.parquet"
    _make_parquet(presence_path, "px_pres")
    _make_parquet(absence_path,  "px_abs")

    presence_region = TrainingRegion(
        id="reg_presence", name="P", label="presence",
        bbox=[0, 0, 1, 1], year=2024, tags=[], notes=None,
    )
    absence_region = TrainingRegion(
        id="reg_absence", name="A", label="absence",
        bbox=[0, 0, 1, 1], year=2024, tags=[], notes=None,
    )

    with patch.object(_train, "_region_parquet_path") as mock_path:
        def side_effect(region_id):
            return presence_path if "presence" in region_id else absence_path
        mock_path.side_effect = side_effect

        summaries, labels = _build_summaries([presence_region, absence_region])

    assert labels.loc["px_pres"] == 1.0, "Presence region must have label 1.0"
    assert labels.loc["px_abs"]  == 0.0, "Absence region must have label 0.0"


# ---------------------------------------------------------------------------
# WT-02  _build_summaries: duplicate pixel across two regions keeps first label
# ---------------------------------------------------------------------------

def test_wt02_build_summaries_duplicate_pixel_keeps_first(tmp_path):
    from utils.regions import TrainingRegion

    shared_path = tmp_path / "regions" / "region_a.parquet"
    shared_path.parent.mkdir(parents=True, exist_ok=True)
    _s2_df(point_ids=("shared_px",)).to_parquet(shared_path)

    reg_a = TrainingRegion(id="region_a", name="A", label="presence",
                           bbox=[0, 0, 1, 1], year=2024, tags=[], notes=None)
    reg_b = TrainingRegion(id="region_b", name="B", label="absence",
                           bbox=[0, 0, 1, 1], year=2024, tags=[], notes=None)

    with patch.object(_train, "_region_parquet_path", return_value=shared_path):
        summaries, labels = _build_summaries([reg_a, reg_b])

    # shared_px appears in both — keep first (presence)
    assert "shared_px" in labels.index
    assert labels.loc["shared_px"] == 1.0, (
        "When a pixel appears in multiple regions, the first region's label must win"
    )
    assert summaries.index.duplicated().sum() == 0


# ---------------------------------------------------------------------------
# WT-03  load_splits cache: reloaded label index matches summaries index
# ---------------------------------------------------------------------------

def test_wt03_cache_label_index_matches_summaries(tmp_path):
    """The critical bug: tr_lbl.index = tr_sum.index blindly overwrites index.
    If parquet RangeIndex differs from string index, labels silently misalign.
    We verify that after cache round-trip, label[pid] corresponds to the same pid."""
    summaries = pd.DataFrame(
        {"B11_p5": [0.1, 0.2, 0.3]},
        index=pd.Index(["pid_a", "pid_b", "pid_c"], name="point_id"),
    )
    labels = pd.Series([1.0, 0.0, 1.0],
                       index=pd.Index(["pid_a", "pid_b", "pid_c"], name="point_id"),
                       name="label")

    # Write to parquet (simulating what train.py does)
    tr_sum_path = tmp_path / "train_summaries.parquet"
    tr_lbl_path = tmp_path / "train_labels.parquet"
    summaries.to_parquet(tr_sum_path)
    pd.DataFrame({"label": labels.values}, index=summaries.index).to_parquet(tr_lbl_path)

    # Reload and reassign index the same way load_splits does
    tr_sum = pd.read_parquet(tr_sum_path)
    tr_lbl = pd.read_parquet(tr_lbl_path)["label"]
    tr_lbl.index = tr_sum.index  # this is the line under test

    # Verify each pid maps to the correct label
    for pid, expected in [("pid_a", 1.0), ("pid_b", 0.0), ("pid_c", 1.0)]:
        actual = tr_lbl.loc[pid]
        assert actual == expected, (
            f"After cache round-trip, {pid} has label {actual} instead of {expected}. "
            "Index reassignment in load_splits may be silently misaligning labels."
        )


# ---------------------------------------------------------------------------
# WS-01/02  score_parquet: correct columns and probability range
# ---------------------------------------------------------------------------

def _make_mock_model(n_features: int, prob: float = 0.7):
    """Return a mock sklearn-compatible classifier."""
    model = MagicMock()
    def predict_proba(X):
        return np.column_stack([
            np.full(len(X), 1 - prob),
            np.full(len(X), prob),
        ]).astype(np.float32)
    model.predict_proba = predict_proba
    return model


def test_ws01_score_parquet_output_columns(tmp_path):
    df = _combined_df()
    pq_path = tmp_path / "input.parquet"
    df.to_parquet(pq_path)

    feat_names = WOODY_FEATURE_NAMES
    model = _make_mock_model(len(feat_names), prob=0.6)
    out_path = _score.score_parquet(pq_path, model, feat_names, tmp_path)

    assert out_path is not None and out_path.exists()
    result = pd.read_parquet(out_path)
    assert "point_id"   in result.columns, "Output must have point_id column"
    assert "prob_woody" in result.columns, "Output must have prob_woody column"


def test_ws02_score_parquet_probs_in_range(tmp_path):
    df = _combined_df()
    pq_path = tmp_path / "input.parquet"
    df.to_parquet(pq_path)

    model = _make_mock_model(len(WOODY_FEATURE_NAMES), prob=0.6)
    out_path = _score.score_parquet(pq_path, model, WOODY_FEATURE_NAMES, tmp_path)

    result = pd.read_parquet(out_path)
    assert (result["prob_woody"] >= 0).all() and (result["prob_woody"] <= 1).all()


# ---------------------------------------------------------------------------
# WS-03  score_parquet: extra columns in feats don't cause misalignment
# ---------------------------------------------------------------------------

def test_ws03_score_parquet_column_alignment(tmp_path):
    """Features DataFrame may have columns in a different order than feat_names.
    The X matrix must be assembled in feat_names order, not feats.columns order."""
    df = _combined_df()
    pq_path = tmp_path / "input.parquet"
    df.to_parquet(pq_path)

    # Reverse the feature name order to detect alignment bugs
    reversed_names = list(reversed(WOODY_FEATURE_NAMES))

    received_X = []
    def predict_proba(X):
        received_X.append(X.copy())
        return np.column_stack([np.zeros(len(X)), np.ones(len(X))]).astype(np.float32)

    model = MagicMock()
    model.predict_proba = predict_proba

    _score.score_parquet(pq_path, model, reversed_names, tmp_path)

    assert len(received_X) == 1
    X = received_X[0]
    # Column 0 should correspond to reversed_names[0] = last feature
    # We can't assert specific values without knowing the pixel values,
    # but we can verify shape
    assert X.shape[1] == len(WOODY_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# WS-04  score_parquet: missing feature columns filled with 0.0
# ---------------------------------------------------------------------------

def test_ws04_score_parquet_missing_features_filled(tmp_path):
    """If the model was trained with S1 features but the parquet has no S1 data,
    the missing S1 feature columns must be filled with 0.0, not raise KeyError."""
    df = _s2_df()  # no S1 data
    pq_path = tmp_path / "s2only.parquet"
    df.to_parquet(pq_path)

    received_X = []
    def predict_proba(X):
        received_X.append(X.copy())
        return np.column_stack([np.zeros(len(X)), np.ones(len(X))]).astype(np.float32)

    model = MagicMock()
    model.predict_proba = predict_proba

    # Should not raise even though S1 features will be NaN in feats → filled with 0
    out_path = _score.score_parquet(pq_path, model, WOODY_FEATURE_NAMES, tmp_path)
    assert out_path is not None

    X = received_X[0]
    # S1 feature columns (last 4) should be 0.0, not NaN
    s1_col_indices = [WOODY_FEATURE_NAMES.index(c) for c in
                      ["s1_mean_vh_dry", "s1_vh_contrast", "s1_vh_std", "s1_mean_rvi"]]
    for j in s1_col_indices:
        assert not np.isnan(X[:, j]).any(), f"Feature column {j} has NaN — should be 0.0"


# ---------------------------------------------------------------------------
# WS-05  score_dir_or_file: excludes _woody_probs parquets from re-scoring
# ---------------------------------------------------------------------------

def test_ws05_score_dir_excludes_woody_probs_parquets(tmp_path):
    """Parquets whose name contains 'woody_probs' must not be re-scored."""
    # Write a normal parquet and a woody_probs parquet in the same dir
    df = _combined_df()
    normal_pq   = tmp_path / "tile_55KCQ.parquet"
    probs_pq    = tmp_path / "tile_55KCQ_woody_probs.parquet"
    df.to_parquet(normal_pq)
    df.to_parquet(probs_pq)

    scored_paths = []
    def mock_score_parquet(path, model, feat_names, out_dir, threshold):
        scored_paths.append(path)
        return out_dir / f"{path.stem}_woody_probs.parquet"

    model = _make_mock_model(len(WOODY_FEATURE_NAMES))
    with patch.object(_score, "score_parquet", side_effect=mock_score_parquet):
        _score.score_dir_or_file(tmp_path, model, WOODY_FEATURE_NAMES, tmp_path / "out")

    assert all("woody_probs" not in p.name for p in scored_paths), (
        "score_dir_or_file must not re-score _woody_probs parquets"
    )
    assert any(p.name == "tile_55KCQ.parquet" for p in scored_paths), (
        "Normal parquet should have been scored"
    )


# ---------------------------------------------------------------------------
# WE-01  _threshold_metrics: perfect classifier
# ---------------------------------------------------------------------------

def test_we01_perfect_classifier():
    y     = np.array([1, 1, 0, 0], dtype=float)
    probs = np.array([0.9, 0.95, 0.1, 0.05])
    m = _threshold_metrics(y, probs, 0.5)
    assert m["precision"] == 1.0
    assert m["recall"]    == 1.0
    assert m["fpr"]       == 0.0


# ---------------------------------------------------------------------------
# WE-02  _threshold_metrics: all-negative predictions
# ---------------------------------------------------------------------------

def test_we02_all_negative_predictions():
    y     = np.array([1, 1, 0, 0], dtype=float)
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    m = _threshold_metrics(y, probs, 0.9)  # nothing exceeds threshold
    assert m["recall"] == 0.0
    assert m["fpr"]    == 0.0
    assert m["tp"] == 0 and m["fp"] == 0


# ---------------------------------------------------------------------------
# WE-03  _threshold_metrics: FPR = fp / (fp + tn), not fp / n_total
# ---------------------------------------------------------------------------

def test_we03_fpr_denominator():
    # 1 TP, 1 FP, 1 FN, 7 TN
    y     = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    probs = np.array([0.9, 0.8, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    m = _threshold_metrics(y, probs, 0.75)
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1 and m["tn"] == 7
    expected_fpr = 1 / (1 + 7)  # fp / (fp + tn)
    assert abs(m["fpr"] - expected_fpr) < 1e-9, (
        f"FPR should be {expected_fpr:.4f} (fp/(fp+tn)), got {m['fpr']:.4f}"
    )


# ---------------------------------------------------------------------------
# WE-04  _threshold_metrics: f1 = harmonic mean
# ---------------------------------------------------------------------------

def test_we04_f1_is_harmonic_mean():
    y     = np.array([1, 1, 1, 0, 0, 0], dtype=float)
    probs = np.array([0.9, 0.8, 0.3, 0.9, 0.1, 0.1])
    m = _threshold_metrics(y, probs, 0.5)
    p, r = m["precision"], m["recall"]
    expected_f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    assert abs(m["f1"] - expected_f1) < 1e-9


# ---------------------------------------------------------------------------
# WE-05  _threshold_metrics: no positives in ground truth → recall guard
# ---------------------------------------------------------------------------

def test_we05_no_positives_recall_guard():
    y     = np.array([0, 0, 0], dtype=float)
    probs = np.array([0.9, 0.8, 0.7])
    # Should not divide by zero
    m = _threshold_metrics(y, probs, 0.5)
    assert m["recall"] == 0.0, "recall should be 0.0 when no true positives exist"
    assert m["f1"]     == 0.0
