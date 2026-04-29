"""Unit tests for S1 feature computation.

S1G — S1 global features (global_features.py)
S1T — S1 time-series snapping (dataset.py snap_s1_to_s2)
S1D — TAMDataset integration with use_s1=True

S1G-1.  S1 globals computed correctly from known VH/VV values.
S1G-2.  S1 globals are NaN when no S1 rows present.
S1G-3.  S1 globals absent from pixel_df (no vh/vv columns) → NaN columns still present.
S1G-4.  dry/wet season split is correct (May–Oct = dry, Nov–Apr = wet).
S1G-5.  compute_global_features returns all 9 feature names.
S1G-6.  S2-only pixel_df produces NaN for all S1 global columns.

S1T-1.  snap_s1_to_s2: S1 observation within window snaps to correct S2 row.
S1T-2.  snap_s1_to_s2: S1 observation outside window (>7 days) produces NaN.
S1T-3.  snap_s1_to_s2: nearest S1 chosen when two candidates equidistant.
S1T-4.  snap_s1_to_s2: VH dB conversion correct (10·log10).
S1T-5.  snap_s1_to_s2: VH−VV ratio correct.
S1T-6.  snap_s1_to_s2: RVI correct (4·VH/(VH+VV)).
S1T-7.  snap_s1_to_s2: output contains only S2 rows (S1 rows not in output).
S1T-8.  snap_s1_to_s2: no S1 data → all S1 columns NaN.
S1T-9.  snap_s1_to_s2: pixel with no S1 data has NaN S1 columns.
S1T-10. snap_s1_to_s2: spatial alignment — S1 only snaps to same point_id.

S1D-1.  TAMDataset with use_s1=True has N_BANDS_S1 features.
S1D-2.  TAMDataset with use_s1=False has N_BANDS features (S2 only).
S1D-3.  TAMDataset use_s1=True: S1 features non-NaN for windows with S1 data.
S1D-4.  TAMDataset use_s1=True: band_mean/band_std shape matches N_BANDS_S1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from tam.core.global_features import (
    GLOBAL_FEATURE_NAMES,
    compute_global_features,
    _compute_s1_globals,
)
from tam.core.dataset import (
    snap_s1_to_s2,
    TAMDataset,
    N_BANDS,
    N_BANDS_S1,
    S1_FEATURE_COLS,
)
from analysis.constants import BANDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DRY_DOYS  = [150, 200, 270]   # May–Oct
_WET_DOYS  = [10, 50, 330]     # Nov–Apr

def _s2_row(point_id: str, date: str, doy: int, **band_kw) -> dict:
    row = {
        "point_id": point_id,
        "date": pd.Timestamp(date),
        "doy": doy,
        "year": pd.Timestamp(date).year,
        "source": "S2",
        "scl_purity": 1.0,
        "scl": 4,
        **{b: 0.2 for b in BANDS},
    }
    row.update(band_kw)
    return row


def _s1_row(point_id: str, date: str, doy: int, vh: float, vv: float) -> dict:
    return {
        "point_id": point_id,
        "date": pd.Timestamp(date),
        "doy": doy,
        "year": pd.Timestamp(date).year,
        "source": "S1",
        "vh": vh,
        "vv": vv,
        "scl_purity": np.nan,
        "scl": np.nan,
        **{b: np.nan for b in BANDS},
    }


def _make_pixel_df(s2_rows: list[dict], s1_rows: list[dict] | None = None) -> pd.DataFrame:
    rows = list(s2_rows)
    if s1_rows:
        rows.extend(s1_rows)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    # Add vh/vv columns if missing
    if "vh" not in df.columns:
        df["vh"] = np.nan
    if "vv" not in df.columns:
        df["vv"] = np.nan
    return df


# ---------------------------------------------------------------------------
# S1G-1: S1 globals computed correctly from known values
# ---------------------------------------------------------------------------

def test_s1_globals_basic_values():
    vh_lin = 0.01   # -20 dB
    vv_lin = 0.02   # ~-17 dB
    vh_db  = 10 * np.log10(vh_lin)
    vv_db  = 10 * np.log10(vv_lin)

    rows = []
    for doy in _DRY_DOYS:
        rows.append({"point_id": "px", "doy": doy, "date": pd.Timestamp("2022-01-01"),
                     "source": "S1", "vh": vh_lin, "vv": vv_lin})
    for doy in _WET_DOYS:
        rows.append({"point_id": "px", "doy": doy, "date": pd.Timestamp("2022-01-01"),
                     "source": "S1", "vh": vh_lin * 0.5, "vv": vv_lin})

    df = pd.DataFrame(rows)
    result = _compute_s1_globals(df)

    assert abs(result["s1_mean_vh_dry"]["px"] - vh_db) < 0.01
    # wet VH is vh_lin*0.5 → dB = vh_db - 3 dB
    expected_contrast = (vh_db - 3.0) - vh_db
    assert abs(result["s1_vh_contrast"]["px"] - expected_contrast) < 0.1
    # mean_rvi pools all rows: dry (vh_lin) and wet (vh_lin*0.5)
    # dry RVI: 4*0.01/(0.02+0.01) = 1.333; wet RVI: 4*0.005/(0.02+0.005) = 0.8
    dry_rvi = 4 * vh_lin / (vv_lin + vh_lin)
    wet_rvi = 4 * (vh_lin * 0.5) / (vv_lin + vh_lin * 0.5)
    expected_rvi = (dry_rvi * len(_DRY_DOYS) + wet_rvi * len(_WET_DOYS)) / (len(_DRY_DOYS) + len(_WET_DOYS))
    assert abs(result["s1_mean_rvi"]["px"] - expected_rvi) < 0.01


# ---------------------------------------------------------------------------
# S1G-2: NaN when no S1 rows
# ---------------------------------------------------------------------------

def test_s1_globals_no_s1_rows():
    df = pd.DataFrame([{"point_id": "px", "doy": 150, "date": pd.Timestamp("2022-06-01"),
                        "source": "S2", "vh": np.nan, "vv": np.nan}])
    result = _compute_s1_globals(df)
    for key in result.values():
        assert key.empty or key.isna().all()


# ---------------------------------------------------------------------------
# S1G-3: no vh/vv columns → NaN columns present in output
# ---------------------------------------------------------------------------

def test_compute_global_features_no_s1_columns():
    rows = []
    for yr in [2021, 2022]:
        for doy in range(1, 366, 30):
            rows.append({
                "point_id": "px", "year": yr, "doy": doy,
                "date": pd.Timestamp(f"{yr}-01-01"),
                "B08": 0.4, "B04": 0.1,
            })
    df = pd.DataFrame(rows)
    result = compute_global_features(df)
    assert set(result.columns) == set(GLOBAL_FEATURE_NAMES)
    for col in ["s1_mean_vh_dry", "s1_vh_contrast", "s1_vh_std", "s1_mean_rvi"]:
        assert result[col].isna().all(), f"{col} should be NaN when no S1 data"


# ---------------------------------------------------------------------------
# S1G-4: dry/wet season split
# ---------------------------------------------------------------------------

def test_s1_globals_dry_wet_split():
    rows = []
    # dry season only
    for doy in [121, 200, 304]:
        rows.append({"point_id": "px", "doy": doy, "date": pd.Timestamp("2022-01-01"),
                     "source": "S1", "vh": 0.01, "vv": 0.02})
    # wet season only (different VH)
    for doy in [1, 60, 340]:
        rows.append({"point_id": "px", "doy": doy, "date": pd.Timestamp("2022-01-01"),
                     "source": "S1", "vh": 0.10, "vv": 0.02})
    df = pd.DataFrame(rows)
    result = _compute_s1_globals(df)
    dry_db = 10 * np.log10(0.01)
    wet_db = 10 * np.log10(0.10)
    assert abs(result["s1_mean_vh_dry"]["px"] - dry_db) < 0.01
    assert abs(result["s1_vh_contrast"]["px"] - (wet_db - dry_db)) < 0.1


# ---------------------------------------------------------------------------
# S1G-5: compute_global_features returns all 9 feature names
# ---------------------------------------------------------------------------

def test_compute_global_features_returns_all_names():
    rows = []
    for yr in [2021, 2022, 2023]:
        for doy in range(1, 366, 20):
            rows.append({
                "point_id": "px", "year": yr, "doy": doy,
                "date": pd.Timestamp(f"{yr}-01-01"),
                "B08": 0.4, "B04": 0.1,
                "source": "S2", "vh": np.nan, "vv": np.nan,
            })
    # Add S1 rows
    for doy in [150, 200, 270, 30, 60]:
        rows.append({
            "point_id": "px", "year": 2022, "doy": doy,
            "date": pd.Timestamp("2022-01-01"),
            "B08": np.nan, "B04": np.nan,
            "source": "S1", "vh": 0.01, "vv": 0.02,
        })
    df = pd.DataFrame(rows)
    result = compute_global_features(df)
    assert list(result.columns) == GLOBAL_FEATURE_NAMES
    assert len(GLOBAL_FEATURE_NAMES) == 9


# ---------------------------------------------------------------------------
# S1G-6: S2-only pixel_df → NaN for S1 globals
# ---------------------------------------------------------------------------

def test_compute_global_features_s2_only_rows():
    rows = []
    for yr in [2021, 2022]:
        for doy in range(1, 366, 20):
            rows.append({
                "point_id": "px", "year": yr, "doy": doy,
                "date": pd.Timestamp(f"{yr}-01-01"),
                "B08": 0.4, "B04": 0.1,
                "source": "S2", "vh": np.nan, "vv": np.nan,
            })
    df = pd.DataFrame(rows)
    result = compute_global_features(df)
    for col in ["s1_mean_vh_dry", "s1_vh_contrast", "s1_vh_std", "s1_mean_rvi"]:
        assert result[col].isna().all()


# ---------------------------------------------------------------------------
# S1T-1: snap_s1_to_s2 snaps to correct S2 row
# ---------------------------------------------------------------------------

def test_snap_s1_snaps_within_window():
    df = _make_pixel_df(
        s2_rows=[_s2_row("px", "2022-06-01", 152)],
        s1_rows=[_s1_row("px", "2022-06-04", 155, vh=0.01, vv=0.02)],  # 3 days later
    )
    out = snap_s1_to_s2(df)
    assert len(out) == 1
    assert not np.isnan(out["s1_vh"].iloc[0])
    assert abs(out["s1_vh"].iloc[0] - 10 * np.log10(0.01)) < 0.001


# ---------------------------------------------------------------------------
# S1T-2: outside window → NaN
# ---------------------------------------------------------------------------

def test_snap_s1_outside_window_is_nan():
    df = _make_pixel_df(
        s2_rows=[_s2_row("px", "2022-06-01", 152)],
        s1_rows=[_s1_row("px", "2022-06-15", 166, vh=0.01, vv=0.02)],  # 14 days
    )
    out = snap_s1_to_s2(df)
    assert np.isnan(out["s1_vh"].iloc[0])


# ---------------------------------------------------------------------------
# S1T-3: nearest S1 chosen when two candidates
# ---------------------------------------------------------------------------

def test_snap_s1_chooses_nearest():
    df = _make_pixel_df(
        s2_rows=[_s2_row("px", "2022-06-05", 156)],
        s1_rows=[
            _s1_row("px", "2022-06-02", 153, vh=0.001, vv=0.002),  # 3 days before
            _s1_row("px", "2022-06-08", 159, vh=0.010, vv=0.020),  # 3 days after
        ],
    )
    out = snap_s1_to_s2(df)
    # Both equidistant — either is valid, just must be non-NaN
    assert not np.isnan(out["s1_vh"].iloc[0])


# ---------------------------------------------------------------------------
# S1T-4: VH dB conversion
# ---------------------------------------------------------------------------

def test_snap_s1_vh_db_conversion():
    vh_lin = 0.005
    df = _make_pixel_df(
        s2_rows=[_s2_row("px", "2022-06-01", 152)],
        s1_rows=[_s1_row("px", "2022-06-01", 152, vh=vh_lin, vv=0.01)],
    )
    out = snap_s1_to_s2(df)
    expected = 10 * np.log10(vh_lin)
    assert abs(out["s1_vh"].iloc[0] - expected) < 1e-4


# ---------------------------------------------------------------------------
# S1T-5: VH−VV ratio
# ---------------------------------------------------------------------------

def test_snap_s1_vh_vv_ratio():
    vh_lin, vv_lin = 0.01, 0.04
    df = _make_pixel_df(
        s2_rows=[_s2_row("px", "2022-06-01", 152)],
        s1_rows=[_s1_row("px", "2022-06-01", 152, vh=vh_lin, vv=vv_lin)],
    )
    out = snap_s1_to_s2(df)
    expected = 10 * np.log10(vh_lin) - 10 * np.log10(vv_lin)
    assert abs(out["s1_vh_vv"].iloc[0] - expected) < 1e-4


# ---------------------------------------------------------------------------
# S1T-6: RVI
# ---------------------------------------------------------------------------

def test_snap_s1_rvi():
    vh_lin, vv_lin = 0.01, 0.03
    df = _make_pixel_df(
        s2_rows=[_s2_row("px", "2022-06-01", 152)],
        s1_rows=[_s1_row("px", "2022-06-01", 152, vh=vh_lin, vv=vv_lin)],
    )
    out = snap_s1_to_s2(df)
    expected = 4 * vh_lin / (vv_lin + vh_lin)
    assert abs(out["s1_rvi"].iloc[0] - expected) < 1e-5


# ---------------------------------------------------------------------------
# S1T-7: output contains only S2 rows
# ---------------------------------------------------------------------------

def test_snap_s1_output_only_s2_rows():
    df = _make_pixel_df(
        s2_rows=[_s2_row("px", "2022-06-01", 152), _s2_row("px", "2022-07-01", 182)],
        s1_rows=[_s1_row("px", "2022-06-02", 153, vh=0.01, vv=0.02)],
    )
    out = snap_s1_to_s2(df)
    assert len(out) == 2
    assert (out["source"] == "S2").all()


# ---------------------------------------------------------------------------
# S1T-8: no S1 data → all S1 columns NaN
# ---------------------------------------------------------------------------

def test_snap_s1_no_s1_data():
    df = _make_pixel_df(
        s2_rows=[_s2_row("px", "2022-06-01", 152)],
        s1_rows=None,
    )
    out = snap_s1_to_s2(df)
    for col in S1_FEATURE_COLS:
        assert col in out.columns
        assert np.isnan(out[col].iloc[0])


# ---------------------------------------------------------------------------
# S1T-9: pixel with no S1 data has NaN S1 columns
# ---------------------------------------------------------------------------

def test_snap_s1_pixel_without_s1_is_nan():
    df = _make_pixel_df(
        s2_rows=[
            _s2_row("px_a", "2022-06-01", 152),
            _s2_row("px_b", "2022-06-01", 152),
        ],
        s1_rows=[_s1_row("px_a", "2022-06-01", 152, vh=0.01, vv=0.02)],
    )
    out = snap_s1_to_s2(df)
    out = out.set_index("point_id")
    assert not np.isnan(out.loc["px_a", "s1_vh"])
    assert np.isnan(out.loc["px_b", "s1_vh"])


# ---------------------------------------------------------------------------
# S1T-10: S1 only snaps to same point_id
# ---------------------------------------------------------------------------

def test_snap_s1_spatial_isolation():
    """S1 obs for px_a must not snap to S2 obs for px_b."""
    df = _make_pixel_df(
        s2_rows=[
            _s2_row("px_a", "2022-06-01", 152),
            _s2_row("px_b", "2022-06-01", 152),
        ],
        s1_rows=[_s1_row("px_a", "2022-06-02", 153, vh=0.99, vv=0.99)],
    )
    out = snap_s1_to_s2(df).set_index("point_id")
    # px_b has no S1 → must be NaN even though px_a's S1 is on the same date
    assert np.isnan(out.loc["px_b", "s1_vh"])
    assert not np.isnan(out.loc["px_a", "s1_vh"])


# ---------------------------------------------------------------------------
# S1D-1: TAMDataset with use_s1=True has N_BANDS_S1 features
# ---------------------------------------------------------------------------

def _make_dataset_df(n_pixels: int = 3, n_s2_per_pixel: int = 20,
                     include_s1: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    for i in range(n_pixels):
        pid = f"px_{i:04d}"
        for obs in range(n_s2_per_pixel):
            doy = (obs * 18 % 365) + 1
            rows.append({
                "point_id": pid, "year": 2022, "doy": doy,
                "date": pd.Timestamp(f"2022-01-01") + pd.Timedelta(days=doy - 1),
                "source": "S2", "scl_purity": 1.0, "scl": 4,
                **{b: 0.2 + i * 0.05 for b in BANDS},
                "vh": np.nan, "vv": np.nan,
            })
        if include_s1:
            for obs in range(8):
                doy = (obs * 45 % 365) + 1
                rows.append({
                    "point_id": pid, "year": 2022, "doy": doy,
                    "date": pd.Timestamp(f"2022-01-01") + pd.Timedelta(days=doy - 1),
                    "source": "S1", "scl_purity": np.nan, "scl": np.nan,
                    **{b: np.nan for b in BANDS},
                    "vh": 0.01 * (i + 1), "vv": 0.02 * (i + 1),
                })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    labels = pd.Series({f"px_{i:04d}": float(i % 2) for i in range(n_pixels)})
    return df, labels


def test_tamdataset_use_s1_n_bands():
    df, labels = _make_dataset_df(include_s1=True)
    ds = TAMDataset(df, labels, use_s1=True)
    sample = ds[0]
    assert sample.bands.shape == (128, N_BANDS_S1), (
        f"Expected (128, {N_BANDS_S1}), got {sample.bands.shape}"
    )


# ---------------------------------------------------------------------------
# S1D-2: TAMDataset with use_s1=False has N_BANDS features
# ---------------------------------------------------------------------------

def test_tamdataset_no_s1_n_bands():
    df, labels = _make_dataset_df(include_s1=True)
    ds = TAMDataset(df, labels, use_s1=False)
    sample = ds[0]
    assert sample.bands.shape == (128, N_BANDS), (
        f"Expected (128, {N_BANDS}), got {sample.bands.shape}"
    )


# ---------------------------------------------------------------------------
# S1D-3: use_s1=True: S1 features non-NaN for windows with S1 data
# ---------------------------------------------------------------------------

def test_tamdataset_s1_features_populated():
    df, labels = _make_dataset_df(include_s1=True)
    ds = TAMDataset(df, labels, use_s1=True)
    # At least one window must have a finite (non-NaN, non-zero) S1 feature value
    for i in range(len(ds)):
        sample = ds[i]
        bands = sample.bands  # (128, 17)
        n_obs = int((sample.doy != 0).sum())
        s1_cols = bands[:n_obs, N_BANDS:]  # last 4 columns for real observations
        if torch.isfinite(s1_cols).any() and s1_cols[torch.isfinite(s1_cols)].abs().max() > 0:
            return
    pytest.fail("No finite non-zero S1 features found in any window")


# ---------------------------------------------------------------------------
# S1D-4: band_mean/band_std shape matches N_BANDS_S1
# ---------------------------------------------------------------------------

def test_tamdataset_band_stats_shape_with_s1():
    df, labels = _make_dataset_df(include_s1=True)
    ds = TAMDataset(df, labels, use_s1=True)
    mean, std = ds.band_stats
    assert mean.shape == (N_BANDS_S1,)
    assert std.shape  == (N_BANDS_S1,)
