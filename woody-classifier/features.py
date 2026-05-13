"""woody-classifier/features.py — compute_woody_features()

Computes ~20 per-pixel annual summary features for the stage-1 woody mask.

S2 features (16):
    B11_p5, B11_p95, B11_std       — SWIR-1 floor, ceiling, stability
    B12_p5, B12_std                 — SWIR-2 axis and stability
    B08_p95, B08_std                — NIR peak and stability
    B8A_p95                         — NIR narrow canopy density
    B05_p95                         — red-edge-1 chlorophyll saturation
    NDVI_p10, NDVI_p90, NDVI_std    — phenological range and variability
    NDWI_p5                         — persistent water detection
    ndvi_amplitude                  — NDVI_p90 − NDVI_p10 (grass vs woody)
    swir_nir_ratio_p5               — B11_p5 / (B08_p95 + ε)  bark/cellulose ratio
    nir_cv                          — B08_std / (mean_B08 + ε)  relative NIR variability

S1 features (4) — adapted from tam/core/global_features.py:
    s1_mean_vh_dry   — mean VH dB May–Oct
    s1_vh_contrast   — mean VH wet − mean VH dry
    s1_vh_std        — temporal std of VH
    s1_mean_rvi      — mean Radar Vegetation Index

All features are summarised over the full annual observation stack; no temporal
sequence is required (avoids the obs-count mismatch that breaks TAM at inference).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Dry season: May 1 – Oct 31
_DRY_DOY_MIN = 121
_DRY_DOY_MAX = 304
_SCL_PURITY_MIN = 0.5
_EPS = 1e-6

WOODY_FEATURE_NAMES: list[str] = [
    "B11_p5", "B11_p95", "B11_std",
    "B12_p5", "B12_std",
    "B08_p95", "B08_std",
    "B8A_p95",
    "B05_p95",
    "NDVI_p10", "NDVI_p90", "NDVI_std",
    "NDWI_p5",
    "ndvi_amplitude",
    "swir_nir_ratio_p5",
    "nir_cv",
    "s1_mean_vh_dry", "s1_vh_contrast", "s1_vh_std", "s1_mean_rvi",
]


def compute_woody_features(pixel_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-pixel woody-mask features from the full observation stack.

    Parameters
    ----------
    pixel_df:
        Raw observations. Must have columns: point_id, date (or doy).
        S2 rows should have source="S2" (or no source column).
        S1 rows should have source="S1" with vh/vv as linear power.
        SCL filter applied automatically when scl_purity is present.

    Returns
    -------
    DataFrame indexed by point_id with columns WOODY_FEATURE_NAMES.
    Missing features (e.g. S1 absent) are NaN.
    """
    df = pixel_df.copy()
    if "doy" not in df.columns:
        df["doy"] = pd.to_datetime(df["date"]).dt.day_of_year

    # Split S2 / S1
    if "source" in df.columns:
        s2 = df[df["source"] == "S2"].copy()
        s1 = df[df["source"] == "S1"].copy()
    else:
        s2 = df.copy()
        s1 = pd.DataFrame()

    # SCL filter on S2
    if "scl_purity" in s2.columns:
        s2 = s2[s2["scl_purity"] >= _SCL_PURITY_MIN]

    # Compute NDVI / NDWI on S2 rows
    if not s2.empty and {"B08", "B04"}.issubset(s2.columns):
        b08 = s2["B08"].values.astype(np.float32)
        b04 = s2["B04"].values.astype(np.float32)
        b03 = s2["B03"].values.astype(np.float32) if "B03" in s2.columns else None
        ndvi_denom = b08 + b04
        s2 = s2.copy()
        s2["_ndvi"] = np.where(ndvi_denom != 0, (b08 - b04) / ndvi_denom, 0.0)
        if b03 is not None:
            ndwi_denom = b03 + b08
            s2["_ndwi"] = np.where(ndwi_denom != 0, (b03 - b08) / ndwi_denom, 0.0)
        else:
            s2["_ndwi"] = np.nan

    result = pd.DataFrame(index=pd.Index([], name="point_id"))

    if not s2.empty:
        s2_feats = _compute_s2_features(s2)
        result = s2_feats
    else:
        result = pd.DataFrame(
            np.nan,
            index=pd.Index(pixel_df["point_id"].unique(), name="point_id"),
            columns=[c for c in WOODY_FEATURE_NAMES if not c.startswith("s1_")],
        )

    if not s1.empty and {"vh", "vv"}.issubset(s1.columns):
        s1_feats = _compute_s1_features(s1)
        for col in ["s1_mean_vh_dry", "s1_vh_contrast", "s1_vh_std", "s1_mean_rvi"]:
            result[col] = s1_feats.get(col, pd.Series(dtype=float))
    else:
        for col in ["s1_mean_vh_dry", "s1_vh_contrast", "s1_vh_std", "s1_mean_rvi"]:
            result[col] = np.nan

    # Guarantee column order
    for col in WOODY_FEATURE_NAMES:
        if col not in result.columns:
            result[col] = np.nan

    return result[WOODY_FEATURE_NAMES]


# ---------------------------------------------------------------------------
# S2 features
# ---------------------------------------------------------------------------

def _compute_s2_features(s2: pd.DataFrame) -> pd.DataFrame:
    """Per-pixel S2 annual summaries. Expects _ndvi and _ndwi columns pre-computed."""
    pids = s2["point_id"].to_numpy()
    unique_pids, inverse = np.unique(pids, return_inverse=True)

    def _col(name: str) -> np.ndarray | None:
        return s2[name].to_numpy(dtype=np.float32) if name in s2.columns else None

    b11 = _col("B11")
    b12 = _col("B12")
    b08 = _col("B08")
    b8a = _col("B8A")
    b05 = _col("B05")
    ndvi = _col("_ndvi")
    ndwi = _col("_ndwi")

    n = len(unique_pids)
    rows = {c: np.full(n, np.nan, dtype=np.float32) for c in [
        "B11_p5", "B11_p95", "B11_std",
        "B12_p5", "B12_std",
        "B08_p95", "B08_std",
        "B8A_p95", "B05_p95",
        "NDVI_p10", "NDVI_p90", "NDVI_std",
        "NDWI_p5",
        "ndvi_amplitude", "swir_nir_ratio_p5", "nir_cv",
    ]}

    for i in range(n):
        mask = inverse == i
        if b11 is not None:
            v = b11[mask]
            v = v[~np.isnan(v)]
            if len(v):
                rows["B11_p5"][i]  = np.percentile(v, 5)
                rows["B11_p95"][i] = np.percentile(v, 95)
                rows["B11_std"][i] = v.std(ddof=1) if len(v) > 1 else 0.0

        if b12 is not None:
            v = b12[mask]
            v = v[~np.isnan(v)]
            if len(v):
                rows["B12_p5"][i]  = np.percentile(v, 5)
                rows["B12_std"][i] = v.std(ddof=1) if len(v) > 1 else 0.0

        b08_vals = None
        if b08 is not None:
            v = b08[mask]
            v = v[~np.isnan(v)]
            if len(v):
                rows["B08_p95"][i] = np.percentile(v, 95)
                rows["B08_std"][i] = v.std(ddof=1) if len(v) > 1 else 0.0
                b08_mean = v.mean()
                rows["nir_cv"][i] = rows["B08_std"][i] / (b08_mean + _EPS)
                b08_vals = v

        if b8a is not None:
            v = b8a[mask]
            v = v[~np.isnan(v)]
            if len(v):
                rows["B8A_p95"][i] = np.percentile(v, 95)

        if b05 is not None:
            v = b05[mask]
            v = v[~np.isnan(v)]
            if len(v):
                rows["B05_p95"][i] = np.percentile(v, 95)

        if ndvi is not None:
            v = ndvi[mask]
            v = v[~np.isnan(v)]
            if len(v):
                p10 = np.percentile(v, 10)
                p90 = np.percentile(v, 90)
                rows["NDVI_p10"][i]       = p10
                rows["NDVI_p90"][i]       = p90
                rows["NDVI_std"][i]       = v.std(ddof=1) if len(v) > 1 else 0.0
                rows["ndvi_amplitude"][i] = p90 - p10

        if ndwi is not None:
            v = ndwi[mask]
            v = v[~np.isnan(v)]
            if len(v):
                rows["NDWI_p5"][i] = np.percentile(v, 5)

        # swir_nir_ratio_p5 = B11_p5 / (B08_p95 + ε)
        b11_p5  = rows["B11_p5"][i]
        b08_p95 = rows["B08_p95"][i]
        if not (np.isnan(b11_p5) or np.isnan(b08_p95)):
            rows["swir_nir_ratio_p5"][i] = b11_p5 / (b08_p95 + _EPS)

    return pd.DataFrame(rows, index=pd.Index(unique_pids, name="point_id"))


# ---------------------------------------------------------------------------
# S1 features — adapted from tam/core/global_features.py:_compute_s1_globals()
# ---------------------------------------------------------------------------

def _compute_s1_features(s1: pd.DataFrame) -> dict[str, pd.Series]:
    """Compute per-pixel S1 features. vh/vv expected as linear power."""
    df = s1.copy()
    if "doy" not in df.columns:
        df["doy"] = pd.to_datetime(df["date"]).dt.day_of_year

    vh_lin = df["vh"].values.astype(np.float32)
    vv_lin = df["vv"].values.astype(np.float32) if "vv" in df.columns else np.full(len(df), np.nan, dtype=np.float32)

    df["_vh_db"] = np.where(vh_lin > 0, 10 * np.log10(vh_lin), np.nan).astype(np.float32)
    df["_vv_db"] = np.where(vv_lin > 0, 10 * np.log10(vv_lin), np.nan).astype(np.float32)

    both = (vh_lin > 0) & (vv_lin > 0)
    denom = vh_lin + vv_lin
    rvi = np.where(both & (denom > 0), 4 * vh_lin / denom, np.nan).astype(np.float32)
    df["_rvi"] = rvi

    dry_mask = (df["doy"] >= _DRY_DOY_MIN) & (df["doy"] <= _DRY_DOY_MAX)
    wet_mask = ~dry_mask

    dry = df[dry_mask & df["_vh_db"].notna()]
    wet = df[wet_mask & df["_vh_db"].notna()]

    mean_vh_dry = dry.groupby("point_id")["_vh_db"].mean().rename("s1_mean_vh_dry")
    mean_vh_wet = wet.groupby("point_id")["_vh_db"].mean()
    vh_contrast  = (mean_vh_wet - mean_vh_dry).rename("s1_vh_contrast")
    vh_std       = df[df["_vh_db"].notna()].groupby("point_id")["_vh_db"].std().rename("s1_vh_std")
    mean_rvi     = df[~np.isnan(df["_rvi"].values)].groupby("point_id")["_rvi"].mean().rename("s1_mean_rvi")

    return {
        "s1_mean_vh_dry": mean_vh_dry,
        "s1_vh_contrast": vh_contrast,
        "s1_vh_std":      vh_std,
        "s1_mean_rvi":    mean_rvi,
    }
