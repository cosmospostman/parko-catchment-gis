"""Shared constants for the spectral time series pipeline.

All band names, quality profiles, SCL masks, and scientific thresholds
are defined here and imported by every other module. Edit once, applies everywhere.
"""

# ---------------------------------------------------------------------------
# Sentinel-2 band names used throughout the pipeline
# ---------------------------------------------------------------------------

BANDS: list[str] = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
SPECTRAL_INDEX_COLS: list[str] = ["NDVI", "NDWI", "EVI", "MAVI", "NDRE", "CI_RE"]

SCL_BAND = "SCL"
AOT_BAND = "AOT"
VZA_BAND = "VZA"  # view zenith angle
SZA_BAND = "SZA"  # sun zenith angle

# ---------------------------------------------------------------------------
# SCL (Scene Classification Layer) clear values
# SCL class codes that indicate usable (non-cloud, non-shadow) pixels:
#   4 = vegetation, 5 = bare soil, 6 = water, 11 = snow/ice (retained — not cloud)
# 7 (unclassified) is excluded: Sen2Cor uses it as a catch-all that includes
# cloud shadow edges and partially shadowed pixels, which can produce near-zero
# SWIR observations that corrupt time series (confirmed at Quaids, May 2026).
# ---------------------------------------------------------------------------

SCL_CLEAR_VALUES: set[int] = {4, 5, 6, 11}

# ---------------------------------------------------------------------------
# Quality profile masks — passed to ObservationQuality.score(mask=...)
#
# Q_ATMOSPHERIC  : cloud + haze only; use for any spectral index computation
# Q_GEOMETRIC    : cloud + haze + geometry; use for anomaly where greenness IS signal
# Q_FULL         : all five components; use for flowering peak detection
# Q_CLOUD_ONLY   : SCL purity only; use for structural features (HAND, texture)
# ---------------------------------------------------------------------------

Q_ATMOSPHERIC: set[str] = {"scl_purity", "aot"}
Q_GEOMETRIC: set[str] = {"scl_purity", "aot", "view_zenith", "sun_zenith"}
Q_FULL: None = None  # all five components
Q_CLOUD_ONLY: set[str] = {"scl_purity"}

# ---------------------------------------------------------------------------
# Spatial validation gate
# Minimum AUC on held-out spatial region required to certify a model for inference.
# train.py refuses to write model_{run_id}.pkl if validation falls below this.
# ---------------------------------------------------------------------------

SPATIAL_VALIDATION_THRESHOLD: float = 0.85

# ---------------------------------------------------------------------------
# Flowering phenology window and detection threshold
#
# FLOWERING_WINDOW : (doy_start, doy_end) — expected day-of-year range for
#                    Parkinsonia aculeata peak greenness in north Queensland.
#                    200 = ~19 Jul, 340 = ~6 Dec.
# FLOWERING_THRESHOLD : minimum flowering index value to count as a detected peak.
# ---------------------------------------------------------------------------

FLOWERING_WINDOW: tuple[int, int] = (200, 340)
FLOWERING_THRESHOLD: float = 0.15


# ---------------------------------------------------------------------------
# Spectral index computation
# ---------------------------------------------------------------------------

UINT16_BAND_SCALE: float = 10_000.0
UINT8_QUALITY_SCALE: float = 100.0


def ensure_float32_bands(df: "pl.DataFrame") -> "pl.DataFrame":
    """Cast uint16 band columns back to float32 reflectance after parquet load.

    New parquets store bands as uint16 ×10000 to halve storage. This function
    is idempotent — float32 columns are left unchanged, so it is safe to call
    on both old (float32) and new (uint16) parquets.
    """
    import polars as pl

    exprs = [
        (pl.col(b).cast(pl.Float32) / UINT16_BAND_SCALE).alias(b)
        for b in BANDS
        if b in df.columns and df[b].dtype == pl.UInt16
    ]
    return df.with_columns(exprs) if exprs else df


def add_spectral_indices(df):
    """Return df with NDVI, NDWI, EVI, MAVI, NDRE, CI_RE columns appended.

    Accepts both polars.DataFrame and pandas.DataFrame. Automatically
    converts uint16 band columns to float32 before computing indices.
    """
    import polars as pl

    if isinstance(df, pl.DataFrame):
        df = ensure_float32_bands(df)

        # Build a single quality guard expression (source==S2 and scl_purity>=0.5).
        # All six index formulas share it so Polars fuses them into one scan instead
        # of recomputing the mask six times via the Signal.quality_mask() path.
        good = pl.lit(True)
        if "source" in df.columns:
            good = good & (pl.col("source") == "S2")
        if "scl_purity" in df.columns:
            good = good & (pl.col("scl_purity") >= 0.5)

        def _safe(expr: pl.Expr, denom: pl.Expr) -> pl.Expr:
            return pl.when(good & (denom != 0)).then(expr).otherwise(pl.lit(None).cast(pl.Float32))

        b02, b03, b04 = pl.col("B02"), pl.col("B03"), pl.col("B04")
        b05, b08      = pl.col("B05"), pl.col("B08")
        b8a, b11      = pl.col("B8A"), pl.col("B11")

        ndvi  = _safe((b08 - b04) / (b08 + b04),          b08 + b04)
        ndwi  = _safe((b03 - b08) / (b03 + b08),          b03 + b08)
        evi_d = b08 + 6 * b04 - 7.5 * b02 + 1
        evi   = _safe(2.5 * (b08 - b04) / evi_d,          evi_d)
        mavi  = _safe((b8a - b11) / (b8a + b11),          b8a + b11)
        ndre  = _safe((b8a - b05) / (b8a + b05),          b8a + b05)
        cire  = _safe(b8a / b05 - 1,                      b05)

        return df.with_columns([
            ndvi.alias("NDVI"),
            ndwi.alias("NDWI"),
            evi.alias("EVI"),
            mavi.alias("MAVI"),
            ndre.alias("NDRE"),
            cire.alias("CI_RE"),
        ])

    # pandas path — for analysis/ scripts that are out of migration scope.
    from signals.ndvi import NDVISignal, NDWISignal, EVISignal
    from signals.mavi import MAVISignal
    from signals.ndre import NDRESignal, CIRESignal

    df = df.copy()
    _pl_df = pl.from_pandas(df)
    df["NDVI"]  = NDVISignal().compute(_pl_df).to_numpy()
    df["NDWI"]  = NDWISignal().compute(_pl_df).to_numpy()
    df["EVI"]   = EVISignal().compute(_pl_df).to_numpy()
    df["MAVI"]  = MAVISignal().compute(_pl_df).to_numpy()
    df["NDRE"]  = NDRESignal().compute(_pl_df).to_numpy()
    df["CI_RE"] = CIRESignal().compute(_pl_df).to_numpy()
    return df
