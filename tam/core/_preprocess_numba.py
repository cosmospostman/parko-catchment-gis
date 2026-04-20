"""tam/core/_preprocess_numba.py — numba kernels for fast inference preprocessing."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def extract_features(
    b02: np.ndarray,  # (N,) float32
    b03: np.ndarray,
    b04: np.ndarray,
    b05: np.ndarray,
    b06: np.ndarray,
    b07: np.ndarray,
    b08: np.ndarray,
    b8a: np.ndarray,
    b11: np.ndarray,
    b12: np.ndarray,
    out: np.ndarray,  # (N, 13) float32 C-contiguous, pre-allocated
) -> None:
    """Fill out with [B02..B12, NDVI, NDWI, EVI] in parallel, no intermediate arrays."""
    N = len(b02)
    for i in prange(N):
        out[i, 0] = b02[i]; out[i, 1] = b03[i]; out[i, 2] = b04[i]
        out[i, 3] = b05[i]; out[i, 4] = b06[i]; out[i, 5] = b07[i]
        out[i, 6] = b08[i]; out[i, 7] = b8a[i]; out[i, 8] = b11[i]; out[i, 9] = b12[i]
        denom = b08[i] + b04[i]
        out[i, 10] = (b08[i] - b04[i]) / denom if denom != 0.0 else 0.0
        denom = b03[i] + b08[i]
        out[i, 11] = (b03[i] - b08[i]) / denom if denom != 0.0 else 0.0
        denom = b08[i] + 6.0 * b04[i] - 7.5 * b02[i] + 1.0
        out[i, 12] = 2.5 * (b08[i] - b04[i]) / denom if denom != 0.0 else 0.0


@njit(parallel=True, cache=True)
def fill_windows(
    feat: np.ndarray,       # (N, F) float32 C-contiguous
    doy_arr: np.ndarray,    # (N,) int32
    valid_starts: np.ndarray,  # (W,) int64
    capped: np.ndarray,     # (W,) int32  — min(length, MAX_SEQ_LEN)
    band_mean: np.ndarray,  # (F,) float32
    band_std: np.ndarray,   # (F,) float32
    bands_out: np.ndarray,  # (W, MAX_SEQ_LEN, F) float32, zeroed
    doy_out: np.ndarray,    # (W, MAX_SEQ_LEN) int64, zeroed
    mask_out: np.ndarray,   # (W, MAX_SEQ_LEN) bool, ones
) -> None:
    """Normalise feat and scatter into padded (W, SEQ, F) tensors in parallel."""
    W = len(valid_starts)
    F = feat.shape[1]
    for k in prange(W):
        s = valid_starts[k]
        c = capped[k]
        for t in range(c):
            for f in range(F):
                bands_out[k, t, f] = (feat[s + t, f] - band_mean[f]) / band_std[f]
            doy_out[k, t] = doy_arr[s + t]
            mask_out[k, t] = False


def warmup() -> None:
    """JIT-compile both kernels with tiny dummy data (call once at startup)."""
    n = 10
    dummy_band = np.zeros(n, dtype=np.float32)
    out = np.zeros((n, 13), dtype=np.float32)
    extract_features(dummy_band, dummy_band, dummy_band, dummy_band, dummy_band,
                     dummy_band, dummy_band, dummy_band, dummy_band, dummy_band, out)

    feat = np.zeros((n, 13), dtype=np.float32)
    doy  = np.zeros(n, dtype=np.int32)
    starts = np.array([0], dtype=np.int64)
    caps   = np.array([5], dtype=np.int32)
    mean   = np.zeros(13, dtype=np.float32)
    std    = np.ones(13, dtype=np.float32)
    bands_out = np.zeros((1, 128, 13), dtype=np.float32)
    doy_out   = np.zeros((1, 128), dtype=np.int64)
    mask_out  = np.ones((1, 128), dtype=np.bool_)
    fill_windows(feat, doy, starts, caps, mean, std, bands_out, doy_out, mask_out)
