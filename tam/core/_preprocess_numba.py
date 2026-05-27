"""tam/core/_preprocess_numba.py — numba kernels for fast inference preprocessing."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def count_s2_s1_per_window(
    is_s1: np.ndarray,      # (N,) bool
    boundaries: np.ndarray, # (W,) int64 — start of each window
    ends: np.ndarray,       # (W,) int64 — one-past-end of each window
    n_s2_out: np.ndarray,   # (W,) int32 — output: S2 count per window
    n_s1_out: np.ndarray,   # (W,) int32 — output: S1 count per window
) -> None:
    """Count S2 and S1 observations per window in parallel."""
    W = len(boundaries)
    for k in prange(W):
        s2 = np.int32(0)
        s1 = np.int32(0)
        for i in range(boundaries[k], ends[k]):
            if is_s1[i]:
                s1 += np.int32(1)
            else:
                s2 += np.int32(1)
        n_s2_out[k] = s2
        n_s1_out[k] = s1


@njit(parallel=True, cache=True)
def compute_window_stats(
    feat: np.ndarray,          # (N, F) float32 C-contiguous
    is_s1: np.ndarray,         # (N,) bool — True = S1 row; used to mask columns
    boundaries: np.ndarray,    # (W,) int64 — start of each window
    ends: np.ndarray,          # (W,) int64 — one-past-end of each window
    n_s2: np.int64,            # number of S2 columns (first n_s2 cols of feat)
    s2_mean_out: np.ndarray,   # (W, n_s2) float32 — output
    s2_std_out: np.ndarray,    # (W, n_s2) float32 — output
    s1_mean_out: np.ndarray,   # (W, n_s1) float32 — output
    s1_std_out: np.ndarray,    # (W, n_s1) float32 — output
) -> None:
    """Compute per-window mean and std for S2 and S1 feature columns in parallel.

    S2 columns (0..n_s2-1) are averaged over non-S1 rows; S1 columns (n_s2..F-1)
    over S1 rows. NaN values are excluded. If a window has no valid rows for a
    modality, mean=0 and std=1 are written (identity transform).

    Minimum std is clipped to 1e-6 to avoid division-by-zero in callers.
    """
    W = len(boundaries)
    F = feat.shape[1]
    n_s1 = F - n_s2
    for k in prange(W):
        s = boundaries[k]
        e = ends[k]

        # --- S2 stats (non-S1 rows, first n_s2 columns) ---
        for f in range(n_s2):
            acc = np.float64(0.0)
            cnt = np.int64(0)
            for i in range(s, e):
                if not is_s1[i]:
                    v = np.float64(feat[i, f])
                    if v == v:  # not NaN
                        acc += v
                        cnt += np.int64(1)
            if cnt > 0:
                mean_f = acc / cnt
                var = np.float64(0.0)
                for i in range(s, e):
                    if not is_s1[i]:
                        v = np.float64(feat[i, f])
                        if v == v:
                            d = v - mean_f
                            var += d * d
                std_f = (var / cnt) ** 0.5
                s2_mean_out[k, f] = np.float32(mean_f)
                s2_std_out[k, f]  = np.float32(max(std_f, 1e-6))
            else:
                s2_mean_out[k, f] = np.float32(0.0)
                s2_std_out[k, f]  = np.float32(1.0)

        # --- S1 stats (S1 rows, columns n_s2..F-1) ---
        for f in range(n_s1):
            acc = np.float64(0.0)
            cnt = np.int64(0)
            for i in range(s, e):
                if is_s1[i]:
                    v = np.float64(feat[i, n_s2 + f])
                    if v == v:
                        acc += v
                        cnt += np.int64(1)
            if cnt > 0:
                mean_f = acc / cnt
                var = np.float64(0.0)
                for i in range(s, e):
                    if is_s1[i]:
                        v = np.float64(feat[i, n_s2 + f])
                        if v == v:
                            d = v - mean_f
                            var += d * d
                std_f = (var / cnt) ** 0.5
                s1_mean_out[k, f] = np.float32(mean_f)
                s1_std_out[k, f]  = np.float32(max(std_f, 1e-6))
            else:
                s1_mean_out[k, f] = np.float32(0.0)
                s1_std_out[k, f]  = np.float32(1.0)


@njit(parallel=True, cache=True)
def compute_window_stats_s2only(
    feat: np.ndarray,          # (N, F) float32 C-contiguous
    boundaries: np.ndarray,    # (W,) int64
    ends: np.ndarray,          # (W,) int64
    mean_out: np.ndarray,      # (W, F) float32 — output
    std_out: np.ndarray,       # (W, F) float32 — output
) -> None:
    """Compute per-window mean and std for all F columns (S2-only path).

    NaN values excluded. If all values in a column are NaN → mean=0, std=1.
    Std clipped to 1e-6 minimum.
    """
    W = len(boundaries)
    F = feat.shape[1]
    for k in prange(W):
        s = boundaries[k]
        e = ends[k]
        for f in range(F):
            acc = np.float64(0.0)
            cnt = np.int64(0)
            for i in range(s, e):
                v = np.float64(feat[i, f])
                if v == v:
                    acc += v
                    cnt += np.int64(1)
            if cnt > 0:
                mean_f = acc / cnt
                var = np.float64(0.0)
                for i in range(s, e):
                    v = np.float64(feat[i, f])
                    if v == v:
                        d = v - mean_f
                        var += d * d
                std_f = (var / cnt) ** 0.5
                mean_out[k, f] = np.float32(mean_f)
                std_out[k, f]  = np.float32(max(std_f, 1e-6))
            else:
                mean_out[k, f] = np.float32(0.0)
                std_out[k, f]  = np.float32(1.0)


@njit(parallel=True, cache=True)
def fill_windows_mixed(
    feat: np.ndarray,        # (N, F) float32 C-contiguous
    is_s1: np.ndarray,       # (N,) bool
    doy_arr: np.ndarray,     # (N,) int32
    valid_starts: np.ndarray, # (W,) int64
    lengths: np.ndarray,     # (W,) int32  — actual window lengths (already ≤ max_seq_len)
    s2_mean: np.ndarray,     # (W, n_s2) float32 — per-window S2 zscore mean; zeros if no zscore
    s2_std: np.ndarray,      # (W, n_s2) float32 — per-window S2 zscore std; ones if no zscore
    s1_mean: np.ndarray,     # (W, n_s1) float32
    s1_std: np.ndarray,      # (W, n_s1) float32
    n_s2: np.int64,          # number of S2 feature columns
    bands_out: np.ndarray,   # (W, max_seq_len, F) float32, zeroed
    doy_out: np.ndarray,     # (W, max_seq_len) int64, zeroed
    mask_out: np.ndarray,    # (W, max_seq_len) bool, ones
    is_s1_out: np.ndarray,   # (W, max_seq_len) bool, zeroed
) -> None:
    """Fill padded window tensors for mixed S2+S1 mode in parallel.

    Applies per-window pixel z-score normalisation: S2 features use s2_mean/s2_std,
    S1 features use s1_mean/s1_std. Pass zero-mean and unit-std arrays to skip
    normalisation for either modality.

    Only handles windows where length ≤ max_seq_len (no subsampling needed).
    Windows requiring S1 subsampling are handled separately in Python.

    NaN values are replaced with 0.0.
    """
    W = len(valid_starts)
    F = feat.shape[1]
    n_s1 = F - n_s2
    for k in prange(W):
        s = valid_starts[k]
        length = lengths[k]
        for t in range(length):
            row = s + t
            src = is_s1[row]
            doy_out[k, t] = doy_arr[row]
            mask_out[k, t] = False
            is_s1_out[k, t] = src
            if src:
                # S1 feature columns (positions n_s2 .. F-1)
                for f in range(n_s1):
                    v = feat[row, n_s2 + f]
                    v = (v - s1_mean[k, f]) / s1_std[k, f]
                    bands_out[k, t, n_s2 + f] = 0.0 if (v != v) else v  # nan→0
            else:
                # S2 feature columns (positions 0 .. n_s2-1)
                for f in range(n_s2):
                    v = feat[row, f]
                    v = (v - s2_mean[k, f]) / s2_std[k, f]
                    bands_out[k, t, f] = 0.0 if (v != v) else v  # nan→0


@njit(parallel=True, cache=True)
def fill_windows_mixed_subsample(
    feat: np.ndarray,          # (N, F) float32 C-contiguous
    is_s1: np.ndarray,         # (N,) bool
    doy_arr: np.ndarray,       # (N,) int32
    valid_starts: np.ndarray,  # (W,) int64
    valid_lengths: np.ndarray, # (W,) int32 — raw window lengths (> max_seq_len)
    s2_mean: np.ndarray,       # (W, n_s2) float32
    s2_std: np.ndarray,        # (W, n_s2) float32
    s1_mean: np.ndarray,       # (W, n_s1) float32
    s1_std: np.ndarray,        # (W, n_s1) float32
    n_s2: np.int64,
    max_seq_len: np.int64,
    bands_out: np.ndarray,     # (W, max_seq_len, F) float32, pre-zeroed
    doy_out: np.ndarray,       # (W, max_seq_len) int64, pre-zeroed
    mask_out: np.ndarray,      # (W, max_seq_len) bool, pre-set to True
    is_s1_out: np.ndarray,     # (W, max_seq_len) bool, pre-zeroed
) -> None:
    """Fill padded window tensors for overlong mixed windows in parallel.

    Priority: keep all S2 obs (up to max_seq_len), then fill remainder with S1
    obs selected by greedy farthest-point DOY subsampling.  All work is in
    numba prange so the Python loop in _preprocess is eliminated.

    NaN values are replaced with 0.0.
    """
    W = len(valid_starts)
    F = feat.shape[1]
    n_s1 = F - n_s2
    for k in prange(W):
        s = valid_starts[k]
        length = valid_lengths[k]

        # Pass 1: collect S2 and S1 local indices within this window
        n_s2_win = np.int64(0)
        n_s1_win = np.int64(0)
        for i in range(length):
            if is_s1[s + i]:
                n_s1_win += np.int64(1)
            else:
                n_s2_win += np.int64(1)

        # Allocate scratch arrays for S2 and S1 local indices (max = window length)
        s2_local = np.empty(length, dtype=np.int64)
        s1_local = np.empty(length, dtype=np.int64)
        j2 = np.int64(0)
        j1 = np.int64(0)
        for i in range(length):
            if is_s1[s + i]:
                s1_local[j1] = np.int64(i)
                j1 += np.int64(1)
            else:
                s2_local[j2] = np.int64(i)
                j2 += np.int64(1)

        # Take S2 obs first (capped at max_seq_len), then fill remainder with S1
        n_keep_s2 = min(n_s2_win, max_seq_len)
        s1_budget = max_seq_len - n_keep_s2

        # Greedy farthest-point DOY subsampling for S1 (inline to avoid Python call)
        n_keep_s1 = np.int64(0)
        kept_s1 = np.empty(length, dtype=np.int64)  # local indices into s1_local
        if n_s1_win > np.int64(0) and s1_budget > np.int64(0):
            if n_s1_win <= s1_budget:
                # Keep all S1 obs — no subsampling needed
                for i in range(n_s1_win):
                    kept_s1[i] = i
                n_keep_s1 = n_s1_win
            else:
                # Farthest-point DOY sampling: anchor at first and last S1 obs
                selected = np.empty(s1_budget, dtype=np.int64)
                selected[0] = np.int64(0)
                n_sel = np.int64(1)
                if s1_budget >= np.int64(2):
                    selected[1] = n_s1_win - np.int64(1)
                    n_sel = np.int64(2)
                while n_sel < s1_budget:
                    best_i = np.int64(0)
                    best_d = np.float32(-1.0)
                    for i in range(n_s1_win):
                        # min distance to any already-selected obs
                        di = doy_arr[s + s1_local[i]]
                        min_dist = np.float32(1e9)
                        for j in range(n_sel):
                            dj = doy_arr[s + s1_local[selected[j]]]
                            dist = np.float32(di - dj) if di >= dj else np.float32(dj - di)
                            if dist < min_dist:
                                min_dist = dist
                        # Skip already-selected
                        already = False
                        for j in range(n_sel):
                            if selected[j] == np.int64(i):
                                already = True
                                break
                        if not already and min_dist > best_d:
                            best_d = min_dist
                            best_i = np.int64(i)
                    selected[n_sel] = best_i
                    n_sel += np.int64(1)
                # Sort selected indices to preserve temporal order
                for i in range(1, n_sel):
                    key = selected[i]
                    j = i - 1
                    while j >= 0 and selected[j] > key:
                        selected[j + 1] = selected[j]
                        j -= 1
                    selected[j + 1] = key
                for i in range(n_sel):
                    kept_s1[i] = selected[i]
                n_keep_s1 = n_sel

        # Merge S2 (first n_keep_s2) + kept S1 in temporal (DOY) order
        # Collect all kept row offsets with their DOY for sorting
        n_out = n_keep_s2 + n_keep_s1
        out_offsets = np.empty(n_out, dtype=np.int64)
        out_doys    = np.empty(n_out, dtype=np.int32)
        for i in range(n_keep_s2):
            off = s2_local[i]
            out_offsets[i] = off
            out_doys[i]    = doy_arr[s + off]
        for i in range(n_keep_s1):
            off = s1_local[kept_s1[i]]
            out_offsets[n_keep_s2 + i] = off
            out_doys[n_keep_s2 + i]    = doy_arr[s + off]

        # Insertion sort by DOY
        for i in range(1, n_out):
            key_off = out_offsets[i]
            key_doy = out_doys[i]
            j = i - 1
            while j >= 0 and out_doys[j] > key_doy:
                out_offsets[j + 1] = out_offsets[j]
                out_doys[j + 1]    = out_doys[j]
                j -= 1
            out_offsets[j + 1] = key_off
            out_doys[j + 1]    = key_doy

        # Write output tensors
        for t in range(n_out):
            row = s + out_offsets[t]
            src = is_s1[row]
            doy_out[k, t]   = out_doys[t]
            mask_out[k, t]  = False
            is_s1_out[k, t] = src
            if src:
                for f in range(n_s1):
                    v = feat[row, n_s2 + f]
                    v = (v - s1_mean[k, f]) / s1_std[k, f]
                    bands_out[k, t, n_s2 + f] = np.float32(0.0) if (v != v) else v
            else:
                for f in range(n_s2):
                    v = feat[row, f]
                    v = (v - s2_mean[k, f]) / s2_std[k, f]
                    bands_out[k, t, f] = np.float32(0.0) if (v != v) else v


@njit(parallel=True, cache=True)
def build_gate_tensors(
    bands_np: np.ndarray,     # (B, T_full, F) float32 — already on CPU
    doy_np: np.ndarray,       # (B, T_full) int64
    mask_np: np.ndarray,      # (B, T_full) bool — True = padding
    is_s1_np: np.ndarray,     # (B, T_full) bool  (may be all-False for S2-only)
    T_gate: np.int64,
    gate_bands_out: np.ndarray,   # (B, T_gate, F) float32, pre-zeroed
    gate_doy_out: np.ndarray,     # (B, T_gate) int64, pre-zeroed
    gate_mask_out: np.ndarray,    # (B, T_gate) bool, pre-set to True
    gate_is_s1_out: np.ndarray,   # (B, T_gate) bool, pre-zeroed
) -> None:
    """Build gate tensors for all batch items in parallel (replaces Python loop).

    For each item, selects T_gate observations via greedy farthest-point DOY
    subsampling from the non-padding positions, then writes them to the output
    tensors in temporal order.
    """
    B = bands_np.shape[0]
    F = bands_np.shape[2]
    for bi in prange(B):
        # Count real (non-padding) observations
        n_real = np.int64(0)
        for t in range(bands_np.shape[1]):
            if not mask_np[bi, t]:
                n_real += np.int64(1)
        if n_real == np.int64(0):
            continue

        T_full = np.int64(bands_np.shape[1])

        if n_real <= T_gate:
            # Keep all real obs in order
            out_t = np.int64(0)
            for t in range(T_full):
                if not mask_np[bi, t]:
                    gate_doy_out[bi, out_t]   = doy_np[bi, t]
                    gate_mask_out[bi, out_t]  = False
                    gate_is_s1_out[bi, out_t] = is_s1_np[bi, t]
                    for f in range(F):
                        gate_bands_out[bi, out_t, f] = bands_np[bi, t, f]
                    out_t += np.int64(1)
        else:
            # Collect real indices
            real_idx = np.empty(n_real, dtype=np.int64)
            ri = np.int64(0)
            for t in range(T_full):
                if not mask_np[bi, t]:
                    real_idx[ri] = np.int64(t)
                    ri += np.int64(1)

            # Greedy farthest-point DOY subsampling
            selected = np.empty(T_gate, dtype=np.int64)
            selected[0] = np.int64(0)           # index into real_idx
            n_sel = np.int64(1)
            if T_gate >= np.int64(2):
                selected[1] = n_real - np.int64(1)
                n_sel = np.int64(2)
            while n_sel < T_gate:
                best_i = np.int64(0)
                best_d = np.float32(-1.0)
                for i in range(n_real):
                    di = doy_np[bi, real_idx[i]]
                    min_dist = np.float32(1e9)
                    for j in range(n_sel):
                        dj = doy_np[bi, real_idx[selected[j]]]
                        dist = np.float32(di - dj) if di >= dj else np.float32(dj - di)
                        if dist < min_dist:
                            min_dist = dist
                    already = False
                    for j in range(n_sel):
                        if selected[j] == np.int64(i):
                            already = True
                            break
                    if not already and min_dist > best_d:
                        best_d = min_dist
                        best_i = np.int64(i)
                selected[n_sel] = best_i
                n_sel += np.int64(1)

            # Sort selected by DOY order
            for i in range(1, n_sel):
                key = selected[i]
                j = i - 1
                while j >= 0 and doy_np[bi, real_idx[selected[j]]] > doy_np[bi, real_idx[key]]:
                    selected[j + 1] = selected[j]
                    j -= 1
                selected[j + 1] = key

            # Write gate tensors
            for i in range(n_sel):
                t = real_idx[selected[i]]
                gate_doy_out[bi, i]   = doy_np[bi, t]
                gate_mask_out[bi, i]  = False
                gate_is_s1_out[bi, i] = is_s1_np[bi, t]
                for f in range(F):
                    gate_bands_out[bi, i, f] = bands_np[bi, t, f]


@njit(parallel=True, cache=True)
def fill_windows_zscore(
    feat: np.ndarray,         # (N, F) float32 C-contiguous
    doy_arr: np.ndarray,      # (N,) int32
    valid_starts: np.ndarray, # (W,) int64
    capped: np.ndarray,       # (W,) int32  — min(length, max_seq_len)
    per_win_mean: np.ndarray, # (W, F) float32 — per-window mean
    per_win_std: np.ndarray,  # (W, F) float32 — per-window std
    bands_out: np.ndarray,    # (W, max_seq_len, F) float32, zeroed
    doy_out: np.ndarray,      # (W, max_seq_len) int64, zeroed
    mask_out: np.ndarray,     # (W, max_seq_len) bool, ones
) -> None:
    """Fill padded window tensors with per-window pixel z-score normalisation.

    Equivalent to fill_windows but uses per-window mean/std instead of global
    scalars, covering the S2-only pixel_zscore path. NaN → 0.0.
    """
    W = len(valid_starts)
    F = feat.shape[1]
    for k in prange(W):
        s = valid_starts[k]
        c = capped[k]
        for t in range(c):
            row = s + t
            doy_out[k, t] = doy_arr[row]
            mask_out[k, t] = False
            for f in range(F):
                v = (feat[row, f] - per_win_mean[k, f]) / per_win_std[k, f]
                bands_out[k, t, f] = 0.0 if (v != v) else v  # nan→0


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
    out: np.ndarray,  # (N, 16) float32 C-contiguous, pre-allocated
) -> None:
    """Fill out with [B02..B12, NDVI, NDWI, EVI, MAVI, NDRE, CI_RE] in parallel.

    Column order must match ALL_FEATURE_COLS in tam/core/dataset.py:
      0-9:   B02 B03 B04 B05 B06 B07 B08 B8A B11 B12
      10-12: NDVI NDWI EVI
      13:    MAVI  = (B08 - B04) / (B08 + B04 + B11)
      14:    NDRE  = (B8A - B05) / (B8A + B05)
      15:    CI_RE = (B07 / B05) - 1
    """
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
        denom = b08[i] + b04[i] + b11[i]
        out[i, 13] = (b08[i] - b04[i]) / denom if denom != 0.0 else 0.0
        denom = b8a[i] + b05[i]
        out[i, 14] = (b8a[i] - b05[i]) / denom if denom != 0.0 else 0.0
        out[i, 15] = (b07[i] / b05[i]) - 1.0 if b05[i] != 0.0 else 0.0


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


@njit(parallel=True, cache=True)
def compute_band_summaries(
    feat: np.ndarray,          # (N, F) float32 C-contiguous — S2 rows only
    boundaries: np.ndarray,    # (W,) int64 — start of each window
    ends: np.ndarray,          # (W,) int64 — one-past-end of each window
    out: np.ndarray,           # (W, F*3) float32 — output: [p5_0, p95_0, std_0, p5_1, ...]
) -> None:
    """Compute per-window p5/p95/std for each feature column in parallel.

    Uses an insertion sort on a scratch buffer to find percentiles — correct and
    fast for the small window sizes (50–200 obs) typical in this dataset.
    NaN values are excluded.  Windows with fewer than 2 valid obs get 0/0/1.
    """
    W = len(boundaries)
    F = feat.shape[1]
    for k in prange(W):
        s = boundaries[k]
        e = ends[k]
        for f in range(F):
            # Collect valid (non-NaN) values for this column into a scratch array.
            # Allocate at max possible length; track actual count.
            max_n = e - s
            buf = np.empty(max_n, dtype=np.float64)
            cnt = np.int64(0)
            for i in range(s, e):
                v = np.float64(feat[i, f])
                if v == v:  # not NaN
                    buf[cnt] = v
                    cnt += np.int64(1)
            if cnt < 2:
                out[k, f * 3]     = np.float32(0.0)
                out[k, f * 3 + 1] = np.float32(0.0)
                out[k, f * 3 + 2] = np.float32(1.0)
                continue
            # Insertion sort the valid values.
            for i in range(1, cnt):
                key = buf[i]
                j = i - 1
                while j >= 0 and buf[j] > key:
                    buf[j + 1] = buf[j]
                    j -= 1
                buf[j + 1] = key
            # Percentile via linear interpolation (matches numpy default).
            def _lerp_percentile(p: np.float64) -> np.float64:
                idx = p * np.float64(cnt - 1)
                lo = np.int64(idx)
                hi = lo + np.int64(1)
                if hi >= cnt:
                    return buf[cnt - 1]
                frac = idx - np.float64(lo)
                return buf[lo] * (np.float64(1.0) - frac) + buf[hi] * frac
            p5  = _lerp_percentile(np.float64(0.05))
            p95 = _lerp_percentile(np.float64(0.95))
            # Std (population, matching np.std default).
            mean = np.float64(0.0)
            for i in range(cnt):
                mean += buf[i]
            mean /= np.float64(cnt)
            var = np.float64(0.0)
            for i in range(cnt):
                d = buf[i] - mean
                var += d * d
            var /= np.float64(cnt)
            out[k, f * 3]     = np.float32(p5)
            out[k, f * 3 + 1] = np.float32(p95)
            out[k, f * 3 + 2] = np.float32(var ** np.float64(0.5))


def warmup() -> None:
    """JIT-compile all kernels with tiny dummy data (call once at startup)."""
    n = 10
    W = 2
    T = 8
    n_s2 = 14
    n_s1 = 4
    F = n_s2 + n_s1

    dummy_band = np.zeros(n, dtype=np.float32)
    out = np.zeros((n, 16), dtype=np.float32)
    extract_features(dummy_band, dummy_band, dummy_band, dummy_band, dummy_band,
                     dummy_band, dummy_band, dummy_band, dummy_band, dummy_band, out)

    feat  = np.zeros((n, 16), dtype=np.float32)
    doy   = np.zeros(n, dtype=np.int32)
    starts = np.array([0, 5], dtype=np.int64)
    caps   = np.array([5, 5], dtype=np.int32)
    mean   = np.zeros(16, dtype=np.float32)
    std    = np.ones(16, dtype=np.float32)
    bands_out = np.zeros((W, T, 16), dtype=np.float32)
    doy_out   = np.zeros((W, T), dtype=np.int64)
    mask_out  = np.ones((W, T), dtype=np.bool_)
    fill_windows(feat, doy, starts, caps, mean, std, bands_out, doy_out, mask_out)

    # fill_windows_zscore
    per_mean = np.zeros((W, 16), dtype=np.float32)
    per_std  = np.ones((W, 16), dtype=np.float32)
    bands_out2 = np.zeros((W, T, 16), dtype=np.float32)
    doy_out2   = np.zeros((W, T), dtype=np.int64)
    mask_out2  = np.ones((W, T), dtype=np.bool_)
    fill_windows_zscore(feat, doy, starts, caps, per_mean, per_std,
                        bands_out2, doy_out2, mask_out2)

    # fill_windows_mixed
    feat_m  = np.zeros((n, F), dtype=np.float32)
    is_s1_w = np.zeros(n, dtype=np.bool_)
    is_s1_w[5:] = True
    s2_mean = np.zeros((W, n_s2), dtype=np.float32)
    s2_std  = np.ones((W, n_s2), dtype=np.float32)
    s1_mean = np.zeros((W, n_s1), dtype=np.float32)
    s1_std  = np.ones((W, n_s1), dtype=np.float32)
    bands_out3 = np.zeros((W, T, F), dtype=np.float32)
    doy_out3   = np.zeros((W, T), dtype=np.int64)
    mask_out3  = np.ones((W, T), dtype=np.bool_)
    is_s1_out3 = np.zeros((W, T), dtype=np.bool_)
    lengths = np.array([5, 5], dtype=np.int32)
    fill_windows_mixed(feat_m, is_s1_w, doy, starts, lengths,
                       s2_mean, s2_std, s1_mean, s1_std, np.int64(n_s2),
                       bands_out3, doy_out3, mask_out3, is_s1_out3)

    # count_s2_s1_per_window
    ends_w = np.array([5, 10], dtype=np.int64)
    n_s2_out = np.zeros(W, dtype=np.int32)
    n_s1_out = np.zeros(W, dtype=np.int32)
    count_s2_s1_per_window(is_s1_w, starts, ends_w, n_s2_out, n_s1_out)

    # compute_window_stats (mixed)
    s2m = np.zeros((W, n_s2), dtype=np.float32)
    s2s = np.ones( (W, n_s2), dtype=np.float32)
    s1m = np.zeros((W, n_s1), dtype=np.float32)
    s1s = np.ones( (W, n_s1), dtype=np.float32)
    compute_window_stats(feat_m, is_s1_w, starts, ends_w, np.int64(n_s2), s2m, s2s, s1m, s1s)

    # compute_window_stats_s2only
    s2_only_m = np.zeros((W, 16), dtype=np.float32)
    s2_only_s = np.ones( (W, 16), dtype=np.float32)
    compute_window_stats_s2only(feat, starts, ends_w, s2_only_m, s2_only_s)

    # compute_band_summaries
    bs_out = np.zeros((W, 16 * 3), dtype=np.float32)
    compute_band_summaries(feat, starts, ends_w, bs_out)

    # fill_windows_mixed_subsample
    lengths_long = np.array([10, 10], dtype=np.int32)
    bands_out4 = np.zeros((W, T, F), dtype=np.float32)
    doy_out4   = np.zeros((W, T), dtype=np.int64)
    mask_out4  = np.ones((W, T), dtype=np.bool_)
    is_s1_out4 = np.zeros((W, T), dtype=np.bool_)
    fill_windows_mixed_subsample(
        feat_m, is_s1_w, doy, starts, lengths_long,
        s2_mean, s2_std, s1_mean, s1_std, np.int64(n_s2), np.int64(T),
        bands_out4, doy_out4, mask_out4, is_s1_out4,
    )

    # build_gate_tensors
    T_full = 8
    T_gate_w = np.int64(4)
    B_gate = 2
    bands_full = np.zeros((B_gate, T_full, n_s2), dtype=np.float32)
    doy_full   = np.zeros((B_gate, T_full), dtype=np.int64)
    mask_full  = np.ones((B_gate, T_full), dtype=np.bool_)
    mask_full[:, :5] = False
    is_s1_full = np.zeros((B_gate, T_full), dtype=np.bool_)
    g_bands = np.zeros((B_gate, T_gate_w, n_s2), dtype=np.float32)
    g_doy   = np.zeros((B_gate, T_gate_w), dtype=np.int64)
    g_mask  = np.ones((B_gate, T_gate_w), dtype=np.bool_)
    g_is_s1 = np.zeros((B_gate, T_gate_w), dtype=np.bool_)
    build_gate_tensors(bands_full, doy_full, mask_full, is_s1_full, T_gate_w,
                       g_bands, g_doy, g_mask, g_is_s1)
