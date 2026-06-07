"""Cascade gating investigation: accuracy and throughput of early-exit at T=8 / T=16.

The premise: run V10 at a very short sequence length (T=8 or T=16) as a cheap Stage 1
gate. Pixels scoring below a threshold are discarded without running the full T=128 pass.
Same checkpoint, same weights — no retraining required.

Measures:
  1. Recall@gate on presence pixels (how many real Parkinsonia pixels survive the gate)
  2. Fraction of absence pixels discarded by the gate (the speedup multiplier)
  3. Throughput of Stage 1 at each T (px/s)
  4. Overall cascade throughput vs full T=128 baseline

Usage:
    python scripts/bench_cascade.py --checkpoint outputs/models/tam-v10-0.860
    python scripts/bench_cascade.py --checkpoint outputs/models/tam-v10-0.860 --device cuda
    python scripts/bench_cascade.py --checkpoint outputs/models/tam-v10-0.860 \\
        --gate-thresholds 0.02 0.05 0.10 0.20 \\
        --stage1-t 8 16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rss_gb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6
    return float("nan")


def _score_at_t(
    *,
    bands: np.ndarray,    # (N, T_full, F) already padded to full seq len
    doy:   np.ndarray,    # (N, T_full) int64
    mask:  np.ndarray,    # (N, T_full) bool — True = padding
    n_obs: np.ndarray,    # (N,) float32
    is_s1: np.ndarray | None,  # (N, T_full) bool | None
    model,
    device: str,
    batch_size: int,
    t_cap: int,
    band_summaries: dict | None,
) -> np.ndarray:
    """Run model inference selecting t_cap real tokens per pixel via farthest-point DOY sampling.

    Matches the gate-augmented training augmentation: tokens are spread maximally
    across the annual arc rather than truncated to the earliest t_cap observations.
    n_obs uses T_full as the denominator so the model's sparsity scalar reads the
    same value it sees during training (n_selected / max_seq_len, not n / t_cap).

    Returns float32 probabilities, shape (N,).
    """
    import torch
    from tam.core.dataset import subsample_obs_indices

    N, T_full, F = bands.shape

    if t_cap < T_full:
        b_out = np.zeros((N, t_cap, F), dtype=np.float32)
        d_out = np.zeros((N, t_cap),    dtype=np.int64)
        m_out = np.ones( (N, t_cap),    dtype=bool)   # True = padding
        s_out = np.zeros((N, t_cap),    dtype=bool) if is_s1 is not None else None

        for i in range(N):
            real_idx = np.where(~mask[i])[0]
            sel      = subsample_obs_indices(real_idx, doy[i, real_idx], t_cap)
            keep     = real_idx[sel]
            n_keep   = len(keep)
            b_out[i, :n_keep] = bands[i, keep]
            d_out[i, :n_keep] = doy[i, keep]
            m_out[i, :n_keep] = False
            if s_out is not None:
                s_out[i, :n_keep] = is_s1[i, keep]

        # n_obs denominator is T_full, matching training and inference behaviour
        n_obs_out = np.clip(
            (~m_out).sum(axis=1).astype(np.float32) / T_full, 0.0, 1.0
        )

        bands_th = torch.from_numpy(b_out)
        doy_th   = torch.from_numpy(d_out)
        mask_th  = torch.from_numpy(m_out)
        n_obs_th = torch.from_numpy(n_obs_out)
        is_s1_th = torch.from_numpy(s_out) if s_out is not None else None
    else:
        bands_th = torch.from_numpy(bands)
        doy_th   = torch.from_numpy(doy)
        mask_th  = torch.from_numpy(mask)
        n_obs_th = torch.from_numpy(n_obs)
        is_s1_th = torch.from_numpy(is_s1) if is_s1 is not None else None

    W = N
    probs_list: list[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, W, batch_size):
            end = min(start + batch_size, W)
            is_s1_batch = is_s1_th[start:end].to(device, non_blocking=True) if is_s1_th is not None else None

            gf_batch = None
            if band_summaries is not None and model.n_annual_features > 0:
                # band_summaries is a plain dict pid->array; we skip for simplicity here
                # (pre-scored into annual_feats array by caller if needed)
                pass

            prob, _ = model(
                bands_th[start:end].to(device, non_blocking=True),
                doy_th[start:end].to(device, non_blocking=True),
                mask_th[start:end].to(device, non_blocking=True),
                n_obs_th[start:end].to(device, non_blocking=True),
                annual_feats=gf_batch,
                is_s1=is_s1_batch,
            )
            probs_list.append(prob.cpu().float().numpy())

    return np.concatenate(probs_list)


def _score_at_t_timed(label: str, **kwargs) -> tuple[np.ndarray, float]:
    """Wrapper that prints throughput and returns (probs, elapsed_s)."""
    N = kwargs["bands"].shape[0]
    t0 = time.perf_counter()
    probs = _score_at_t(**kwargs)
    elapsed = time.perf_counter() - t0
    pps = N / elapsed if elapsed > 0 else float("inf")
    print(f"  {label:<30}  N={N:>7,}  elapsed={elapsed:6.2f}s  {pps:>8,.0f} px/s")
    return probs, elapsed


# ---------------------------------------------------------------------------
# Gate analysis
# ---------------------------------------------------------------------------

def analyse_gate(
    *,
    labels: np.ndarray,         # (N,) float32 — 0=absence, 1=presence
    probs_full: np.ndarray,     # (N,) full T=128 scores (reference)
    probs_gate: np.ndarray,     # (N,) gate T=8/16 scores
    t_gate: int,
    thresholds: list[float],
    elapsed_full: float,
    elapsed_gate: float,
    n_pixels: int,
) -> None:
    presence = labels == 1.0
    absence  = labels == 0.0
    n_pres = presence.sum()
    n_abs  = absence.sum()

    print(f"\n  --- T={t_gate} gate analysis  (presence={n_pres:,}  absence={n_abs:,}) ---")
    print(f"  {'Threshold':>10}  {'Recall@gate':>11}  {'Abs discarded':>13}  "
          f"{'Cascade speedup':>15}  {'Gate survivors':>14}")
    print(f"  {'-'*10}  {'-'*11}  {'-'*13}  {'-'*15}  {'-'*14}")

    t_ratio = elapsed_gate / elapsed_full if elapsed_full > 0 else 0.0

    for thresh in thresholds:
        # Pixels that pass the gate (prob >= thresh)
        gate_pass = probs_gate >= thresh

        recall = gate_pass[presence].mean() if n_pres else float("nan")
        abs_discarded_frac = (~gate_pass[absence]).mean() if n_abs else float("nan")

        # Cascade speedup: gate runs on all pixels, full model only on survivors.
        # speedup = 1 / (gate_fraction + survivor_fraction * 1.0)
        # where gate_fraction = t_gate/128 (proportional cost)
        survivor_frac = gate_pass.mean()
        gate_cost_frac = t_gate / 128  # relative to full T=128
        cascade_frac = gate_cost_frac + survivor_frac
        speedup = (1.0 / cascade_frac) if cascade_frac > 0 else float("inf")

        survivors = gate_pass.sum()
        print(f"  {thresh:>10.3f}  {recall:>10.1%}  {abs_discarded_frac:>12.1%}  "
              f"  {speedup:>12.2f}×  {survivors:>10,} / {n_pixels:,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    import torch
    from tam.core.train import load_tam
    from tam.core.score import (
        _compute_s2_pixel_zscore_stats,
        _compute_pixel_s1_stats_mixed,
        _preprocess,
        _PreparedBatch,
        _ZscoreArrays,
    )
    from tam.core.dataset import ALL_FEATURE_COLS, V10_S1_FEATURE_COLS

    checkpoint_dir = Path(args.checkpoint)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nbench_cascade  checkpoint={checkpoint_dir.name}  device={device}")
    print(f"  stage1 T values: {args.stage1_t}")
    print(f"  gate thresholds: {args.gate_thresholds}")

    # -----------------------------------------------------------------------
    # 1. Load checkpoint
    # -----------------------------------------------------------------------
    print("\n[1/4] Loading checkpoint...")
    model, band_mean, band_std, annual_feat_mean, annual_feat_std = load_tam(
        checkpoint_dir, device=device
    )
    with open(checkpoint_dir / "tam_config.json") as fh:
        cfg = json.load(fh)

    s2_cols = list(cfg.get("feature_cols", ALL_FEATURE_COLS))
    s1_cols = list(cfg.get("s1_feature_cols", V10_S1_FEATURE_COLS))
    mixed   = cfg.get("use_s1", True)
    pixel_zscore = cfg.get("inference_pixel_zscore", False)
    t_full  = args.t_full or cfg.get("max_seq_len", 128)

    print(f"  max_seq_len={t_full}  mixed={mixed}  pixel_zscore={pixel_zscore}")
    print(f"  s2_cols ({len(s2_cols)}): {s2_cols}")
    print(f"  s1_cols ({len(s1_cols)}): {s1_cols}")

    # -----------------------------------------------------------------------
    # 2. Load val parquet + labels
    # -----------------------------------------------------------------------
    print("\n[2/4] Loading val pixels...")
    val_df = pl.read_parquet(checkpoint_dir / "prep_val_pixel_df.parquet")
    with open(checkpoint_dir / "pixel_df_labels.json") as fh:
        label_map: dict[str, float] = json.load(fh)

    # Limit to one year per pixel (most recent) to keep memory bounded
    max_year = val_df["year"].max()
    if args.limit_years:
        val_df = val_df.filter(pl.col("year") == max_year)
        print(f"  Limited to year={max_year}")

    n_rows = len(val_df)
    n_pids = val_df["point_id"].n_unique()
    print(f"  {n_rows:,} rows  {n_pids:,} unique pixels")

    # -----------------------------------------------------------------------
    # 3. Preprocess: one _preprocess call to build padded tensors at T=t_full
    # -----------------------------------------------------------------------
    print("\n[3/4] Pre-processing (building padded tensors at T=128)...")

    # Pre-pass for pixel z-score stats (needed for mixed-mode _preprocess).
    # Write val_df to a temp parquet so the pre-pass functions can read it.
    pixel_zscore_stats = None
    s1_zscore_stats    = None
    if pixel_zscore and mixed:
        import tempfile
        print("  Running pixel z-score pre-pass...")
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        val_df.write_parquet(tmp_path)
        year_parquets_tmp = [(int(max_year), tmp_path)]

        from tam.core.score import _compute_s2_pixel_zscore_stats, _compute_pixel_s1_stats_mixed
        from concurrent.futures import ThreadPoolExecutor as _TPE
        with _TPE(max_workers=2) as ex:
            f_s2 = ex.submit(_compute_s2_pixel_zscore_stats,
                             year_parquets=year_parquets_tmp, feature_cols=s2_cols,
                             scl_purity_min=0.5)
            f_s1 = ex.submit(_compute_pixel_s1_stats_mixed,
                             year_parquets=year_parquets_tmp, s1_feature_cols=s1_cols)
        _raw_s2 = f_s2.result()
        _raw_s1 = f_s1.result()
        tmp_path.unlink(missing_ok=True)
        _s2_n = len(next(iter(_raw_s2[0].values()))) if _raw_s2[0] else len(s2_cols)
        _s1_n = len(next(iter(_raw_s1[0].values()))) if _raw_s1[0] else len(s1_cols)
        pixel_zscore_stats = _ZscoreArrays(*_raw_s2, n_feat=_s2_n)
        s1_zscore_stats    = _ZscoreArrays(*_raw_s1, n_feat=_s1_n)
        print(f"  z-score stats: {len(_raw_s2[0])} S2 pixels, {len(_raw_s1[0])} S1 pixels")

    t0_prep = time.perf_counter()
    prepared: _PreparedBatch | None = _preprocess(
        chunk=val_df,
        band_mean=band_mean,
        band_std=band_std,
        scl_purity_min=0.5,
        min_obs_per_year=8,
        pin=False,
        mixed=mixed,
        pixel_zscore=pixel_zscore,
        pixel_zscore_stats=pixel_zscore_stats,
        s1_zscore_stats=s1_zscore_stats,
        s2_feature_cols=s2_cols,
        s1_feature_cols=s1_cols,
        max_seq_len=t_full,
    )
    t_prep = time.perf_counter() - t0_prep

    if prepared is None:
        print("ERROR: _preprocess returned None — no valid pixels in val set")
        sys.exit(1)

    bands_th, doy_th, mask_th, n_obs_th, pids, years, is_s1_th = prepared
    N = len(pids)
    print(f"  Preprocessed {N:,} pixel-years  ({t_prep:.1f}s)")

    # Extract numpy arrays (CPU)
    bands_np = bands_th.numpy()
    doy_np   = doy_th.numpy()
    mask_np  = mask_th.numpy()
    n_obs_np = n_obs_th.numpy()
    is_s1_np = is_s1_th.numpy() if is_s1_th is not None else None

    # Build label array aligned with pids
    label_arr = np.array([label_map.get(pid, 0.0) for pid in pids], dtype=np.float32)
    n_pres = (label_arr == 1.0).sum()
    n_abs  = (label_arr == 0.0).sum()
    print(f"  Labels: {n_pres:,} presence  {n_abs:,} absence")

    # -----------------------------------------------------------------------
    # 4. Inference at full T and each stage-1 T
    # -----------------------------------------------------------------------
    print(f"\n[4/4] Inference  (batch_size={args.batch_size})")

    common = dict(
        bands=bands_np, doy=doy_np, mask=mask_np, n_obs=n_obs_np, is_s1=is_s1_np,
        model=model, device=device, batch_size=args.batch_size, band_summaries=None,
    )

    # Warmup: one small forward pass to JIT any lazy ops
    print("  Warming up...")
    _ = _score_at_t(**common, t_cap=t_full)

    print("\n  Throughput:")
    probs_full, elapsed_full = _score_at_t_timed(f"T={t_full} (full)", t_cap=t_full, **common)

    gate_results: dict[int, tuple[np.ndarray, float]] = {}
    for t_gate in sorted(args.stage1_t):
        probs_gate, elapsed_gate = _score_at_t_timed(f"T={t_gate} (gate)", t_cap=t_gate, **common)
        gate_results[t_gate] = (probs_gate, elapsed_gate)

    # -----------------------------------------------------------------------
    # 5. Accuracy / cascade analysis
    # -----------------------------------------------------------------------
    print("\n[Analysis]")
    print(f"  Full T={t_full} baseline:  {N/elapsed_full:,.0f} px/s")

    for t_gate, (probs_gate, elapsed_gate) in gate_results.items():
        analyse_gate(
            labels=label_arr,
            probs_full=probs_full,
            probs_gate=probs_gate,
            t_gate=t_gate,
            thresholds=args.gate_thresholds,
            elapsed_full=elapsed_full,
            elapsed_gate=elapsed_gate,
            n_pixels=N,
        )

    # Summary: correlation between gate and full scores
    print("\n  Score correlation (gate vs full):")
    for t_gate, (probs_gate, _) in gate_results.items():
        r = float(np.corrcoef(probs_full, probs_gate)[0, 1])
        print(f"    T={t_gate:>3} vs T={t_full}: r={r:.4f}")

    # Presence pixel score distribution at gate T
    print("\n  Presence pixel gate scores (should be high to be safe):")
    for t_gate, (probs_gate, _) in gate_results.items():
        pres_scores = probs_gate[label_arr == 1.0]
        print(f"    T={t_gate:>3}  p05={np.percentile(pres_scores,5):.3f}  "
              f"p25={np.percentile(pres_scores,25):.3f}  "
              f"p50={np.percentile(pres_scores,50):.3f}  "
              f"p95={np.percentile(pres_scores,95):.3f}")

    print("\n  Absence pixel gate scores (low = safely discarded):")
    for t_gate, (probs_gate, _) in gate_results.items():
        abs_scores = probs_gate[label_arr == 0.0]
        print(f"    T={t_gate:>3}  p05={np.percentile(abs_scores,5):.3f}  "
              f"p25={np.percentile(abs_scores,25):.3f}  "
              f"p50={np.percentile(abs_scores,50):.3f}  "
              f"p95={np.percentile(abs_scores,95):.3f}")

    print(f"\nDone.  Peak RSS: {rss_gb():.2f} GB")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V10 cascade gate accuracy + throughput investigation")
    parser.add_argument("--checkpoint", required=True,
                        help="Checkpoint directory (e.g. outputs/models/tam-v10-0.860)")
    parser.add_argument("--stage1-t", type=int, nargs="+", default=[8, 16],
                        help="Stage-1 sequence lengths to test (default: 8 16)")
    parser.add_argument("--gate-thresholds", type=float, nargs="+",
                        default=[0.02, 0.05, 0.10, 0.20],
                        help="Gate discard thresholds (pixel scores below this are dropped)")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default=None,
                        help="torch device (default: cuda if available, else cpu)")
    parser.add_argument("--limit-years", action="store_true", default=True,
                        help="Score only the most recent year per pixel (default: True)")
    parser.add_argument("--all-years", dest="limit_years", action="store_false",
                        help="Score all years in the val parquet")
    parser.add_argument("--t-full", type=int, default=None,
                        help="Override max_seq_len from checkpoint (useful for testing T=128 on a T=64 checkpoint)")
    args = parser.parse_args()
    main(args)
