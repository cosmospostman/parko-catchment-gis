"""explain.py — Attention + gradient saliency for TAM inference on Kowanyama pixels.

For the top-N and bottom-N scoring pixels from a TAM pixel ranking CSV, produces:
  - Attention heatmap: which acquisitions the model weighted (DOY × layer/head)
  - Gradient saliency: which (acquisition, band) cells drove the classification

Usage
-----
    python explain.py
    python explain.py --checkpoint outputs/tam-frenchs --ranking outputs/tam-kowanyama/tam_pixel_ranking.csv
    python explain.py --n 5 --out outputs/explain
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from tam.core.dataset import BAND_COLS, MAX_SEQ_LEN, MIN_OBS_PER_YEAR
from tam.core.model import TAMClassifier
from tam.core.train import load_tam
from utils.location import get as get_location

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy checkpoint support (learned DOY embedding, pre-sinusoidal refactor)
# ---------------------------------------------------------------------------

class _LegacyTAMClassifier(TAMClassifier):
    """TAMClassifier variant with a learned DOY embedding (nn.Embedding).

    Used to load checkpoints saved before the switch to fixed sinusoidal
    DOY encoding.  The forward and get_attention_weights methods are
    identical to the parent except they use self.doy_embed instead of
    _doy_encoding().
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        import torch.nn as nn
        self.doy_embed = nn.Embedding(366, self.d_model, padding_idx=0)

    def forward(self, bands, doy, key_padding_mask):
        import torch
        x = self.band_proj(bands) + self.doy_embed(doy)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        valid = (~key_padding_mask).float().unsqueeze(-1)
        x_pool = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        logit = self.head(x_pool).squeeze(-1)
        prob = torch.sigmoid(logit)
        return prob, logit

    def get_attention_weights(self, bands, doy, key_padding_mask):
        self.eval()
        import torch
        with torch.no_grad():
            x = self.band_proj(bands) + self.doy_embed(doy)
            attn_weights = []
            for layer in self.encoder.layers:
                attn_out, w = layer.self_attn(
                    x, x, x,
                    key_padding_mask=key_padding_mask,
                    need_weights=True,
                    average_attn_weights=False,
                )
                attn_weights.append(w.squeeze(0))
                x2 = layer.norm1(x + layer.dropout1(attn_out))
                x2 = layer.norm2(x2 + layer.dropout2(layer.linear2(
                    layer.dropout(layer.activation(layer.linear1(x2)))
                )))
                x = x2
        return [w.cpu().numpy() for w in attn_weights]


def _load_tam_compat(
    checkpoint_dir: Path,
    device: str,
) -> tuple[TAMClassifier, np.ndarray, np.ndarray]:
    """Load a TAM checkpoint, handling both sinusoidal and legacy learned-DOY variants."""
    import json, torch
    from tam.core.config import TAMConfig

    with open(checkpoint_dir / "tam_config.json") as fh:
        cfg = TAMConfig.from_dict(json.load(fh))

    sd = torch.load(checkpoint_dir / "tam_model.pt", map_location=device, weights_only=True)
    if "doy_embed.weight" in sd:
        logger.info("Checkpoint uses learned DOY embedding (legacy) — loading with _LegacyTAMClassifier")
        model = _LegacyTAMClassifier(
            d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
            d_ff=cfg.d_ff, dropout=cfg.dropout,
        )
    else:
        model = TAMClassifier.from_config(cfg)

    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()

    stats = np.load(checkpoint_dir / "tam_band_stats.npz")
    return model, stats["mean"].astype(np.float32), stats["std"].astype(np.float32)

# Defaults
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "tam-frenchs"
DEFAULT_RANKING    = PROJECT_ROOT / "outputs" / "tam-kowanyama" / "tam_pixel_ranking.csv"
DEFAULT_PARQUET    = PROJECT_ROOT / "data" / "kowanyama.parquet"
DEFAULT_OUT        = PROJECT_ROOT / "outputs" / "explain"
DEFAULT_N          = 5
SCL_PURITY_MIN     = 0.5


# ---------------------------------------------------------------------------
# Pixel loading
# ---------------------------------------------------------------------------

def _load_pixel_windows(
    parquet: Path,
    point_ids: list[str],
    band_mean: np.ndarray,
    band_std: np.ndarray,
    end_year: int | None = None,
) -> dict[str, list[tuple[int, np.ndarray, np.ndarray]]]:
    """Load observation windows for the requested point_ids from parquet.

    Returns
    -------
    dict mapping point_id → list of (year, bands_norm [T, 10], doy [T])
    sorted by year, then DOY within year. Only years with >= MIN_OBS_PER_YEAR
    clear observations are included.
    """
    import pyarrow.parquet as pq

    needed = set(point_ids)
    read_cols = ["point_id", "date", "scl_purity"] + BAND_COLS
    pf = pq.ParquetFile(parquet)

    rows: list[pd.DataFrame] = []
    for rg in range(pf.metadata.num_row_groups):
        chunk = pf.read_row_group(rg, columns=read_cols).to_pandas()
        chunk = chunk[chunk["point_id"].isin(needed)]
        if not chunk.empty:
            rows.append(chunk)

    if not rows:
        return {}

    df = pd.concat(rows, ignore_index=True)
    df = df[df["scl_purity"] >= SCL_PURITY_MIN]
    df = df.dropna(subset=BAND_COLS)
    dates = pd.to_datetime(df["date"])
    df["year"] = dates.dt.year
    df["doy"]  = dates.dt.day_of_year

    if end_year is not None:
        df = df[df["year"] <= end_year]

    result: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for pid, pgrp in df.groupby("point_id"):
        windows = []
        for yr, ygrp in pgrp.groupby("year"):
            ygrp = ygrp.sort_values("doy")
            if len(ygrp) < MIN_OBS_PER_YEAR:
                continue
            n = min(len(ygrp), MAX_SEQ_LEN)
            raw = ygrp[BAND_COLS].values[:n].astype(np.float32)
            bands_norm = (raw - band_mean) / band_std
            doy = ygrp["doy"].values[:n].astype(np.int32)
            windows.append((int(yr), bands_norm, doy))
        if windows:
            result[pid] = sorted(windows, key=lambda t: t[0])
    return result


def _best_year_window(
    windows: list[tuple[int, np.ndarray, np.ndarray]],
    end_year: int,
    decay: float = 0.7,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Return the year window with highest decay weight (most recent non-empty year)."""
    best = max(windows, key=lambda t: np.exp(-decay * (end_year - t[0])))
    return best


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------

def _get_attention(
    model: TAMClassifier,
    bands_norm: np.ndarray,
    doy: np.ndarray,
    device: str,
) -> tuple[float, list[np.ndarray]]:
    """Run forward + attention extraction for one window.

    Returns (prob, attn_weights) where attn_weights is list of (n_heads, T, T).
    T = actual sequence length (not padded MAX_SEQ_LEN).
    """
    n = len(bands_norm)

    bands_t = torch.zeros(1, MAX_SEQ_LEN, len(BAND_COLS))
    bands_t[0, :n] = torch.from_numpy(bands_norm)

    doy_t = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.long)
    doy_t[0, :n] = torch.from_numpy(doy.astype(np.int64))

    mask_t = torch.ones(1, MAX_SEQ_LEN, dtype=torch.bool)
    mask_t[0, :n] = False

    bands_t = bands_t.to(device)
    doy_t   = doy_t.to(device)
    mask_t  = mask_t.to(device)

    with torch.no_grad():
        prob, _ = model(bands_t, doy_t, mask_t)

    raw_attn = model.get_attention_weights(bands_t, doy_t, mask_t)
    # Layers may be tensors (sinusoidal model) or numpy arrays (legacy model)
    attn_layers = [
        (w.cpu().numpy() if hasattr(w, "cpu") else w)[:, :n, :n]
        for w in raw_attn
    ]

    return float(prob.item()), attn_layers


# ---------------------------------------------------------------------------
# Gradient saliency
# ---------------------------------------------------------------------------

def _get_gradient_saliency(
    model: TAMClassifier,
    bands_norm: np.ndarray,
    doy: np.ndarray,
    device: str,
) -> np.ndarray:
    """Input × gradient saliency map, shape (T, N_BANDS).

    Uses integrated-gradient-style input×grad to give signed attribution.
    Positive values → increasing band pushes toward presence.
    """
    n = len(bands_norm)

    bands_t = torch.zeros(1, MAX_SEQ_LEN, len(BAND_COLS), requires_grad=True)
    with torch.no_grad():
        bands_t.data[0, :n] = torch.from_numpy(bands_norm)

    doy_t = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.long)
    doy_t[0, :n] = torch.from_numpy(doy.astype(np.int64))

    mask_t = torch.ones(1, MAX_SEQ_LEN, dtype=torch.bool)
    mask_t[0, :n] = False

    bands_t = bands_t.to(device)
    doy_t   = doy_t.to(device)
    mask_t  = mask_t.to(device)

    with torch.enable_grad():
        _, logit = model(bands_t, doy_t, mask_t)
        logit.backward()

    grad = bands_t.grad[0, :n].cpu().numpy()          # (T, N_BANDS)
    inp  = bands_t.data[0, :n].cpu().numpy()           # (T, N_BANDS)
    saliency = inp * grad                              # input × gradient
    return saliency


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def _doy_to_month(doy_arr: np.ndarray) -> list[str]:
    dates = pd.to_datetime(doy_arr, unit="D", origin=pd.Timestamp("2000-01-01") - pd.Timedelta(days=1))
    return [d.strftime("%b") for d in dates]


def _plot_pixel(
    point_id: str,
    prob_tam: float,
    year: int,
    doy: np.ndarray,
    attn_layers: list[np.ndarray],
    saliency: np.ndarray,
    out_path: Path,
) -> None:
    """Single-pixel figure: attention heatmap + saliency heatmap."""
    n = len(doy)
    n_layers = len(attn_layers)

    fig = plt.figure(figsize=(14, 4 + 3 * n_layers))
    fig.suptitle(
        f"{point_id}  |  prob_tam={prob_tam:.3f}  |  year={year}  |  T={n} obs",
        fontsize=11, fontweight="bold",
    )

    n_rows = n_layers + 1
    gs = gridspec.GridSpec(n_rows, 1, figure=fig, hspace=0.5)

    x_labels = [f"{d}\n{_doy_to_month([d])[0]}" for d in doy]

    # --- Attention: one subplot per layer, mean across heads ---
    for li, attn in enumerate(attn_layers):
        ax = fig.add_subplot(gs[li])
        # Mean across heads, then mean-pool over query dimension → per-key importance
        attn_mean = attn.mean(axis=0).mean(axis=0)  # (T,)
        ax.bar(range(n), attn_mean, color="steelblue", alpha=0.8)
        ax.set_xticks(range(n))
        ax.set_xticklabels(x_labels, fontsize=6, rotation=0)
        ax.set_ylabel("Attention\nweight", fontsize=8)
        ax.set_title(f"Layer {li+1} attention (mean over heads, mean-pooled over queries)", fontsize=8)
        ax.set_xlim(-0.5, n - 0.5)

    # --- Saliency: band × acquisition heatmap ---
    ax_sal = fig.add_subplot(gs[n_layers])
    sal_display = saliency.T  # (N_BANDS, T)
    vmax = np.percentile(np.abs(sal_display), 95) or 1e-6
    im = ax_sal.imshow(
        sal_display, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, interpolation="nearest",
    )
    ax_sal.set_yticks(range(len(BAND_COLS)))
    ax_sal.set_yticklabels(BAND_COLS, fontsize=8)
    ax_sal.set_xticks(range(n))
    ax_sal.set_xticklabels(x_labels, fontsize=6, rotation=0)
    ax_sal.set_title("Input × gradient saliency  (red = pushes toward presence, blue = absence)", fontsize=8)
    plt.colorbar(im, ax=ax_sal, fraction=0.02, pad=0.02)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _plot_summary(
    results: list[dict],
    out_path: Path,
) -> None:
    """Summary figure: mean saliency per band across all explained pixels."""
    if not results:
        return

    sal_stack = np.stack([r["saliency"].mean(axis=0) for r in results])  # (N_pixels, N_BANDS)
    mean_sal = sal_stack.mean(axis=0)
    std_sal  = sal_stack.std(axis=0)

    # Mean absolute attention weight per acquisition DOY (binned to month)
    all_doy_attn: list[tuple[int, float]] = []
    for r in results:
        doy = r["doy"]
        for li, attn in enumerate(r["attn_layers"]):
            attn_per_key = attn.mean(axis=0).mean(axis=0)  # (T,)
            for d, a in zip(doy, attn_per_key):
                all_doy_attn.append((int(d), float(a)))

    doa_df = pd.DataFrame(all_doy_attn, columns=["doy", "attn"])
    doa_df["month"] = ((doa_df["doy"] - 1) // 30).clip(0, 11)
    monthly_attn = doa_df.groupby("month")["attn"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(
        f"Explanation summary — {len(results)} pixels  "
        f"(prob range {min(r['prob'] for r in results):.2f}–{max(r['prob'] for r in results):.2f})",
        fontsize=11, fontweight="bold",
    )

    # Band saliency
    ax = axes[0]
    colors = ["tomato" if v > 0 else "steelblue" for v in mean_sal]
    ax.barh(BAND_COLS, mean_sal, xerr=std_sal, color=colors, alpha=0.85, capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean input×gradient saliency\n(positive = pushes toward presence)")
    ax.set_title("Band importance (mean across pixels)")

    # Monthly attention
    ax2 = axes[1]
    months_present = monthly_attn.index.tolist()
    ax2.bar(months_present, monthly_attn.values, color="steelblue", alpha=0.8)
    ax2.set_xticks(range(12))
    ax2.set_xticklabels(_MONTH_LABELS, fontsize=8)
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Mean attention weight")
    ax2.set_title("Seasonal attention (which months the model looked at)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _select_bbox_ids(
    ranking: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    n: int,
) -> list[str]:
    """Return up to n representative point_ids within bbox from the ranking.

    Picks the highest, lowest, and evenly-spaced median pixels within the bbox
    so we get a spread of model confidence rather than all-high or all-low.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    in_box = ranking[
        ranking["lon"].between(lon_min, lon_max) &
        ranking["lat"].between(lat_min, lat_max)
    ].sort_values("prob_tam", ascending=False)

    if in_box.empty:
        return []

    if len(in_box) <= n:
        return in_box["point_id"].tolist()

    # Always include highest and lowest; fill remainder from evenly-spaced interior
    indices = sorted(set(
        [0, len(in_box) - 1] +
        [int(i * (len(in_box) - 1) / (n - 1)) for i in range(n)]
    ))[:n]
    return in_box.iloc[indices]["point_id"].tolist()


def explain(
    checkpoint: Path,
    ranking_csv: Path,
    parquet: Path,
    out_dir: Path,
    n: int,
    end_year: int,
    device: str | None,
    bbox: tuple[float, float, float, float] | None = None,
) -> None:
    device = device or "cpu"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading checkpoint from %s", checkpoint)
    model, band_mean, band_std = _load_tam_compat(checkpoint, device=device)

    logger.info("Loading pixel ranking from %s", ranking_csv)
    ranking = pd.read_csv(ranking_csv).sort_values("prob_tam", ascending=False)
    ranking = ranking.dropna(subset=["prob_tam"])

    if bbox is not None:
        if "lon" not in ranking.columns or "lat" not in ranking.columns:
            raise ValueError("Ranking CSV must have lon/lat columns to use --bbox")
        all_ids = _select_bbox_ids(ranking, bbox, n)
        if not all_ids:
            logger.error("No pixels found in bbox %s", bbox)
            return
        logger.info("Explaining %d pixels sampled from bbox %s", len(all_ids), bbox)
        # Treat them all as one group for summary purposes
        top_ids, bottom_ids = all_ids, []
    else:
        top_ids    = ranking.head(n)["point_id"].tolist()
        bottom_ids = ranking.tail(n)["point_id"].tolist()
        all_ids    = list(dict.fromkeys(top_ids + bottom_ids))
        logger.info(
            "Explaining %d pixels: top-%d (high prob) + bottom-%d (low prob)",
            len(all_ids), n, n,
        )

    logger.info("Loading observations from %s", parquet)
    windows_by_pid = _load_pixel_windows(parquet, all_ids, band_mean, band_std, end_year=end_year)

    missing = set(all_ids) - set(windows_by_pid)
    if missing:
        logger.warning("No qualifying observations found for %d pixels: %s", len(missing), missing)

    results_top:    list[dict] = []
    results_bottom: list[dict] = []

    for group_name, pid_list, results_list in [
        ("top",    top_ids,    results_top),
        ("bottom", bottom_ids, results_bottom),
    ]:
        for pid in pid_list:
            if pid not in windows_by_pid:
                continue

            windows = windows_by_pid[pid]
            year, bands_norm, doy = _best_year_window(windows, end_year)

            prob, attn_layers = _get_attention(model, bands_norm, doy, device)
            saliency = _get_gradient_saliency(model, bands_norm, doy, device)

            prob_tam = float(ranking.loc[ranking["point_id"] == pid, "prob_tam"].iloc[0])

            stem = f"explain_{group_name}_{pid}"
            _plot_pixel(
                point_id=pid,
                prob_tam=prob_tam,
                year=year,
                doy=doy,
                attn_layers=attn_layers,
                saliency=saliency,
                out_path=out_dir / f"{stem}.png",
            )

            results_list.append({
                "point_id":   pid,
                "prob":       prob_tam,
                "year":       year,
                "doy":        doy,
                "attn_layers": attn_layers,
                "saliency":   saliency,
            })

    _plot_summary(results_top,    out_dir / "explain_summary_top.png")
    _plot_summary(results_bottom, out_dir / "explain_summary_bottom.png")
    _plot_summary(results_top + results_bottom, out_dir / "explain_summary_all.png")

    logger.info("Done — outputs in %s", out_dir)


def _resolve_parquet(args_parquet: str | None, year: int | None = None) -> Path:
    if args_parquet:
        return Path(args_parquet)
    loc = get_location("kowanyama")
    years = loc.parquet_years()
    if not years:
        raise FileNotFoundError(f"No annual parquets found for {loc.id}")
    y = year if year is not None else years[-1]
    return loc.parquet_path(y)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="TAM attention + saliency explanation")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                        help=f"Checkpoint directory (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--ranking", default=str(DEFAULT_RANKING),
                        help=f"Pixel ranking CSV (default: {DEFAULT_RANKING})")
    parser.add_argument("--parquet", default=None,
                        help="Pixel parquet (default: kowanyama location parquet)")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help=f"Output directory (default: {DEFAULT_OUT})")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"Number of top/bottom pixels to explain (default: {DEFAULT_N})")
    parser.add_argument("--end-year", type=int, default=2024,
                        help="Reference year for picking best annual window (default: 2024)")
    parser.add_argument("--device", default=None, help="cpu / cuda (auto-detect if omitted)")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                        help="Explain pixels within this bbox instead of top/bottom from ranking")
    args = parser.parse_args()

    bbox = tuple(args.bbox) if args.bbox else None

    explain(
        checkpoint  = Path(args.checkpoint),
        ranking_csv = Path(args.ranking),
        parquet     = _resolve_parquet(args.parquet, year=args.end_year),
        out_dir     = Path(args.out),
        n           = args.n,
        end_year    = args.end_year,
        device      = args.device,
        bbox        = bbox,
    )
