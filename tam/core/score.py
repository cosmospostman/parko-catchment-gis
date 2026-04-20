"""tam/core/score.py — Chunked inference for TAMClassifier."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tam.core.dataset import TAMDataset, collate_fn, BAND_COLS, INDEX_COLS
from tam.core.model import TAMClassifier

logger = logging.getLogger(__name__)


def _score_chunk(
    chunk: pd.DataFrame,
    model: TAMClassifier,
    band_mean: np.ndarray,
    band_std: np.ndarray,
    pixel_year_probs: dict[str, dict[int, list[float]]],
    scl_purity_min: float,
    min_obs_per_year: int,
    batch_size: int,
    device: str,
) -> None:
    """Score one in-memory chunk and accumulate results into pixel_year_probs (in-place).

    pixel_year_probs maps point_id -> {year -> [prob, ...]} so that the caller
    can apply year-weighted aggregation after all chunks are processed.
    """
    ds = TAMDataset(
        chunk, labels=None,
        band_mean=band_mean, band_std=band_std,
        scl_purity_min=scl_purity_min,
        min_obs_per_year=min_obs_per_year,
    )
    if len(ds) == 0:
        return
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    with torch.no_grad():
        for batch in loader:
            prob, _ = model(
                batch["bands"].to(device),
                batch["doy"].to(device),
                batch["mask"].to(device),
            )
            for pid, year, p in zip(batch["point_id"], batch["year"], prob.cpu().numpy()):
                year_int = int(year)
                pixel_year_probs.setdefault(pid, {}).setdefault(year_int, []).append(float(p))


def aggregate_year_probs(
    pixel_year_probs: dict[str, dict[int, list[float]]],
    end_year: int,
    decay: float = 0.7,
) -> pd.DataFrame:
    """Aggregate per-(pixel, year) probabilities into a single score per pixel.

    Each year's mean probability is weighted by exp(-decay * (end_year - year)),
    so years closer to end_year receive higher weight. This makes the score
    reflect current presence while still benefiting from multi-year signal.

    Parameters
    ----------
    pixel_year_probs:
        Maps point_id -> {year -> [prob, ...]}
    end_year:
        The reference year (typically the last year of the inference window).
    decay:
        Exponential decay rate (per year back from end_year). Default 0.7
        gives ~12% relative weight to a year 3 years old vs the current year.
    """
    records = []
    for pid, year_probs in pixel_year_probs.items():
        total_w = 0.0
        total_wp = 0.0
        for yr, probs in year_probs.items():
            w = float(np.exp(-decay * (end_year - yr)))
            total_w += w
            total_wp += w * float(np.mean(probs))
        records.append({"point_id": pid, "prob_tam": total_wp / total_w if total_w > 0 else 0.0})
    if not records:
        return pd.DataFrame(columns=["point_id", "prob_tam"])
    return pd.DataFrame(records)


def score_pixels_chunked(
    parquet: Path,
    model: TAMClassifier,
    band_mean: np.ndarray,
    band_std: np.ndarray,
    scl_purity_min: float = 0.5,
    min_obs_per_year: int = 8,
    batch_size: int = 256,
    device: str | None = None,
    tile_id: str | None = None,
    end_year: int | None = None,
    decay: float = 0.7,
) -> pd.DataFrame:
    """Score all pixels in parquet one row-group at a time to bound peak RAM.

    Returns a DataFrame with columns: point_id, prob_tam
    (exponentially decay-weighted mean probability across years, anchored at end_year).
    """
    import pyarrow.parquet as pq

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    pf = pq.ParquetFile(parquet)
    n_rg = pf.metadata.num_row_groups
    pixel_year_probs: dict[str, dict[int, list[float]]] = {}
    tile_prefix = f"_{tile_id}_" if tile_id else None
    parquet_cols = set(pf.schema_arrow.names)
    available_index_cols = [c for c in INDEX_COLS if c in parquet_cols]
    read_cols = ["point_id", "date", "scl_purity"] + (["item_id"] if tile_id else []) + BAND_COLS + available_index_cols
    leftover: pd.DataFrame = pd.DataFrame()

    for rg in range(n_rg):
        chunk = pf.read_row_group(rg, columns=read_cols).to_pandas()
        if tile_prefix:
            chunk = chunk[chunk["item_id"].str.contains(tile_prefix, regex=False)]
        chunk = chunk[chunk["scl_purity"] >= scl_purity_min]
        dates = pd.to_datetime(chunk["date"])
        chunk["year"] = dates.dt.year
        chunk["doy"]  = dates.dt.day_of_year

        # Prepend any pixels that straddled the previous row-group boundary
        if not leftover.empty:
            chunk = pd.concat([leftover, chunk], ignore_index=True)

        is_last = (rg == n_rg - 1)
        if chunk.empty:
            leftover = pd.DataFrame()
            continue
        if not is_last:
            boundary_pid = chunk["point_id"].iloc[-1]
            leftover = chunk[chunk["point_id"] == boundary_pid].copy()
            chunk = chunk[chunk["point_id"] != boundary_pid]
        else:
            leftover = pd.DataFrame()

        if chunk.empty:
            continue
        _score_chunk(chunk, model, band_mean, band_std, pixel_year_probs,
                     scl_purity_min, min_obs_per_year, batch_size, device)
        logger.info("Scored row group %d/%d  (%d pixels so far)", rg + 1, n_rg, len(pixel_year_probs))

    # Flush any remaining leftover (last pixel in file)
    if not leftover.empty:
        _score_chunk(leftover, model, band_mean, band_std, pixel_year_probs,
                     scl_purity_min, min_obs_per_year, batch_size, device)

    inferred_end_year = end_year or max(
        yr for yp in pixel_year_probs.values() for yr in yp
    )
    return aggregate_year_probs(pixel_year_probs, end_year=inferred_end_year, decay=decay)
