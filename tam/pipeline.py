"""tam/pipeline.py — End-to-end TAM train/inference pipeline.

Usage
-----
    # Train and score:
    python -m tam.pipeline --location longreach --train

    # Score only (requires existing checkpoint):
    python -m tam.pipeline --location longreach
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.common import label_pixels, save_pixel_ranking, summarise
from signals._shared import ensure_pixel_sorted
from tam.config import TAMConfig
from tam.dataset import TAMDataset, collate_fn, BAND_COLS
from tam.model import TAMClassifier
from tam.train import train_tam, load_tam
from utils.heatmap import plot_prob_heatmaps
from utils.location import get as get_location

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

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
    read_cols = ["point_id", "date", "scl_purity"] + (["item_id"] if tile_id else []) + BAND_COLS
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


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(
    location_id: str,
    do_train: bool = True,
    scl_purity_min: float = 0.5,
    n_epochs: int = 100,
    patience: int = 15,
    device: str | None = None,
    tile_id: str | None = None,
    checkpoint_dir: Path | None = None,
    end_year: int | None = None,
    decay: float = 0.7,
) -> None:
    loc = get_location(location_id)
    parquet = loc.parquet_path()
    if not parquet.exists():
        logger.error("Parquet not found: %s — run pixel_collector first", parquet)
        sys.exit(1)

    out_dir = PROJECT_ROOT / "outputs" / f"tam-{loc.id}"
    if tile_id:
        out_dir = out_dir / tile_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Location: %s  parquet: %s  tile: %s", loc.name, parquet, tile_id or "all")

    # --- Load pixel coords only (cheap) to resolve labels before full load ---
    logger.info("Resolving labels ...")
    import pyarrow.parquet as pq
    from concurrent.futures import ThreadPoolExecutor, as_completed
    pf_coords = pq.ParquetFile(parquet)
    read_coord_cols = ["point_id", "lon", "lat"] + (["item_id"] if tile_id else [])
    tile_prefix = f"_{tile_id}_" if tile_id else None
    n_rg_coords = pf_coords.metadata.num_row_groups

    def _read_coord_rg(rg: int) -> pd.DataFrame:
        pf = pq.ParquetFile(parquet)
        chunk = pf.read_row_group(rg, columns=read_coord_cols).to_pandas()
        if tile_prefix:
            chunk = chunk[chunk["item_id"].str.contains(tile_prefix, regex=False)]
        return chunk[["point_id", "lon", "lat"]].drop_duplicates("point_id")

    coord_chunks = []
    n_done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_read_coord_rg, rg): rg for rg in range(n_rg_coords)}
        for fut in as_completed(futures):
            chunk = fut.result()
            if not chunk.empty:
                coord_chunks.append(chunk)
            n_done += 1
            if n_done % 100 == 0:
                logger.info("  coords %d/%d row groups", n_done, n_rg_coords)

    pixel_coords = (
        pd.concat(coord_chunks, ignore_index=True)
        .groupby("point_id")[["lon", "lat"]]
        .first()
        .reset_index()
    )
    logger.info("Unique pixels after tile filter: %d", len(pixel_coords))

    # --- Labels from YAML sub-bboxes ----------------------------------------
    labelled = label_pixels(pixel_coords, loc)
    labelled_known = labelled.dropna(subset=["is_presence"])
    labels = labelled_known.set_index("point_id")["is_presence"].map(
        {True: 1.0, False: 0.0}
    )
    logger.info(
        "Labeled pixels — presence: %d  absence: %d",
        (labels == 1).sum(), (labels == 0).sum(),
    )

    # --- Train or load checkpoint -------------------------------------------
    parquet = ensure_pixel_sorted(parquet)
    if do_train:
        logger.info("Loading parquet (labeled pixels only for training) ...")
        labeled_ids = set(labels.index)
        pf = pq.ParquetFile(parquet)
        chunks = []
        for rg in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["point_id", "lon", "lat", "date", "scl_purity"] + BAND_COLS)
            pdf = tbl.to_pandas()
            pdf = pdf[pdf["point_id"].isin(labeled_ids)]
            if not pdf.empty:
                chunks.append(pdf)
        pixel_df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        pixel_df = pixel_df[pixel_df["scl_purity"] >= scl_purity_min]
        _dates = pd.to_datetime(pixel_df["date"])
        pixel_df["year"] = _dates.dt.year
        pixel_df["doy"]  = _dates.dt.day_of_year
        logger.info("Loaded %d observations for %d labeled pixels", len(pixel_df), pixel_df["point_id"].nunique())

        logger.info("Training TAM ...")
        cfg = TAMConfig(
            n_epochs=n_epochs,
            patience=patience,
            scl_purity_min=scl_purity_min,
        )
        model, band_mean, band_std = (
            train_tam(
                pixel_df=pixel_df,
                labels=labels,
                pixel_coords=pixel_coords,
                out_dir=out_dir,
                cfg=cfg,
                device=device,
            ),
            *_load_band_stats(out_dir),
        )
        # Reload from checkpoint for cleaner interface
        model, band_mean, band_std = load_tam(out_dir, device=device)
    else:
        load_dir = checkpoint_dir or out_dir
        logger.info("Loading checkpoint from %s ...", load_dir)
        model, band_mean, band_std = load_tam(load_dir, device=device)

    # --- Score all pixels via chunked row-group inference -------------------
    logger.info("Scoring all pixels (chunked) ...")
    scores = score_pixels_chunked(
        parquet, model, band_mean, band_std,
        scl_purity_min=scl_purity_min, device=device, tile_id=tile_id,
        end_year=end_year, decay=decay,
    )

    # Merge with coords + labels
    scored = pixel_coords.merge(scores, on="point_id", how="left")
    scored = scored.merge(
        labelled[["point_id", "is_presence"]], on="point_id", how="left"
    )

    # Rank (1 = highest probability)
    scored["rank"] = scored["prob_tam"].rank(ascending=False, method="first").astype("Int64")

    # --- Outputs ------------------------------------------------------------
    stem = f"tam_{loc.id}" + (f"_{tile_id}" if tile_id else "")
    summarise(scored, loc)
    save_pixel_ranking(scored, out_dir / "tam_pixel_ranking.csv", features=["prob_tam"])
    plot_prob_heatmaps(scored.dropna(subset=["prob_tam"]), loc, out_dir, stem=stem, prob_col="prob_tam")
    logger.info("Done — outputs in %s", out_dir)


def _load_band_stats(out_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    stats = np.load(out_dir / "tam_band_stats.npz")
    return stats["mean"], stats["std"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="TAM Parkinsonia classifier pipeline")
    parser.add_argument("--location",   required=True, help="Location ID (e.g. longreach)")
    parser.add_argument("--train",      action="store_true", help="Train a new model")
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--patience",   type=int, default=15, help="Early stopping patience (default 15)")
    parser.add_argument("--scl-purity", type=float, default=0.5)
    parser.add_argument("--device",     default=None, help="cpu / cuda (auto-detect if omitted)")
    parser.add_argument("--tile",        default=None, help="Restrict to one S2 tile_id (e.g. 54LWH)")
    parser.add_argument("--checkpoint",  default=None, help="Load checkpoint from this dir instead of the default output dir")
    parser.add_argument("--end-year",    type=int, default=None, help="Reference year for recency weighting (default: latest year in data)")
    parser.add_argument("--decay",       type=float, default=0.7, help="Exponential decay rate per year back from end-year (default 0.7)")
    args = parser.parse_args()

    run(
        location_id=args.location,
        do_train=args.train,
        scl_purity_min=args.scl_purity,
        n_epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        tile_id=args.tile,
        checkpoint_dir=Path(args.checkpoint) if args.checkpoint else None,
        end_year=args.end_year,
        decay=args.decay,
    )
