"""tam/pipeline.py — TAM train/score CLI.

Usage
-----
    # Train a named experiment
    python -m tam.pipeline train --experiment v1_spectral

    # Score any location with an existing checkpoint
    python -m tam.pipeline score --checkpoint outputs/tam-v1_spectral \\
        --location longreach-8x8km --end-year 2024

    # Legacy: train and score with a location (backwards compat)
    python -m tam.pipeline legacy --location longreach --train
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tam.utils import label_pixels, save_pixel_ranking, summarise
from signals._shared import ensure_pixel_sorted
from tam.core.config import TAMConfig
from tam.core.dataset import BAND_COLS
from tam.core.score import score_pixels_chunked
from tam.core.train import load_tam, train_tam
from utils.location import get as get_location

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# train subcommand
# ---------------------------------------------------------------------------

def _cmd_train(args: argparse.Namespace) -> None:
    """Train a named experiment, loading pixels from data/training/tiles/."""
    import importlib
    import pyarrow.parquet as pq

    exp_module = importlib.import_module(f"tam.experiments.{args.experiment}")
    exp = exp_module.EXPERIMENT

    from utils.regions import select_regions
    from utils.training_collector import tile_ids_for_regions, tile_parquet_path

    regions = select_regions(exp.region_ids)
    tile_ids = tile_ids_for_regions(exp.region_ids)
    logger.info("Experiment: %s  tiles: %d  regions: %s", exp.name, len(tile_ids), exp.region_ids)

    # Load labeled pixels from training tile parquets
    chunks: list[pd.DataFrame] = []
    for tid in tile_ids:
        path = tile_parquet_path(tid)
        if not path.exists():
            logger.warning("Missing tile parquet: %s — run training_collector first", path)
            continue
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["point_id", "lon", "lat", "date", "scl_purity"] + exp.feature_cols)
            chunks.append(tbl.to_pandas())

    if not chunks:
        logger.error("No training data found for experiment %s", exp.name)
        sys.exit(1)

    pixel_df = pd.concat(chunks, ignore_index=True)

    # Apply date filter
    dates = pd.to_datetime(pixel_df["date"])
    dates = pd.to_datetime(pixel_df["date"])
    pixel_df["year"] = dates.dt.year
    pixel_df["doy"]  = dates.dt.day_of_year

    # Per-region year pinning: drop observations outside [year-5, year] for
    # regions that carry a `year` field (guards against post-clearance imagery).
    pinned_regions = [r for r in regions if r.year is not None]
    if pinned_regions:
        # For each pixel_id, find which pinned region it falls in and derive
        # the allowed year window; pixels in no pinned region are kept as-is.
        keep_mask = pd.Series(True, index=pixel_df.index)
        for region in pinned_regions:
            lon_min, lat_min, lon_max, lat_max = region.bbox
            in_region = (
                pixel_df["lon"].between(lon_min, lon_max) &
                pixel_df["lat"].between(lat_min, lat_max)
            )
            out_of_window = in_region & (
                (pixel_df["year"] < region.year - 5) |
                (pixel_df["year"] > region.year)
            )
            keep_mask &= ~out_of_window
        n_dropped = (~keep_mask).sum()
        if n_dropped:
            logger.info("Year-pinning: dropped %d observations outside region windows", n_dropped)
        pixel_df = pixel_df[keep_mask].reset_index(drop=True)

    # Build labels from regions
    pixel_coords = (
        pixel_df[["point_id", "lon", "lat"]]
        .drop_duplicates("point_id")
        .reset_index(drop=True)
    )
    labelled = label_pixels(pixel_coords, regions)
    labelled_known = labelled.dropna(subset=["is_presence"])
    labels = labelled_known.set_index("point_id")["is_presence"].map({True: 1.0, False: 0.0})

    logger.info(
        "Labeled pixels — presence: %d  absence: %d",
        (labels == 1).sum(), (labels == 0).sum(),
    )

    out_dir = PROJECT_ROOT / "outputs" / f"tam-{exp.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_kwargs = dict(exp.train_kwargs)
    model_kwargs = dict(exp.model_kwargs)

    cfg = TAMConfig(
        n_epochs=args.epochs or 100,
        patience=args.patience or 15,
        scl_purity_min=args.scl_purity,
        **{k: v for k, v in train_kwargs.items() if k in TAMConfig.__dataclass_fields__},
    )

    train_tam(
        pixel_df=pixel_df[pixel_df["point_id"].isin(labels.index)],
        labels=labels,
        pixel_coords=pixel_coords,
        out_dir=out_dir,
        cfg=cfg,
        device=args.device,
    )
    logger.info("Checkpoint saved to %s", out_dir)


# ---------------------------------------------------------------------------
# score subcommand
# ---------------------------------------------------------------------------

def _cmd_score(args: argparse.Namespace) -> None:
    """Score a location using an existing checkpoint."""
    import pyarrow.parquet as pq
    from concurrent.futures import ThreadPoolExecutor, as_completed

    checkpoint_dir = Path(args.checkpoint)
    loc = get_location(args.location)
    parquet = loc.parquet_path()

    if not parquet.exists():
        logger.error("Parquet not found: %s — run pixel_collector first", parquet)
        sys.exit(1)

    tile_id = getattr(args, "tile", None)
    out_csv = Path(args.out) if getattr(args, "out", None) else None
    out_dir = out_csv.parent if out_csv else checkpoint_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Location: %s  parquet: %s  checkpoint: %s", loc.name, parquet, checkpoint_dir)

    # Load pixel coords for labeling (cached sidecar to avoid scanning 2000+ row groups)
    tile_prefix = f"_{tile_id}_" if tile_id else None
    coords_cache = loc.coords_cache_path(tile_id=tile_id)

    if coords_cache and coords_cache.exists():
        logger.info("Loading pixel coords from cache ...")
        pixel_coords = pd.read_parquet(coords_cache)
    else:
        logger.info("Resolving labels ...")
        pf_coords = pq.ParquetFile(parquet)
        read_coord_cols = ["point_id", "lon", "lat"] + (["item_id"] if tile_id else [])
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
        if coords_cache:
            pixel_coords.to_parquet(coords_cache, index=False)
            logger.info("Cached pixel coords to %s", coords_cache)

    logger.info("Unique pixels after tile filter: %d", len(pixel_coords))

    labelled = label_pixels(pixel_coords, loc)

    parquet_sorted = ensure_pixel_sorted(parquet)
    logger.info("Loading checkpoint from %s ...", checkpoint_dir)
    model, band_mean, band_std = load_tam(checkpoint_dir, device=args.device)

    logger.info("Scoring all pixels (chunked) ...")
    scores = score_pixels_chunked(
        parquet_sorted, model, band_mean, band_std,
        scl_purity_min=args.scl_purity,
        device=args.device,
        tile_id=tile_id,
        end_year=args.end_year,
        decay=args.decay,
        batch_size=args.batch_size,
        n_total_pixels=len(pixel_coords),
    )

    scored = pixel_coords.merge(scores, on="point_id", how="left")
    scored = scored.merge(labelled[["point_id", "is_presence"]], on="point_id", how="left")
    scored["rank"] = scored["prob_tam"].rank(ascending=False, method="first").astype("Int64")

    summarise(scored, loc, prob_col="prob_tam")
    csv_path = out_csv if out_csv else out_dir / "tam_pixel_ranking.csv"
    save_pixel_ranking(scored, csv_path, features=["prob_tam"])
    logger.info("Done — outputs in %s", out_dir)


# ---------------------------------------------------------------------------
# legacy subcommand (backwards compat: --location + optional --train)
# ---------------------------------------------------------------------------

def _cmd_legacy(args: argparse.Namespace) -> None:
    """Original single-location train+score flow."""
    import pyarrow.parquet as pq
    from concurrent.futures import ThreadPoolExecutor, as_completed

    loc = get_location(args.location)
    parquet = loc.parquet_path()
    if not parquet.exists():
        logger.error("Parquet not found: %s — run pixel_collector first", parquet)
        sys.exit(1)

    tile_id = getattr(args, "tile", None)
    out_dir = PROJECT_ROOT / "outputs" / f"tam-{loc.id}"
    if tile_id:
        out_dir = out_dir / tile_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Location: %s  parquet: %s  tile: %s", loc.name, parquet, tile_id or "all")

    logger.info("Resolving labels ...")
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

    labelled = label_pixels(pixel_coords, loc)
    labelled_known = labelled.dropna(subset=["is_presence"])
    labels = labelled_known.set_index("point_id")["is_presence"].map(
        {True: 1.0, False: 0.0}
    )
    logger.info(
        "Labeled pixels — presence: %d  absence: %d",
        (labels == 1).sum(), (labels == 0).sum(),
    )

    parquet_sorted = ensure_pixel_sorted(parquet)

    if args.train:
        logger.info("Loading parquet (labeled pixels only for training) ...")
        labeled_ids = set(labels.index)
        pf = pq.ParquetFile(parquet_sorted)
        chunks = []
        for rg in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["point_id", "lon", "lat", "date", "scl_purity"] + BAND_COLS)
            pdf = tbl.to_pandas()
            pdf = pdf[pdf["point_id"].isin(labeled_ids)]
            if not pdf.empty:
                chunks.append(pdf)
        pixel_df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        pixel_df = pixel_df[pixel_df["scl_purity"] >= args.scl_purity]
        _dates = pd.to_datetime(pixel_df["date"])
        pixel_df["year"] = _dates.dt.year
        pixel_df["doy"]  = _dates.dt.day_of_year
        logger.info("Loaded %d observations for %d labeled pixels", len(pixel_df), pixel_df["point_id"].nunique())

        logger.info("Training TAM ...")
        cfg = TAMConfig(
            n_epochs=args.epochs,
            patience=args.patience,
            scl_purity_min=args.scl_purity,
        )
        train_tam(
            pixel_df=pixel_df,
            labels=labels,
            pixel_coords=pixel_coords,
            out_dir=out_dir,
            cfg=cfg,
            device=args.device,
        )

    checkpoint_dir = Path(args.checkpoint) if args.checkpoint else out_dir
    logger.info("Loading checkpoint from %s ...", checkpoint_dir)
    model, band_mean, band_std = load_tam(checkpoint_dir, device=args.device)

    logger.info("Scoring all pixels (chunked) ...")
    scores = score_pixels_chunked(
        parquet_sorted, model, band_mean, band_std,
        scl_purity_min=args.scl_purity, device=args.device, tile_id=tile_id,
        end_year=args.end_year, decay=args.decay, batch_size=args.batch_size,
    )

    scored = pixel_coords.merge(scores, on="point_id", how="left")
    scored = scored.merge(labelled[["point_id", "is_presence"]], on="point_id", how="left")
    scored["rank"] = scored["prob_tam"].rank(ascending=False, method="first").astype("Int64")

    summarise(scored, loc, prob_col="prob_tam")
    save_pixel_ranking(scored, out_dir / "tam_pixel_ranking.csv", features=["prob_tam"])
    logger.info("Done — outputs in %s", out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common_score_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--scl-purity",  type=float, default=0.5)
    p.add_argument("--device",      default=None, help="cpu / cuda (auto-detect if omitted)")
    p.add_argument("--end-year",    type=int, default=None)
    p.add_argument("--decay",       type=float, default=0.7)
    p.add_argument("--batch-size",  type=int, default=4096)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="TAM Parkinsonia classifier pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    p_train = sub.add_parser("train", help="Train a named experiment")
    p_train.add_argument("--experiment", required=True, help="Experiment module name (e.g. v1_spectral)")
    p_train.add_argument("--epochs",   type=int, default=None)
    p_train.add_argument("--patience", type=int, default=None)
    p_train.add_argument("--scl-purity", type=float, default=0.5)
    p_train.add_argument("--device", default=None)

    # --- score ---
    p_score = sub.add_parser("score", help="Score a location with an existing checkpoint")
    p_score.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    p_score.add_argument("--location",   required=True, help="Location ID")
    p_score.add_argument("--tile",       default=None, help="Restrict to one S2 tile_id")
    p_score.add_argument("--out",        default=None, help="Output CSV path (overrides default in checkpoint dir)")
    _add_common_score_args(p_score)

    # --- legacy ---
    p_legacy = sub.add_parser("legacy", help="Original single-location train+score flow")
    p_legacy.add_argument("--location",   required=True)
    p_legacy.add_argument("--train",      action="store_true")
    p_legacy.add_argument("--epochs",     type=int, default=100)
    p_legacy.add_argument("--patience",   type=int, default=15)
    p_legacy.add_argument("--tile",       default=None)
    p_legacy.add_argument("--checkpoint", default=None)
    _add_common_score_args(p_legacy)

    args = parser.parse_args()

    if args.command == "train":
        _cmd_train(args)
    elif args.command == "score":
        _cmd_score(args)
    elif args.command == "legacy":
        _cmd_legacy(args)
