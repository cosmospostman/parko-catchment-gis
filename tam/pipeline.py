"""tam/pipeline.py — TAM train/score CLI.

Usage
-----
    # Train a named experiment
    python -m tam.pipeline train --experiment v1_spectral

    # Score any location with an existing checkpoint
    python -m tam.pipeline score --checkpoint outputs/tam-v1_spectral \\
        --location longreach-8x8km --end-year 2024
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tam.utils import label_pixels, save_pixel_ranking, summarise
from signals._shared import ensure_pixel_sorted
from tam.core.config import TAMConfig
from tam.core.score import score_location_years
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

    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / f"tam-{exp.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_kwargs = dict(exp.train_kwargs)
    model_kwargs = dict(exp.model_kwargs)

    overrides = {k: v for k, v in {
        "lr":              args.lr,
        "dropout":         args.dropout,
        "n_layers":        args.n_layers,
        "obs_dropout_min": args.obs_dropout_min,
    }.items() if v is not None}
    cfg = TAMConfig(
        n_epochs=args.epochs or TAMConfig.__dataclass_fields__["n_epochs"].default,
        patience=args.patience or TAMConfig.__dataclass_fields__["patience"].default,
        scl_purity_min=args.scl_purity,
        **{k: v for k, v in train_kwargs.items() if k in TAMConfig.__dataclass_fields__},
        **overrides,
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

    tile_paths_by_year = loc.parquet_tile_paths()
    if not tile_paths_by_year:
        logger.error("No tile parquets found for %s — run `python cli/location.py fetch %s --years ...` first", loc.id, loc.id)
        sys.exit(1)
    if args.end_year:
        tile_paths_by_year = {y: ps for y, ps in tile_paths_by_year.items() if y <= args.end_year}
    if not tile_paths_by_year:
        logger.error("No parquets found for years up to %d", args.end_year)
        sys.exit(1)

    years = sorted(tile_paths_by_year)
    out_csv = Path(args.out) if getattr(args, "out", None) else None
    out_dir = out_csv.parent if out_csv else checkpoint_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Location: %s  years: %s  checkpoint: %s", loc.name, years, checkpoint_dir)

    # Load pixel coords from all tile parquets of the first year.
    first_year = years[0]
    coords_cache = loc.coords_cache_path(first_year)
    first_year_parquets = tile_paths_by_year[first_year]

    if coords_cache.exists():
        logger.info("Loading pixel coords from cache ...")
        pixel_coords = pd.read_parquet(coords_cache)
    else:
        logger.info("Resolving pixel coords from %d tile parquet(s) for year %d ...", len(first_year_parquets), first_year)
        coord_chunks = []
        for tile_parquet in first_year_parquets:
            pf_coords = pq.ParquetFile(tile_parquet)
            n_rg_coords = pf_coords.metadata.num_row_groups

            def _read_coord_rg(rg: int, _path: Path = tile_parquet) -> pd.DataFrame:
                pf = pq.ParquetFile(_path)
                chunk = pf.read_row_group(rg, columns=["point_id", "lon", "lat"]).to_pandas()
                return chunk.drop_duplicates("point_id")

            n_done = 0
            with ThreadPoolExecutor(max_workers=8) as ex:
                futures = {ex.submit(_read_coord_rg, rg): rg for rg in range(n_rg_coords)}
                for fut in as_completed(futures):
                    chunk = fut.result()
                    if not chunk.empty:
                        coord_chunks.append(chunk)
                    n_done += 1
                    if n_done % 100 == 0:
                        logger.info("  coords %d/%d row groups (%s)", n_done, n_rg_coords, tile_parquet.name)

        pixel_coords = (
            pd.concat(coord_chunks, ignore_index=True)
            .groupby("point_id")[["lon", "lat"]]
            .first()
            .reset_index()
        )
        pixel_coords.to_parquet(coords_cache, index=False)
        logger.info("Cached pixel coords to %s", coords_cache)

    logger.info("Unique pixels: %d", len(pixel_coords))

    labelled = label_pixels(pixel_coords, loc)

    # Flat list of (year, path) — one entry per tile per year.
    year_parquets = [
        (y, ensure_pixel_sorted(p))
        for y, paths in sorted(tile_paths_by_year.items())
        for p in paths
    ]

    logger.info("Loading checkpoint from %s ...", checkpoint_dir)
    model, band_mean, band_std = load_tam(checkpoint_dir, device=args.device)

    logger.info("Scoring %d tile-year parquets ...", len(year_parquets))
    scores = score_location_years(
        year_parquets=year_parquets,
        model=model,
        band_mean=band_mean,
        band_std=band_std,
        scl_purity_min=args.scl_purity,
        device=args.device,
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
    p_train.add_argument("--output-dir", default=None, help="Override output directory (default: outputs/tam-<experiment>)")
    p_train.add_argument("--epochs",           type=int,   default=None)
    p_train.add_argument("--patience",         type=int,   default=None)
    p_train.add_argument("--lr",               type=float, default=None)
    p_train.add_argument("--dropout",          type=float, default=None)
    p_train.add_argument("--obs-dropout-min",  type=int,   default=None,
                         help="Subsample each training window to Uniform(N, n) obs (default: off)")
    p_train.add_argument("--n-layers", type=int,   default=None)
    p_train.add_argument("--scl-purity", type=float, default=0.5)
    p_train.add_argument("--device", default=None)

    # --- score ---
    p_score = sub.add_parser("score", help="Score a location with an existing checkpoint")
    p_score.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    p_score.add_argument("--location",   required=True, help="Location ID")
    p_score.add_argument("--out",        default=None, help="Output CSV path (overrides default in checkpoint dir)")
    _add_common_score_args(p_score)

    args = parser.parse_args()

    if args.command == "train":
        _cmd_train(args)
    elif args.command == "score":
        _cmd_score(args)
