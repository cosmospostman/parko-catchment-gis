"""tam/pipeline.py — TAM train/score CLI.

Usage
-----
    # Train a named experiment
    python -m tam.pipeline train --experiment v1_spectral

    # Score any location with an existing checkpoint
    python -m tam.pipeline score --checkpoint outputs/tam-v1_spectral \\
        --location longreach --years 2022 2023 2024
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tam.utils import label_pixels, save_pixel_ranking, summarise
from signals._shared import ensure_pixel_sorted
from tam.core.config import TAMConfig
from tam.core.score import score_location_years, score_tiles_chunked
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
        available = set(pf.schema_arrow.names)
        # Always load source/vh/vv when present — needed for S1 global features
        # even when use_s1=False (time series S1 disabled but globals still wanted).
        s1_cols = [c for c in ("source", "vh", "vv", "scl") if c in available]
        # B08/B04 needed by compute_global_features (noise filter + S2 globals)
        # even in S1-only experiments where they are not model input features.
        s2_global_cols = [c for c in ("B08", "B04") if c in available and c not in exp.feature_cols]
        base_cols = ["point_id", "lon", "lat", "date", "scl_purity"]
        read_cols = base_cols + [c for c in exp.feature_cols if c in available] + s1_cols + s2_global_cols
        for rg in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(rg, columns=read_cols)
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

    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / f"tam-{exp.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_kwargs = dict(exp.train_kwargs)
    model_kwargs = dict(exp.model_kwargs)

    overrides = {k: v for k, v in {
        "lr":                    args.lr,
        "dropout":               args.dropout,
        "n_layers":              args.n_layers,
        "d_model":               args.d_model,
        "obs_dropout_min":       args.obs_dropout_min,
        "spatial_stride":        args.spatial_stride,
        "band_noise_std":        args.band_noise_std,
        "weight_decay":          args.weight_decay,
        "presence_min_dry_ndvi": args.presence_min_dry_ndvi,
        "presence_min_rec_p":    args.presence_min_rec_p,
        "presence_grass_nir_cv": args.presence_grass_nir_cv,
        "s1_despeckle_window":   args.s1_despeckle_window,
    }.items() if v is not None}
    if args.val_sites:
        overrides["val_sites"] = tuple(args.val_sites)
    if args.stride_exclude_sites:
        overrides["stride_exclude_sites"] = tuple(args.stride_exclude_sites)
    positional = {"n_epochs", "patience", "scl_purity_min"}
    cfg = TAMConfig(
        n_epochs=args.epochs or train_kwargs.pop("n_epochs", TAMConfig.__dataclass_fields__["n_epochs"].default),
        patience=args.patience or train_kwargs.pop("patience", TAMConfig.__dataclass_fields__["patience"].default),
        scl_purity_min=args.scl_purity,
        **{k: v for k, v in model_kwargs.items() if k in TAMConfig.__dataclass_fields__ and k not in overrides},
        **{k: v for k, v in train_kwargs.items() if k in TAMConfig.__dataclass_fields__ and k not in overrides and k not in positional},
        **overrides,
    )

    _, best_val_auc = train_tam(
        pixel_df=pixel_df[pixel_df["point_id"].isin(labels.index)],
        labels=labels,
        pixel_coords=pixel_coords,
        out_dir=out_dir,
        cfg=cfg,
        device=args.device,
    )
    logger.info("Checkpoint saved to %s  best_val_auc=%.3f", out_dir, best_val_auc)


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
    if args.years:
        tile_paths_by_year = {y: ps for y, ps in tile_paths_by_year.items() if y in args.years}
    if not tile_paths_by_year:
        logger.error("No parquets found for years %s", args.years)
        sys.exit(1)

    years = sorted(tile_paths_by_year)
    end_year = max(years)
    if getattr(args, "out", None):
        out_csv = Path(args.out)
    else:
        out_csv = PROJECT_ROOT / "outputs" / loc.id / f"{checkpoint_dir.name}.csv"
    out_dir = out_csv.parent
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

    with open(checkpoint_dir / "tam_config.json") as _fh:
        _cfg_dict = json.load(_fh)
    _n_bands = _cfg_dict.get("n_bands", 13)
    s1_only = _n_bands == 4
    pixel_zscore = _cfg_dict.get("pixel_zscore", False)
    s1_despeckle_window = _cfg_dict.get("s1_despeckle_window", 0)

    if getattr(args, "out_parquet", False):
        # Build {tile_id: [(year, pixel-sorted-path), ...]} for score_tiles_chunked
        tile_year_map: dict[str, list[tuple[int, Path]]] = {}
        for y, paths in sorted(tile_paths_by_year.items()):
            for p in paths:
                tid = p.stem  # e.g. "54LWH"
                tile_year_map.setdefault(tid, []).append((y, ensure_pixel_sorted(p)))

        parquet_out_dir = (
            PROJECT_ROOT / "outputs" / loc.id / checkpoint_dir.name / str(end_year)
        )
        logger.info(
            "Scoring %d tiles → parquet shards in %s ...",
            len(tile_year_map), parquet_out_dir,
        )
        final_paths = score_tiles_chunked(
            tile_year_parquets=tile_year_map,
            model=model,
            band_mean=band_mean,
            band_std=band_std,
            out_dir=parquet_out_dir,
            scl_purity_min=args.scl_purity,
            device=args.device,
            end_year=end_year,
            decay=args.decay,
            batch_size=args.batch_size,
            n_tile_workers=getattr(args, "n_tile_workers", 1),
            s1_only=s1_only,
            s1_despeckle_window=s1_despeckle_window,
        )
        logger.info("Done — %d tile parquets in %s", len(final_paths), parquet_out_dir)
        return

    logger.info("Scoring %d tile-year parquets ...", len(year_parquets))
    scores = score_location_years(
        year_parquets=year_parquets,
        model=model,
        band_mean=band_mean,
        band_std=band_std,
        scl_purity_min=args.scl_purity,
        device=args.device,
        end_year=end_year,
        decay=args.decay,
        batch_size=args.batch_size,
        n_total_pixels=len(pixel_coords),
        s1_only=s1_only,
        pixel_zscore=pixel_zscore,
        s1_despeckle_window=s1_despeckle_window,
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
    p.add_argument("--years",       type=int, nargs="+", default=None, metavar="YEAR")
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
    p_train.add_argument("--n-layers",        type=int,   default=None)
    p_train.add_argument("--d-model",         type=int,   default=None)
    p_train.add_argument("--spatial-stride",        type=int,  default=None,
                         help="Thin training pixels spatially — use every Nth pixel per region (default: 1, no thinning)")
    p_train.add_argument("--stride-exclude-sites",  nargs="+", default=None,
                         help="Site prefixes exempt from spatial stride (e.g. --stride-exclude-sites barkly stockholm)")
    p_train.add_argument("--band-noise-std",  type=float, default=None,
                         help="Std of per-window band offset in normalised space (default: 0.5)")
    p_train.add_argument("--weight-decay",    type=float, default=None)
    p_train.add_argument("--val-sites",       nargs="+",  default=None,
                         help="Hold out these sites entirely for validation (e.g. --val-sites frenchs barcoorah)")
    p_train.add_argument("--presence-min-dry-ndvi", type=float, default=None,
                         help="Min dry-season median NDVI for presence pixels (default: 0.10)")
    p_train.add_argument("--presence-min-rec-p",    type=float, default=None,
                         help="Min NDVI amplitude for presence pixels (default: 0.20)")
    p_train.add_argument("--s1-despeckle-window",  type=int,   default=None,
                         help="Temporal despeckle window for S1 (rolling median over N acquisitions). 0=off, default=3. Other reasonable values: 5, 7.")
    p_train.add_argument("--presence-grass-nir-cv", type=float, default=None,
                         help="Max NIR CV for presence pixels — filters grass (default: 0.20)")
    p_train.add_argument("--scl-purity", type=float, default=0.5)
    p_train.add_argument("--device", default=None)

    # --- score ---
    p_score = sub.add_parser("score", help="Score a location with an existing checkpoint")
    p_score.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    p_score.add_argument("--location",   required=True, help="Location ID")
    p_score.add_argument("--out",          default=None, help="Output CSV path (overrides default in checkpoint dir)")
    p_score.add_argument("--out-parquet", action="store_true", help="Write tile-sharded parquet instead of CSV")
    p_score.add_argument("--n-tile-workers", type=int, default=1, help="Parallel tile workers for parquet output (default: 1)")
    _add_common_score_args(p_score)

    args = parser.parse_args()

    if args.command == "train":
        _cmd_train(args)
    elif args.command == "score":
        _cmd_score(args)
