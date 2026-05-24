"""tam/pipeline.py — TAM train/score CLI.

Usage
-----
    # Train a named experiment
    python -m tam.pipeline train --experiment v1_spectral

    # Score any location with an existing checkpoint
    python -m tam.pipeline score --checkpoint outputs/models/tam-v1_spectral \\
        --location longreach --years 2022 2023 2024
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tam.utils import label_pixels, save_pixel_ranking, summarise
from utils.parquet_utils import ensure_pixel_sorted
from tam.core.config import TAMConfig
from tam.core.score import score_location_years, score_tiles_chunked
from tam.core.train import load_tam, train_tam
from utils.location import get as get_location

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# train subcommand
# ---------------------------------------------------------------------------

def _rss_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return float("nan")


def _pixel_df_cache_key(
    tile_specs: list[tuple[Path, list[str], int]],
    exp_name: str,
    feature_cols: list[str],
    region_ids: list[str],
    want_pixel_zscore: bool,
    want_s1_data: bool,
    load_stride: int,
) -> str:
    """Stable hash covering all inputs that determine the content of pixel_df_cache.parquet."""
    import hashlib as _hl
    parts: list[str] = [
        exp_name,
        ",".join(sorted(feature_cols)),
        ",".join(sorted(region_ids)),
        f"zscore={want_pixel_zscore}",
        f"s1={want_s1_data}",
        f"stride={load_stride}",
    ]
    for path, cols, _ in sorted(tile_specs, key=lambda x: str(x[0])):
        st = path.stat()
        parts.append(f"{path}:{st.st_mtime_ns}:{st.st_size}:{','.join(sorted(cols))}")
    return _hl.md5("\n".join(parts).encode()).hexdigest()


def _cmd_train(args: argparse.Namespace) -> None:
    """Train a named experiment, loading pixels from data/training/tiles/."""
    import importlib
    import subprocess
    import pyarrow.parquet as pq

    try:
        exp_module = importlib.import_module(f"tam.experiments.{args.experiment}")
    except ModuleNotFoundError:
        available = sorted(
            p.stem for p in Path("tam/experiments").glob("*.py") if not p.stem.startswith("_")
        )
        raise SystemExit(
            f"Unknown experiment '{args.experiment}'. Available: {', '.join(available)}"
        )
    exp = exp_module.EXPERIMENT

    from utils.regions import select_regions
    from utils.training_collector import tile_ids_for_regions, tile_parquet_path

    regions = select_regions(exp.region_ids)
    tile_ids = tile_ids_for_regions(exp.region_ids)
    logger.info("Experiment: %s  tiles: %d  regions: %s", exp.name, len(tile_ids), exp.region_ids)

    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / "models" / f"tam-{exp.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load labeled pixels from training tile parquets — row groups read in parallel.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _use_s1 = args.use_s1 if args.use_s1 is not None else exp.train_kwargs.get("use_s1", True)
    _want_s1_data = _use_s1 is not False

    # Build (path, read_cols, n_row_groups) list for tiles that exist.
    tile_specs: list[tuple[Path, list[str], int]] = []
    for tid in tile_ids:
        path = tile_parquet_path(tid)
        if not path.exists():
            logger.warning("Missing tile parquet: %s — run training_collector first", path)
            continue
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        s1_cols = [c for c in (
            ("source", "vh", "vv", "scl") if _want_s1_data else ("source", "scl")
        ) if c in available]
        # vh/vv always needed for the VH heuristic presence filter, even in S2-only runs.
        for _c in ("source", "vh", "vv"):
            if _c in available and _c not in s1_cols:
                s1_cols.append(_c)
        # B08/B04 needed by compute_global_features even in S1-only experiments.
        s2_global_cols = [c for c in ("B08", "B04") if c in available and c not in exp.feature_cols]
        base_cols = ["point_id", "lon", "lat", "date", "scl_purity"]
        read_cols = base_cols + [c for c in exp.feature_cols if c in available] + s1_cols + s2_global_cols
        tile_specs.append((path, read_cols, pf.metadata.num_row_groups))

    _want_pixel_zscore = exp.train_kwargs.get("pixel_zscore", False)
    _load_stride = args.spatial_stride or exp.train_kwargs.get("spatial_stride", 1) or 1

    # --- Cache check: skip tile loading if pixel_df_cache is up to date ------
    _cache_parquet = out_dir / "pixel_df_cache.parquet"
    _cache_key_path = out_dir / "pixel_df_cache.key"
    _cache_key = _pixel_df_cache_key(
        tile_specs, exp.name, list(exp.feature_cols), list(exp.region_ids),
        _want_pixel_zscore, _want_s1_data, _load_stride,
    )
    _cache_hit = (
        _cache_parquet.exists()
        and _cache_key_path.exists()
        and _cache_key_path.read_text().strip() == _cache_key
    )

    if _cache_hit:
        logger.info("pixel_df cache hit — skipping tile load (key=%s)", _cache_key[:12])
        # Load supporting files written alongside the cache.
        band_summaries: pl.DataFrame | None = None
        _bs_cache = out_dir / "pixel_df_band_summaries.parquet"
        if _bs_cache.exists():
            band_summaries = pl.from_arrow(pq.read_table(_bs_cache))
            logger.info("Band summaries from cache: %d pixels", len(band_summaries))
    else:
        def _read_tile(path: Path, cols: list[str], n_rg: int) -> pl.DataFrame:
            """Read all row groups of one tile with bounded parallelism."""
            pf = pq.ParquetFile(path)
            rg_chunks: list[pl.DataFrame] = []
            # 4 threads per tile overlaps I/O without blowing memory across tiles.
            n_workers = min(4, n_rg)
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futures = {ex.submit(pf.read_row_group, rg, columns=cols): rg for rg in range(n_rg)}
                for fut in as_completed(futures):
                    rg_chunks.append(pl.from_arrow(fut.result()))
            return pl.concat(rg_chunks)

        # Filter to known training region point_ids before accumulating tiles.
        # point_id format is <region_id>_<row>_<col>; strip the trailing two numeric
        # segments to recover the region prefix and match against the experiment set.
        # Applying this per-tile keeps each tile's footprint small before concat.
        import re as _re
        _suffix_re = _re.compile(r"_\d+_\d+$")
        known_region_ids = set(exp.region_ids)

        def _filter_to_regions(df: pl.DataFrame) -> pl.DataFrame:
            return df.filter(
                df["point_id"].str.replace(_suffix_re.pattern, "", literal=False).is_in(known_region_ids)
            )

        # Load-time spatial stride: thin unique point_ids per tile before accumulating.
        # This mirrors the train_tam spatial_stride logic but runs early enough to keep
        # tile_dfs small — at 100M+ rows the accumulated object arrays OOM before concat.
        # stride=1 is a no-op. The same stride is re-applied in train_tam to labels only,
        # which is harmless (the strided pixels are simply absent from pixel_df).
        def _stride_tile(df: pl.DataFrame, stride: int) -> pl.DataFrame:
            if stride <= 1:
                return df
            # Sort by lat/lon for a geographically systematic (reproducible) sample.
            keep = set(
                df.select(["point_id", "lat", "lon"])
                .unique("point_id")
                .sort(["lat", "lon"])["point_id"]
                .gather(list(range(0, df["point_id"].n_unique(), stride)))
                .to_list()
            )
            return df.filter(pl.col("point_id").is_in(keep))

        # Read tiles one at a time, filter and thin immediately — peak memory per
        # iteration is one raw tile, not the accumulation of all tiles.
        # If band summaries are needed, compute them per tile here so we never need
        # an S2-only copy of the full pixel_df later (avoids a ~25 GB transient spike).
        _want_band_summaries = exp.train_kwargs.get("use_band_summaries", True)
        _bs_feature_cols: list[str] | None = None
        if _want_band_summaries:
            from tam.core.train import _compute_band_summaries
            from tam.core.dataset import V9_FEATURE_COLS as _V9_FEATURE_COLS
            _bs_feature_cols = _V9_FEATURE_COLS

        _zscore_feature_cols: list[str] | None = (
            list(exp.train_kwargs["feature_cols_override"]) if "feature_cols_override" in exp.train_kwargs
            else list(exp.feature_cols)
        ) if _want_pixel_zscore else None

        def _apply_pixel_zscore(df: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
            """Z-score each pixel's S2 band values by its own multi-year mean/std.

            All rows for a given point_id reside in the same tile parquet, so
            computing over the concat'd frame is equivalent to per-tile computation.

            Implementation: sort by point_id codes (int32 when Categorical, avoids
            string comparison), then use np.add.reduceat for vectorised per-pixel
            mean/std without a group_by+join (which leaves phantom jemalloc arenas).
            """
            s2_cols = [c for c in feature_cols if c in df.columns]
            if not s2_cols:
                return df

            has_source = "source" in df.columns
            if has_source:
                s2_mask_arr = (df["source"] == "S2").to_numpy()
            else:
                s2_mask_arr = np.ones(len(df), dtype=bool)

            # Sort by point_id. If Categorical, sort by int32 codes (faster, avoids
            # materialising the full string array for argsort).
            pid_col = df["point_id"]
            if pid_col.dtype == pl.Categorical:
                sort_key = pid_col.to_physical().to_numpy()  # int32 codes
            else:
                sort_key = pid_col.to_numpy()
            sort_idx = np.argsort(sort_key, kind="stable")
            inv_idx = np.empty_like(sort_idx)
            inv_idx[sort_idx] = np.arange(len(df))

            pid_sorted = sort_key[sort_idx]
            s2_mask_sorted = s2_mask_arr[sort_idx]

            # S2-only rows in sorted order; compute per-pixel boundaries.
            s2_idx_sorted = np.where(s2_mask_sorted)[0]
            s2_pid_sorted = pid_sorted[s2_idx_sorted]
            s2_bounds = np.nonzero(s2_pid_sorted[1:] != s2_pid_sorted[:-1])[0] + 1 if len(s2_pid_sorted) > 1 else np.array([], dtype=np.int64)
            s2_starts = np.concatenate([[0], s2_bounds])
            s2_ns = np.diff(np.concatenate([s2_starts, [len(s2_pid_sorted)]]))
            del pid_sorted, s2_pid_sorted, s2_bounds  # free before per-col loop

            updated_cols: dict[str, np.ndarray] = {}
            for c in s2_cols:
                arr = df[c].to_numpy().astype(np.float32, copy=False)
                arr_sorted = arr[sort_idx].astype(np.float64)  # float64 for accumulation (ddof=1)
                s2_vals = arr_sorted[s2_idx_sorted]
                group_sums = np.add.reduceat(s2_vals, s2_starts)
                group_means = group_sums / s2_ns
                mean_per_s2 = np.repeat(group_means, s2_ns)
                diffs_sq = (s2_vals - mean_per_s2) ** 2
                group_var_sums = np.add.reduceat(diffs_sq, s2_starts)
                group_stds = np.maximum(np.sqrt(group_var_sums / np.maximum(s2_ns - 1, 1)), 1e-6)
                std_per_s2 = np.repeat(group_stds, s2_ns)
                arr_sorted[s2_idx_sorted] = (s2_vals - mean_per_s2) / std_per_s2
                updated_cols[c] = arr_sorted[inv_idx].astype(np.float32)

            return df.with_columns([
                pl.Series(c, updated_cols[c]) for c in s2_cols
            ])

        tile_dfs: list[pl.DataFrame] = []
        band_summary_dfs: list[pl.DataFrame] = []
        for path, cols, n_rg in tile_specs:
            logger.info("Loading tile %s (%d row groups) ...", path.name, n_rg)
            tile_df = _stride_tile(_filter_to_regions(_read_tile(path, cols, n_rg)), _load_stride)
            if len(tile_df) > 0:
                if _want_band_summaries and _bs_feature_cols is not None:
                    band_summary_dfs.append(_compute_band_summaries(tile_df, _bs_feature_cols))
                # Cast point_id to Categorical now so the accumulated tile_dfs use
                # 4 bytes/row instead of ~29 bytes/row for the string column.
                if "point_id" in tile_df.columns:
                    tile_df = tile_df.with_columns(pl.col("point_id").cast(pl.Categorical))
                tile_dfs.append(tile_df)
            del tile_df
            gc.collect()
            logger.info("  RSS after tile: %.1f GB", _rss_gb())

        if not tile_dfs:
            logger.error("No training data found for experiment %s", exp.name)
            sys.exit(1)

        band_summaries = None
        if band_summary_dfs:
            band_summaries = pl.concat(band_summary_dfs)
            del band_summary_dfs
            gc.collect()
            logger.info("Band summaries precomputed per tile: %d pixels", len(band_summaries))

        pixel_df = pl.concat(tile_dfs)
        del tile_dfs
        gc.collect()
        logger.info(
            "After concat: %d rows  estimated_size=%.1f GB  RSS=%.1f GB",
            len(pixel_df), pixel_df.estimated_size() / 1e9, _rss_gb(),
        )

        # Apply pixel zscore once on the full frame rather than per-tile.
        # Per-tile zscore leaves phantom jemalloc arenas after each tile that never
        # return to the OS; across 15 tiles this accumulates ~15-20 GB of phantom RSS.
        # A single pass on the concat'd frame incurs the arena cost only once.
        # point_id is already Categorical from the per-tile cast above, so np.argsort
        # on the codes (int32) is faster and cheaper than sorting strings.
        if _want_pixel_zscore and _zscore_feature_cols is not None:
            logger.info("Applying pixel zscore on full frame (%d rows)  RSS=%.1f GB ...", len(pixel_df), _rss_gb())
            pixel_df = _apply_pixel_zscore(pixel_df, _zscore_feature_cols)
            gc.collect()
            logger.info("After pixel zscore: estimated_size=%.1f GB  RSS=%.1f GB", pixel_df.estimated_size() / 1e9, _rss_gb())

        # Parse date column and derive year/doy.
        pixel_df = pixel_df.with_columns(
            pl.col("date").cast(pl.Utf8).str.to_date().alias("_date_parsed")
        ).with_columns(
            pl.col("_date_parsed").dt.year().alias("year"),
            pl.col("_date_parsed").dt.ordinal_day().alias("doy"),
        ).drop("_date_parsed")

        # Per-region year pinning: drop observations outside [min(years), max(years)]
        # (guards against post-clearance imagery; window is now explicit in the YAML).
        drop_mask = pl.lit(False)
        for region in regions:
            lon_min, lat_min, lon_max, lat_max = region.bbox
            in_region = (
                pl.col("lon").is_between(lon_min, lon_max) &
                pl.col("lat").is_between(lat_min, lat_max)
            )
            out_of_window = in_region & (
                (pl.col("year") < min(region.years)) |
                (pl.col("year") > max(region.years))
            )
            drop_mask = drop_mask | out_of_window
        pixel_df = pixel_df.filter(~drop_mask)
        gc.collect()

        # Build labels and filter pixel_df to labeled pixels only.
        pixel_coords = (
            pixel_df.select(["point_id", "lon", "lat"])
            .unique("point_id")
        )
        labelled = label_pixels(pixel_coords, regions)
        labelled_known = labelled.filter(pl.col("is_presence").is_not_null())
        labels: dict[str, float] = {
            row[0]: 1.0 if row[1] else 0.0
            for row in labelled_known.select(["point_id", "is_presence"]).iter_rows()
        }
        labeled_pids = set(labels.keys())
        pixel_df = pixel_df.filter(pl.col("point_id").is_in(labeled_pids))

        # Write cache — cast Categorical back to String so the worker can reload
        # without needing to reconstruct the category dictionary.
        logger.info("Writing pixel_df cache: %d rows  RSS=%.1f GB ...", len(pixel_df), _rss_gb())
        _str_cols = [c for c in pixel_df.columns if pixel_df[c].dtype == pl.Categorical]
        _cache_df = pixel_df.with_columns([pl.col(c).cast(pl.String) for c in _str_cols]) if _str_cols else pixel_df
        _cache_df.write_parquet(_cache_parquet)
        del _cache_df, pixel_df
        gc.collect()
        logger.info("pixel_df cache written: RSS=%.1f GB", _rss_gb())

        _cache_key_path.write_text(_cache_key)
        logger.info("Cache key written: %s", _cache_key[:12])

        pixel_coords.write_parquet(out_dir / "pixel_df_pixel_coords.parquet")
        if band_summaries is not None:
            band_summaries.write_parquet(out_dir / "pixel_df_band_summaries.parquet")
        with open(out_dir / "pixel_df_labels.json", "w") as _fh:
            json.dump(labels, _fh)

    # --- Build cfg (needed whether cache hit or miss) -------------------------
    if _cache_hit:
        with open(out_dir / "pixel_df_labels.json") as _fh:
            labels = {k: float(v) for k, v in json.load(_fh).items()}
        pixel_coords = pl.from_arrow(pq.read_table(out_dir / "pixel_df_pixel_coords.parquet"))

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
        "presence_min_vh_dry_db":     args.presence_min_vh_dry_db,
        "presence_ndvi_rescue_vh_db": args.presence_ndvi_rescue_vh_db,
        "presence_ndvi_rescue_min":   args.presence_ndvi_rescue_min,
        "s1_despeckle_window":   args.s1_despeckle_window,
        "batch_size":            args.batch_size,
        "doy_phase_shift":       args.doy_phase_shift,
        "use_s1":                args.use_s1,
    }.items() if v is not None}
    if args.val_sites:
        overrides["val_sites"] = tuple(args.val_sites)
    if exp.val_region_ids:
        overrides["val_region_ids"] = tuple(exp.val_region_ids)
    if args.stride_exclude_sites:
        overrides["stride_exclude_sites"] = tuple(args.stride_exclude_sites)
    # Auto-correct n_bands when --use-s1 is enabled from CLI but the experiment was S2-only.
    if _use_s1 not in (False, None) and "n_bands" not in overrides:
        from tam.core.dataset import S1_FEATURE_COLS
        _active_s1 = list(args.s1_features) if args.s1_features else S1_FEATURE_COLS
        if args.s1_features:
            overrides["s1_feature_cols"] = tuple(args.s1_features)
        base_n_bands = model_kwargs.get("n_bands", train_kwargs.get("n_bands", len(exp.feature_cols)))
        if base_n_bands == len(exp.feature_cols):
            overrides["n_bands"] = len(exp.feature_cols) + len(_active_s1)
    # S2 pixel zscore was already applied per-tile above. Tell train_tam/TAMDataset
    # not to repeat it. S1 pixel zscore (s1_vh/s1_vv) is a minor effect and is
    # skipped here; it can be restored later by moving lin_to_db upstream too.
    if _want_pixel_zscore:
        train_kwargs["pixel_zscore"] = False
        # Record that inference must apply pixel z-score even though the model
        # itself was trained on pre-zscored data (pixel_zscore=False in TAMConfig).
        train_kwargs["inference_pixel_zscore"] = True

    positional = {"n_epochs", "patience", "scl_purity_min"}
    cfg = TAMConfig(
        n_epochs=args.epochs or train_kwargs.pop("n_epochs", TAMConfig.__dataclass_fields__["n_epochs"].default),
        patience=args.patience or train_kwargs.pop("patience", TAMConfig.__dataclass_fields__["patience"].default),
        scl_purity_min=args.scl_purity,
        **{k: v for k, v in model_kwargs.items() if k in TAMConfig.__dataclass_fields__ and k not in overrides},
        **{k: v for k, v in train_kwargs.items() if k in TAMConfig.__dataclass_fields__ and k not in overrides and k not in positional},
        **overrides,
    )

    # --- Spawn training subprocess -------------------------------------------
    # The parent process has ~20 GB of phantom jemalloc arenas from tile loading
    # and DataFrame operations. These can never be returned to the OS in-process.
    # The worker starts fresh (zero phantom arenas), reads pixel_df via PyArrow
    # (not jemalloc), builds TAMDataset, trains, and saves the checkpoint. When
    # the worker exits the OS unconditionally reclaims all its memory.
    _worker_args = {
        "labels":  labels,
        "cfg":     {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")},
        "device":  args.device,
    }
    with open(out_dir / "worker_args.json", "w") as _fh:
        json.dump(_worker_args, _fh, default=lambda x: list(x) if isinstance(x, (tuple, set)) else x)

    logger.info("Spawning training worker (fresh process, zero phantom arenas) ...")
    result = subprocess.run(
        [sys.executable, "-m", "tam._train_worker",
         str(out_dir), str(out_dir), args.experiment],
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"Training worker exited with code {result.returncode}")

    if _want_pixel_zscore:
        _cfg_path = out_dir / "tam_config.json"
        with open(_cfg_path) as _fh:
            _saved = json.load(_fh)
        _saved["inference_pixel_zscore"] = True
        with open(_cfg_path, "w") as _fh:
            json.dump(_saved, _fh, indent=2)

    best_val_auc = 0.0
    _cfg_path = out_dir / "tam_config.json"
    if _cfg_path.exists():
        with open(_cfg_path) as _fh:
            best_val_auc = json.load(_fh).get("best_val_auc", 0.0)
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
        out_csv = PROJECT_ROOT / "outputs" / "scores" / loc.id / f"{checkpoint_dir.name}.csv"
    out_dir = out_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Location: %s  years: %s  checkpoint: %s", loc.name, years, checkpoint_dir)

    # Load pixel coords from all tile parquets of the first year.
    first_year = years[0]
    coords_cache = loc.coords_cache_path(first_year)
    first_year_parquets = tile_paths_by_year[first_year]

    if coords_cache.exists():
        logger.info("Loading pixel coords from cache ...")
        pixel_coords = pl.read_parquet(coords_cache)
    else:
        logger.info("Resolving pixel coords from %d tile parquet(s) for year %d ...", len(first_year_parquets), first_year)
        coord_chunks = []
        for tile_parquet in first_year_parquets:
            pf_coords = pq.ParquetFile(tile_parquet)
            n_rg_coords = pf_coords.metadata.num_row_groups

            def _read_coord_rg(rg: int, _path: Path = tile_parquet) -> pl.DataFrame:
                pf = pq.ParquetFile(_path)
                chunk = pl.from_arrow(pf.read_row_group(rg, columns=["point_id", "lon", "lat"]))
                return chunk.unique(subset=["point_id"])

            n_done = 0
            with ThreadPoolExecutor(max_workers=8) as ex:
                futures = {ex.submit(_read_coord_rg, rg): rg for rg in range(n_rg_coords)}
                for fut in as_completed(futures):
                    chunk = fut.result()
                    if not chunk.is_empty():
                        coord_chunks.append(chunk)
                    n_done += 1
                    if n_done % 100 == 0:
                        logger.info("  coords %d/%d row groups (%s)", n_done, n_rg_coords, tile_parquet.name)

        pixel_coords = (
            pl.concat(coord_chunks)
            .group_by("point_id")
            .agg([pl.col("lon").first(), pl.col("lat").first()])
        )
        pixel_coords.write_parquet(coords_cache)
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
    model, band_mean, band_std, global_feat_mean, global_feat_std = load_tam(checkpoint_dir, device=args.device)

    with open(checkpoint_dir / "tam_config.json") as _fh:
        _cfg_dict = json.load(_fh)
    # inference_pixel_zscore: True means the model was trained on pre-zscored data
    # and inference must apply pixel z-score before feeding the model.
    # Falls back to band-stats heuristic for old checkpoints that predate this flag.
    import numpy as _np
    if _cfg_dict.get("inference_pixel_zscore"):
        pixel_zscore = True
    else:
        pixel_zscore = _cfg_dict.get("pixel_zscore", False)
        _bm = band_mean[~_np.isnan(band_mean)]
        _bs = band_std[~_np.isnan(band_std)]
        if not pixel_zscore and len(_bm) > 0 and abs(float(_bm.mean())) < 0.1 and abs(float(_bs.mean()) - 1.0) < 0.1:
            logger.warning("band_mean≈0/band_std≈1 but no pixel_zscore flag — inferring inference_pixel_zscore=True from stats")
            pixel_zscore = True
    s1_despeckle_window = _cfg_dict.get("s1_despeckle_window", 0)

    # Determine scoring mode from config/weights
    use_s1_cfg = _cfg_dict.get("use_s1", False)
    s1_feature_cols_cfg = _cfg_dict.get("s1_feature_cols", None)  # e.g. ["s1_vh", "s1_vv"]
    s1_only = model.n_bands == 4
    mixed   = (not s1_only) and bool(use_s1_cfg)

    # S2 and S1 feature columns for mixed mode
    s2_feature_cols_cfg = _cfg_dict.get("feature_cols", None)   # the 14 S2 cols for v10
    s1_feature_cols_cfg = s1_feature_cols_cfg or (["s1_vh", "s1_vv"] if mixed else None)

    # summary_feature_cols: used for global band-summary head; must match global_feat_mean shape.
    # For mixed models feature_cols in config = S2 cols only (14), matching n_global_features//3.
    summary_feature_cols = s2_feature_cols_cfg

    # In non-mixed S2-only mode: validate feature_cols length against model.n_bands
    feature_cols = s2_feature_cols_cfg
    if not mixed and feature_cols is not None and len(feature_cols) != model.n_bands:
        logger.warning(
            "feature_cols in config has %d entries but model.n_bands=%d — ignoring for preprocessing",
            len(feature_cols), model.n_bands,
        )
        feature_cols = None

    if getattr(args, "out_parquet", False):
        # Build {tile_id: [(year, pixel-sorted-path), ...]} for score_tiles_chunked
        tile_year_map: dict[str, list[tuple[int, Path]]] = {}
        for y, paths in sorted(tile_paths_by_year.items()):
            for p in paths:
                tid = p.stem  # e.g. "54LWH"
                tile_year_map.setdefault(tid, []).append((y, ensure_pixel_sorted(p)))

        parquet_out_dir = (
            PROJECT_ROOT / "outputs" / "scores" / loc.id / checkpoint_dir.name / str(end_year)
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
        mixed=mixed,
        pixel_zscore=pixel_zscore,
        s1_despeckle_window=s1_despeckle_window,
        feature_cols=feature_cols,
        s2_feature_cols=s2_feature_cols_cfg if mixed else None,
        s1_feature_cols=s1_feature_cols_cfg if mixed else None,
        summary_feature_cols=summary_feature_cols,
        global_feat_mean=global_feat_mean,
        global_feat_std=global_feat_std,
    )
    scored = (
        pixel_coords
        .join(scores.select(["point_id", "prob_tam"]), on="point_id", how="left")
        .join(labelled.select(["point_id", "is_presence"]), on="point_id", how="left")
        .with_columns(
            pl.col("prob_tam")
            .rank(descending=True, method="ordinal")
            .cast(pl.Int64)
            .alias("rank")
        )
    )

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
    _log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, _log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="TAM Parkinsonia classifier pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    p_train = sub.add_parser("train", help="Train a named experiment")
    p_train.add_argument("--experiment", required=True, help="Experiment module name (e.g. v1_spectral)")
    p_train.add_argument("--output-dir", default=None, help="Override output directory (default: outputs/models/tam-<experiment>)")
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
    p_train.add_argument("--doy-phase-shift", type=lambda x: x.lower() in ("true", "1", "yes"),
                         default=None, metavar="BOOL",
                         help="Enable/disable DOY phase shift encoding (true/false)")
    p_train.add_argument("--presence-min-vh-dry-db", type=float, default=None,
                         help="Strict VH floor (dB): drop presence pixel-years unconditionally below this (default: -21.0)")
    p_train.add_argument("--presence-ndvi-rescue-vh-db", type=float, default=None,
                         help="Looser VH floor (dB) used only when dry-season NDVI passes (default: -23.0)")
    p_train.add_argument("--presence-ndvi-rescue-min", type=float, default=None,
                         help="Min dry-season NDVI to rescue a pixel-year between the two VH floors (default: 0.50)")
    p_train.add_argument("--s1-despeckle-window",  type=int,   default=None,
                         help="Temporal despeckle window for S1 (rolling median over N acquisitions). 0=off, default=3. Other reasonable values: 5, 7.")
    p_train.add_argument("--batch-size",           type=int,   default=None)
    p_train.add_argument("--scl-purity", type=float, default=0.5)
    p_train.add_argument("--use-s1", type=lambda x: x.lower() in ("true", "1", "yes") if x.lower() not in ("s1_only",) else "s1_only",
                         default=None, help="Enable S1: true/false or 's1_only'")
    p_train.add_argument("--s1-features", nargs="+", default=None,
                         metavar="COL", help="S1 feature cols to use (default: all 4). E.g. --s1-features s1_vh s1_vv")
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
