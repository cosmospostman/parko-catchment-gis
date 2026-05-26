"""tam/_prep_worker.py — Preprocessing subprocess for train_tam.

Run as:
    python -m tam._prep_worker <work_dir> <out_dir> <experiment>

Reads:
  <work_dir>/pixel_df_cache.parquet
  <work_dir>/pixel_df_pixel_coords.parquet
  <work_dir>/pixel_df_band_summaries.parquet  (optional)
  <work_dir>/worker_args.json

Writes:
  <work_dir>/prep_train_pixel_df.parquet
  <work_dir>/prep_val_pixel_df.parquet
  <work_dir>/prep_results.json  — train_py_labels, val_py_labels, global_feat_df path

On exit all jemalloc arenas are reclaimed by the OS. The training worker
(_train_worker.py) then loads only the sliced parquets and builds datasets
from a clean ~0 GB baseline.
"""

from __future__ import annotations

import gc
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S,%f",
)
logger = logging.getLogger(__name__)


def _rss_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return float("nan")


def _log_rss(tag: str) -> None:
    logger.info("RSS %s: %.1f GB", tag, _rss_gb())


def main(work_dir: Path, out_dir: Path, experiment: str) -> None:
    from tam.core.config import TAMConfig
    from tam.core.constants import DRY_DOY_MIN as _DRY_DOY_MIN, DRY_DOY_MAX as _DRY_DOY_MAX
    from tam.core.dataset import BAND_COLS, S1_FEATURE_COLS, V9_FEATURE_COLS, lin_to_db
    from tam.core.global_features import GLOBAL_FEATURE_NAMES
    from tam.core.train import (
        _apply_presence_filter,
        _compute_band_summaries,
        _load_or_compute_global_features,
        _site_class,
        region_holdout_split,
        site_holdout_split,
        spatial_split,
    )

    args_path = work_dir / "worker_args.json"
    with open(args_path) as f:
        worker_args = json.load(f)

    logger.info("Prep worker start: RSS=%.1f GB", _rss_gb())

    # --- Load pixel_df -------------------------------------------------------
    _cache_path = work_dir / "pixel_df_cache.parquet"
    _cache_schema = pq.ParquetFile(_cache_path).schema_arrow.names
    _read_cols = [c for c in _cache_schema if c not in {"lon", "lat"}] or None
    _arrow_tbl = pq.read_table(_cache_path, columns=_read_cols)
    pixel_df = pl.from_arrow(_arrow_tbl)
    del _arrow_tbl
    gc.collect()
    _str_cols = [c for c in ("point_id", "source") if c in pixel_df.columns
                 and pixel_df[c].dtype == pl.String]
    if _str_cols:
        pixel_df = pixel_df.with_columns([pl.col(c).cast(pl.Categorical) for c in _str_cols])
        gc.collect()
    logger.info("pixel_df loaded: %d rows × %d cols  estimated_size=%.1f GB  RSS=%.1f GB",
                len(pixel_df), pixel_df.width, pixel_df.estimated_size() / 1e9, _rss_gb())

    labels: dict[str, float] = {k: float(v) for k, v in worker_args["labels"].items()}
    pixel_coords = pl.from_arrow(pq.read_table(work_dir / "pixel_df_pixel_coords.parquet"))

    band_summaries: pl.DataFrame | None = None
    bs_path = work_dir / "pixel_df_band_summaries.parquet"
    if bs_path.exists():
        band_summaries = pl.from_arrow(pq.read_table(bs_path))
        logger.info("Band summaries loaded: %d pixels", len(band_summaries))

    cfg_dict = worker_args["cfg"]
    _tuple_fields = {"val_sites", "val_region_ids", "stride_exclude_sites", "s1_feature_cols", "feature_cols_override"}
    for _f in _tuple_fields:
        if _f in cfg_dict and cfg_dict[_f] is not None:
            cfg_dict[_f] = tuple(cfg_dict[_f])
    cfg = TAMConfig(**{k: v for k, v in cfg_dict.items() if k in TAMConfig.__dataclass_fields__})

    # --- Label split ---------------------------------------------------------
    if cfg.val_region_ids:
        train_labels, val_labels = region_holdout_split(labels, cfg.val_region_ids)
    elif cfg.val_sites:
        train_labels, val_labels = site_holdout_split(labels, cfg.val_sites)
    else:
        train_labels, val_labels = spatial_split(labels, pixel_coords, cfg.val_frac)

    # --- Column trim ---------------------------------------------------------
    _feature_cols_override = list(cfg.feature_cols_override) if cfg.feature_cols_override else None
    _s1_feature_cols_override = list(cfg.s1_feature_cols) if cfg.s1_feature_cols else None
    _active_s1_cols = _s1_feature_cols_override or S1_FEATURE_COLS
    _s1_raw = ["vh", "vv"] if cfg.use_s1 not in (False, None) else []
    _feature_cols_base = set(_feature_cols_override) if _feature_cols_override else set(BAND_COLS)
    _keep_cols = {"point_id", "date", "year", "doy", "scl_purity", "scl", "source"} | \
                 _feature_cols_base | set(_active_s1_cols) | set(_s1_raw)
    _trim_cols = [c for c in pixel_df.columns if c in _keep_cols]
    if len(_trim_cols) < pixel_df.width:
        pixel_df = pixel_df.select(_trim_cols)
    gc.collect()
    _log_rss("after column trim")

    # --- SCL=6 exclusion -----------------------------------------------------
    if "scl" in pixel_df.columns and "source" in pixel_df.columns:
        n_before = len(pixel_df)
        pixel_df = pixel_df.filter(
            ~((pl.col("source") == "S2") & (pl.col("scl") == 6))
        )
        logger.info("SCL=6 exclusion: removed %d observations", n_before - len(pixel_df))
    if "scl" in pixel_df.columns:
        pixel_df = pixel_df.drop("scl")
    _log_rss("after SCL exclusion")

    # --- Global features -----------------------------------------------------
    _has_source = "source" in pixel_df.columns
    global_feat_df: pl.DataFrame | None = None
    if cfg.use_band_summaries:
        if band_summaries is not None:
            global_feat_df = band_summaries
            logger.info("Using precomputed band summaries: %d pixels, %d features",
                        len(global_feat_df), global_feat_df.width - 1)
        else:
            _s2_n = int((pixel_df["source"] == "S2").sum()) if _has_source else len(pixel_df)
            logger.info("Computing band summaries (%d rows) ...", _s2_n)
            global_feat_df = _compute_band_summaries(pixel_df, V9_FEATURE_COLS)
    elif cfg.n_global_features > 0:
        global_feat_df = _load_or_compute_global_features(pixel_df, out_dir, GLOBAL_FEATURE_NAMES)
    _log_rss("after global features")

    # --- Presence filter slim extracts ---------------------------------------
    _presence_filter_s1_slim: pl.DataFrame | None = None
    _presence_filter_s2_slim: pl.DataFrame | None = None
    if cfg.presence_min_vh_dry_db > -99 and _has_source and "vh" in pixel_df.columns:
        _doy_col = "doy" if "doy" in pixel_df.columns else None
        _date_col = "doy" if _doy_col else "date"
        _presence_pids = {pid for pid, lbl in labels.items() if lbl == 1.0}
        _presence_filter_s1_slim = (
            pixel_df.lazy()
            .filter(
                (pl.col("source") == "S1") &
                pl.col("point_id").is_in(_presence_pids) &
                pl.col(_date_col).is_between(_DRY_DOY_MIN, _DRY_DOY_MAX)
            )
            .select(["point_id", "year", "vh", _date_col])
            .collect()
        )
        _has_scl = "scl" in pixel_df.columns
        _s2_ndvi_cols = ["point_id", "year", _date_col] + (["scl"] if _has_scl else [])
        _ndvi_expr = (pl.col("B08") - pl.col("B04")) / (pl.col("B08") + pl.col("B04"))
        _presence_filter_s2_slim = (
            pixel_df.lazy()
            .filter(
                (pl.col("source") == "S2") &
                pl.col("point_id").is_in(_presence_pids) &
                pl.col(_date_col).is_between(_DRY_DOY_MIN, _DRY_DOY_MAX)
            )
            .with_columns(_ndvi_expr.alias("NDVI"))
            .select(_s2_ndvi_cols + ["NDVI"])
            .collect()
        )

    # --- Drop S1 rows if not needed by TAMDataset ----------------------------
    if "source" in pixel_df.columns and cfg.use_s1 not in (True, "mixed", "s1_only"):
        pixel_df = pixel_df.filter(pl.col("source") == "S2")
        gc.collect()
    logger.info("pixel_df after S1 drop: %d rows  estimated_size=%.1f GB",
                len(pixel_df), pixel_df.estimated_size() / 1e9)

    # --- Spatial stride ------------------------------------------------------
    noise_removed: dict[tuple[str, str], int] = {}
    stride_removed: dict[tuple[str, str], int] = {}
    if cfg.spatial_stride > 1:
        candidate_pids = set(train_labels.keys())
        coord_subset = pixel_coords.filter(pl.col("point_id").is_in(candidate_pids))
        if cfg.stride_exclude_sites:
            exc_sites = set(cfg.stride_exclude_sites)
            _site_map = {pid: _site_class(pid)[0] for pid in coord_subset["point_id"].unique().to_list()}
            coord_subset = coord_subset.with_columns(
                pl.col("point_id").replace(_site_map).alias("_site")
            )
            excluded_pids = set(coord_subset.filter(pl.col("_site").is_in(exc_sites))["point_id"].to_list())
            to_stride = coord_subset.filter(~pl.col("_site").is_in(exc_sites))
        else:
            excluded_pids: set[str] = set()
            to_stride = coord_subset
        strided_pids = set(
            to_stride.sort(["lat", "lon"])["point_id"].to_list()[::cfg.spatial_stride]
        ) | excluded_pids
        removed_by_stride = set(train_labels.keys()) - strided_pids
        for pid in removed_by_stride:
            key = _site_class(pid)
            stride_removed[key] = stride_removed.get(key, 0) + 1
        train_labels = {k: v for k, v in train_labels.items() if k in strided_pids}

    # --- Broadcast pixel labels → pixel-year labels --------------------------
    labeled_pids = set(labels.keys())
    pixel_years = (
        pixel_df.filter(pl.col("point_id").is_in(labeled_pids))
        .select(["point_id", "year"])
        .unique()
    )

    def _broadcast(lbl: dict[str, float]) -> dict[tuple[str, int], float]:
        lbl_df = pl.DataFrame({"point_id": list(lbl.keys()), "_label": list(lbl.values())}).with_columns(
            pl.col("point_id").cast(pl.Categorical)
        )
        joined = pixel_years.join(lbl_df, on="point_id", how="inner")
        return {
            (pid, yr): label
            for pid, yr, label in zip(
                joined["point_id"].to_list(),
                joined["year"].to_list(),
                joined["_label"].to_list(),
            )
        }

    train_py_labels = _broadcast(train_labels)
    val_py_labels   = _broadcast(val_labels)

    # --- Presence filter -----------------------------------------------------
    all_py = {**train_py_labels, **val_py_labels}
    pid_to_sc: dict[str, tuple[str, str]] = {pid: _site_class(pid) for pid in labeled_pids}
    raw_counts: dict[tuple[str, str], int] = Counter(pid_to_sc[k[0]] for k in all_py)

    if _presence_filter_s1_slim is not None:
        s1_slim = _presence_filter_s1_slim
        s2_slim = _presence_filter_s2_slim
        _doy_col = "doy" if "doy" in s1_slim.columns else None
        _date_col = "doy" if _doy_col else "date"
        _has_scl_slim = s2_slim is not None and "scl" in s2_slim.columns

        if len(s1_slim) > 0:
            doy_vals = s1_slim[_date_col].to_numpy()
            vh_lin   = s1_slim["vh"].cast(pl.Float32).to_numpy()
            vh_db    = lin_to_db(vh_lin)
            dry_mask = (doy_vals >= _DRY_DOY_MIN) & (doy_vals <= _DRY_DOY_MAX) & np.isfinite(vh_db)
            dry_s1   = pl.DataFrame({
                "point_id": s1_slim["point_id"].to_numpy()[dry_mask],
                "year":     s1_slim["year"].to_numpy()[dry_mask],
                "_vh_db":   vh_db[dry_mask].astype(np.float32),
            })
            _mean_vh = dry_s1.group_by(["point_id", "year"]).agg(pl.col("_vh_db").mean().alias("mean_vh"))
            mean_vh_dry_py: dict[tuple[str, int], float] = {
                (pid, yr): val
                for pid, yr, val in zip(
                    _mean_vh["point_id"].to_list(),
                    _mean_vh["year"].to_list(),
                    _mean_vh["mean_vh"].to_list(),
                )
            }
            mean_ndvi_dry_py: dict[tuple[str, int], float] | None = None
            if s2_slim is not None and len(s2_slim) > 0:
                s2_doy = s2_slim[_date_col].to_numpy()
                s2_dry_mask = (s2_doy >= _DRY_DOY_MIN) & (s2_doy <= _DRY_DOY_MAX)
                if _has_scl_slim:
                    s2_dry_mask &= np.isin(s2_slim["scl"].to_numpy(), [4.0, 5.0])
                dry_s2_df = s2_slim.filter(
                    pl.Series(s2_dry_mask) & pl.col("NDVI").is_not_null()
                )
                if len(dry_s2_df) > 0:
                    _mean_ndvi = dry_s2_df.group_by(["point_id", "year"]).agg(
                        pl.col("NDVI").mean().alias("mean_ndvi")
                    )
                    mean_ndvi_dry_py = {
                        (pid, yr): val
                        for pid, yr, val in zip(
                            _mean_ndvi["point_id"].to_list(),
                            _mean_ndvi["year"].to_list(),
                            _mean_ndvi["mean_ndvi"].to_list(),
                        )
                    }
            train_py_labels = _apply_presence_filter(train_py_labels, mean_vh_dry_py, cfg, pid_to_sc, noise_removed, mean_ndvi_dry_py)
            val_py_labels   = _apply_presence_filter(val_py_labels,   mean_vh_dry_py, cfg, pid_to_sc, noise_removed, mean_ndvi_dry_py)
            ndvi_note = (f", NDVI rescue: VH>={cfg.presence_ndvi_rescue_vh_db:.1f} dB"
                         f" & NDVI>={cfg.presence_ndvi_rescue_min:.2f}"
                         if mean_ndvi_dry_py is not None else "")
            logger.info("Presence filter (mean_vh_dry < %.1f dB%s): removed %d presence pixel-years",
                        cfg.presence_min_vh_dry_db, ndvi_note, sum(noise_removed.values()))
        del s1_slim, s2_slim
        gc.collect()
    _log_rss("after presence filter")

    # --- Slice and write train/val pixel_dfs ---------------------------------
    train_pids_ds = set(k[0] for k in train_py_labels)
    val_pids_ds   = set(k[0] for k in val_py_labels)

    # Slice global features
    if global_feat_df is not None and not cfg.use_band_summaries:
        global_feat_df = global_feat_df.select(
            ["point_id"] + global_feat_df.columns[1:cfg.n_global_features + 1]
        )

    if "source" in pixel_df.columns and cfg.use_s1 not in (True, "mixed", "s1_only"):
        pixel_df = pixel_df.drop("source")
        gc.collect()

    train_pixel_df = pixel_df.filter(pl.col("point_id").is_in(train_pids_ds))
    val_pixel_df   = pixel_df.filter(pl.col("point_id").is_in(val_pids_ds))
    del pixel_df
    gc.collect()
    _log_rss("after split + free pixel_df")

    # Write sliced parquets — sorted by (point_id, year, date) so TAMDataset
    # can skip its internal sort (saves ~10 GB peak RSS at dataset construction).
    # Cast Categorical back to String for parquet portability.
    def _write_df(df: pl.DataFrame, path: Path) -> None:
        cat_cols = [c for c in df.columns if df[c].dtype == pl.Categorical]
        out = df.with_columns([pl.col(c).cast(pl.String) for c in cat_cols]) if cat_cols else df
        sort_cols = [c for c in ("point_id", "year", "date") if c in out.columns]
        if sort_cols:
            out = out.sort(sort_cols)
        out.write_parquet(path)

    train_path = work_dir / "prep_train_pixel_df.parquet"
    val_path   = work_dir / "prep_val_pixel_df.parquet"
    _write_df(train_pixel_df, train_path)
    logger.info("Wrote train_pixel_df: %d rows  %.1f GB  RSS=%.1f GB",
                len(train_pixel_df), train_pixel_df.estimated_size() / 1e9, _rss_gb())
    del train_pixel_df
    gc.collect()

    _write_df(val_pixel_df, val_path)
    logger.info("Wrote val_pixel_df: %d rows  %.1f GB  RSS=%.1f GB",
                len(val_pixel_df), val_pixel_df.estimated_size() / 1e9, _rss_gb())
    del val_pixel_df
    gc.collect()

    # Write global features if present
    global_feat_path: str | None = None
    if global_feat_df is not None:
        global_feat_path = str(work_dir / "prep_global_feat_df.parquet")
        global_feat_df.write_parquet(global_feat_path)
        logger.info("Wrote global_feat_df: %d pixels  %d features", len(global_feat_df), global_feat_df.width - 1)

    # Serialise label dicts — keys are (point_id, year) tuples, JSON needs strings
    def _labels_to_json(d: dict[tuple[str, int], float]) -> dict[str, float]:
        return {f"{pid}\x00{yr}": lbl for (pid, yr), lbl in d.items()}

    prep_results = {
        "train_py_labels": _labels_to_json(train_py_labels),
        "val_py_labels":   _labels_to_json(val_py_labels),
        "global_feat_path": global_feat_path,
    }
    with open(work_dir / "prep_results.json", "w") as f:
        json.dump(prep_results, f)
    logger.info("Prep worker done: RSS=%.1f GB", _rss_gb())


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <work_dir> <out_dir> <experiment>", file=sys.stderr)
        sys.exit(1)
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
