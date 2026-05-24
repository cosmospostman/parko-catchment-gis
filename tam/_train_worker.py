"""tam/_train_worker.py — Subprocess worker for train_tam.

Run as:
    python -m tam._train_worker <work_dir> <out_dir> <experiment>

work_dir and out_dir are the same directory (the model output dir).
Reads:
  <work_dir>/pixel_df_cache.parquet  — full pixel_df, already filtered+zscored
  <work_dir>/pixel_df_pixel_coords.parquet
  <work_dir>/pixel_df_band_summaries.parquet  (optional)
  <work_dir>/worker_args.json        — labels, cfg dict, device

The parent process spawns this as a subprocess so that all Polars/jemalloc
allocations (TAMDataset construction, training loop) are freed when this
process exits — the OS reclaims all arenas unconditionally.
"""

from __future__ import annotations

import gc
import importlib
import json
import logging
import sys
from pathlib import Path

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


def main(work_dir: Path, out_dir: Path, experiment: str) -> None:
    args_path = work_dir / "worker_args.json"
    with open(args_path) as f:
        worker_args = json.load(f)

    logger.info("Worker start: RSS=%.1f GB", _rss_gb())

    # Read all dataframes via PyArrow so they live in PyArrow's allocator, not
    # jemalloc. pl.from_arrow wraps PyArrow buffers zero-copy (no jemalloc alloc).
    pixel_df = pl.from_arrow(pq.read_table(work_dir / "pixel_df_cache.parquet"))
    logger.info(
        "pixel_df loaded: %d rows × %d cols  estimated_size=%.1f GB  RSS=%.1f GB",
        len(pixel_df), pixel_df.width,
        pixel_df.estimated_size() / 1e9,
        _rss_gb(),
    )

    labels: dict[str, float] = {k: float(v) for k, v in worker_args["labels"].items()}
    pixel_coords = pl.from_arrow(pq.read_table(work_dir / "pixel_df_pixel_coords.parquet"))

    band_summaries: pl.DataFrame | None = None
    bs_path = work_dir / "pixel_df_band_summaries.parquet"
    if bs_path.exists():
        band_summaries = pl.from_arrow(pq.read_table(bs_path))
        logger.info("Band summaries loaded: %d pixels", len(band_summaries))

    from tam.core.config import TAMConfig
    from tam.core.train import train_tam

    cfg_dict = worker_args["cfg"]
    # TAMConfig fields that are lists in JSON need to stay as lists (not tuples),
    # but tuple fields (val_sites, val_region_ids, etc.) are stored as lists in JSON.
    _tuple_fields = {"val_sites", "val_region_ids", "stride_exclude_sites", "s1_feature_cols", "feature_cols_override"}
    for _f in _tuple_fields:
        if _f in cfg_dict and cfg_dict[_f] is not None:
            cfg_dict[_f] = tuple(cfg_dict[_f])
    cfg = TAMConfig(**{k: v for k, v in cfg_dict.items() if k in TAMConfig.__dataclass_fields__})

    logger.info("Calling train_tam: RSS=%.1f GB", _rss_gb())

    # Use the holder+closure pattern so del pixel_df inside train_tam hits refcount=0.
    _pixel_df_holder = [pixel_df]
    del pixel_df
    gc.collect()

    def _call() -> tuple:
        return train_tam(
            pixel_df=_pixel_df_holder.pop(),
            labels=labels,
            pixel_coords=pixel_coords,
            out_dir=out_dir,
            cfg=cfg,
            device=worker_args.get("device"),
            precomputed_band_summaries=band_summaries,
        )

    _, best_val_auc = _call()
    logger.info("Worker done: best_val_auc=%.4f  RSS=%.1f GB", best_val_auc, _rss_gb())


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <work_dir> <out_dir> <experiment>", file=sys.stderr)
        sys.exit(1)
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
