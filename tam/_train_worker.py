"""tam/_train_worker.py — Subprocess worker for train_tam.

Run as:
    python -m tam._train_worker <work_dir> <out_dir> <experiment>

work_dir is the pixel cache directory (shared across sweep runs).
out_dir is the per-run checkpoint directory.
Reads:
  <work_dir>/pixel_df_cache.parquet  — full pixel_df, already filtered+zscored
  <work_dir>/pixel_df_pixel_coords.parquet
  <work_dir>/pixel_df_band_summaries.parquet  (optional)
  <out_dir>/worker_args.json        — labels, cfg dict, device

The parent process spawns this as a subprocess so that all Polars/jemalloc
allocations (TAMDataset construction, training loop) are freed when this
process exits — the OS reclaims all arenas unconditionally.
"""

from __future__ import annotations

import gc
import json
import logging
import subprocess
import sys
from pathlib import Path

import torch
import polars as pl
import pyarrow.parquet as pq

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_out_dir = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path(".")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_out_dir / "train.log", mode="a"),
    ],
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
    args_path = out_dir / "worker_args.json"
    with open(args_path) as f:
        worker_args = json.load(f)

    logger.info("Worker start: RSS=%.1f GB", _rss_gb())

    # --- Stage 1: preprocessing subprocess -----------------------------------
    # Spawn _prep_worker as a separate process so that all Polars/jemalloc arenas
    # from preprocessing (column trim, SCL filter, presence filter, split) are
    # reclaimed by the OS on exit before dataset construction begins here.
    prep_results_path = out_dir / "prep_results.json"
    logger.info("Spawning prep worker ...")
    _prep_proc = subprocess.run(
        [sys.executable, "-m", "tam._prep_worker",
         str(work_dir), str(out_dir), experiment],
        check=True,
    )
    logger.info("Prep worker done: RSS=%.1f GB", _rss_gb())

    with open(prep_results_path) as f:
        prep_results = json.load(f)

    # Decode label dicts — keys were serialised as "point_id\x00year"
    def _labels_from_json(d: dict[str, float]) -> dict[tuple[str, int], float]:
        result = {}
        for key, lbl in d.items():
            pid, yr = key.split("\x00", 1)
            result[(pid, int(yr))] = lbl
        return result

    train_py_labels = _labels_from_json(prep_results["train_py_labels"])
    val_py_labels   = _labels_from_json(prep_results["val_py_labels"])

    # --- Stage 2: pass parquet paths to train_tam (no loading here) ----------
    # Pass paths rather than loaded DataFrames so this process stays at ~2 GB
    # RSS while the dataset subprocess reads the parquets in the child.
    from tam.core.config import TAMConfig
    from tam.core.train import train_tam

    cfg_dict = worker_args["cfg"]
    _tuple_fields = {"val_sites", "val_region_ids", "stride_exclude_sites", "s1_feature_cols", "feature_cols_override"}
    for _f in _tuple_fields:
        if _f in cfg_dict and cfg_dict[_f] is not None:
            cfg_dict[_f] = tuple(cfg_dict[_f])
    cfg = TAMConfig(**{k: v for k, v in cfg_dict.items() if k in TAMConfig.__dataclass_fields__})

    annual_feat_df: pl.DataFrame | None = None
    if prep_results.get("annual_feat_path"):
        import pyarrow.parquet as _pq2
        annual_feat_df = pl.from_arrow(_pq2.read_table(prep_results["annual_feat_path"]))
        logger.info("annual_feat_df: %d pixels  %d features  RSS=%.1f GB",
                    len(annual_feat_df), annual_feat_df.width - 1, _rss_gb())

    logger.info("Calling train_tam (precomputed_split paths): RSS=%.1f GB", _rss_gb())

    precomputed_split = {
        "train_pixel_df": out_dir / "prep_train_pixel_df.parquet",
        "val_pixel_df":   out_dir / "prep_val_pixel_df.parquet",
        "train_py_labels": train_py_labels,
        "val_py_labels":   val_py_labels,
        "annual_feat_df":  annual_feat_df,
    }
    _split_holder = [precomputed_split]
    del precomputed_split
    gc.collect()

    def _call() -> tuple:
        return train_tam(
            pixel_df=pl.DataFrame(),   # unused when precomputed_split is provided
            labels={},                 # unused when precomputed_split is provided
            pixel_coords=pl.DataFrame(),
            out_dir=out_dir,
            cfg=cfg,
            device=worker_args.get("device"),
            precomputed_split=_split_holder.pop(),
        )

    _, best_val_auc = _call()
    logger.info("Worker done: best_val_auc=%.4f  RSS=%.1f GB", best_val_auc, _rss_gb())

    # Remove prep parquets — large intermediate files, not needed after training.
    for _p in (
        out_dir / "prep_train_pixel_df.parquet",
        out_dir / "prep_val_pixel_df.parquet",
        out_dir / "prep_annual_feat_df.parquet",
    ):
        try:
            _p.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <work_dir> <out_dir> <experiment>", file=sys.stderr)
        sys.exit(1)
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
