"""sweep_zscore_etna_landsend.py — Hyperparam sweep over v8_s1_zscore_nr_etna_landsend.

Grid:
  lr:      1e-5, 5e-5
  d_model: 64, 128

= 4 runs total. All other settings identical to v8_s1_zscore_nr_etna_landsend.

Usage:
    python sweep_zscore_etna_landsend.py
    python sweep_zscore_etna_landsend.py --out outputs/models/sweep_zscore_etna_landsend
    python sweep_zscore_etna_landsend.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def _setup_logging(log_path: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )


logger = logging.getLogger("sweep_zscore")


def run_one(run_id: str, out_dir: Path, lr: float, d_model: int) -> float | None:
    """Train one variant, tee all output to a per-run log, return best val_auc."""
    import importlib
    import pyarrow.parquet as pq
    import pandas as pd

    from tam.core.config import TAMConfig
    from tam.core.train import train_tam
    from tam.utils import label_pixels
    from utils.regions import select_regions
    from utils.training_collector import tile_ids_for_regions, tile_parquet_path

    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "train.log"

    # Add a per-run file handler so all output (including tam.core.train epoch lines)
    # goes to both console and the run's own log file.
    run_handler = logging.FileHandler(run_log)
    run_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.addHandler(run_handler)

    try:
        return _run(run_id, out_dir, lr, d_model,
                    importlib, pq, pd, TAMConfig, train_tam,
                    label_pixels, select_regions, tile_ids_for_regions, tile_parquet_path)
    finally:
        root.removeHandler(run_handler)
        run_handler.close()


def _run(run_id, out_dir, lr, d_model,
         importlib, pq, pd, TAMConfig, train_tam,
         label_pixels, select_regions, tile_ids_for_regions, tile_parquet_path):

    exp = importlib.import_module("tam.experiments.v8_s1_zscore_nr_etna_landsend").EXPERIMENT

    cfg = TAMConfig(
        d_model=d_model,
        n_layers=2,
        dropout=0.5,
        n_bands=4,
        n_global_features=0,
        lr=lr,
        weight_decay=0.1,
        n_epochs=60,
        patience=15,
        band_noise_std=0.0,
        obs_dropout_min=4,
        doy_density_norm=True,
        doy_phase_shift=True,
        pixel_zscore=True,
        use_s1="s1_only",
        val_sites=("etna",),
    )

    logger.info("=" * 60)
    logger.info("START %s  lr=%s  d_model=%d", run_id, lr, d_model)
    logger.info("=" * 60)

    regions = select_regions(exp.region_ids)
    tile_ids = tile_ids_for_regions(exp.region_ids)

    chunks = []
    for tid in tile_ids:
        path = tile_parquet_path(tid)
        if not path.exists():
            logger.error("Missing tile parquet: %s", path)
            return None
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        s1_cols = [c for c in ("source", "vh", "vv", "scl") if c in available]
        # B08/B04 needed by compute_global_features (noise filter) even in S1-only experiments
        s2_global_cols = [c for c in ("B08", "B04") if c in available and c not in exp.feature_cols]
        base_cols = ["point_id", "lon", "lat", "date", "scl_purity"]
        read_cols = base_cols + [c for c in exp.feature_cols if c in available] + s1_cols + s2_global_cols
        for rg in range(pf.metadata.num_row_groups):
            chunks.append(pf.read_row_group(rg, columns=read_cols).to_pandas())

    pixel_df = pd.concat(chunks, ignore_index=True)
    dates = pd.to_datetime(pixel_df["date"])
    pixel_df["year"] = dates.dt.year
    pixel_df["doy"]  = dates.dt.day_of_year

    pixel_coords = (
        pixel_df[["point_id", "lon", "lat"]]
        .drop_duplicates("point_id")
        .reset_index(drop=True)
    )
    labelled = label_pixels(pixel_coords, regions).dropna(subset=["is_presence"])
    labels = labelled.set_index("point_id")["is_presence"].map({True: 1.0, False: 0.0})

    try:
        _, best_val_auc = train_tam(
            pixel_df=pixel_df[pixel_df["point_id"].isin(labels.index)],
            labels=labels,
            pixel_coords=pixel_coords,
            out_dir=out_dir,
            cfg=cfg,
        )
    except Exception as exc:
        logger.error("Run %s failed: %s", run_id, exc, exc_info=True)
        return None

    logger.info("END %s  best_val_auc=%.4f", run_id, best_val_auc)
    return best_val_auc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/models/sweep_zscore_etna_landsend")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_out = Path(args.out)
    base_out.mkdir(parents=True, exist_ok=True)

    _setup_logging(base_out / "sweep.log")

    grid = [
        # (lr,    d_model)
        (1e-5,   64),
        (5e-5,   64),
        (1e-5,  128),
        (5e-5,  128),
    ]

    summary_path = base_out / "summary.tsv"
    header = "run_id\tlr\td_model\tbest_val_auc"
    logger.info(header.replace("\t", "  "))
    with open(summary_path, "w") as fh:
        fh.write(header + "\n")

    for lr, d_model in grid:
        run_id = f"lr{lr:.0e}_dm{d_model}"
        out_dir = base_out / run_id

        if args.dry_run:
            logger.info("DRY RUN — would run %s", run_id)
            continue

        val_auc = run_one(run_id, out_dir, lr, d_model)

        if val_auc is not None:
            attn_dir = out_dir / "attention"
            if not (attn_dir.exists() and any(attn_dir.glob("*.png"))):
                logger.info("Running attention viz for %s", run_id)
                subprocess.run(
                    [sys.executable, "-m", "tam.viz_attention",
                     "--checkpoint", str(out_dir),
                     "--experiment", "v8_s1_zscore_nr_etna_landsend"],
                    cwd=PROJECT_ROOT,
                )

        val_str = f"{val_auc:.4f}" if val_auc is not None else "FAILED"
        row = f"{run_id}\t{lr:.0e}\t{d_model}\t{val_str}"
        logger.info(row.replace("\t", "  "))
        with open(summary_path, "a") as fh:
            fh.write(row + "\n")

    logger.info("Done. Results: %s", summary_path)
    if summary_path.exists():
        print("\n" + "=" * 50)
        print("SWEEP RESULTS")
        print("=" * 50)
        print(summary_path.read_text())


if __name__ == "__main__":
    main()
