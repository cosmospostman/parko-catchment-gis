"""sweep_v9_reg.py — Regularisation / capacity sweep on V9-SPECTRAL.

Context: best known result is val_auc=0.755 (mean_pool, d_model=128, n_layers=2,
d_ff=64, dropout=0.5, weight_decay=0.1). The train/val gap (~0.13) indicates
overfitting, not a capacity ceiling. This sweep tests two independent levers:

  1. Weight decay (0.1 → 0.3 / 0.5) — stronger L2 on current arch
  2. Smaller model (d_model=64/32, n_layers=1/2) — reduce parameter count

d_ff is kept at 0.5× d_model throughout, preserving the bottleneck FFN effect.
All other hyperparameters match the best v9 config.

Runs:
  baseline       d_model=128  n_layers=2  d_ff=64   wd=0.1   reference
  wd_03          d_model=128  n_layers=2  d_ff=64   wd=0.3   stronger L2
  wd_05          d_model=128  n_layers=2  d_ff=64   wd=0.5   aggressive L2
  small          d_model=64   n_layers=1  d_ff=32   wd=0.1   half capacity
  small_wd       d_model=64   n_layers=1  d_ff=32   wd=0.3   small + L2
  small_deep     d_model=64   n_layers=2  d_ff=32   wd=0.1   small + depth
  small_deep_wd  d_model=64   n_layers=2  d_ff=32   wd=0.3   small + deep + L2
  tiny           d_model=32   n_layers=1  d_ff=16   wd=0.1   minimal capacity

Usage
-----
    python sweeps/sweep_v9_reg.py
    python sweeps/sweep_v9_reg.py --dry-run
    python sweeps/sweep_v9_reg.py --out outputs/my-sweep-dir
    python sweeps/sweep_v9_reg.py --runs baseline small tiny
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_BASE = dict(
    doy_phase_shift=False,
    dropout=0.5,
    obs_dropout_min=4,
    lr=5e-5,
    band_noise_std=0.05,
)

RUNS = [
    {**_BASE, "run_id": "baseline",      "d_model": 128, "n_layers": 2, "d_ff": 64,  "weight_decay": 0.1},
    {**_BASE, "run_id": "wd_03",         "d_model": 128, "n_layers": 2, "d_ff": 64,  "weight_decay": 0.3},
    {**_BASE, "run_id": "wd_05",         "d_model": 128, "n_layers": 2, "d_ff": 64,  "weight_decay": 0.5},
    {**_BASE, "run_id": "small",         "d_model": 64,  "n_layers": 1, "d_ff": 32,  "weight_decay": 0.1},
    {**_BASE, "run_id": "small_wd",      "d_model": 64,  "n_layers": 1, "d_ff": 32,  "weight_decay": 0.3},
    {**_BASE, "run_id": "small_deep",    "d_model": 64,  "n_layers": 2, "d_ff": 32,  "weight_decay": 0.1},
    {**_BASE, "run_id": "small_deep_wd", "d_model": 64,  "n_layers": 2, "d_ff": 32,  "weight_decay": 0.3},
    {**_BASE, "run_id": "tiny",          "d_model": 32,  "n_layers": 1, "d_ff": 16,  "weight_decay": 0.1},
]

RUN_IDS = [r["run_id"] for r in RUNS]


def _setup_logging(sweep_log: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(sweep_log),
        ],
        force=True,
    )


logger = logging.getLogger("sweep_v9_reg")


def _load_pixels(base_out: Path):
    import importlib
    import pyarrow.parquet as pq
    from tam.utils import label_pixels
    from utils.regions import select_regions
    from utils.training_collector import tile_ids_for_regions, tile_parquet_path

    exp = importlib.import_module("tam.experiments.v9_spectral").EXPERIMENT
    regions = select_regions(exp.region_ids)
    tile_ids = tile_ids_for_regions(exp.region_ids)

    chunks = []
    for tid in tile_ids:
        path = tile_parquet_path(tid)
        if not path.exists():
            logger.error("Missing tile parquet: %s — run training_collector first", path)
            sys.exit(1)
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        base_cols = ["point_id", "lon", "lat", "date", "scl_purity"]
        s1_cols   = [c for c in ("source", "vh", "vv", "scl") if c in available]
        s2_extra  = [c for c in ("B08", "B04") if c in available and c not in exp.feature_cols]
        read_cols = base_cols + [c for c in exp.feature_cols if c in available] + s1_cols + s2_extra
        read_cols = list(dict.fromkeys(read_cols))
        for rg in range(pf.metadata.num_row_groups):
            chunks.append(pl.from_arrow(pf.read_row_group(rg, columns=read_cols)))

    pixel_df = pl.concat(chunks).with_columns([
        pl.col("date").cast(pl.Date).dt.year().alias("year"),
        pl.col("date").cast(pl.Date).dt.ordinal_day().alias("doy"),
    ])

    pixel_coords = pixel_df.select(["point_id", "lon", "lat"]).unique("point_id")
    labelled = label_pixels(pixel_coords, regions).filter(pl.col("is_presence").is_not_null())
    labels = {
        row["point_id"]: 1.0 if row["is_presence"] else 0.0
        for row in labelled.iter_rows(named=True)
    }

    logger.info("Pixels loaded: %d unique points", len(pixel_coords))
    return pixel_df, pixel_coords, labels


def run_one(run: dict, out_dir: Path, pixel_df, pixel_coords, labels, device) -> float | None:
    import importlib
    from tam.core.config import TAMConfig
    from tam.core.dataset import V9_FEATURE_COLS
    from tam.core.train import train_tam

    run_id = run["run_id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    run_handler = logging.FileHandler(out_dir / "train.log")
    run_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.addHandler(run_handler)

    n_params_est = (
        run["d_model"] * len(V9_FEATURE_COLS)
        + run["n_layers"] * (
            4 * run["d_model"] ** 2
            + 2 * run["d_model"] * run["d_ff"]
        )
    )
    logger.info("=" * 60)
    logger.info(
        "START %s  d_model=%d  n_layers=%d  d_ff=%d  wd=%.2f  (~%dk params)",
        run_id, run["d_model"], run["n_layers"], run["d_ff"],
        run["weight_decay"], n_params_est // 1000,
    )
    logger.info("=" * 60)

    cfg = TAMConfig(
        d_model=run["d_model"],
        n_heads=max(1, run["d_model"] // 32),  # head_dim fixed at 32
        n_layers=run["n_layers"],
        d_ff=run["d_ff"],
        dropout=run["dropout"],
        n_bands=len(V9_FEATURE_COLS),
        n_annual_features=0,
        lr=run["lr"],
        weight_decay=run["weight_decay"],
        n_epochs=60,
        patience=15,
        band_noise_std=run["band_noise_std"],
        obs_dropout_min=run["obs_dropout_min"],
        doy_density_norm=True,
        doy_phase_shift=run["doy_phase_shift"],
        pixel_zscore=True,
        use_s1=False,
        val_region_ids=tuple(
            importlib.import_module("tam.experiments.v9_spectral").EXPERIMENT.val_region_ids
        ),
        feature_cols_override=tuple(V9_FEATURE_COLS),
        max_seq_len=64,
    )

    try:
        _, best_val_auc = train_tam(
            pixel_df=pixel_df.filter(pl.col("point_id").is_in(set(labels))),
            labels=labels,
            pixel_coords=pixel_coords,
            out_dir=out_dir,
            cfg=cfg,
            device=device,
        )
    except Exception as exc:
        logger.error("Run %s failed: %s", run_id, exc, exc_info=True)
        return None
    finally:
        root.removeHandler(run_handler)
        run_handler.close()

    logger.info("END %s  best_val_auc=%.4f", run_id, best_val_auc)
    return best_val_auc


def main() -> None:
    parser = argparse.ArgumentParser(description="V9-SPECTRAL regularisation / capacity sweep")
    parser.add_argument("--out", default="outputs/sweep-v9-reg",
                        help="Base output directory (default: outputs/sweep-v9-reg)")
    parser.add_argument("--device", default=None, help="cpu / cuda (auto-detect if omitted)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runs", nargs="+", choices=RUN_IDS, default=None,
                        metavar="RUN_ID",
                        help=f"Subset of runs to execute (default: all). Choices: {RUN_IDS}")
    args = parser.parse_args()

    base_out = PROJECT_ROOT / args.out
    base_out.mkdir(parents=True, exist_ok=True)

    _setup_logging(base_out / "sweep.log")

    runs = [r for r in RUNS if args.runs is None or r["run_id"] in args.runs]

    header = "run_id\td_model\tn_layers\td_ff\tweight_decay\tbest_val_auc"
    logger.info(header.replace("\t", "  "))

    summary_path = base_out / "summary.tsv"
    if not summary_path.exists():
        with open(summary_path, "w") as fh:
            fh.write(header + "\n")

    if args.dry_run:
        for run in runs:
            logger.info(
                "DRY RUN — %s  d_model=%d  n_layers=%d  d_ff=%d  wd=%.2f",
                run["run_id"], run["d_model"], run["n_layers"],
                run["d_ff"], run["weight_decay"],
            )
        return

    pixel_df, pixel_coords, labels = _load_pixels(base_out)

    for run in runs:
        run_id = run["run_id"]
        out_dir = base_out / run_id

        val_auc = run_one(run, out_dir, pixel_df, pixel_coords, labels, args.device)

        val_str = f"{val_auc:.4f}" if val_auc is not None else "FAILED"
        row = (
            f"{run_id}\t{run['d_model']}\t{run['n_layers']}\t"
            f"{run['d_ff']}\t{run['weight_decay']}\t{val_str}"
        )
        logger.info(row.replace("\t", "  "))
        with open(summary_path, "a") as fh:
            fh.write(row + "\n")

    logger.info("Done. Results: %s", summary_path)
    print("\n" + "=" * 60)
    print("SWEEP RESULTS")
    print("=" * 60)
    print(summary_path.read_text())


if __name__ == "__main__":
    main()
