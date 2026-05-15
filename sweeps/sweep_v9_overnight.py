"""sweep_v9_overnight.py — Overnight ablation sweep on V9-SPECTRAL.

Context: baseline val_auc=0.755 was on a tighter dataset. This sweep reflects
the expanded training set (new regions, Frenchs val sites, relaxed noise filter
admitting more sparse/juvenile presence pixels). Expected baseline is lower.

Baseline: doy_phase_shift=True, dropout=0.5, obs_dropout_min=4,
          lr=5e-5, band_noise_std=0.05

Runs (each isolates one variable against the baseline):
  1. baseline            — all base values
  2. no_phase_shift      — doy_phase_shift=False
  3. lr_1e4              — lr=1e-4
  4. lr_2e4              — lr=2e-4
  5. noise_03            — band_noise_std=0.03
  6. noise_05            — band_noise_std=0.05  (== baseline, explicit reference point)

Usage
-----
    python sweeps/sweep_v9_overnight.py
    python sweeps/sweep_v9_overnight.py --dry-run
    python sweeps/sweep_v9_overnight.py --out outputs/my-dir
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Baseline values
_BASE = dict(
    doy_phase_shift=True,
    dropout=0.5,
    obs_dropout_min=4,
    lr=5e-5,
    band_noise_std=0.05,
)

RUNS = [
    {**_BASE, "run_id": "baseline"},
    {**_BASE, "run_id": "no_phase_shift",  "doy_phase_shift": False},
    {**_BASE, "run_id": "lr_1e4",          "lr": 1e-4},
    {**_BASE, "run_id": "lr_2e4",          "lr": 2e-4},
    {**_BASE, "run_id": "noise_03",        "band_noise_std": 0.03},
    {**_BASE, "run_id": "noise_05",        "band_noise_std": 0.05},
]


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


logger = logging.getLogger("sweep_v9_overnight")


def _load_pixels(base_out: Path):
    import importlib
    import pyarrow.parquet as pq
    import pandas as pd
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

    logger.info("=" * 60)
    logger.info(
        "START %s  phase_shift=%s  dropout=%.2f  obs_dropout_min=%d  lr=%g  band_noise_std=%.2f",
        run_id, run["doy_phase_shift"], run["dropout"],
        run["obs_dropout_min"], run["lr"], run["band_noise_std"],
    )
    logger.info("=" * 60)

    cfg = TAMConfig(
        d_model=128,
        n_layers=2,
        n_heads=4,
        dropout=run["dropout"],
        n_bands=len(V9_FEATURE_COLS),
        n_global_features=0,
        lr=run["lr"],
        weight_decay=0.1,
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
        use_band_summaries=True,
        feature_cols_override=tuple(V9_FEATURE_COLS),
    )

    try:
        _, best_val_auc = train_tam(
            pixel_df=pixel_df[pixel_df["point_id"].isin(labels.index)],
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
    parser = argparse.ArgumentParser(description="V9-SPECTRAL overnight ablation sweep")
    parser.add_argument("--out", default="outputs/sweep-v9-overnight",
                        help="Base output directory (default: outputs/sweep-v9-overnight)")
    parser.add_argument("--device", default=None, help="cpu / cuda (auto-detect if omitted)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_out = PROJECT_ROOT / args.out
    base_out.mkdir(parents=True, exist_ok=True)

    _setup_logging(base_out / "sweep.log")

    header = "run_id\tdoy_phase_shift\tdropout\tobs_dropout_min\tlr\tband_noise_std\tbest_val_auc"
    logger.info(header.replace("\t", "  "))

    summary_path = base_out / "summary.tsv"
    if not summary_path.exists():
        with open(summary_path, "w") as fh:
            fh.write(header + "\n")

    if args.dry_run:
        for run in RUNS:
            logger.info(
                "DRY RUN — %s  phase_shift=%s  dropout=%.2f  obs_dropout_min=%d  lr=%g  band_noise_std=%.2f",
                run["run_id"], run["doy_phase_shift"], run["dropout"],
                run["obs_dropout_min"], run["lr"], run["band_noise_std"],
            )
        return

    pixel_df, pixel_coords, labels = _load_pixels(base_out)

    for run in RUNS:
        run_id = run["run_id"]
        out_dir = base_out / run_id

        val_auc = run_one(run, out_dir, pixel_df, pixel_coords, labels, args.device)

        val_str = f"{val_auc:.4f}" if val_auc is not None else "FAILED"
        row = (
            f"{run_id}\t{run['doy_phase_shift']}\t{run['dropout']:.2f}\t"
            f"{run['obs_dropout_min']}\t{run['lr']}\t{run['band_noise_std']:.2f}\t{val_str}"
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
