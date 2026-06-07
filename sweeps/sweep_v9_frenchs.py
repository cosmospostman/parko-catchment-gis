"""sweep_v9_frenchs.py — Single-param ablations from the Frenchs baseline.

Baseline: d_model=64, lr=2e-5, obs_dropout_min=15, use_band_summaries=True,
          dropout=0.5, weight_decay=0.1, n_layers=2, Etna holdout.

Six runs — one param changed at a time to identify the next improvement lever.

Usage
-----
    python sweeps/sweep_v9_frenchs.py
    python sweeps/sweep_v9_frenchs.py --dry-run
    python sweeps/sweep_v9_frenchs.py --out outputs/my-dir
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Baseline hyperparams
_BASE = dict(
    d_model=64,
    lr=2e-5,
    obs_dropout_min=15,
    use_band_summaries=True,
    dropout=0.5,
    weight_decay=0.1,
    n_layers=2,
)

RUNS = [
    {"run_id": "baseline",      **_BASE},
    {"run_id": "no_summaries",  **{**_BASE, "use_band_summaries": False}},
    {"run_id": "dropout_low",   **{**_BASE, "dropout": 0.3}},
    {"run_id": "wd_high",       **{**_BASE, "weight_decay": 0.3}},
    {"run_id": "n_layers_1",    **{**_BASE, "n_layers": 1}},
    {"run_id": "obs_drop_8",    **{**_BASE, "obs_dropout_min": 8}},
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


logger = logging.getLogger("sweep_v9_frenchs")


def _load_pixels(exp, base_out: Path):
    import pyarrow.parquet as pq

    from tam.utils import label_pixels
    from utils.regions import select_regions
    from utils.training_collector import tile_ids_for_regions, tile_parquet_path

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
        extra_cols = [c for c in ("source", "scl", "scl_purity") if c in available]
        s2_annual_cols = [c for c in ("B08", "B04") if c in available and c not in exp.feature_cols]
        base_cols = ["point_id", "lon", "lat", "date"]
        read_cols = (
            base_cols
            + [c for c in exp.feature_cols if c in available]
            + extra_cols
            + s2_annual_cols
        )
        seen: set[str] = set()
        read_cols = [c for c in read_cols if not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]
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

    return pixel_df, pixel_coords, labels


def run_one(run_id: str, out_dir: Path, pixel_df, pixel_coords, labels,
            device: str | None, **kwargs) -> float | None:
    from tam.core.config import TAMConfig
    from tam.core.dataset import V9_FEATURE_COLS
    from tam.core.train import train_tam

    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "train.log"

    run_handler = logging.FileHandler(run_log)
    run_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.addHandler(run_handler)

    logger.info("=" * 60)
    logger.info("START %s  %s", run_id, "  ".join(f"{k}={v}" for k, v in kwargs.items()))
    logger.info("=" * 60)

    cfg = TAMConfig(
        # Fixed architecture / training constants
        n_heads=4,
        d_ff=64,
        n_bands=len(V9_FEATURE_COLS),
        n_annual_features=0,
        n_epochs=60,
        patience=15,
        band_noise_std=0.03,
        doy_density_norm=True,
        doy_phase_shift=True,
        pixel_zscore=True,
        use_s1=False,
        val_sites=("etna",),
        feature_cols_override=tuple(V9_FEATURE_COLS),
        # Variable params from run config
        d_model=kwargs["d_model"],
        lr=kwargs["lr"],
        obs_dropout_min=kwargs["obs_dropout_min"],
        use_band_summaries=kwargs["use_band_summaries"],
        dropout=kwargs["dropout"],
        weight_decay=kwargs["weight_decay"],
        n_layers=kwargs["n_layers"],
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
    parser = argparse.ArgumentParser(description="V9 Frenchs ablation sweep")
    parser.add_argument("--out", default="outputs/sweep-v9-frenchs",
                        help="Base output directory (default: outputs/sweep-v9-frenchs)")
    parser.add_argument("--device", default=None, help="cpu / cuda (auto-detect if omitted)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_out = PROJECT_ROOT / args.out
    base_out.mkdir(parents=True, exist_ok=True)

    _setup_logging(base_out / "sweep.log")

    summary_path = base_out / "summary.tsv"
    header = "run_id\td_model\tlr\tobs_dropout_min\tuse_band_summaries\tdropout\tweight_decay\tn_layers\tbest_val_auc"
    logger.info(header.replace("\t", "  "))
    if not summary_path.exists():
        with open(summary_path, "w") as fh:
            fh.write(header + "\n")

    if not args.dry_run:
        import importlib
        exp = importlib.import_module("tam.experiments.v9_spectral").EXPERIMENT
        logger.info("Loading pixels for v9_spectral ...")
        pixel_df, pixel_coords, labels = _load_pixels(exp, base_out)
        logger.info("Pixels loaded: %d unique points", len(pixel_coords))
    else:
        pixel_df = pixel_coords = labels = None

    for run in RUNS:
        run_id = run["run_id"]
        out_dir = base_out / run_id
        kwargs = {k: v for k, v in run.items() if k != "run_id"}

        if args.dry_run:
            logger.info("DRY RUN — would run %s  %s", run_id,
                        "  ".join(f"{k}={v}" for k, v in kwargs.items()))
            continue

        val_auc = run_one(
            run_id=run_id,
            out_dir=out_dir,
            pixel_df=pixel_df,
            pixel_coords=pixel_coords,
            labels=labels,
            device=args.device,
            **kwargs,
        )

        val_str = f"{val_auc:.4f}" if val_auc is not None else "FAILED"
        row = "\t".join([
            run_id,
            str(kwargs["d_model"]),
            str(kwargs["lr"]),
            str(kwargs["obs_dropout_min"]),
            str(kwargs["use_band_summaries"]),
            str(kwargs["dropout"]),
            str(kwargs["weight_decay"]),
            str(kwargs["n_layers"]),
            val_str,
        ])
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
