"""sweep_v9_spectral.py — S2-only TAM sweep, Etna holdout.

Four runs over two axes: band summaries (False / True) × lr (1e-5 / 5e-5).
All outputs written under outputs/sweep-v9-spectral/.

Usage
-----
    python sweeps/sweep_v9_spectral.py
    python sweeps/sweep_v9_spectral.py --dry-run
    python sweeps/sweep_v9_spectral.py --out outputs/my-dir
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RUNS = [
    {"run_id": "s2_nosumm_1e-5", "use_band_summaries": False, "lr": 1e-5},
    {"run_id": "s2_nosumm_5e-5", "use_band_summaries": False, "lr": 5e-5},
    {"run_id": "s2_summ_1e-5",   "use_band_summaries": True,  "lr": 1e-5},
    {"run_id": "s2_summ_5e-5",   "use_band_summaries": True,  "lr": 5e-5},
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


logger = logging.getLogger("sweep_v9_spectral")


def _load_pixels(exp, base_out: Path):
    """Load and label pixels for v9_spectral."""
    import pyarrow.parquet as pq
    import pandas as pd

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
        # Read source/scl for S2 filtering; B08/B04 for noise filter if not already in feature_cols.
        extra_cols = [c for c in ("source", "scl", "scl_purity") if c in available]
        s2_global_cols = [c for c in ("B08", "B04") if c in available and c not in exp.feature_cols]
        base_cols = ["point_id", "lon", "lat", "date"]
        read_cols = (
            base_cols
            + [c for c in exp.feature_cols if c in available]
            + extra_cols
            + s2_global_cols
        )
        # Deduplicate while preserving order
        seen: set[str] = set()
        read_cols = [c for c in read_cols if not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]
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

    return pixel_df, pixel_coords, labels


def run_one(
    run_id: str,
    out_dir: Path,
    pixel_df,
    pixel_coords,
    labels,
    use_band_summaries: bool,
    lr: float,
    device: str | None,
) -> float | None:
    """Train one variant, tee output to a per-run log, return best val_auc."""
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
    logger.info("START %s  use_band_summaries=%s  lr=%g", run_id, use_band_summaries, lr)
    logger.info("=" * 60)

    cfg = TAMConfig(
        # Architecture
        d_model=128,
        n_layers=2,
        dropout=0.5,
        n_bands=len(V9_FEATURE_COLS),  # 11; overridden by band summaries at runtime
        n_global_features=0,
        # Training
        lr=lr,
        weight_decay=0.1,
        n_epochs=60,
        patience=15,
        band_noise_std=0.03,
        obs_dropout_min=4,
        doy_density_norm=True,
        doy_phase_shift=True,
        pixel_zscore=True,
        use_s1=False,
        val_sites=("etna",),
        # V9-specific
        use_band_summaries=use_band_summaries,
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
    parser = argparse.ArgumentParser(description="V9-SPECTRAL S2-only sweep")
    parser.add_argument("--out", default="outputs/sweep-v9-spectral",
                        help="Base output directory (default: outputs/sweep-v9-spectral)")
    parser.add_argument("--device", default=None, help="cpu / cuda (auto-detect if omitted)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_out = PROJECT_ROOT / args.out
    base_out.mkdir(parents=True, exist_ok=True)

    _setup_logging(base_out / "sweep.log")

    summary_path = base_out / "summary.tsv"
    header = "run_id\tuse_band_summaries\tlr\tbest_val_auc"
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

        if args.dry_run:
            logger.info("DRY RUN — would run %s  use_band_summaries=%s  lr=%g",
                        run_id, run["use_band_summaries"], run["lr"])
            continue

        val_auc = run_one(
            run_id=run_id,
            out_dir=out_dir,
            pixel_df=pixel_df,
            pixel_coords=pixel_coords,
            labels=labels,
            use_band_summaries=run["use_band_summaries"],
            lr=run["lr"],
            device=args.device,
        )

        val_str = f"{val_auc:.4f}" if val_auc is not None else "FAILED"
        row = f"{run_id}\t{run['use_band_summaries']}\t{run['lr']}\t{val_str}"
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
