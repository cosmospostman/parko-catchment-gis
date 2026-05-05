"""sweep_v8_roper_despeckle_laso.py — two-phase sweep for v8_roper.

Phase 1: s1_despeckle_window in {0, 3, 5, 7}, Etna holdout.
Phase 2: LASO — best despeckle value, each site held out in turn.

All outputs written under outputs/sweep-speckle-laso/.

Usage
-----
    python sweeps/sweep_v8_roper_despeckle_laso.py
    python sweeps/sweep_v8_roper_despeckle_laso.py --dry-run
    python sweeps/sweep_v8_roper_despeckle_laso.py --despeckle-only
    python sweeps/sweep_v8_roper_despeckle_laso.py --laso-only --best-despeckle 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DESPECKLE_VALUES = [0, 3, 5, 7]

# All sites present in v8_roper region list.
# cloncurry is absence-only — val AUC will be uninformative for that holdout.
LASO_SITES = [
    "norman_road",
    "cloncurry",
    "etna",
    "landsend",
    "lake_mueller",
    "corfield",
    "roper",
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


logger = logging.getLogger("sweep_v8_roper_despeckle_laso")


def _load_pixels(exp, base_out: Path):
    """Load and label pixels for v8_roper — shared across all runs."""
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
        s1_cols = [c for c in ("source", "vh", "vv", "scl") if c in available]
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

    return pixel_df, pixel_coords, labels


def run_one(
    run_id: str,
    out_dir: Path,
    pixel_df,
    pixel_coords,
    labels,
    s1_despeckle_window: int,
    val_sites: tuple[str, ...],
    device: str | None,
) -> float | None:
    """Train one variant, tee output to a per-run log, return best val_auc."""
    import pandas as pd
    from tam.core.config import TAMConfig
    from tam.core.train import train_tam

    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "train.log"

    run_handler = logging.FileHandler(run_log)
    run_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.addHandler(run_handler)

    logger.info("=" * 60)
    logger.info("START %s  despeckle=%d  val_sites=%s", run_id, s1_despeckle_window, val_sites)
    logger.info("=" * 60)

    cfg = TAMConfig(
        # Architecture
        d_model=128,
        n_layers=2,
        dropout=0.5,
        n_bands=4,
        n_global_features=0,
        # Training
        lr=5e-5,
        weight_decay=0.1,
        n_epochs=60,
        patience=15,
        band_noise_std=0.0,
        obs_dropout_min=4,
        doy_density_norm=True,
        doy_phase_shift=True,
        pixel_zscore=True,
        use_s1="s1_only",
        val_sites=val_sites,
        s1_despeckle_window=s1_despeckle_window,
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


def phase1_despeckle(
    base_out: Path,
    pixel_df,
    pixel_coords,
    labels,
    device: str | None,
    dry_run: bool,
    summary_path: Path,
) -> None:
    logger.info("=" * 60)
    logger.info("PHASE 1: despeckle sweep  val_sites=(etna,)")
    logger.info("=" * 60)

    for w in DESPECKLE_VALUES:
        run_id = f"despeckle{w}_val_etna"
        out_dir = base_out / run_id

        if dry_run:
            logger.info("DRY RUN — would run %s", run_id)
            continue

        val_auc = run_one(run_id, out_dir, pixel_df, pixel_coords, labels,
                          s1_despeckle_window=w, val_sites=("etna",), device=device)

        val_str = f"{val_auc:.4f}" if val_auc is not None else "FAILED"
        row = f"{run_id}\t{w}\tetna\t{val_str}"
        logger.info(row.replace("\t", "  "))
        with open(summary_path, "a") as fh:
            fh.write(row + "\n")


def phase2_laso(
    base_out: Path,
    pixel_df,
    pixel_coords,
    labels,
    best_despeckle: int,
    device: str | None,
    dry_run: bool,
    summary_path: Path,
    sites: list[str] | None = None,
) -> None:
    logger.info("=" * 60)
    logger.info("PHASE 2: LASO sweep  despeckle=%d", best_despeckle)
    logger.info("=" * 60)

    for site in (sites if sites is not None else LASO_SITES):
        run_id = f"laso_{site}_despeckle{best_despeckle}"
        out_dir = base_out / run_id

        if dry_run:
            logger.info("DRY RUN — would run %s", run_id)
            continue

        val_auc = run_one(run_id, out_dir, pixel_df, pixel_coords, labels,
                          s1_despeckle_window=best_despeckle, val_sites=(site,), device=device)

        val_str = f"{val_auc:.4f}" if val_auc is not None else "FAILED"
        row = f"{run_id}\t{best_despeckle}\t{site}\t{val_str}"
        logger.info(row.replace("\t", "  "))
        with open(summary_path, "a") as fh:
            fh.write(row + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="v8_roper despeckle + LASO sweep")
    parser.add_argument("--out", default="outputs/sweep-speckle-laso",
                        help="Base output directory (default: outputs/sweep-speckle-laso)")
    parser.add_argument("--device", default=None, help="cpu / cuda (auto-detect if omitted)")
    parser.add_argument("--best-despeckle", type=int, default=None,
                        help="Despeckle value to use for LASO phase (required with --laso-only)")
    parser.add_argument("--laso-only", action="store_true",
                        help="Run only phase 2 (requires --best-despeckle)")
    parser.add_argument("--despeckle-only", action="store_true",
                        help="Run only phase 1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sites", default=None,
                        help="Comma-separated list of sites for phase 2 (default: all LASO_SITES)")
    args = parser.parse_args()

    if args.laso_only and args.best_despeckle is None:
        parser.error("--laso-only requires --best-despeckle")

    base_out = PROJECT_ROOT / args.out
    base_out.mkdir(parents=True, exist_ok=True)

    _setup_logging(base_out / "sweep.log")

    summary_path = base_out / "summary.tsv"
    header = "run_id\tdespeckle\tval_site\tbest_val_auc"
    logger.info(header.replace("\t", "  "))
    if not summary_path.exists():
        with open(summary_path, "w") as fh:
            fh.write(header + "\n")

    if not args.dry_run:
        import importlib
        exp = importlib.import_module("tam.experiments.v8_roper").EXPERIMENT
        logger.info("Loading pixels for v8_roper ...")
        pixel_df, pixel_coords, labels = _load_pixels(exp, base_out)
        logger.info("Pixels loaded: %d unique points", len(pixel_coords))
    else:
        pixel_df = pixel_coords = labels = None

    if not args.laso_only:
        phase1_despeckle(base_out, pixel_df, pixel_coords, labels,
                         args.device, args.dry_run, summary_path)

    if not args.despeckle_only:
        best = args.best_despeckle if args.best_despeckle is not None else 3
        if args.best_despeckle is None and not args.laso_only:
            logger.info(
                "Phase 1 complete — pass --best-despeckle <val> --laso-only to run phase 2 "
                "with the winning value. Continuing with despeckle=%d.", best
            )
        sites = [s.strip() for s in args.sites.split(",")] if args.sites else None
        phase2_laso(base_out, pixel_df, pixel_coords, labels,
                    best, args.device, args.dry_run, summary_path, sites=sites)

    logger.info("Done. Results: %s", summary_path)
    if summary_path.exists():
        print("\n" + "=" * 50)
        print("SWEEP RESULTS")
        print("=" * 50)
        print(summary_path.read_text())


if __name__ == "__main__":
    main()
