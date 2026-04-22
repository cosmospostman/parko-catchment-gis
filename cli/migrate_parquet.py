"""Migrate a parquet file in-place to the optimised schema.

Applies _optimise_schema (lon/lat → float32, date → date32) and _WRITE_OPTS
(ZSTD-3, dictionary encoding) to an existing parquet file without needing a
full second copy on disk.

Algorithm:
  1. Stream row groups in chunks of --chunk-rgs.
  2. Write optimised chunks to <file>.migrating.parquet.
  3. On success, atomically rename migrating → original.
  4. On any error, delete the partial .migrating file and exit non-zero.

Usage:
  python cli/migrate_parquet.py data/training/tiles/54LWH.parquet
  python cli/migrate_parquet.py data/training/tiles/*.parquet --verify
  python cli/migrate_parquet.py data/training/tiles/54LWH.parquet --dry-run
"""

import argparse
import os
import sys
from pathlib import Path


def _migrate(path: Path, chunk_rgs: int, dry_run: bool, verify: bool) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from signals._shared import _optimise_schema, _WRITE_OPTS

    src_size = os.path.getsize(path)
    pf = pq.ParquetFile(path)
    meta = pf.metadata
    n_rgs = meta.num_row_groups
    total_rows = meta.num_rows
    orig_compression = meta.row_group(0).column(0).compression
    orig_schema = pf.schema_arrow

    print(f"{path.name}: {n_rgs} row groups, {total_rows:,} rows, "
          f"{src_size/1e6:.1f} MB, compression={orig_compression}")

    if dry_run:
        has_date = "date" in orig_schema.names
        lon_type = orig_schema.field("lon").type if "lon" in orig_schema.names else None
        print(f"  [dry-run] would cast: lon/lat→float32, date→date32={has_date}, "
              f"lon currently={lon_type}")
        print(f"  [dry-run] would apply: {_WRITE_OPTS}")
        return

    migrating = path.with_suffix(".migrating.parquet")
    migrating.unlink(missing_ok=True)

    writer: pq.ParquetWriter | None = None
    written_rows = 0
    try:
        for chunk_start in range(0, n_rgs, chunk_rgs):
            chunk_end = min(chunk_start + chunk_rgs, n_rgs)
            tables = []
            for rg in range(chunk_start, chunk_end):
                tables.append(pf.read_row_group(rg))
            tbl = _optimise_schema(pa.concat_tables(tables))
            if writer is None:
                writer = pq.ParquetWriter(str(migrating), tbl.schema, **_WRITE_OPTS)
            writer.write_table(tbl)
            written_rows += len(tbl)
            print(f"  row groups {chunk_start+1}–{chunk_end}/{n_rgs}: "
                  f"{written_rows:,} rows written")
    except Exception:
        if writer is not None:
            writer.close()
        migrating.unlink(missing_ok=True)
        raise

    if writer is not None:
        writer.close()

    if verify:
        check = pq.ParquetFile(migrating)
        if check.metadata.num_rows != total_rows:
            migrating.unlink(missing_ok=True)
            raise RuntimeError(
                f"Row count mismatch: expected {total_rows}, got {check.metadata.num_rows}"
            )
        print(f"  verified: {check.metadata.num_rows:,} rows")

    migrating.replace(path)
    new_size = os.path.getsize(path)
    saving = src_size - new_size
    print(f"  done: {src_size/1e6:.1f} MB → {new_size/1e6:.1f} MB "
          f"({saving/1e6:+.1f} MB, {saving/src_size*100:.1f}% saving)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path, help="Parquet files to migrate")
    parser.add_argument("--chunk-rgs", type=int, default=50,
                        help="Row groups per write batch (default: 50)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report schema changes without writing")
    parser.add_argument("--verify", action="store_true",
                        help="Row-count check after migration")
    args = parser.parse_args()

    errors = []
    for path in args.files:
        if not path.exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            errors.append(path)
            continue
        try:
            _migrate(path, args.chunk_rgs, args.dry_run, args.verify)
        except Exception as e:
            print(f"ERROR migrating {path}: {e}", file=sys.stderr)
            errors.append(path)

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
