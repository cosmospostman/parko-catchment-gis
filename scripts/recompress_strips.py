"""Recompress strip parquets in-place with zstd, then move to dest_dir."""
import sys
from pathlib import Path
import pyarrow.parquet as pq

def recompress(src: Path, dest_dir: Path) -> None:
    dest = dest_dir / src.name
    if dest.exists() and dest != src:
        print(f"  skip {src.name} (already at dest)")
        return
    tmp = src.with_suffix(".zstd_tmp.parquet")
    print(f"  recompress {src.name} ({src.stat().st_size / 1e9:.1f} GB) ...", flush=True)
    pf = pq.ParquetFile(src)
    writer = None
    try:
        for batch in pf.iter_batches():
            import pyarrow as pa
            tbl = pa.Table.from_batches([batch])
            if writer is None:
                writer = pq.ParquetWriter(str(tmp), tbl.schema, compression="zstd", write_statistics=False)
            writer.write_table(tbl)
        if writer:
            writer.close()
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    print(f"  compressed → {tmp.stat().st_size / 1e9:.1f} GB, moving to {dest} ...", flush=True)
    import shutil
    shutil.move(str(tmp), str(dest))
    src.unlink()
    print(f"  done {src.name}")

if __name__ == "__main__":
    src_dir  = Path(sys.argv[1])
    dest_dir = Path(sys.argv[2])
    dest_dir.mkdir(parents=True, exist_ok=True)
    for strip in sorted(src_dir.glob("*_strip_??.parquet")):
        recompress(strip, dest_dir)
