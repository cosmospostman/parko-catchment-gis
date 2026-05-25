"""Minimal merge loop diagnostic — instruments the heap loop to catch infinite cycles."""
from __future__ import annotations
import sys, heapq, time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _block_keys(blk):
    pid_col = blk.column("point_id").combine_chunks()
    parts = pc.split_pattern(pid_col, "_", max_splits=2, reverse=True)
    northings = pc.list_element(parts, 2).cast(pa.int32()).to_numpy(zero_copy_only=False)
    dates = blk.column("date").combine_chunks().cast(pa.int32()).to_numpy(zero_copy_only=False)
    return northings, dates


def run_debug(shard_paths: list[Path], out_path: Path, schema: pa.Schema):
    pfiles = [pq.ParquetFile(p) for p in shard_paths]
    n_rgs  = [pf.metadata.num_row_groups for pf in pfiles]
    blocks = [pf.read_row_group(0) for pf in pfiles]
    keys   = [_block_keys(blk) for blk in blocks]
    rg_idx = [0] * len(pfiles)
    pos    = [0] * len(pfiles)

    heap: list[tuple] = []
    for i, blk in enumerate(blocks):
        if blk is not None and len(blk) > 0:
            ns, ds = keys[i]
            heapq.heappush(heap, (int(ns[0]), int(ds[0]), i))

    def _load_block(i):
        rg_idx[i] += 1
        if rg_idx[i] < n_rgs[i]:
            blocks[i] = pfiles[i].read_row_group(rg_idx[i])
            keys[i]   = _block_keys(blocks[i])
            pos[i]    = 0
            ns, ds    = keys[i]
            heapq.heappush(heap, (int(ns[0]), int(ds[0]), i))
        else:
            blocks[i] = None
            keys[i]   = None

    iters = 0
    rows_emitted = 0
    last_state = None
    stuck_count = 0
    t_start = time.time()
    TIMEOUT_S = 10

    while heap:
        _, _, i = heap[0]
        blk = blocks[i]
        current_state = (i, pos[i], rg_idx[i], heap[0])

        if current_state == last_state:
            stuck_count += 1
            if stuck_count >= 3:
                print(f"\n*** INFINITE LOOP DETECTED at iter {iters} ***")
                print(f"  stream i={i}  pos={pos[i]}  rg_idx={rg_idx[i]}")
                print(f"  heap[0]={heap[0]}")
                print(f"  heap[1]={heap[1] if len(heap)>1 else 'N/A'}")
                ns, ds = keys[i]
                print(f"  block len={len(blk)}  ns[pos]={ns[pos[i]]}  ds[pos]={ds[pos[i]]}")
                last = len(blk) - 1
                print(f"  last_key=({ns[last]}, {ds[last]})")
                if len(heap) > 1:
                    next_n, next_d = heap[1][0], heap[1][1]
                    print(f"  next_key=({next_n}, {next_d})")
                    sub_ns = ns[pos[i]:]
                    rel = int(np.searchsorted(sub_ns, next_n, side="left"))
                    cut = pos[i] + rel
                    print(f"  searchsorted rel={rel}  cut={cut}  pos={pos[i]}")
                    # advance cut for date
                    cut2 = cut
                    while cut2 < len(blk) and int(ns[cut2]) == next_n and int(ds[cut2]) < next_d:
                        cut2 += 1
                    print(f"  cut after date walk={cut2}  (emitting {cut2 - pos[i]} rows)")
                return
        else:
            stuck_count = 0
            last_state = current_state

        if len(heap) == 1:
            heapq.heappop(heap)
            rows_emitted += len(blk) - pos[i]
            pos[i] = 0; rg_idx[i] += 1
            while rg_idx[i] < n_rgs[i]:
                rows_emitted += pfiles[i].read_row_group(rg_idx[i]).num_rows
                rg_idx[i] += 1
            break

        next_n, next_d = heap[1][0], heap[1][1]
        ns, ds = keys[i]
        last = len(blk) - 1
        last_key = (int(ns[last]), int(ds[last]))

        heapq.heappop(heap)
        if last_key <= (next_n, next_d):
            rows_emitted += len(blk) - pos[i]
            pos[i] = 0
            _load_block(i)
        else:
            sub_ns = ns[pos[i]:]
            cut = pos[i] + int(np.searchsorted(sub_ns, next_n, side="left"))
            while cut < len(blk) and int(ns[cut]) == next_n and int(ds[cut]) < next_d:
                cut += 1
            if cut == pos[i]:
                cut = pos[i] + 1  # tie-break: always advance
            rows_emitted += cut - pos[i]
            pos[i] = cut
            if pos[i] < len(blk):
                heapq.heappush(heap, (int(ns[pos[i]]), int(ds[pos[i]]), i))
            else:
                _load_block(i)

        iters += 1
        if iters % 50_000 == 0:
            elapsed = time.time() - t_start
            print(f"  iter={iters:,}  rows_emitted={rows_emitted:,}  heap_size={len(heap)}  elapsed={elapsed:.1f}s")
            if elapsed > TIMEOUT_S:
                print(f"\n*** TIMEOUT after {elapsed:.1f}s at iter {iters:,} ***")
                print(f"  last state: stream={i}  pos={pos[i]}  rg={rg_idx[i]}  heap[0]={heap[0]}")
                return

    print(f"\nCompleted: {iters:,} iterations  {rows_emitted:,} rows")


def profile_emit(shard_paths: list[Path], schema: pa.Schema):
    """Profile _emit and _conform_table cost in isolation."""
    import pyarrow.parquet as pq
    from utils.parquet_utils import _conform_table
    import tempfile

    pf = pq.ParquetFile(shard_paths[0])
    blk = pf.read_row_group(0)
    n = len(blk)
    print(f"\nProfiling _conform_table on block of {n:,} rows ({len(schema)} cols)...")

    # Full block
    t0 = time.time()
    for _ in range(5):
        _conform_table(blk, schema)
    dt = (time.time() - t0) / 5
    print(f"  Full block ({n:,} rows): {dt*1000:.1f}ms  →  {n/dt:,.0f} rows/s")

    # Single-row slice
    single = blk.slice(0, 1)
    t0 = time.time()
    N = 10_000
    for _ in range(N):
        _conform_table(single, schema)
    dt = (time.time() - t0) / N
    print(f"  Single-row slice:       {dt*1000:.3f}ms  →  {1/dt:,.0f} rows/s  (old hot path)")

    # Write cost: ParquetWriter.write_table for one full block
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=True) as f:
        writer = pq.ParquetWriter(f.name, schema, compression="zstd")
        t0 = time.time()
        for _ in range(5):
            writer.write_table(blk)
        dt = (time.time() - t0) / 5
        writer.close()
    print(f"  write_table full block: {dt*1000:.1f}ms  →  {n/dt:,.0f} rows/s")

    # pa.concat_tables cost
    bufs = [blk] * 4
    t0 = time.time()
    for _ in range(10):
        pa.concat_tables(bufs)
    dt = (time.time() - t0) / 10
    print(f"  concat_tables(4 blocks): {dt*1000:.1f}ms  →  {4*n/dt:,.0f} rows/s")


if __name__ == "__main__":
    import tempfile, datetime
    from scripts.perf_merge import _make_combined_schema, build_shards

    schema = _make_combined_schema()
    with tempfile.TemporaryDirectory(prefix="debug_merge_") as tmp:
        tmp_dir = Path(tmp)
        print("Building 2 small interleaved shards (worst case)...")
        shard_paths, total = build_shards(tmp_dir, n_pixels=500, n_shards=2,
                                          n_dates=10, mode="interleaved",
                                          schema=schema, row_group_size=1000)
        print(f"Total rows: {total:,}")
        print("\nRunning instrumented merge loop...")
        run_debug(shard_paths, tmp_dir / "out.parquet", schema)

        print("\n--- Emit/conform profiling ---")
        (tmp_dir / "prof").mkdir()
        shard_paths2, _ = build_shards(tmp_dir / "prof", n_pixels=50_000, n_shards=1,
                                        n_dates=40, mode="sequential",
                                        schema=schema, row_group_size=5_000_000)
        profile_emit(shard_paths2, schema)
