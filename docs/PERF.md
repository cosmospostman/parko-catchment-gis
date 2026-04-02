# Performance Notes

## Instance

- **`c7gn.4xlarge`** (16 vCPU ARM64/Graviton3, 32 GB, 50 Gbit/s network), `us-west-2`.
- Pipeline is **CPU-bound** with the EBS local cache — 94–96% CPU utilisation observed, disk at 15–36% util.

## Worker configuration

- **16 fetch workers** is the current setting (`--fetch-workers 16`).
- Workers run in a `ProcessPoolExecutor` with `fork` context — each worker has its own GIL, giving true CPU parallelism.
- `dask scheduler="synchronous"` inside each worker — no thread spawning per tile.
- `fetch_s` p50 ~7s, p90 ~11s (dominated by stackstac COG header reads and numpy compute).

## Tile size

- **512px** tiles at 10 m resolution = 5.12 km × 5.12 km.
- 1024px tiles OOM on 32 GB with 16 workers — avoid.

## EBS local cache

- Sentinel-2 COGs are synced to `/mnt/s2cache` before running the pipeline.
- `LOCAL_S2_ROOT=/mnt/s2cache` rewrites asset hrefs to local paths.
- EBS read latency ~1.2 ms vs ~30 ms per range request over S3 — eliminates network as bottleneck.
- Without the cache, fetch_s would return to 63–128s (network-bound), even with more workers.

## ARM64 / Graviton notes

- `rasterio==1.4.4` — use this version; `1.5.0` had no ARM64 wheel on PyPI at time of setup.
- Rebuild `.venv` from scratch on ARM64 — cannot copy an x86 venv.
