# Performance Notes

## Instance & Network

- **`c6a.4xlarge`**: 6.25 Gbit/s baseline, 12.5 Gbit/s burst. Sustained ~5 Gbit/s observed.
- **Network is the bottleneck** — CPU at ~23%, memory comfortable, network saturated at 16 fetch workers.
- Adding workers beyond 16 gives no throughput gain and introduces connection errors.
- **`c7gn.4xlarge`** ($0.998/hr vs $0.612/hr) has 50 Gbit/s network — likely 3-4x faster per run, cheaper overall for network-bound jobs.
- **DO droplet**: ~2.5 Gbit/s sustained — half the EC2 throughput due to leaving the AWS network.

## S3 & Routing

- EC2 in `us-west-2` (same region as Sentinel-2 COGs) ✅
- **S3 Gateway Endpoint** added — S3 traffic routes via AWS backbone, bypassing the internet gateway. Was not configured initially.
- Gateway endpoint doesn't change DNS resolution (still shows public IPs) but traffic takes the private path.
- Single TCP connection to S3 caps at ~1-2 Gbit/s — parallel connections required to saturate bandwidth.

## Worker Configuration

- **16 fetch workers, 16 compute workers** is the sweet spot for `c6a.4xlarge`.
- 24-32 fetch workers causes `getaddrinfo() thread failed to start` (OS DNS thread exhaustion).
- **nscd** installed on EC2 to cache DNS responses — reduces DNS thread pressure.
- **HTTP/2 multiplexing** enabled (`GDAL_HTTP_VERSION=2`, `GDAL_HTTP_MULTIPLEX=YES`) — multiple range requests share one connection.

## Tile Size

- **512px** is the right balance for `c6a.4xlarge` — 1024px causes OOM.
- Larger tiles = fewer requests = better overhead ratio, but doesn't increase throughput once network-saturated.
- Tile size affects efficiency and memory, not raw Gbit/s.

## EBS

- `c6a.4xlarge` EBS baseline is 4,750 Mbps — potential hidden bottleneck if scratch tiles are on EBS.
- Writing compressed tiles (deflate) reduces EBS pressure.

## COG Request Overhead

- Each tile triggers ~20-50 range requests (header + data per scene per band).
- 512px gives 4x better payload-to-overhead ratio vs 256px.
- `GDAL_HTTP_PERSISTENT=YES` keeps connections warm — amortises TLS/TCP handshake cost.
- Sequential header→data dependency within each COG limits per-connection parallelism.

## Migrating to c7gn (ARM64/Graviton3)

- `rasterio==1.5.0` was a pre-release with no ARM64 wheel on PyPI — pinned to `1.4.4` which has ARM64 wheels.
- All other dependencies (`numpy`, `xarray`, `geopandas`, `shapely`, `pyproj`, `stackstac`, `odc-stac`, `dask`, `scikit-learn`, `matplotlib`) have ARM64 wheels available.
- Rebuild `.venv` from scratch on the new instance — cannot copy an x86 venv to ARM64.
