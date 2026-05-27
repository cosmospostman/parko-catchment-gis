# Fetch Proxy — VM Setup Guide

The fetch proxy runs on a cloud VM co-located with Element84's S3 COG bucket
(us-west-2). It extracts pixel parquets at Gbps-rate S3 reads and streams
only the compressed sorted output back to the workstation (~60 GB/tile/year
instead of ~5 TB).

## VM Spec

**Recommended:** AWS `t3a.xlarge` (4 vCPU, 16 GB RAM, 20 GB SSD) in `us-west-2`.  
**Minimum:** `t3a.large` (2 vCPU, 8 GB RAM) — limited extraction parallelism.  
**Alternative:** DigitalOcean 8 GB droplet (SFO) — simpler pricing, ~$48/mo.

## 1. VM Setup

```bash
# Ubuntu 22.04 / 24.04
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3-pip git gdal-bin

# Clone the repo (or rsync from workstation)
git clone https://github.com/your-org/parko-catchment-gis.git
cd parko-catchment-gis

# Create env and install server deps
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r proxy/requirements-server.txt
```

## 2. AWS Credentials

The server reads COGs from Element84's public S3 bucket. No credentials needed
for public data. If using MPC signing for S1, set:

```bash
export AZURE_CLIENT_ID=...
export AZURE_CLIENT_SECRET=...
export AZURE_TENANT_ID=...
```

## 3. SSH Tunnel (workstation)

```bash
ssh -N -L 8765:localhost:8765 ubuntu@<vm-ip>
```

Keep this running in a separate terminal. The proxy listens only on
`localhost:8765` — the SSH key is the only auth.

## 4. Start the Server

On the VM:

```bash
cd parko-catchment-gis
source .venv/bin/activate
python -m proxy.server
```

Or with environment overrides:

```bash
PROXY_MAX_CONCURRENT=64 PROXY_N_WORKERS=4 python -m proxy.server
```

The server prints `Uvicorn running on http://127.0.0.1:8765` when ready.

## 5. Run a Fetch (workstation)

```bash
python cli/location.py fetch mitchell --years 2021 --proxy http://localhost:8765
```

Or programmatically:

```python
from utils.location import get
loc = get("mitchell")
loc.fetch(years=[2021], proxy_url="http://localhost:8765")
```

## 6. Tuning Knobs

All knobs are environment variables on the VM — no code changes needed:

| Bottleneck | Variable | Default |
|---|---|---|
| Fetch too slow | `PROXY_MAX_CONCURRENT` | 32 |
| Extract too slow | `PROXY_N_WORKERS` | auto-scaled |
| Merge out of RAM | `PROXY_MERGE_MEM_GB` | 2 |
| Port | `PROXY_PORT` | 8765 |

To find the bottleneck, watch the server logs. Each strip logs:

```
[strip 0000]  fetch:   t=0s
[strip 0000]  extract: t=Xs
[strip 0000]  merge:   t=Ys
[strip 0000]  stream:  t=Zs
[strip 0000]  done in Ws
[strip 0001]  fetch:   t=Ys    ← started overlapping strip 0's stream
```

If strip N+1's merge finishes after strip N's stream ends, there's an idle
gap — the WAN pipe went dry. Increase `PROXY_MAX_CONCURRENT` or `PROXY_N_WORKERS`,
or upsize the VM.

## 7. Verification

After one tile runs, confirm output matches a direct local fetch:

```bash
# On workstation — compare row counts and a sample of values
python -c "
import pyarrow.parquet as pq
proxy = pq.ParquetFile('data/pixels/mitchell/2021/54LWH.parquet')
local = pq.ParquetFile('data/pixels/mitchell_local/2021/54LWH.parquet')
print('proxy rows:', proxy.metadata.num_rows)
print('local rows:', local.metadata.num_rows)
"
```

Expected: identical row counts and NBAR-corrected band values.

## 8. Strip-level Resume

If the connection drops mid-tile, re-run the same command. The client detects
which strips are already complete (`.parquet` files in `tmp/<tile>/<year>/`)
and resumes from the first missing strip. The VM re-runs the STAC search
(cheap) and skips already-received strips.

## 9. Peak Disk on VM

~7 GB at any moment — well within the 20 GB recommended disk:

- 1 S2 `.npz` in flight during fetch/extract: ~570 MB
- All S2 scene parquets (until merge completes): ~630 MB  
- S1 strip parquet: ~25 MB
- 1 sorted strip shard (during stream): ~5.5 GB
