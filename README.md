# Parkinsonia Catchment GIS

Detects and maps *Parkinsonia aculeata* (parkinsonia) across a catchment using
Sentinel-2 NDVI composites, flood extent, flowering index, and a random-forest
classifier. Outputs priority-patch polygons and year-on-year change detection.

## Virtual environment

Create the venv once:

```bash
python3 -m venv .venv
```

Activate it before running any code:

```bash
source .venv/bin/activate
```

## Setup

Install dependencies (with the venv active):

```bash
pip install -r requirements.txt
```

For running tests, also install dev dependencies:

```bash
pip install -r requirements-dev.txt
```

## Configuration

All operational data lives outside the repo under a single `BASE_DIR`. Edit
[config.sh](config.sh) to set its location (defaults to `/data/mrc-parko`):

```bash
export BASE_DIR="/data/mrc-parko"   # change this line
```

You must also place the catchment boundary file at:

```
$BASE_DIR/mitchell_catchment.geojson
```

Fetch it from the Queensland Government drainage basins dataset:

```bash
curl -s "https://spatial-gis.information.qld.gov.au/arcgis/rest/services/InlandWaters/DrainageBasins/MapServer/1/query?where=BASIN_NAME+LIKE+%27%25Mitchell%25%27&f=geojson&outFields=*" \
  -o "${BASE_DIR}/mitchell_catchment.geojson"
```

This queries the Queensland spatial data service for the Mitchell drainage basin and writes it directly to the expected path. Run it after sourcing `config.sh` so `$BASE_DIR` is set.

Source the config before running anything:

```bash
source config.sh
```

All other paths (`cache/`, `working/`, `outputs/`, `logs/`) are derived from
`BASE_DIR` automatically and created at runtime.

## Running the pipeline

```bash
source config.sh
./run.sh YEAR
```

Example:

```bash
./run.sh 2024
```

Optional flags:

| Flag | Description |
|---|---|
| `--composite-start MM-DD` | Start of Sentinel-2 composite window (default `05-01`) |
| `--composite-end MM-DD` | End of composite window (default `10-31`) |
| `--from-step N` | Resume from step N, skipping earlier steps |
| `--to-step N` | Stop after step N (e.g. `--to-step 3` runs steps 1–3 only) |
| `--only-step N` | Run a single step only |
| `--rebuild-baseline` | Recompute the NDVI baseline (Landsat 1986–present) |
| `--force` | Clear step sentinels and re-run all steps for the year |
| `--dry-run` | Print steps that would run without executing anything |
| `--tile-size N` | Spatial tile size in pixels (default `512`) |
| `--fetch-workers N` | Number of concurrent tile workers (default `16`) |

The pipeline runs 7 steps in sequence. Completed steps are recorded as
sentinels in `$WORKING_DIR` and skipped on re-runs unless `--force` is passed.
Logs are written to `$BASE_DIR/logs/`.

## Run pipeline steps 1–4 on a cloud instance

Steps 1–4 are CPU and I/O intensive and require large remote datasets. Run them on an EC2 instance with EBS-cached copies of the Sentinel-2 COGs and Sentinel-1 GRD scenes.

**Why step 4 runs on EC2:** The Sentinel-1 flood extent step fetches ~128 GB of GRD scenes from `s3://sentinel-s1-l1c` (us-east-1). Streaming this locally is impractical; caching to EBS makes re-runs fast and keeps all remote data transfer within AWS.

**Instance spec:** Ubuntu 24.04 LTS, `c7gn.4xlarge` (16 vCPU ARM64/Graviton3, 32 GB), region `us-west-2`. Add your SSH key during creation. 100 GiB `gp3` root volume.

**First-time setup:**

```bash
scp $BASE_DIR/mitchell_catchment.geojson ubuntu@<instance-ip>:/data/mrc-parko/
ssh ubuntu@<instance-ip>
```

```bash
sudo apt-get update && sudo apt-get install -y python3-pip python3-venv git gdal-bin libgdal-dev
git clone https://github.com/cosmospostman/parko-catchment-gis.git parko-catchment-gis
cd parko-catchment-gis
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install GDAL==3.8.4
sudo mkdir -p /data/mrc-parko && sudo chown $USER /data/mrc-parko
```

> **Note:** install GDAL 3.8.4 via pip after `requirements.txt` to ensure the Python bindings match the system library version. Do not add any GDAL PPAs — use the version from the standard Ubuntu 24.04 repos only.

**Each run** — attach both EBS cache volumes, then run the pipeline. The pipeline's step 0 generates manifests and syncs both S2 and S1 scenes automatically. See [EBS-SETUP.md](EBS-SETUP.md) for volume setup. Use `tmux` so the pipeline survives SSH disconnection:

```bash
ssh ubuntu@<instance-ip>
cd parko-catchment-gis && source .venv/bin/activate
tmux new -s pipeline

# Run steps 1–4 with local caches (step 0 syncs S2 and S1 automatically)
export LOCAL_S2_ROOT=/mnt/ebs/s2cache
export LOCAL_S1_ROOT=/mnt/ebs/s1cache
./run.sh 2025 --to-step 4
# Ctrl+B then D to detach; tmux attach -t pipeline to return
```

**Copy outputs back** once all four steps complete:

```bash
scp -r ubuntu@<instance-ip>:/data/mrc-parko/outputs/2025 ./outputs/
```

Then snapshot the EBS volumes, detach, and delete them (see [EBS-SETUP.md](EBS-SETUP.md)). Stop or terminate the instance.

## Run the remainder of the pipeline

Steps 5–7 work entirely from the GeoTIFFs and GeoPackages written by steps 1–4 and can be run locally. With the outputs copied back:

```bash
source config.sh
./run.sh 2025 --from-step 5
```

## Running tests

```bash
source config.sh
pytest
```

Tests live in [tests/](tests/) and run with the settings in [pytest.ini](pytest.ini).
