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

The pipeline runs 7 steps in sequence. Completed steps are recorded as
sentinels in `$WORKING_DIR` and skipped on re-runs unless `--force` is passed.
Logs are written to `$BASE_DIR/logs/`.

## Run pipeline steps 1–3 on DigitalOcean

Steps 1–3 download several terabytes of COG tiles from `sentinel-cogs.s3.us-west-2.amazonaws.com`. Running them on a DigitalOcean droplet in San Francisco (SFO3) avoids routing that traffic over a local internet connection.

**Droplet spec:** Ubuntu 24.04 LTS, `c2-8vcpu-16gb` (CPU-optimised), region SFO3. Add your SSH key during creation.

**First-time setup** — copy the catchment file up, then SSH in and install dependencies:

```bash
scp $BASE_DIR/mitchell_catchment.geojson root@<droplet-ip>:/data/mrc-parko/
ssh root@<droplet-ip>
```

```bash
sudo apt-get update && sudo apt-get install -y python3-pip python3-venv git gdal-bin libgdal-dev
git clone <your-repo-url> parko-catchment-gis
cd parko-catchment-gis
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sudo mkdir -p /data/mrc-parko && sudo chown $USER /data/mrc-parko
```

**Each run** — use `tmux` so the pipeline survives SSH disconnection:

```bash
ssh root@<droplet-ip>
cd parko-catchment-gis && source .venv/bin/activate
tmux new -s pipeline
./run.sh 2025 --to-step 3
# Ctrl+B then D to detach; tmux attach -t pipeline to return
```

**Copy outputs back** once all three steps complete:

```bash
scp -r root@<droplet-ip>:/data/mrc-parko/outputs/2025 ./outputs/
```

Then power off or destroy the droplet.

## Run the remainder of the pipeline

Steps 4–7 work entirely from the GeoTIFFs written by steps 1–3 and can be run locally. With the outputs copied back from DO:

```bash
source config.sh
./run.sh 2025 --from-step 4
```

## Running tests

```bash
source config.sh
pytest
```

Tests live in [tests/](tests/) and run with the settings in [pytest.ini](pytest.ini).
