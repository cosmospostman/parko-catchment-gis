# Parkinsonia Catchment GIS

Detects and maps *Parkinsonia aculeata* (parkinsonia) across a catchment using
Sentinel-2 spectral time series. A training pipeline builds a random-forest
classifier from ALA occurrence records; an inference pipeline applies it to
produce probability and priority-patch rasters.

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
[config.py](config.py) to set its location (defaults to `/data/mrc-parko`):

```python
BASE_DIR = Path("/data/mrc-parko")   # change this line
```

You must also place the catchment boundary file at:

```
$BASE_DIR/mitchell_catchment.geojson
```

Fetch it from the Queensland Government drainage basins dataset:

```bash
curl -s "https://spatial-gis.information.qld.gov.au/arcgis/rest/services/InlandWaters/DrainageBasins/MapServer/1/query?where=BASIN_NAME+LIKE+%27%25Mitchell%25%27&f=geojson&outFields=*" \
  -o "/data/mrc-parko/mitchell_catchment.geojson"
```

## Running the pipeline

**Collect training pixels** — fetch per-tile Sentinel-2 parquets for labeled regions:

```bash
python -m utils.training_collector ensure --all \
    --start 2020-01-01 --end 2025-12-31
```

**Train** — fit a TAM classifier for a named experiment:

```bash
python -m tam.pipeline train --experiment v1_spectral
```

**Score** — apply the trained model to a location:

```bash
python -m tam.pipeline score \
    --checkpoint outputs/tam-v1_spectral \
    --location frenchs --end-year 2025
```

## Running tests

```bash
pytest
```

Tests live in [tests/](tests/) and run with the settings in [pytest.ini](pytest.ini).

## Deployment (DigitalOcean droplet)

After cloning the repo on the droplet:

```bash
# 1. Bootstrap — installs Deno, sets up Python venv, fetches ALA sightings
bash ui/production/setup.sh

# 2. Copy ranking CSVs from your dev machine (run locally)
rsync -avz --include="*/" --include="*.csv" --exclude="*" \
  outputs/ <droplet>:/path/to/repo/outputs/

# 3. Start the server inside screen (auto-respawns on crash, port 80)
screen -S parko bash ui/production/run.sh
```

To refresh sightings data later without re-running full setup:

```bash
bash ui/production/fetch-sightings.sh
```
