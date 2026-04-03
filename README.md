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

**Training** — fetch chips and train a classifier:

```bash
python pipelines/train.py --year 2024
```

**Inference** — apply the trained model to produce output rasters:

```bash
python pipelines/infer.py --year 2025
```

## Running tests

```bash
pytest
```

Tests live in [tests/](tests/) and run with the settings in [pytest.ini](pytest.ini).
