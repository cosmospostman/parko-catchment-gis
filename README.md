# Parkinsonia Catchment GIS

Detects and maps *Parkinsonia aculeata* (parkinsonia) across a catchment using
Sentinel-2 NDVI composites, flood extent, flowering index, and a random-forest
classifier. Outputs priority-patch polygons and year-on-year change detection.

## Setup

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
export BASE_DIR="${HOME}/parko-data"   # change this line
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
| `--only-step N` | Run a single step only |
| `--rebuild-baseline` | Recompute the NDVI baseline (Landsat 1986–present) |
| `--force` | Clear step sentinels and re-run all steps for the year |
| `--dry-run` | Print steps that would run without executing anything |

The pipeline runs 7 steps in sequence. Completed steps are recorded as
sentinels in `$WORKING_DIR` and skipped on re-runs unless `--force` is passed.
Logs are written to `$BASE_DIR/logs/`.

## Running tests

```bash
source config.sh
pytest
```

Tests live in [tests/](tests/) and run with the settings in [pytest.ini](pytest.ini).
