# Mitchell Parkinsonia GIS Pipeline — Implementation Plan

---

## 1. Repository scaffolding and config layer

**`config.sh`** — single source of truth for all filesystem paths, always sourced before any Python subprocess. Beyond the DESIGN.md sketch, add `CODE_DIR` (so Python scripts can resolve `data/` without hardcoding) and `LOG_DIR` (for persistent run logs under `OUTPUTS_DIR`).

**`config.py`** — imported by every script; fails loudly at import time if any required env var is absent. All scientific parameters are defined here as script-level constants (not from env — they belong in version-controlled code):

- Path constants: `BASE_DIR`, `CACHE_DIR`, `WORKING_DIR`, `OUTPUTS_DIR`, `CODE_DIR`, `CATCHMENT_GEOJSON`
- Runtime constants: `YEAR` (int), `COMPOSITE_START`, `COMPOSITE_END`
- Analysis constants: `TARGET_CRS = "EPSG:7844"`, `TARGET_RESOLUTION = 10`, `CLOUD_COVER_MAX = 20`, `COMPOSITE_BANDS`, `FLOWERING_WINDOW_START/END`, `FLOOD_SEASON_START/END`, `BASELINE_START_YEAR = 1986`, `PROBABILITY_THRESHOLD = 0.6`, `MIN_PATCH_AREA_HA = 0.25`, `RF_N_ESTIMATORS = 500`, `SPATIAL_BLOCK_SIZE_KM = 50`, `NAN_FRACTION_MAX = 0.20`, `CATCHMENT_MEDIAN_NDVI_MIN/MAX`, `NDVI_ANOMALY_MIN/MAX_MEAN`, `NDVI_ANOMALY_MIN/MAX_STD`, `FLOWERING_ANOMALY_CORRELATION_MAX = 0.70`, `TARGET_OVERALL_ACCURACY = 0.85`, `TARGET_RECALL = 0.80`, `CHANGE_DETECTION_MEAN_TOLERANCE = 0.05`
- Data source constants — things a pipeline operator might change without touching analysis code:
  - Endpoint URLs: `STAC_ENDPOINT_ELEMENT84`, `STAC_ENDPOINT_CDSE`, `DEA_S3_BUCKET`, `ALA_API_BASE`
  - Collection names: `S2_COLLECTION = "sentinel-2-l2a"`, `S1_COLLECTION = "sentinel-1-grd"`, `DEA_COLLECTION = "ga_ls_ard_3"`, `FC_COLLECTION = "ga_ls_fc_3"`
  - Species query: `ALA_SPECIES_QUERY = "Parkinsonia aculeata"`
- **Output filename template functions**: `ndvi_median_path(year)`, `ndvi_anomaly_path(year)`, `flowering_index_path(year)`, etc. — centralised so a rename requires editing one file.

The test for whether a constant belongs in `config.py` vs. an individual script: *would a pipeline operator plausibly need to change this without touching analysis code?* Collection names and endpoints — yes. Band lists, chunk sizes, and algorithm parameters — no; those stay in the script that uses them, where their meaning is clear from context.

---

## 2. Utility modules (`utils/`)

Five small modules warranted to avoid copy-paste across seven scripts:

| Module | Responsibilities |
|---|---|
| `utils/io.py` | `write_cog()`, `read_raster()`, `ensure_output_dirs(year)` |
| `utils/stac.py` | `search_sentinel2()`, `search_sentinel1()`, `load_stackstac()`, `load_dea_landsat()` — all mockable in tests |
| `utils/mask.py` | `apply_scl_mask()`, `apply_s2cloudless_mask()`, `apply_habitat_mask()` |
| `utils/sar.py` | `preprocess_s1_scene()` — wrapper around sarsen/s1-reader; isolated so tests can mock it |
| `utils/verification.py` | `check_ndvi_range()`, `check_nan_fraction()`, `check_value_range()`, `check_crs()`, `check_geometry_validity()` — all raise `AssertionError` with descriptive messages; shared between verify scripts and tests |
| `utils/report.py` | `VerificationReport` class; `save_report()` / `load_report()` roundtrip to JSON with atomic append |
| `utils/quicklook.py` | `save_quicklook()` — PNG thumbnail per step |

---

## 3. CLI harness (`run.sh`) — ergonomics beyond the sketch

**Argument interface:**
```
./run.sh YEAR [--composite-start MM-DD] [--composite-end MM-DD]
              [--from-step N] [--only-step N] [--dry-run] [--rebuild-baseline]
```

**Automatic resume:** After each step+verify pair passes, the harness writes a sentinel file to `$WORKING_DIR/.step_NN_complete_YYYY_<git-sha>`. On the next run, `run_step` checks for the sentinel and skips the step if found, logging `[SKIP] step already complete`. Re-running `./run.sh 2025` after a failure automatically picks up from the failed step with no manual intervention. Sentinels are keyed to both the year and the short git SHA so a code change between runs invalidates them and forces a re-run from that step onward.

**Manual resume / step selection:** `--from-step N` overrides sentinel logic and forces restart from step N (useful when you want to re-run a step whose sentinel exists, e.g. after fixing a bug). `--only-step N` runs exactly one step and its verify. `--force` clears all sentinels for the given year before running.

**Persistent log file:** Every run writes to `$LOG_DIR/run_YYYY_TIMESTAMP.log` via `tee`; terminal and log get the same output.

**Structured step timing:** Wall-clock start/end per step; summary table printed at completion.

**Better error reporting:** On failure, the harness extracts the last 20 lines of stderr (the traceback tail) and prints them with a highlighted `FAILED:` prefix. No hunting through logs.

**Pre-flight checks** (run before any analysis):
- `mitchell_catchment.geojson` exists
- `WORKING_DIR` and `OUTPUTS_DIR` are writable
- `python -c "import stackstac, odc, rioxarray, sklearn, geopandas"` succeeds
- For year > 1: prior year's probability raster exists (Step 07 prerequisite)

**Exit codes:**
- `0` — all steps passed
- `1` — analysis step crashed
- `2` — verify step failed (analysis ran but science checks didn't pass — different problem)
- `3` — pre-flight check failed

**Verify script naming fix:** The DESIGN.md sketch has a sed substitution bug. Pass both names explicitly: `run_step "01_ndvi_composite" "01_verify_ndvi_composite"`.

**Git tag suggestion:** Printed at end but not auto-executed (auto-tagging every run creates noise).

---

## 4. Analysis scripts (one per step)

Each script:
- Has a **script-level constants block** importing from `config.py` and declaring any step-specific overrides
- Has a `main()` function (required for testability — tests call `main()` directly)
- Uses `logging` (not `print`) so the harness captures structured output

### Step-specific constants

| Step | Key script-level constants (algorithm details — not in `config.py`) |
|---|---|
| 01 | `SCL_CLEAR_CLASSES = [4,5,6]`, `S2CLOUDLESS_PROB_MAX = 0.4`, `DASK_CHUNK_SPATIAL = 2048` |
| 02 | `DEA_BANDS = ["nbart_red","nbart_nir"]`, `FC_PV_BAND = "pv"`, `RESAMPLING_METHOD = "bilinear"`, `BASELINE_CACHE_FILENAME` |
| 03 | `FLOWERING_BANDS = ["B03","B08","B05","B06"]`, `GREEN_NIR_RATIO_NODATA`, `NDRE_NODATA` |
| 04 | `S1_POLARISATIONS = ["VV","VH"]`, `VV_OPEN_WATER_THRESHOLD_DB = -14.0`, `FLOOD_UNION_SIMPLIFY_TOLERANCE = 20` |
| 05 | `FEATURE_NAMES`, `GLCM_KERNEL_SIZE = 7`, `GLCM_ENABLED = True`, `PSEUDO_ABSENCE_BUFFER_KM = 2.0`, `RF_CLASS_WEIGHT = "balanced"`, `MODEL_CACHE_PATH` |
| 06 | `TIER_THRESHOLDS = {"A":0.85,"B":0.75,...}`, `SEED_FLUX_STREAM_ORDER_WEIGHTS`, `KOWANYAMA_COORDS`, `OUTPUT_ATTRIBUTES` |
| 07 | `SIGNIFICANT_INCREASE_THRESHOLD = 0.15`, `SIGNIFICANT_DECREASE_THRESHOLD = -0.15` |

Step 02 checks for a `--rebuild-baseline` env flag and reuses the cached Landsat baseline otherwise (the most expensive one-time download). Step 04 checks at startup whether `sarsen` is importable and logs a clear SNAP fallback warning if not. Step 07 exits 0 with a log message (not a failure) when it's a Year 1 run with no prior raster.

---

## 5. Verification scripts

Each verify script:
1. Loads the corresponding analysis output
2. Runs assertions via `utils.verification` helpers
3. Appends a `VerificationReport` entry to `verification_report_YYYY.json`
4. Exits 0 (pass) or 2 (fail — not a crash)
5. Prints a one-line summary: `[PASS] Step 01: NDVI composite — 4 checks passed`

| Script | Key assertions |
|---|---|
| 01 | NDVI in [-1,1]; NaN fraction < 0.20; catchment median in [0.15, 0.50]; CRS correct |
| 02 | Anomaly mean near zero (±0.05); std in [0.03, 0.20]; NaN fraction; CRS |
| 03 | Green/NIR ratio in [0.01, 10]; NDRE in [-1,1]; **correlation with NDVI anomaly < 0.70** |
| 04 | All geometries valid; CRS; total flood area in plausible range (500–15,000 km²) |
| 05 | CV accuracy ≥ 0.85; recall ≥ 0.80; **top feature is `ndvi_anomaly` or `flowering_index`** (not geography) |
| 06 | All geometries valid; area ≥ 0.25 ha; CRS = EPSG:7844; all attributes present; tier distribution has ≥ 2 distinct values |
| 07 | Change mean near zero (±0.05); values in [-1,1]; CRS/resolution match |

---

## 6. Test suite

### Structure

```
tests/
├── conftest.py                    # shared fixtures
├── fixtures/                      # synthetic rasters/vectors, mini catchment GeoJSON
├── unit/                          # utils/config tests
├── analysis/                      # test_0N_*.py per step
├── verify/                        # test_0N_verify_*.py per step
└── integration/
    └── test_harness.py            # run.sh argument parsing, exit codes
```

### Fixtures (`conftest.py`)

- `tmp_dirs(tmp_path)` — sets all required env vars to `tmp_path` subdirs; tears down after test
- `synthetic_ndvi_raster` — 50×50 DataArray, values from N(0.35, 0.08), GDA2020 CRS, seeded with `np.random.default_rng(42)`
- `synthetic_anomaly_raster`, `synthetic_probability_raster` (Beta(2,5)), `synthetic_flood_gdf`, `synthetic_patches_gdf`

### Mocking strategy

External calls are never made in tests:

| External call | Mock mechanism |
|---|---|
| `pystac_client.Client.open()` | `pytest-mock` patch; returns synthetic `pystac.Item` list |
| `stackstac.stack()` | Patched to return pre-built DataArray fixture |
| `odc.stac.load()` | Patched similarly |
| ALA API | `responses` library; returns canned JSON with synthetic occurrence records |
| `utils.sar.preprocess_s1_scene()` | Patched to return synthetic sigma-naught array |

All tests use `dask.config.set(scheduler="synchronous")` — no distributed cluster, total suite runtime target < 30 seconds.

### Key test cases per layer

**Unit (`tests/unit/`):**
- `test_config.py`: raises `KeyError` if env var missing; `YEAR` cast to int; path template functions return expected paths
- `test_utils_verification.py`: each assertion function passes for valid data; raises `AssertionError` with descriptive message for bad data

**Analysis (`tests/analysis/`):**
- Happy-path `main()` call with mocked loaders; assert output file written, values in range
- Step 02: assert baseline cache is reused on second call (mock call count check)
- Step 06: edge case — probability raster entirely below threshold produces empty patches file
- Step 07: "Year 1 run" path — no prior raster present → exits 0, no output file created; CRS mismatch → clear error

**Verification (`tests/verify/`):**
- Happy path passes; each failure mode raises the expected `AssertionError` with the right message
- Step 05: `dist_to_watercourse` as top feature → fails with geographic overfitting message
- Step 06: single-tier output → fails; invalid geometry → fails; wrong CRS → fails

**Integration (`tests/integration/test_harness.py`):**
- Tests `run.sh` via `subprocess.run()` with stub analysis scripts (Python one-liners that write a sentinel file and exit 0)
- `./run.sh` with no args → exit 3, usage message
- `./run.sh 2025 --dry-run` → exit 0, no scripts executed
- Pre-flight failure (non-writable dir) → exit 3, clear message
- `--only-step 1` → only step 1 and its verify run
- `--from-step 3` → steps 1 and 2 skipped
- Retry after partial run: sentinel files for steps 1 and 2 present → steps 1 and 2 auto-skipped, step 3 runs
- `--force` → all sentinels cleared, all steps run regardless

---

## 7. Implementation order

1. `config.sh`, `config.py`
2. `utils/io.py`, `utils/verification.py`, `utils/report.py`
3. `utils/stac.py`, `utils/mask.py`, `utils/sar.py`, `utils/quicklook.py`
4. `run.sh` (all step names now known)
5. `tests/conftest.py` + `tests/fixtures/`
6. Steps 01–07: implement `analysis/0N_*.py` + `verify/0N_verify_*.py` + their tests together as a unit, in order
7. `tests/integration/test_harness.py` (last — needs all step stubs to exist)

`requirements.txt` pins major versions (`stackstac>=0.5,<0.6`); `requirements-dev.txt` covers `pytest`, `pytest-cov`, `pytest-mock`, `pytest-env`, `responses`.

---

## Notes

**Landsat baseline cache invalidation:** The Landsat baseline (`ndvi_baseline_median.tif`) embeds its year range in the GeoTIFF `IMAGEDESCRIPTION` tag. Step 02 reads this tag and rebuilds automatically if the cached range differs from `BASELINE_START_YEAR` to `YEAR-1`.

**`verification_report_YYYY.json` accumulation:** Each verify script appends its entry to the same JSON file. `utils.report.save_report()` reads the existing list, appends the new entry, and writes the whole file back atomically.

**`data/drainage_network.gpkg`:** Required by Steps 05 and 06 (Strahler stream order for seed flux ranking). Not committed to Git due to size — document in README as a required manual download.
