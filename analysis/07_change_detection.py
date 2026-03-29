"""
Step 07 — Year-on-year probability change detection.

Produces: change_detection_{year}.tif  (COG, EPSG:7844, float32 in [-1,1])

Exits 0 with a log message (no output file) when no prior-year raster exists (Year 1 run).
"""
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def main() -> None:
    import config
    from utils.io import ensure_output_dirs, read_raster, write_cog
    from utils.quicklook import save_quicklook

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        stream=sys.stdout,
    )

    ensure_output_dirs(config.YEAR)

    prior_year = config.YEAR - 1
    prior_path = config.probability_raster_path(prior_year)

    if not prior_path.exists():
        logger.info(
            "No prior-year probability raster found for %d — "
            "this appears to be a Year 1 run. Skipping change detection.",
            prior_year,
        )
        return

    current_path = config.probability_raster_path(config.YEAR)
    if not current_path.exists():
        raise FileNotFoundError(f"Step 05 output not found: {current_path}")

    current_da = read_raster(current_path).squeeze()
    prior_da   = read_raster(prior_path).squeeze()

    # Check CRS match
    current_crs = str(current_da.rio.crs)
    prior_crs   = str(prior_da.rio.crs)
    if current_crs.upper() != prior_crs.upper():
        raise ValueError(
            f"CRS mismatch between years: current={current_crs}, prior={prior_crs}"
        )

    # Reproject prior to match current grid
    prior_reproj = prior_da.rio.reproject_match(current_da)

    change = current_da - prior_reproj
    change = change.clip(-1.0, 1.0)
    change = change.rio.write_crs(config.TARGET_CRS)

    out_path = config.change_detection_path(config.YEAR)
    write_cog(change, out_path)
    logger.info("Written: %s", out_path)

    # Log summary statistics
    valid = change.values[~np.isnan(change.values)]
    if len(valid) > 0:
        logger.info(
            "Change stats: mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
            float(np.mean(valid)), float(np.std(valid)),
            float(np.min(valid)), float(np.max(valid)),
        )

    ql_path = Path(str(out_path).replace(".tif", "_quicklook.png"))
    save_quicklook(
        change,
        ql_path,
        vmin=-0.3,
        vmax=0.3,
        cmap="RdBu_r",
        title=f"Probability Change {prior_year}→{config.YEAR}",
    )


if __name__ == "__main__":
    main()
