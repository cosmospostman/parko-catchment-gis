"""Verify Step 05 — plausibility map."""
import logging
import sys

logger = logging.getLogger(__name__)


def main() -> None:
    import numpy as np
    import rioxarray as rxr
    import geopandas as gpd
    import config
    from utils.verification import check_geometry_validity
    from utils.report import VerificationReport, save_report

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                        stream=sys.stdout)

    step = "05_plausibility_map"
    checks_passed = 0
    checks_failed = 0
    messages = []
    status = "PASS"

    raster_path = config.plausibility_map_path(config.YEAR)
    zones_path  = config.plausibility_zones_path(config.YEAR)

    plaus = rxr.open_rasterio(str(raster_path)).squeeze()
    vals  = plaus.values.astype(float)
    valid = vals[np.isfinite(vals)]
    gdf   = gpd.read_file(str(zones_path))

    def check_value_range():
        if valid.size == 0:
            raise AssertionError("All pixels are NaN — no valid plausibility values")
        lo, hi = float(valid.min()), float(valid.max())
        if lo < 0.0 or hi > 1.0:
            raise AssertionError(
                f"Plausibility values outside [0, 1]: min={lo:.4f}, max={hi:.4f}"
            )

    def check_nan_fraction():
        nan_frac = float(np.isnan(vals).mean())
        if nan_frac >= config.NAN_FRACTION_MAX:
            raise AssertionError(
                f"NaN fraction {nan_frac:.1%} >= {config.NAN_FRACTION_MAX:.1%} — check input rasters"
            )

    def check_crs_raster():
        crs_str = str(plaus.rio.crs)
        if "7855" not in crs_str and "GDA2020" not in crs_str:
            raise AssertionError(f"Plausibility raster CRS unexpected: {crs_str}")

    def check_crs_zones():
        if gdf.empty:
            return  # empty is valid; CRS check would be vacuous
        actual = str(gdf.crs)
        if "7855" not in actual and "GDA2020" not in actual:
            raise AssertionError(f"Plausibility zones CRS unexpected: {actual}")

    def check_zones_geometry():
        if not gdf.empty:
            check_geometry_validity(gdf, "plausibility zones")

    def check_area_ha_attribute():
        if gdf.empty:
            return
        if "area_ha" not in gdf.columns:
            raise AssertionError("Plausibility zones missing 'area_ha' attribute")
        computed = gdf.to_crs(config.TARGET_CRS).geometry.area / 1e4
        max_diff = float((gdf["area_ha"] - computed).abs().max())
        if max_diff > 0.01:
            raise AssertionError(
                f"area_ha attribute deviates from geometry area by {max_diff:.4f} ha"
            )

    checks = [
        ("value_range",      check_value_range),
        ("nan_fraction",     check_nan_fraction),
        ("crs_raster",       check_crs_raster),
        ("crs_zones",        check_crs_zones),
        ("zones_geometry",   check_zones_geometry),
        ("area_ha_attribute", check_area_ha_attribute),
    ]

    for name, fn in checks:
        try:
            fn()
            checks_passed += 1
            messages.append(f"PASS {name}")
        except AssertionError as exc:
            checks_failed += 1
            status = "FAIL"
            messages.append(f"FAIL {name}: {exc}")
            logger.error("Check failed [%s]: %s", name, exc)

    report = VerificationReport(
        step=step, year=config.YEAR, status=status,
        checks_passed=checks_passed, checks_failed=checks_failed,
        messages=messages,
    )
    save_report(report, config.verification_report_path(config.YEAR))

    icon = "[PASS]" if status == "PASS" else "[FAIL]"
    print(f"{icon} Step 05: Plausibility map — {checks_passed} checks passed"
          + (f", {checks_failed} failed" if checks_failed else ""))

    if status != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
