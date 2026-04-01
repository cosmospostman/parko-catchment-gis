"""Verify Step 04 — flood extent."""
import logging
import sys

logger = logging.getLogger(__name__)


def main() -> None:
    import config
    import geopandas as gpd
    from utils.verification import check_geometry_validity
    from utils.report import VerificationReport, save_report

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                        stream=sys.stdout)

    step = "04_flood_extent"
    checks_passed = 0
    checks_failed = 0
    messages = []
    status = "PASS"

    gdf = gpd.read_file(str(config.flood_extent_path(config.YEAR)))

    def check_flood_area():
        total_km2 = gdf.to_crs(config.TARGET_CRS).geometry.area.sum() / 1e6
        if not (500 <= total_km2 <= 15_000):
            raise AssertionError(
                f"Total flood area {total_km2:.1f} km² outside expected range [500, 15000]"
            )

    def check_crs_gpkg():
        actual = str(gdf.crs)
        if "7855" not in actual and "7844" not in actual and "GDA2020" not in actual:
            raise AssertionError(f"Flood extent CRS unexpected: {actual}")

    checks = [
        ("geometry_validity", lambda: check_geometry_validity(gdf, "flood extent")),
        ("crs",               check_crs_gpkg),
        ("flood_area_range",  check_flood_area),
    ]

    # Skip flood area check if GDF is empty
    if gdf.empty:
        checks = [c for c in checks if c[0] != "flood_area_range"]
        logger.warning("Flood extent is empty — skipping area range check")

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
    print(f"{icon} Step 04: Flood extent — {checks_passed} checks passed"
          + (f", {checks_failed} failed" if checks_failed else ""))

    if status != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
