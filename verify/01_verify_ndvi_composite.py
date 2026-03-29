"""Verify Step 01 — NDVI composite."""
import logging
import sys

logger = logging.getLogger(__name__)


def main() -> None:
    import config
    from utils.io import read_raster
    from utils.verification import (
        check_ndvi_range, check_nan_fraction, check_catchment_median, check_crs
    )
    from utils.report import VerificationReport, save_report

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        stream=sys.stdout,
    )

    step = "01_ndvi_composite"
    checks_passed = 0
    checks_failed = 0
    messages = []
    status = "PASS"

    path = config.ndvi_median_path(config.YEAR)
    da = read_raster(path)
    if da.ndim == 3:
        da = da.squeeze()

    checks = [
        ("ndvi_range",       lambda: check_ndvi_range(da, "NDVI median")),
        ("nan_fraction",     lambda: check_nan_fraction(da, config.NAN_FRACTION_MAX, "NDVI median")),
        ("catchment_median", lambda: check_catchment_median(
            da, config.CATCHMENT_MEDIAN_NDVI_MIN, config.CATCHMENT_MEDIAN_NDVI_MAX, "NDVI median")),
        ("crs",              lambda: check_crs(da, config.TARGET_CRS, "NDVI median")),
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
        step=step,
        year=config.YEAR,
        status=status,
        checks_passed=checks_passed,
        checks_failed=checks_failed,
        messages=messages,
    )
    save_report(report, config.verification_report_path(config.YEAR))

    icon = "[PASS]" if status == "PASS" else "[FAIL]"
    print(f"{icon} Step 01: NDVI composite — {checks_passed} checks passed"
          + (f", {checks_failed} failed" if checks_failed else ""))

    if status != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
