"""Verify Step 07 — change detection."""
import logging
import sys
import numpy as np

logger = logging.getLogger(__name__)


def main() -> None:
    import config
    from utils.io import read_raster
    from utils.verification import check_nan_fraction, check_crs, check_value_range
    from utils.report import VerificationReport, save_report

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                        stream=sys.stdout)

    step = "07_change_detection"
    checks_passed = 0
    checks_failed = 0
    messages = []
    status = "PASS"

    change_path = config.change_detection_path(config.YEAR)

    if not change_path.exists():
        # Year 1 run — no output expected, that's fine
        logger.info("Change detection output not found — Year 1 run, nothing to verify")
        report = VerificationReport(
            step=step, year=config.YEAR, status="PASS",
            checks_passed=0, checks_failed=0,
            messages=["Year 1 run — no prior raster, change detection skipped"],
        )
        save_report(report, config.verification_report_path(config.YEAR))
        print("[PASS] Step 07: Change detection — Year 1 run, skipped")
        return

    da = read_raster(change_path)
    if da.ndim == 3:
        da = da.squeeze()

    valid = da.values[~np.isnan(da.values)]
    change_mean = float(np.mean(valid)) if len(valid) > 0 else float("nan")

    def check_change_mean():
        tol = config.CHANGE_DETECTION_MEAN_TOLERANCE
        if abs(change_mean) > tol:
            raise AssertionError(
                f"Change detection mean {change_mean:.4f} exceeds tolerance ±{tol}"
            )

    # Check resolution matches current probability raster
    def check_resolution():
        prob_da = read_raster(config.probability_raster_path(config.YEAR))
        if prob_da.ndim == 3:
            prob_da = prob_da.squeeze()
        prob_res = prob_da.rio.resolution()
        change_res = da.rio.resolution()
        if abs(abs(prob_res[0]) - abs(change_res[0])) > 1.0:
            raise AssertionError(
                f"Change raster resolution {change_res} differs from "
                f"probability raster resolution {prob_res}"
            )

    checks = [
        ("value_range",   lambda: check_value_range(da, -1.0, 1.0, "change raster")),
        ("nan_fraction",  lambda: check_nan_fraction(da, config.NAN_FRACTION_MAX, "change raster")),
        ("crs",           lambda: check_crs(da, config.TARGET_CRS, "change raster")),
        ("change_mean",   check_change_mean),
        ("resolution",    check_resolution),
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
        details={"change_mean": change_mean},
    )
    save_report(report, config.verification_report_path(config.YEAR))

    icon = "[PASS]" if status == "PASS" else "[FAIL]"
    print(f"{icon} Step 07: Change detection — {checks_passed} checks passed"
          + (f", {checks_failed} failed" if checks_failed else ""))

    if status != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
