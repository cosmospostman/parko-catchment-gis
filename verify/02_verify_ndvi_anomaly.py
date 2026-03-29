"""Verify Step 02 — NDVI anomaly."""
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

    step = "02_ndvi_anomaly"
    checks_passed = 0
    checks_failed = 0
    messages = []
    status = "PASS"

    path = config.ndvi_anomaly_path(config.YEAR)
    da = read_raster(path)
    if da.ndim == 3:
        da = da.squeeze()

    valid = da.values[~np.isnan(da.values)]
    anomaly_mean = float(np.mean(valid)) if len(valid) > 0 else float("nan")
    anomaly_std  = float(np.std(valid))  if len(valid) > 0 else float("nan")

    def check_mean():
        tol = config.CHANGE_DETECTION_MEAN_TOLERANCE
        if abs(anomaly_mean) > tol:
            raise AssertionError(
                f"NDVI anomaly mean {anomaly_mean:.4f} exceeds tolerance ±{tol} "
                "(baseline may not be well-calibrated)"
            )

    def check_std():
        if not (config.NDVI_ANOMALY_MIN_STD <= anomaly_std <= config.NDVI_ANOMALY_MAX_STD):
            raise AssertionError(
                f"NDVI anomaly std {anomaly_std:.4f} outside expected range "
                f"[{config.NDVI_ANOMALY_MIN_STD}, {config.NDVI_ANOMALY_MAX_STD}]"
            )

    checks = [
        ("anomaly_mean",  check_mean),
        ("anomaly_std",   check_std),
        ("nan_fraction",  lambda: check_nan_fraction(da, config.NAN_FRACTION_MAX, "NDVI anomaly")),
        ("crs",           lambda: check_crs(da, config.TARGET_CRS, "NDVI anomaly")),
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
        details={"anomaly_mean": anomaly_mean, "anomaly_std": anomaly_std},
    )
    save_report(report, config.verification_report_path(config.YEAR))

    icon = "[PASS]" if status == "PASS" else "[FAIL]"
    print(f"{icon} Step 02: NDVI anomaly — {checks_passed} checks passed"
          + (f", {checks_failed} failed" if checks_failed else ""))

    if status != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
