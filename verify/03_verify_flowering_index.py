"""Verify Step 03 — flowering index."""
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

    step = "03_flowering_index"
    checks_passed = 0
    checks_failed = 0
    messages = []
    status = "PASS"

    flowering_da = read_raster(config.flowering_index_path(config.YEAR))
    if flowering_da.ndim == 3:
        flowering_da = flowering_da.squeeze()

    ndvi_da = read_raster(config.ndvi_anomaly_path(config.YEAR))
    if ndvi_da.ndim == 3:
        ndvi_da = ndvi_da.squeeze()

    # Align flowering to NDVI grid for correlation check
    flowering_aligned = flowering_da.rio.reproject_match(ndvi_da)

    fi_vals   = flowering_aligned.values.ravel()
    ndvi_vals = ndvi_da.values.ravel()
    valid_mask = ~(np.isnan(fi_vals) | np.isnan(ndvi_vals))
    fi_v   = fi_vals[valid_mask]
    ndvi_v = ndvi_vals[valid_mask]

    def check_correlation():
        if len(fi_v) < 10:
            return  # not enough data to check
        corr = float(np.corrcoef(fi_v, ndvi_v)[0, 1])
        if abs(corr) >= config.FLOWERING_ANOMALY_CORRELATION_MAX:
            raise AssertionError(
                f"Flowering index vs NDVI anomaly correlation {corr:.3f} >= "
                f"{config.FLOWERING_ANOMALY_CORRELATION_MAX} — signals not independent"
            )

    checks = [
        ("green_nir_range", lambda: check_value_range(
            flowering_da, 0.01, 10.0, "green/NIR ratio")),
        ("nan_fraction",    lambda: check_nan_fraction(
            flowering_da, config.NAN_FRACTION_MAX, "flowering index")),
        ("crs",             lambda: check_crs(
            flowering_da, config.TARGET_CRS, "flowering index")),
        ("ndvi_correlation", check_correlation),
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
    print(f"{icon} Step 03: Flowering index — {checks_passed} checks passed"
          + (f", {checks_failed} failed" if checks_failed else ""))

    if status != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
