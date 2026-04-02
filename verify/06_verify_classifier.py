"""Verify Step 05 — classifier."""
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

GEOGRAPHIC_FEATURES = {"dist_to_watercourse"}


def main() -> None:
    import config
    from utils.io import read_raster
    from utils.verification import check_nan_fraction, check_crs, check_value_range
    from utils.report import VerificationReport, save_report

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                        stream=sys.stdout)

    step = "06_classifier"
    checks_passed = 0
    checks_failed = 0
    messages = []
    status = "PASS"

    prob_path = config.probability_raster_path(config.YEAR)
    prob_da = read_raster(prob_path)
    if prob_da.ndim == 3:
        prob_da = prob_da.squeeze()

    # Load cached model for feature importance check
    model_cache = Path(config.CACHE_DIR) / f"rf_model_{config.YEAR}.pkl"
    cv_scores = None
    top_feature = None
    if model_cache.exists():
        with open(model_cache, "rb") as f:
            cache = pickle.load(f)
        clf = cache["model"]
        feature_names = cache.get("feature_names", [])
        cv_scores = cache.get("cv_scores")
        importances = cache.get("feature_importances", clf.feature_importances_)
        if len(importances) == len(feature_names):
            top_idx = int(np.argmax(importances))
            top_feature = feature_names[top_idx]

    def check_cv_accuracy():
        if cv_scores is None:
            raise AssertionError("CV scores not found in model cache")
        mean_acc = float(np.mean(cv_scores))
        if mean_acc < config.TARGET_OVERALL_ACCURACY:
            raise AssertionError(
                f"CV accuracy {mean_acc:.3f} < target {config.TARGET_OVERALL_ACCURACY}"
            )

    def check_top_feature():
        if top_feature is None:
            raise AssertionError("Could not determine top feature from model cache")
        if top_feature in GEOGRAPHIC_FEATURES:
            raise AssertionError(
                f"Top feature is '{top_feature}' (geographic proxy) — "
                "model may be geographically overfitting rather than learning spectral signal"
            )

    checks = [
        ("prob_range",    lambda: check_value_range(prob_da, 0.0, 1.0, "probability raster")),
        ("nan_fraction",  lambda: check_nan_fraction(prob_da, config.NAN_FRACTION_MAX, "probability raster")),
        ("crs",           lambda: check_crs(prob_da, config.TARGET_CRS, "probability raster")),
        ("cv_accuracy",   check_cv_accuracy),
        ("top_feature",   check_top_feature),
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
        details={"top_feature": top_feature, "cv_mean": float(np.mean(cv_scores)) if cv_scores is not None else None},
    )
    save_report(report, config.verification_report_path(config.YEAR))

    icon = "[PASS]" if status == "PASS" else "[FAIL]"
    print(f"{icon} Step 05: Classifier — {checks_passed} checks passed"
          + (f", {checks_failed} failed" if checks_failed else ""))

    if status != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
