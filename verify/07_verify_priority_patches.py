"""Verify Step 06 — priority patches."""
import logging
import sys

logger = logging.getLogger(__name__)

REQUIRED_ATTRIBUTES = [
    "tier", "area_ha", "prob_mean", "prob_max",
    "dist_to_kowanyama_km", "seed_flux_score", "stream_order",
]


def main() -> None:
    import config
    import geopandas as gpd
    from utils.verification import check_geometry_validity
    from utils.report import VerificationReport, save_report

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                        stream=sys.stdout)

    step = "07_priority_patches"
    checks_passed = 0
    checks_failed = 0
    messages = []
    status = "PASS"

    gdf = gpd.read_file(str(config.priority_patches_path(config.YEAR)))

    def check_crs_gpkg():
        actual = str(gdf.crs)
        if "7844" not in actual and "GDA2020" not in actual:
            raise AssertionError(f"Priority patches CRS unexpected: {actual}")

    def check_min_area():
        if gdf.empty:
            return
        too_small = gdf[gdf["area_ha"] < config.MIN_PATCH_AREA_HA]
        if len(too_small) > 0:
            raise AssertionError(
                f"{len(too_small)} patches below minimum area "
                f"({config.MIN_PATCH_AREA_HA} ha)"
            )

    def check_attributes():
        missing = [a for a in REQUIRED_ATTRIBUTES if a not in gdf.columns]
        if missing:
            raise AssertionError(f"Missing attributes: {missing}")

    def check_tier_diversity():
        if gdf.empty:
            return
        n_tiers = gdf["tier"].nunique()
        if n_tiers < 2:
            raise AssertionError(
                f"Only {n_tiers} distinct tier value(s) — expected >= 2"
            )

    checks = [
        ("geometry_validity", lambda: check_geometry_validity(gdf, "priority patches")),
        ("crs",               check_crs_gpkg),
        ("min_area",          check_min_area),
        ("attributes",        check_attributes),
        ("tier_diversity",    check_tier_diversity),
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
    print(f"{icon} Step 06: Priority patches — {checks_passed} checks passed"
          + (f", {checks_failed} failed" if checks_failed else ""))

    if status != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
