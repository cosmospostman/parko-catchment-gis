"""Integration tests for run.sh argument parsing and exit codes."""
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
RUN_SH = PROJECT_ROOT / "run.sh"


def _setup_stub_env(tmp_path: Path) -> dict:
    """Create a minimal environment for run.sh integration tests."""
    base = tmp_path / "base"
    cache = tmp_path / "cache"
    working = tmp_path / "working"
    outputs = tmp_path / "outputs"
    logs = outputs / "logs"
    for d in [base, cache, working, outputs, logs]:
        d.mkdir(parents=True, exist_ok=True)

    # Catchment GeoJSON
    import geopandas as gpd
    from shapely.geometry import box
    catchment = gpd.GeoDataFrame(
        {"name": ["mitchell"]},
        geometry=[box(141.0, -17.0, 143.0, -15.0)],
        crs="EPSG:4326",
    )
    catchment_path = base / "mitchell_catchment.geojson"
    catchment.to_file(str(catchment_path), driver="GeoJSON")

    env = os.environ.copy()
    env.update({
        "BASE_DIR":          str(base),
        "CACHE_DIR":         str(cache),
        "WORKING_DIR":       str(working),
        "OUTPUTS_DIR":       str(outputs),
        "CODE_DIR":          str(PROJECT_ROOT),
        "LOG_DIR":           str(logs),
        "CATCHMENT_GEOJSON": str(catchment_path),
    })
    return env, working, outputs


def _write_stub_scripts(project_root: Path, step_nums=range(1, 8)):
    """Write stub analysis and verify scripts that exit 0."""
    step_map = {
        1: ("01_ndvi_composite",    "01_verify_ndvi_composite"),
        2: ("02_ndvi_anomaly",      "02_verify_ndvi_anomaly"),
        3: ("03_flowering_index",   "03_verify_flowering_index"),
        4: ("04_flood_extent",      "04_verify_flood_extent"),
        5: ("05_classifier",        "05_verify_classifier"),
        6: ("06_priority_patches",  "06_verify_priority_patches"),
        7: ("07_change_detection",  "07_verify_change_detection"),
    }
    analysis_dir = project_root / "analysis"
    verify_dir   = project_root / "verify"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    verify_dir.mkdir(parents=True, exist_ok=True)

    for n in step_nums:
        a_name, v_name = step_map[n]
        for d, name in [(analysis_dir, a_name), (verify_dir, v_name)]:
            p = d / f"{name}.py"
            if not p.exists():
                p.write_text('import sys; sys.exit(0)\n')


@pytest.mark.skipif(not RUN_SH.exists(), reason="run.sh not found")
def test_no_args_exits_3():
    """run.sh with no args must exit 3 and print usage."""
    result = subprocess.run(
        ["bash", str(RUN_SH)],
        capture_output=True, text=True,
    )
    assert result.returncode == 3
    assert "Usage" in result.stderr or "Usage" in result.stdout


@pytest.mark.skipif(not RUN_SH.exists(), reason="run.sh not found")
def test_dry_run_exits_0(tmp_path):
    """./run.sh 2025 --dry-run must exit 0 without executing scripts."""
    env, working, outputs = _setup_stub_env(tmp_path)
    result = subprocess.run(
        ["bash", str(RUN_SH), "2025", "--dry-run"],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "Dry run" in result.stdout


@pytest.mark.skipif(not RUN_SH.exists(), reason="run.sh not found")
def test_preflight_fails_nonwritable_dir(tmp_path):
    """Pre-flight must fail with exit 3 when OUTPUTS_DIR is not writable."""
    env, working, outputs = _setup_stub_env(tmp_path)
    outputs.chmod(0o444)  # read-only
    try:
        result = subprocess.run(
            ["bash", str(RUN_SH), "2025", "--dry-run"],
            capture_output=True, text=True, env=env,
        )
        assert result.returncode == 3
    finally:
        outputs.chmod(0o755)


@pytest.mark.skipif(not RUN_SH.exists(), reason="run.sh not found")
def test_sentinel_auto_resume(tmp_path):
    """Steps with existing sentinels must be auto-skipped."""
    env, working, outputs = _setup_stub_env(tmp_path)

    # Get git SHA (or "nogit")
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        sha = "nogit"

    # Plant sentinels for steps 1 and 2
    for nn in ["01", "02"]:
        sentinel = working / f".step_{nn}_complete_2025_{sha}"
        sentinel.touch()

    # Write stub scripts for all steps
    _write_stub_scripts(tmp_path / "stubs", step_nums=range(1, 8))
    env["CODE_DIR"] = str(tmp_path / "stubs")
    # Also need stub analysis/verify dirs
    _write_stub_scripts(PROJECT_ROOT, step_nums=range(1, 8))

    result = subprocess.run(
        ["bash", str(RUN_SH), "2025"],
        capture_output=True, text=True, env=env,
        timeout=60,
    )
    output = result.stdout + result.stderr
    assert "SKIP" in output, f"Expected SKIP in output:\n{output}"
