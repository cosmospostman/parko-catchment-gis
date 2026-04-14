"""Shared fixtures and skip guard for science validation tests.

Science tests require real Sentinel-2 chip data staged by:
    python pipelines/train.py load-testdata

If chips are not staged (sentinel file absent), the entire science/ directory
is skipped — the main pytest run is unaffected.
"""

from __future__ import annotations

import datetime
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIXTURE_DIR   = PROJECT_ROOT / "tests" / "fixtures"
CHIPS_DIR     = FIXTURE_DIR / "chips"
SENTINEL_FILE = FIXTURE_DIR / ".fixture_commit"
REPORT_PATH   = Path(__file__).parent / "report.md"

# ---------------------------------------------------------------------------
# Skip guard — runs before collection
# ---------------------------------------------------------------------------

collect_ignore_glob: list[str] = []


def pytest_configure(config: pytest.Config) -> None:
    """Skip all science tests if chip data has not been staged."""
    if not SENTINEL_FILE.exists():
        collect_ignore_glob.append(str(Path(__file__).parent / "*.py"))


# ---------------------------------------------------------------------------
# ReportCollector — session-scoped result accumulator
# ---------------------------------------------------------------------------

class ReportCollector:
    """Accumulate per-signal results and write report.md at session end."""

    def __init__(self) -> None:
        self._sections: list[tuple[str, str]] = []  # (heading, body)
        self._start: datetime.datetime = datetime.datetime.now(tz=datetime.timezone.utc)

    def add(self, heading: str, body: str) -> None:
        self._sections.append((heading, body))

    def write(self, science_points: pd.DataFrame) -> None:
        n_presence = int((science_points["label"] == 1).sum())
        n_absence  = int((science_points["label"] == 0).sum())

        commit = "unknown"
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            pass

        lines = [
            "# Science Validation Report",
            f"Generated: {self._start.strftime('%Y-%m-%dT%H:%M:%SZ')}",
            f"Commit: {commit}",
            f"Points: {n_presence} presence, {n_absence} absence",
            "",
        ]
        for heading, body in self._sections:
            lines.append(f"## {heading}")
            lines.append(body)
            lines.append("")

        REPORT_PATH.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def report_collector() -> ReportCollector:
    return ReportCollector()


@pytest.fixture(scope="session", autouse=True)
def _write_report_at_session_end(
    request: pytest.FixtureRequest,
    report_collector: ReportCollector,
    science_points: pd.DataFrame,
) -> Any:
    """Autouse fixture: write report.md at the end of the session."""
    yield
    report_collector.write(science_points)


@pytest.fixture(scope="session")
def science_points() -> pd.DataFrame:
    """Load science_points.csv — columns: point_id, lon, lat, label."""
    return pd.read_csv(FIXTURE_DIR / "science_points.csv")


@pytest.fixture(scope="session")
def observations(science_points: pd.DataFrame) -> dict[str, list]:
    """Extract and quality-score observations for all science points.

    Returns dict keyed by point_id. All years pooled.
    Reads chips from tests/fixtures/chips/.
    """
    import json
    import pystac
    from utils.chip_store import DiskChipStore
    from analysis.timeseries.extraction import extract_observations
    from analysis.primitives.quality import ArchiveStats, score_observation

    points: list[tuple[str, float, float]] = [
        (row.point_id, row.lon, row.lat)
        for row in science_points.itertuples()
    ]

    items_json = FIXTURE_DIR / "stac_items.json"
    items = [pystac.Item.from_dict(d) for d in json.loads(items_json.read_text())]

    store = DiskChipStore(inputs_dir=CHIPS_DIR)
    raw_obs = extract_observations(items, points, store)

    # Compute ArchiveStats and apply greenness_z scoring
    archive_stats = ArchiveStats.from_observations(raw_obs)
    scored = [score_observation(obs, archive_stats) for obs in raw_obs]

    # Group by point_id
    by_point: dict[str, list] = {pid: [] for pid, *_ in points}
    for obs in scored:
        by_point[obs.point_id].append(obs)

    return by_point


@pytest.fixture(scope="session")
def observations_by_year(observations: dict[str, list]) -> dict[int, dict[str, list]]:
    """Split observations by acquisition year.

    Returns dict[year, dict[point_id, list[Observation]]].
    Years present: 2020 (dry), 2021 (typical), 2025 (wet/recent).
    """
    by_year: dict[int, dict[str, list]] = {}
    for point_id, obs_list in observations.items():
        for obs in obs_list:
            year = obs.date.year
            by_year.setdefault(year, {}).setdefault(point_id, []).append(obs)
    return by_year


@pytest.fixture(scope="session")
def waveforms(observations: dict[str, list]) -> dict[str, dict]:
    """Run extract_waveform_features on each point's observations.

    Returns dict keyed by point_id; points returning {} are excluded.
    """
    from analysis.timeseries.waveform import extract_waveform_features
    from analysis.primitives.indices import flowering_index

    result: dict[str, dict] = {}
    for point_id, obs_list in observations.items():
        wf = extract_waveform_features(obs_list, index_fn=flowering_index)
        if wf:
            result[point_id] = wf
    return result


@pytest.fixture(scope="session")
def features(waveforms: dict[str, dict], observations: dict[str, list], science_points: pd.DataFrame) -> pd.DataFrame:
    """Assemble full feature vectors.

    Returns DataFrame with point_id, label, and all feature columns.
    """
    from analysis.timeseries.features import assemble_feature_vector

    labels = science_points.set_index("point_id")["label"].to_dict()
    structural = {"HAND": 0.0, "dist_to_water": 0.0}

    rows = []
    for point_id, wf in waveforms.items():
        obs_list = observations.get(point_id, [])
        fv = assemble_feature_vector(wf, structural, obs_list)
        fv["point_id"] = point_id
        fv["label"] = labels[point_id]
        rows.append(fv)

    return pd.DataFrame(rows)
