"""Smoke tests for utils/console_plot.py.

Verifies that each plot function runs without error and produces output.
Does not assert exact character layouts — those would be brittle.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from analysis.primitives.indices import flowering_index
from analysis.timeseries.observation import Observation, ObservationQuality
from utils.console_plot import plot_distributions, plot_doy_calendar, plot_waveform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(year: int, month: int, day: int, b07: float = 0.20) -> Observation:
    bands = {"B05": 0.08, "B07": b07, "B08": 0.40, "B11": 0.12, "B04": 0.05}
    quality = ObservationQuality(
        scl_purity=0.9, aot=0.8, view_zenith=0.85, sun_zenith=0.75, greenness_z=0.6
    )
    return Observation(
        point_id="test",
        date=datetime(year, month, day, tzinfo=timezone.utc),
        bands=bands,
        quality=quality,
    )


@pytest.fixture()
def obs_list() -> list[Observation]:
    return [
        _make_obs(2021, 3, 15),
        _make_obs(2021, 8, 20, b07=0.38),
        _make_obs(2021, 9, 15, b07=0.40),
        _make_obs(2022, 3, 10),
        _make_obs(2022, 8, 25, b07=0.37),
        _make_obs(2022, 9, 20, b07=0.39),
    ]


# ---------------------------------------------------------------------------
# plot_waveform
# ---------------------------------------------------------------------------

def test_plot_waveform_runs(obs_list, capsys):
    plot_waveform(obs_list, flowering_index, title="smoke test")
    out = capsys.readouterr().out
    assert len(out) > 0


def test_plot_waveform_empty(capsys):
    plot_waveform([], flowering_index)
    out = capsys.readouterr().out
    assert "no observations" in out


def test_plot_waveform_single_obs(capsys):
    plot_waveform([_make_obs(2021, 9, 1)], flowering_index)
    out = capsys.readouterr().out
    assert len(out) > 0


# ---------------------------------------------------------------------------
# plot_distributions
# ---------------------------------------------------------------------------

def test_plot_distributions_runs(capsys):
    presence = [0.6, 0.55, 0.65, 0.58, 0.62]
    absence = [0.35, 0.30, 0.40, 0.28, 0.38]
    plot_distributions(presence, absence, title="smoke test")
    out = capsys.readouterr().out
    assert "median" in out


def test_plot_distributions_empty(capsys):
    plot_distributions([], [])
    out = capsys.readouterr().out
    assert "no data" in out


def test_plot_distributions_one_sided(capsys):
    plot_distributions([0.5, 0.6], [])
    out = capsys.readouterr().out
    assert "median" in out


# ---------------------------------------------------------------------------
# plot_doy_calendar
# ---------------------------------------------------------------------------

def test_plot_doy_calendar_runs(capsys):
    doys = [215, 220, 245, 260, 275, 280, 230]
    plot_doy_calendar(doys, title="smoke test")
    out = capsys.readouterr().out
    assert "Aug" in out
    assert "flowering" in out


def test_plot_doy_calendar_empty(capsys):
    plot_doy_calendar([])
    out = capsys.readouterr().out
    assert "no DOY" in out


def test_plot_doy_calendar_outside_window(capsys):
    plot_doy_calendar([60, 90, 120], title="no window hits")
    out = capsys.readouterr().out
    assert "Aug" in out  # month rows always rendered
