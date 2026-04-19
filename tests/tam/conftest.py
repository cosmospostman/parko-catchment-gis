"""Shared fixtures for TAM test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tam.config import TAMConfig
from tam.dataset import BAND_COLS
from tam.model import TAMClassifier


@pytest.fixture
def band_cols() -> list[str]:
    return list(BAND_COLS)


@pytest.fixture
def pixel_df(band_cols) -> pd.DataFrame:
    """30 observations for two pixels across one year.
    Seeded for reproducibility. Both pixels have ≥8 clear observations."""
    rng = np.random.default_rng(42)
    rows = []
    for pid in ["px_pres", "px_abs"]:
        dates = pd.date_range("2023-01-15", periods=30, freq="12D")
        for d in dates:
            rows.append({
                "point_id": pid,
                "date": str(d.date()),
                "scl_purity": 1.0,
                "year": 2023,
                **{b: float(rng.uniform(0.01, 0.5)) for b in band_cols},
            })
    return pd.DataFrame(rows)


@pytest.fixture
def labels() -> pd.Series:
    return pd.Series({"px_pres": 1.0, "px_abs": 0.0})


@pytest.fixture
def pixel_coords() -> pd.DataFrame:
    return pd.DataFrame({
        "point_id": ["px_pres", "px_abs"],
        "lon": [144.0, 144.1],
        "lat": [-23.0, -23.5],  # px_abs is further south → goes to val set
    })


@pytest.fixture
def default_cfg() -> TAMConfig:
    return TAMConfig()


@pytest.fixture
def small_model() -> TAMClassifier:
    cfg = TAMConfig(d_model=16, n_heads=2, n_layers=1, d_ff=32)
    return TAMClassifier.from_config(cfg)
