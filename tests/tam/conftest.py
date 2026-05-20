"""Shared fixtures for TAM test suite."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from tam.core.config import TAMConfig
from tam.core.dataset import BAND_COLS
from tam.core.model import TAMClassifier


@pytest.fixture
def band_cols() -> list[str]:
    return list(BAND_COLS)


@pytest.fixture
def pixel_df(band_cols) -> pl.DataFrame:
    """30 observations for two pixels across one year.
    Seeded for reproducibility. Both pixels have ≥8 clear observations."""
    rng = np.random.default_rng(42)
    rows = []
    import datetime
    for pid in ["px_pres", "px_abs"]:
        start = datetime.date(2023, 1, 15)
        dates = [start + datetime.timedelta(days=12 * i) for i in range(30)]
        for d in dates:
            rows.append({
                "point_id": pid,
                "date": str(d),
                "scl_purity": 1.0,
                "year": 2023,
                **{b: float(rng.uniform(0.01, 0.5)) for b in band_cols},
            })
    return pl.DataFrame(rows)


@pytest.fixture
def labels() -> dict[str, float]:
    return {"px_pres": 1.0, "px_abs": 0.0}


@pytest.fixture
def pixel_coords() -> pl.DataFrame:
    return pl.DataFrame({
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
