"""tam/core/experiment.py — Experiment dataclass for named TAM model variants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from tam.core.dataset import BAND_COLS


@dataclass
class Experiment:
    name: str                            # used to name checkpoint dir (e.g. "v1_spectral")
    train_region_ids: list[str]          # training regions (mutually exclusive with val_region_ids)
    val_region_ids: list[str] = field(default_factory=list)   # held-out validation regions
    feature_cols: list[str] = field(default_factory=list)     # band columns to use (subset of BAND_COLS)
    model_kwargs: dict = field(default_factory=dict)   # passed to TAMClassifier(...)
    train_kwargs: dict = field(default_factory=dict)   # passed to train_tam(...)
    preprocess: list[Callable] = field(default_factory=list)  # applied to tile df before training

    @property
    def region_ids(self) -> list[str]:
        """All region IDs (train + val) — used by data-loading callers."""
        return self.train_region_ids + self.val_region_ids
