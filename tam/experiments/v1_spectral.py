"""tam/experiments/v1_spectral.py — First experiment: spectral bands only."""

from tam.core.dataset import BAND_COLS
from tam.core.experiment import Experiment

EXPERIMENT = Experiment(
    name="v1_spectral",
    region_ids=["longreach_presence", "longreach_absence"],
    feature_cols=BAND_COLS,
    model_kwargs={},
    train_kwargs={},
)
