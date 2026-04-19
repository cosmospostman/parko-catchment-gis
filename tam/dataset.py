"""tam/dataset.py — re-export from tam.core.dataset for backwards compatibility."""
from tam.core.dataset import (
    BAND_COLS, N_BANDS, MAX_SEQ_LEN, MIN_OBS_PER_YEAR,
    TAMSample, collate_fn, TAMDataset,
)

__all__ = ["BAND_COLS", "N_BANDS", "MAX_SEQ_LEN", "MIN_OBS_PER_YEAR",
           "TAMSample", "collate_fn", "TAMDataset"]
