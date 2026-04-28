"""tam/experiments/v7_nassau_only.py — Diagnostic: single-site training on Nassau only.

Nassau has riparian native vegetation as absence — a hard negative context unlike
Frenchs (which is dominated by ocean/water/bare soil absence). Tests whether the
model can learn Parkinsonia against spectrally similar native riparian vegetation.

Noise filter disabled (thresholds set to floor/ceiling) because Nassau presence
pixels were fully removed by default thresholds in the LOSO sweep — likely because
Cape York monsoonal Parkinsonia has lower rec_p than the arid/semi-arid calibration.
"""

from tam.core.dataset import ALL_FEATURE_COLS
from tam.core.experiment import Experiment

EXPERIMENT = Experiment(
    name="v7_nassau_only",
    region_ids=[
        "nassau_presence_1", "nassau_presence_2", "nassau_presence_3", "nassau_presence_4",
        "nassau_absence_1", "nassau_absence_2", "nassau_absence_3",
    ],
    feature_cols=ALL_FEATURE_COLS,
    model_kwargs={
        "d_model":  64,
        "n_layers": 2,
        "dropout":  0.5,
    },
    train_kwargs={
        "lr":                    1e-5,
        "weight_decay":          0.1,
        "n_epochs":              60,
        "patience":              10,
        "spatial_stride":        1,
        "band_noise_std":        0.03,
        "obs_dropout_min":       6,
        "presence_min_dry_ndvi": 0.0,
        "presence_min_rec_p":    0.0,
        "presence_grass_nir_cv": 1.0,
    },
)
