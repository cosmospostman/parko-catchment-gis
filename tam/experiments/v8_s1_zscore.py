"""v8_s1_zscore — S1 temporal, per-pixel z-score, phase shift, multisite.

Tests whether per-pixel z-scoring fixes the between-site backscatter offset
that causes spatial incoherence at Beaudesert and poor Lake Mueller transfer.

Key difference from v8_s1_phase_multisite_v2: pixel_zscore=True removes each
pixel's own multi-year mean/std before feeding to the transformer. Model must
learn temporal curve shape, not absolute backscatter level.

Hyperparams: best from sweep_multisite_phase (lr5e-6, dropout=0.7, d_model=64).
Train: NR + Cloncurry + Barcoorah + Stockholm. Val holdout: Lake Mueller.
"""

from tam.core.dataset import S1_FEATURE_COLS
from tam.core.experiment import Experiment

EXPERIMENT = Experiment(
    name="v8_s1_zscore",
    region_ids=[
        "norman_road_presence_1", "norman_road_presence_2",
        "norman_road_presence_3", "norman_road_presence_4",
        "norman_road_presence_5", "norman_road_presence_6",
        "norman_road_presence_7", "norman_road_presence_8",
        "norman_road_presence_9",
        "norman_road_absence_1", "norman_road_absence_2",
        "norman_road_absence_3", "norman_road_absence_4",
        "norman_road_absence_5", "norman_road_absence_7",
        "cloncurry_absence_1", "cloncurry_absence_2", "cloncurry_absence_3",
        "cloncurry_absence_4", "cloncurry_absence_5", "cloncurry_absence_6",
        "cloncurry_absence_7",
        "barcoorah_presence", "barcoorah_presence_2", "barcoorah_presence_3",
        "barcoorah_absence_lake", "barcoorah_absence_woodland",
        "barcoorah_absence_2", "barcoorah_absence_3",
        "stockholm_presence_1", "stockholm_presence_2",
        "stockholm_absence_1",
        "lake_mueller_presence", "lake_mueller_presence_2", "lake_mueller_presence_3",
        "lake_mueller_absence", "lake_mueller_absence_2",
    ],
    feature_cols=S1_FEATURE_COLS,
    model_kwargs={
        "d_model":  64,
        "n_layers": 2,
        "dropout":  0.7,
        "n_bands":  4,
        "n_global_features": 0,
    },
    train_kwargs={
        "lr":              5e-6,
        "weight_decay":    0.1,
        "n_epochs":        80,
        "patience":        20,
        "band_noise_std":  0.0,
        "obs_dropout_min": 4,
        "doy_density_norm": True,
        "doy_phase_shift": True,
        "pixel_zscore":    True,
        "use_s1":          "s1_only",
        "val_sites":       ("lake_mueller",),
    },
)
