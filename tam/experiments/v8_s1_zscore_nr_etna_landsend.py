"""v8_s1_zscore_nr_etna_landsend — v8_s1_zscore_nr + Etna Creek and Landsend sites.

Extends the best-performing S1 baseline (0.816 val AUC) with two geographically
diverse sites southwest of Norman Road to test generalisation. No holdout added
so results are directly comparable to v8_s1_zscore_nr.
"""

from tam.core.dataset import S1_FEATURE_COLS
from tam.core.experiment import Experiment

EXPERIMENT = Experiment(
    name="v8_s1_zscore_nr_etna_landsend",
    region_ids=[
        # Norman Road
        "norman_road_presence_1", "norman_road_presence_2",
        "norman_road_presence_3", "norman_road_presence_4",
        "norman_road_presence_5", "norman_road_presence_6",
        "norman_road_presence_7", "norman_road_presence_8",
        "norman_road_presence_9",
        "norman_road_absence_1", "norman_road_absence_2",
        "norman_road_absence_3", "norman_road_absence_4",
        "norman_road_absence_5", "norman_road_absence_7",
        # Cloncurry
        "cloncurry_absence_1", "cloncurry_absence_2", "cloncurry_absence_3",
        "cloncurry_absence_4", "cloncurry_absence_5", "cloncurry_absence_6",
        "cloncurry_absence_7",
        # Etna Creek
        "etna_presence_1", "etna_presence_2", "etna_presence_3", "etna_presence_4",
        "etna_absence_1", "etna_absence_2", "etna_absence_3", "etna_absence_4",
        "etna_absence_5",
        # Landsend
        "landsend_presence_1", "landsend_presence_2", "landsend_presence_3",
        "landsend_presence_4",
        "landsend_absence_1", "landsend_absence_2", "landsend_absence_3",
    ],
    feature_cols=S1_FEATURE_COLS,
    model_kwargs={
        "d_model":  64,
        "n_layers": 2,
        "dropout":  0.5,
        "n_bands":  4,
        "n_global_features": 0,
    },
    train_kwargs={
        "lr":              1e-5,
        "weight_decay":    0.1,
        "n_epochs":        60,
        "patience":        15,
        "band_noise_std":  0.0,
        "obs_dropout_min": 4,
        "doy_density_norm": True,
        "doy_phase_shift": True,
        "pixel_zscore":    True,
        "use_s1":          "s1_only",
    },
)
