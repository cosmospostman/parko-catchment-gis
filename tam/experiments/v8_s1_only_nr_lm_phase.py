"""v8_s1_only_nr_lm_phase — S1 temporal only, NR+LM, full-year phase shift augmentation.

Same as v8_s1_only_nr_lm but with doy_phase_shift=True: each training window
is randomly shifted by 0–364 days with wraparound, forcing the model to learn
the shape of the VH/VV curve relative to concurrent moisture indicators rather
than absolute calendar timing. Tests whether phase-invariant training improves
cross-site transferability.
"""

from tam.core.dataset import S1_FEATURE_COLS
from tam.core.experiment import Experiment

EXPERIMENT = Experiment(
    name="v8_s1_only_nr_lm_phase",
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
        "lake_mueller_presence", "lake_mueller_presence_2", "lake_mueller_presence_3",
        "lake_mueller_absence", "lake_mueller_absence_2",
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
        "patience":        20,
        "band_noise_std":  0.0,
        "obs_dropout_min": 4,
        "doy_density_norm": True,
        "doy_phase_shift": True,
        "use_s1":          "s1_only",
    },
)
