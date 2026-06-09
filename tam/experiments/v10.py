"""v10 — joint S1+S2 TAM, identical site set to v9_spectral.

S1+S2 joint leg of the three-way comparison (S2-only / S1-only / joint S1+S2).
Feature set: B02 B03 B04 B05 B06 B07 B08 B8A B11 B12 + NDVI + NDWI + MAVI + NDRE + CI_RE (15 S2)
             + s1_vh + s1_vv + s1_vh_vv + s1_rvi (4 S1) = 19 bands total.
"""

from tam.core.dataset import V10_FEATURE_COLS

V10_S1_FEATURE_COLS = ["s1_vh", "s1_vv", "s1_vh_vv", "s1_rvi"]
from tam.core.experiment import Experiment

EXPERIMENT = Experiment(
    name="v10",
    train_region_ids=[
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
        # Etna Creek — train-only regions
        "etna_presence_2",
        "etna_presence_5", "etna_presence_6", "etna_presence_7",
        "etna_presence_8", "etna_presence_9",
        "etna_absence_6", "etna_absence_7",
        "etna_absence_8", "etna_absence_9", "etna_absence_10",
        "etna_absence_11", "etna_absence_12",
        # Landsend — train-only regions
        "landsend_presence_1", "landsend_presence_2", "landsend_presence_3",
        "landsend_presence_4", "landsend_presence_5", "landsend_presence_6",
        "landsend_presence_7",
        "landsend_sparse_presence_1", "landsend_sparse_presence_2",
        "landsend_sparse_presence_3", "landsend_sparse_presence_4",
        "landsend_sparse_presence_5",
        "landsend_absence_1", "landsend_absence_2", "landsend_absence_3",
        "landsend_absence_grass_1", "landsend_absence_grass_2",
        "landsend_absence_riverbed_1", "landsend_absence_riverbed_2",
        "landsend_absence_riverbed_3",
        # Lake Mueller
        "lake_mueller_presence", "lake_mueller_presence_2",
        "lake_mueller_presence_3", "lake_mueller_presence_4",
        "lake_mueller_absence", "lake_mueller_absence_2",
        "lake_mueller_absence_3",
        "lake_mueller_absence_4", "lake_mueller_absence_5",
        "lake_mueller_absence_6",
        # Corfield
        "corfield_presence_1", "corfield_presence_2", "corfield_presence_3",
        "corfield_presence_4", "corfield_presence_5", "corfield_presence_6",
        "corfield_absence_1", "corfield_absence_2", "corfield_absence_3",
        # Roper River
        "roper_presence_1", "roper_presence_2", "roper_presence_3",
        "roper_presence_4",
        "roper_absence_1", "roper_absence_2", "roper_absence_3",
        # Hughenden — train-only regions
        "hughenden_presence_4",
        "hughenden_absence_3", "hughenden_absence_5",
        # Burdekin — train-only regions
        "burdekin_presence_1", "burdekin_presence_2",
        "burdekin_absence_1", "burdekin_absence_2", "burdekin_absence_3",
        "burdekin_absence_4", "burdekin_absence_5",
        "burdekin_absence_6", "burdekin_absence_7", "burdekin_absence_8",
        # Maria Downs — train-only regions
        "maria_downs_presence", "maria_downs_absence",
        "maria_downs_presence_2", "maria_downs_absence_2",
        # Rupert Creek — train-only regions
        "rupert_ck_presence_1", "rupert_ck_presence_2", "rupert_ck_presence_3",
        "rupert_ck_presence_sparse_1",
        "rupert_ck_absence_1", "rupert_ck_absence_2", "rupert_ck_absence_3",
        # Quaids — Cape York Peninsula (monsoonal savanna)
        "quaids_absence_1", "quaids_absence_3", "quaids_absence_5",
        "quaids_absence_7", "quaids_absence_9",
        # Mitchell River — train regions
        "mitchell_presence_1", "mitchell_presence_2", "mitchell_presence_3",
        "mitchell_presence_4", "mitchell_presence_5",
        "mitchell_absence_mangrove_1", "mitchell_absence_mangrove_2",
        "mitchell_absence_mangrove_3",
        "mitchell_absence_bare_1", "mitchell_absence_bare_2",
        "mitchell_absence_riparian_1", "mitchell_absence_riparian_2",
        "mitchell_absence_water_1", "mitchell_absence_water_2",
        "mitchell_absence_water_3",
    ],
    val_region_ids=[
        # Etna Creek
        "etna_presence_1", "etna_presence_3", "etna_presence_4",
        "etna_absence_1", "etna_absence_2", "etna_absence_3", "etna_absence_4",
        "etna_absence_5",
        # Landsend
        "landsend_presence_8",
        "landsend_absence_4", "landsend_absence_5",
        # Hughenden
        "hughenden_presence_1", "hughenden_presence_2", "hughenden_presence_3",
        "hughenden_absence_1", "hughenden_absence_2",
        "hughenden_absence_6", "hughenden_absence_7",
        # Burdekin
        "burdekin_val_presence_1",
        "burdekin_val_absence_1", "burdekin_val_absence_2",
        "burdekin_val_absence_3", "burdekin_val_absence_4",
        # Maria Downs
        "maria_downs_val_presence_1", "maria_downs_val_absence_1",
        # Rupert Creek
        "rupert_ck_val_presence_1", "rupert_ck_val_absence_1",
        # Quaids — held out for monsoonal generalisation check
        "quaids_val_absence_2", "quaids_val_absence_4", "quaids_val_absence_6",
        "quaids_val_absence_8a", "quaids_val_absence_8b", "quaids_val_absence_10",
        # Mitchell River — held out for monsoonal generalisation check
        "mitchell_val_presence_1", "mitchell_val_presence_2",
        "mitchell_val_presence_3", "mitchell_val_presence_4",
        "mitchell_val_absence_mangrove_1", "mitchell_val_absence_bare_1",
        "mitchell_val_absence_riparian_1", "mitchell_val_absence_water_1",
    ],
    feature_cols=V10_FEATURE_COLS,
    model_kwargs={
        "d_model":  256,
        "d_ff":     1024,
        "n_layers": 3,
        "dropout":  0.5,
        "n_bands":  len(V10_FEATURE_COLS) + len(V10_S1_FEATURE_COLS),  # 18
        "n_annual_features": -1,           # auto: set from band summaries width at training time
    },
    train_kwargs={
        "lr":                    5e-5,
        "weight_decay":          0.1,
        "batch_size":            2048,
        "n_epochs":              60,
        "patience":              15,
        "band_noise_std":        0.05,
        "obs_dropout_min":       4,
        "p_gate":                0.0,
        "T_gate":                8,
        "doy_density_norm":      True,
        "doy_phase_shift":       False,
        "pixel_zscore":          True,
        "use_s1":                True,
        "max_seq_len":           128,
        "feature_cols_override": tuple(V10_FEATURE_COLS),
        "s1_feature_cols":       tuple(V10_S1_FEATURE_COLS),  # s1_vh, s1_vv, s1_vh_vv, s1_rvi
    },
)
