"""v9_spectral — S2-only TAM, v8_roper site set + Frenchs.

S2-only leg of the three-way comparison (S2-only / S1-only / joint S1+S2).
Feature set: B02 B03 B04 B05 B07 B08 B8A B11 B12 + NDVI + NDWI + MAVI + NDRE + CI_RE (14 bands).
B06 and EVI excluded. Pixel z-score normalisation identical to V8.
Frenchs added to address bare-soil confusion and restore monsoonal site diversity.
"""

from tam.core.dataset import V9_FEATURE_COLS
from tam.core.experiment import Experiment

EXPERIMENT = Experiment(
    name="v9_spectral",
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
        # Frenchs — Cape York Peninsula (monsoonal savanna-riparian)
        "frenchs_presence_1", "frenchs_presence_2", "frenchs_presence_3",
        "frenchs_presence_4",
        "frenchs_absence_bare_soil_2", "frenchs_absence_bare_soil_3",
        "frenchs_absence_mangrove", "frenchs_absence_ocean",
        "frenchs_absence_riparian_woodland", "frenchs_absence_riparian",
        "frenchs_absence_5", "frenchs_absence_6", "frenchs_absence_7",
        "frenchs_absence_water_1", "frenchs_absence_water_2", "frenchs_absence_water_3",
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
        # Frenchs — held out for monsoonal generalisation check
        "frenchs_presence_5", "frenchs_presence_6",
        "frenchs_absence_savanna", "frenchs_absence_4", "frenchs_absence_8",
    ],
    feature_cols=V9_FEATURE_COLS,
    model_kwargs={
        "d_model":  256,
        "n_layers": 3,
        "dropout":  0.5,
        "n_bands":  len(V9_FEATURE_COLS),  # 14
        "n_global_features": 0,            # overridden at runtime by use_band_summaries
    },
    train_kwargs={
        "lr":                    5e-5,
        "weight_decay":          0.1,
        "n_epochs":              60,
        "patience":              15,
        "band_noise_std":        0.05,
        "obs_dropout_min":       4,
        "doy_density_norm":      True,
        "doy_phase_shift":       False,
        "pixel_zscore":          True,
        "use_s1":                False,
        "use_band_summaries":    True,
        "max_seq_len":           64,
        "feature_cols_override": tuple(V9_FEATURE_COLS),
    },
)
