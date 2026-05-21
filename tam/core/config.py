"""tam/core/config.py — TAMConfig: single source of truth for all TAM hyperparameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TAMConfig:
    # Model architecture
    d_model:    int   = 32
    n_heads:    int   = 4
    n_layers:   int   = 1
    d_ff:       int   = 64
    dropout:    float = 0.3
    use_n_obs:         bool  = True   # append normalised obs-count to pooled repr before head
    n_global_features: int   = 9     # 5 S2 globals + 4 S1 globals (0 to disable)
    use_s1:  bool | str  = True   # True=mixed S1+S2 interleaved, "s1_only"=S1 sequence only, False=S2 only
    s1_feature_cols: tuple | None = None  # subset of S1_FEATURE_COLS to use; None=all four
    use_band_summaries: bool = False  # if True, append per-band [p5, p95, std] as global features
    feature_cols_override: tuple | None = None  # if set, replaces default S2 feature cols (e.g. V9_FEATURE_COLS)

    # Data (mirrors dataset.py constants — change both together)
    # 13 S2 features + 4 S1 features (s1_vh, s1_vv, s1_vh_vv, s1_rvi) when use_s1=True or "mixed"
    n_bands:          int   = 17
    max_seq_len:      int   = 128
    min_obs_per_year: int   = 8
    scl_purity_min:   float = 0.5
    doy_jitter:       int   = 7   # ±days of DOY shift applied per window during training
    doy_phase_shift:  bool  = False  # if True, random full-year phase shift with wraparound (overrides doy_jitter)
    pixel_zscore:     bool  = False  # if True, z-score each pixel's S1 bands by its own multi-year mean/std
    # Temporal despeckle window for S1 backscatter (rolling median over N acquisitions per pixel).
    # Applied to linear vh/vv before dB conversion. 0 = disabled.
    # Conservative starting value is 3; other reasonable values are 5, 7.
    s1_despeckle_window: int = 0
    band_noise_std:   float = 0.0 # std of per-window band offset in normalised space (training only)

    # Training
    n_epochs:        int   = 30
    batch_size:      int   = 1024
    lr:              float = 1e-4
    weight_decay:    float = 1e-3
    val_frac:        float = 0.2
    val_sites:       tuple = ()   # if non-empty, hold out these sites entirely instead of using val_frac
    val_region_ids:  tuple = ()   # if non-empty, hold out exactly these regions (takes precedence over val_sites)
    patience:        int   = 5
    min_delta:       float = 1e-4
    cvar_alpha:      float = 0.25  # tail fraction for CVaR val metric (bottom alpha of site AUCs)
    obs_dropout_min: int   = 0   # if >0, subsample each window to Uniform(obs_dropout_min, n) during training
    warmup_freeze_epochs: int = 0  # if >0, freeze temporal stream for first N epochs (head-only warmup)
    doy_density_norm: bool = False  # if True, weight mean pool by inverse DOY observation frequency
    spatial_stride:       int   = 1   # if >1, thin training pixels spatially (every Nth pixel per region)
    stride_exclude_sites: tuple = ()  # site prefixes exempt from spatial stride (e.g. small/sparse sites)

    # Presence pixel filter: drop presence pixel-years with low dry-season VH unless rescued by NDVI.
    # Logic: drop if mean_vh_dry < presence_min_vh_dry_db
    #        AND NOT (mean_vh_dry >= presence_ndvi_rescue_vh_db AND mean_ndvi_dry >= presence_ndvi_rescue_min)
    presence_min_vh_dry_db:      float = -21.0  # strict VH floor — drop unconditionally below this
    presence_ndvi_rescue_vh_db:  float = -23.0  # looser VH floor used only when NDVI passes
    presence_ndvi_rescue_min:    float = 0.50   # min dry-season NDVI to trigger rescue

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TAMConfig":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})
