"""tam/core/config.py — TAMConfig: single source of truth for all TAM hyperparameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TAMConfig:
    # Model architecture
    d_model:  int   = 64
    n_heads:  int   = 4
    n_layers: int   = 2
    d_ff:     int   = 128
    dropout:  float = 0.3

    # Data (mirrors dataset.py constants — change both together)
    n_bands:          int   = 13
    max_seq_len:      int   = 128
    min_obs_per_year: int   = 8
    scl_purity_min:   float = 0.5
    doy_jitter:       int   = 7   # ±days of DOY shift applied per window during training

    # Training
    n_epochs:     int   = 100
    batch_size:   int   = 1024
    lr:           float = 1e-4
    weight_decay: float = 1e-3
    val_frac:     float = 0.2
    patience:     int   = 15
    min_delta:    float = 1e-4

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TAMConfig":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})
