"""tam/core/model.py — TAMClassifier: temporal attention model for Parkinsonia detection."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from tam.core.config import TAMConfig
from tam.core.dataset import N_BANDS, MAX_SEQ_LEN


def _doy_encoding(doy: torch.Tensor, d_model: int) -> torch.Tensor:
    """Circular sinusoidal DOY encoding.

    DOY is periodic (day 365 ≈ day 1), so we project onto sin/cos of the
    annual cycle and harmonics rather than using learned embeddings.  Padding
    positions (doy == 0) are left as the zero vector.

    Parameters
    ----------
    doy : (B, T) int64, values 1–365; 0 = padding
    d_model : must be even

    Returns
    -------
    (B, T, d_model) float32
    """
    B, T = doy.shape
    device = doy.device

    # Number of sin/cos pairs — fills d_model dimensions
    n_pairs = d_model // 2

    # Angular frequencies: ω_k = 2π·k / 365  for k = 1 … n_pairs
    k = torch.arange(1, n_pairs + 1, dtype=torch.float32, device=device)  # (n_pairs,)
    omega = 2.0 * math.pi * k / 365.0                                     # (n_pairs,)

    angle = doy.float().unsqueeze(-1) * omega.unsqueeze(0).unsqueeze(0)   # (B, T, n_pairs)

    enc = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)         # (B, T, d_model)

    # Zero out padding positions so they don't contribute to attention
    enc = enc * (doy != 0).float().unsqueeze(-1)
    return enc


class TAMClassifier(nn.Module):
    """Transformer encoder that maps an annual pixel time-series to P(Parkinsonia).

    Architecture
    ------------
    band_proj  : Linear(N_BANDS, d_model)
    doy_enc    : circular sinusoidal DOY encoding (no learnable params)
    encoder    : TransformerEncoder (num_layers × TransformerEncoderLayer)
    pool       : mean over non-padded positions
    head       : Linear(d_model, 1) → sigmoid

    Parameters
    ----------
    d_model        : model dimension (must be even for circular DOY encoding)
    n_heads        : number of attention heads (must divide d_model)
    n_layers       : number of encoder layers
    d_ff           : feedforward hidden dimension inside each encoder layer
    dropout        : dropout probability
    """

    def __init__(
        self,
        d_model:  int   = 64,
        n_heads:  int   = 4,
        n_layers: int   = 2,
        d_ff:     int   = 128,
        dropout:  float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.n_layers = n_layers
        self.d_ff     = d_ff
        self.dropout  = dropout

        self.band_proj = nn.Linear(N_BANDS, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    # ------------------------------------------------------------------
    def forward(
        self,
        bands: torch.Tensor,              # (B, T, N_BANDS)
        doy:   torch.Tensor,              # (B, T)  int, 0=padding
        key_padding_mask: torch.Tensor,   # (B, T)  bool, True=padding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (prob, logit), each shape (B,)."""
        x = self.band_proj(bands) + _doy_encoding(doy, self.d_model)  # (B, T, d_model)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # Mean pool over non-padded positions
        valid = (~key_padding_mask).float().unsqueeze(-1)  # (B, T, 1)
        x_pool = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)  # (B, d_model)

        logit = self.head(x_pool).squeeze(-1)   # (B,)
        prob  = torch.sigmoid(logit)
        return prob, logit

    # ------------------------------------------------------------------
    def get_attention_weights(
        self,
        bands: torch.Tensor,            # (1, T, N_BANDS)
        doy:   torch.Tensor,            # (1, T)
        key_padding_mask: torch.Tensor, # (1, T)
    ) -> list[torch.Tensor]:
        """Return per-layer attention weight matrices for a single sample.

        Returns
        -------
        List of length n_layers.  Each tensor has shape (n_heads, T, T).
        attn[layer][head, query, key] = attention weight.
        """
        self.eval()
        with torch.no_grad():
            x = self.band_proj(bands) + _doy_encoding(doy, self.d_model)  # (1, T, d_model)

            attn_weights: list[torch.Tensor] = []
            for layer in self.encoder.layers:
                # Run self-attention manually to capture weights
                attn_out, w = layer.self_attn(
                    x, x, x,
                    key_padding_mask=key_padding_mask,
                    need_weights=True,
                    average_attn_weights=False,  # keep per-head: (1, n_heads, T, T)
                )
                attn_weights.append(w.squeeze(0))  # (n_heads, T, T)

                # Continue through the rest of the encoder layer
                x2 = layer.norm1(x + layer.dropout1(attn_out))
                x2 = layer.norm2(x2 + layer.dropout2(layer.linear2(
                    layer.dropout(layer.activation(layer.linear1(x2)))
                )))
                x = x2

        return attn_weights

    # ------------------------------------------------------------------
    def config(self) -> dict:
        return {
            "d_model":       self.d_model,
            "n_heads":       self.n_heads,
            "n_layers":      self.n_layers,
            "d_ff":          self.d_ff,
            "dropout":       self.dropout,
            "n_bands":       N_BANDS,
            "max_seq_len":   MAX_SEQ_LEN,
        }

    @classmethod
    def from_config(cls, cfg: TAMConfig) -> "TAMClassifier":
        return cls(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
        )
