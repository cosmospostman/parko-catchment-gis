"""tam/model.py — TAMClassifier: temporal attention model for Parkinsonia detection."""

from __future__ import annotations

import torch
import torch.nn as nn

from tam.dataset import N_BANDS, MAX_SEQ_LEN


class TAMClassifier(nn.Module):
    """Transformer encoder that maps an annual pixel time-series to P(Parkinsonia).

    Architecture
    ------------
    band_proj  : Linear(N_BANDS, d_model)
    doy_embed  : Embedding(366, d_model)  — learned DOY positional encoding, 0=padding
    encoder    : TransformerEncoder (num_layers × TransformerEncoderLayer)
    pool       : mean over non-padded positions
    head       : Linear(d_model, 1) → sigmoid

    Parameters
    ----------
    d_model        : model dimension (projection + DOY embedding size)
    n_heads        : number of attention heads (must divide d_model)
    n_layers       : number of encoder layers
    d_ff           : feedforward hidden dimension inside each encoder layer
    dropout        : dropout probability
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.n_layers = n_layers

        self.band_proj = nn.Linear(N_BANDS, d_model)
        self.doy_embed = nn.Embedding(366, d_model, padding_idx=0)

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
        x = self.band_proj(bands) + self.doy_embed(doy)   # (B, T, d_model)
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
            x = self.band_proj(bands) + self.doy_embed(doy)   # (1, T, d_model)

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
            "d_model":  self.d_model,
            "n_heads":  self.n_heads,
            "n_layers": self.n_layers,
            "d_ff":     self.encoder.layers[0].linear1.out_features,
            "dropout":  self.encoder.layers[0].dropout.p,
            "n_bands":  N_BANDS,
            "max_seq_len": MAX_SEQ_LEN,
        }
