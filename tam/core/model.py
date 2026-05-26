"""tam/core/model.py — TAMClassifier: temporal attention model for Parkinsonia detection."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from tam.core.config import TAMConfig
from tam.core.dataset import N_BANDS, MAX_SEQ_LEN  # N_BANDS used as default only


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
        d_model:    int   = 64,
        n_heads:    int   = 4,
        n_layers:   int   = 2,
        d_ff:       int   = 128,
        dropout:    float = 0.1,
        n_bands:           int   = N_BANDS,
        use_n_obs:         bool  = True,
        n_global_features: int   = 0,
        doy_density_norm:  bool  = False,
    ) -> None:
        super().__init__()
        self.d_model            = d_model
        self.n_heads            = n_heads
        self.n_layers           = n_layers
        self.d_ff               = d_ff
        self.dropout            = dropout
        self.n_bands            = n_bands
        self.use_n_obs          = use_n_obs
        self.n_global_features  = n_global_features
        self.doy_density_norm   = doy_density_norm

        # DOY frequency tables: shape (366,) — index by DOY (1-365), 0 unused.
        # Registered as buffers so they move with the model to the correct device.
        # Populated by set_doy_frequencies() before training; default to uniform.
        # Separate tables for S2 and S1 so cloud-driven S2 sampling bias doesn't
        # corrupt the S1 weights (and vice versa for S1 coverage gaps).
        self.register_buffer(
            "doy_inv_freq",
            torch.ones(366, dtype=torch.float32),
        )
        self.register_buffer(
            "doy_inv_freq_s1",
            torch.ones(366, dtype=torch.float32),
        )

        self.band_proj = nn.Linear(n_bands, d_model)
        self.pre_head_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        head_in = d_model + (1 if use_n_obs else 0) + n_global_features
        self.head = nn.Linear(head_in, 1)
        nn.init.constant_(self.head.bias, 0.1)  # positive bias → always starts predicting presence > 0.5

    # ------------------------------------------------------------------
    def set_doy_frequencies(
        self,
        s2_counts: "np.ndarray",
        s1_counts: "np.ndarray | None" = None,
    ) -> None:
        """Populate inverse-frequency weights from per-source (366,) observation count arrays.

        Counts are indexed by DOY (1-indexed). Positions with zero count get weight 1.0.
        Weights are normalised so the mean weight over observed DOYs ≈ 1, keeping the
        pooled representation on the same scale as uniform pooling.

        s1_counts defaults to s2_counts if not provided (backwards-compatible).
        """
        import numpy as np

        def _make_inv(counts: "np.ndarray") -> "np.ndarray":
            counts = np.array(counts, dtype=np.float32)
            inv = np.where(counts > 0, 1.0 / counts, 1.0)
            observed = counts > 0
            if observed.sum() > 0:
                inv[observed] /= inv[observed].mean()
            return inv

        self.doy_inv_freq = torch.from_numpy(_make_inv(s2_counts)).to(self.doy_inv_freq.device)
        s1 = s1_counts if s1_counts is not None else s2_counts
        self.doy_inv_freq_s1 = torch.from_numpy(_make_inv(s1)).to(self.doy_inv_freq_s1.device)

    # ------------------------------------------------------------------
    def forward(
        self,
        bands:            torch.Tensor,          # (B, T, N_BANDS)
        doy:              torch.Tensor,          # (B, T)  int, 0=padding
        key_padding_mask: torch.Tensor,          # (B, T)  bool, True=padding
        n_obs:            torch.Tensor,          # (B,)    float32, n / MAX_SEQ_LEN
        global_feats:     torch.Tensor | None = None,  # (B, n_global_features)
        is_s1:            torch.Tensor | None = None,  # (B, T)  bool, True=S1 obs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (prob, logit), each shape (B,)."""
        x = self.band_proj(bands) + _doy_encoding(doy, self.d_model)  # (B, T, d_model)

        # Rows where every position is masked cause softmax over all-inf attention
        # logits → NaN. Unmask such rows before the encoder; the mean pool below
        # still produces zero for them because valid is all-False.
        all_masked = key_padding_mask.all(dim=1, keepdim=True)  # (B, 1)
        safe_mask = key_padding_mask & ~all_masked               # (B, T)

        x = self.encoder(x, src_key_padding_mask=safe_mask)

        # Pool over non-padded positions, optionally weighted by inverse DOY frequency.
        # When is_s1 is provided, use separate S2/S1 weight tables so that cloud-driven
        # S2 sampling bias and S1 coverage gaps are corrected independently.
        valid = (~key_padding_mask).float()  # (B, T)
        if self.doy_density_norm:
            doy_idx = doy.clamp(0, 365)
            s2_w = self.doy_inv_freq[doy_idx]      # (B, T)
            if is_s1 is not None:
                s1_w = self.doy_inv_freq_s1[doy_idx]  # (B, T)
                doy_w = torch.where(is_s1, s1_w, s2_w)
            else:
                doy_w = s2_w
            pool_w = (valid * doy_w).unsqueeze(-1)     # (B, T, 1)
        else:
            pool_w = valid.unsqueeze(-1)               # (B, T, 1)
        x_pool = (x * pool_w).sum(dim=1) / pool_w.sum(dim=1).clamp(min=1)  # (B, d_model)

        if self.use_n_obs:
            x_pool = torch.cat([x_pool, n_obs.unsqueeze(-1)], dim=-1)

        if self.n_global_features > 0:
            if global_feats is None:
                global_feats = torch.zeros(x_pool.shape[0], self.n_global_features,
                                           dtype=x_pool.dtype, device=x_pool.device)
            x_pool = torch.cat([x_pool, global_feats], dim=-1)

        logit = self.head(self.pre_head_dropout(x_pool)).squeeze(-1)   # (B,)
        prob  = torch.sigmoid(logit)
        return prob, logit

    # ------------------------------------------------------------------
    def forward_varlen(
        self,
        bands_flat:   torch.Tensor,   # (total_tokens, N_BANDS)  float32
        doy_flat:     torch.Tensor,   # (total_tokens,)           int64
        cu_seqlens:   torch.Tensor,   # (B+1,)                    int32, cumulative token counts
        max_seqlen:   int,            # longest sequence in batch
        n_obs:        torch.Tensor,   # (B,)                      float32
        global_feats: torch.Tensor | None = None,
        is_s1_flat:   torch.Tensor | None = None,   # (total_tokens,) bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Varlen forward using flash-attn — no padding, O(sum T_i²) attention.

        Requires flash_attn to be installed.  Falls back to the padded forward()
        if flash_attn is not available (raises ImportError upstream).

        Input is a concatenated flat token tensor (no padding); cu_seqlens marks
        where each sequence starts/ends (same convention as flash_attn_varlen_func).
        """
        from flash_attn.cute.interface import flash_attn_varlen_func
        import torch.nn.functional as F

        d_model  = self.d_model
        n_heads  = self.n_heads
        head_dim = d_model // n_heads
        B = cu_seqlens.shape[0] - 1

        # Project bands + add DOY encoding (flat — no batch dim)
        # _doy_encoding expects (B, T) but we have (1, total_tokens)
        doy_2d = doy_flat.unsqueeze(0)                                # (1, total_tokens)
        doy_enc = _doy_encoding(doy_2d, d_model).squeeze(0)          # (total_tokens, d_model)
        # band_proj and layer norms run in float32 (model weights are float32).
        # flash_attn requires float16 or bfloat16 for Q/K/V; we cast only for that kernel.
        x = self.band_proj(bands_flat.float()) + doy_enc              # (total_tokens, d_model) fp32

        # Run each encoder layer with flash varlen attention
        for layer in self.encoder.layers:
            # Extract Q/K/V weight/bias from the layer's MultiheadAttention
            mha = layer.self_attn
            # in_proj_weight: (3*d_model, d_model); in_proj_bias: (3*d_model,)
            qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias)  # (total, 3*d_model) fp32
            q, k, v = qkv.chunk(3, dim=-1)

            # Reshape to (total_tokens, n_heads, head_dim) and cast to bf16 for flash_attn
            q = q.view(-1, n_heads, head_dim).to(torch.bfloat16)
            k = k.view(-1, n_heads, head_dim).to(torch.bfloat16)
            v = v.view(-1, n_heads, head_dim).to(torch.bfloat16)

            softmax_scale = head_dim ** -0.5
            attn_out = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=softmax_scale,
                causal=False,
            )  # (total_tokens, n_heads, head_dim) bf16
            attn_out = attn_out.view(-1, d_model).float()             # back to fp32

            # Output projection + residual + norm1 + FFN + residual + norm2
            attn_out = F.linear(attn_out, mha.out_proj.weight, mha.out_proj.bias)
            x2 = layer.norm1(x + layer.dropout1(attn_out))
            ffn_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(x2))))
            x = layer.norm2(x2 + layer.dropout2(ffn_out))

        # Pool each sequence: mean over its tokens (with optional DOY density weighting)
        x_pool = torch.zeros(B, d_model, dtype=x.dtype, device=x.device)
        for i in range(B):
            s, e = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
            if e <= s:
                continue
            tok = x[s:e]                                              # (T_i, d_model)
            if self.doy_density_norm:
                doy_i = doy_flat[s:e].clamp(0, 365)
                s2_w = self.doy_inv_freq[doy_i]                      # (T_i,)
                if is_s1_flat is not None:
                    is1 = is_s1_flat[s:e]
                    s1_w = self.doy_inv_freq_s1[doy_i]
                    w = torch.where(is1, s1_w, s2_w)
                else:
                    w = s2_w
                w = w.unsqueeze(-1)                                   # (T_i, 1)
                x_pool[i] = (tok * w).sum(0) / w.sum().clamp(min=1)
            else:
                x_pool[i] = tok.mean(0)

        if self.use_n_obs:
            x_pool = torch.cat([x_pool, n_obs.unsqueeze(-1)], dim=-1)

        if self.n_global_features > 0:
            if global_feats is None:
                global_feats = torch.zeros(B, self.n_global_features,
                                           dtype=x_pool.dtype, device=x_pool.device)
            x_pool = torch.cat([x_pool, global_feats], dim=-1)

        logit = self.head(self.pre_head_dropout(x_pool)).squeeze(-1)  # (B,)
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
            "d_model":            self.d_model,
            "n_heads":            self.n_heads,
            "n_layers":           self.n_layers,
            "d_ff":               self.d_ff,
            "dropout":            self.dropout,
            "n_bands":            self.n_bands,
            "use_n_obs":          self.use_n_obs,
            "n_global_features":  self.n_global_features,
            "doy_density_norm":   self.doy_density_norm,
            "max_seq_len":        getattr(self, "_max_seq_len", MAX_SEQ_LEN),
            # Data/inference config — needed by score pipeline
            "use_s1":             getattr(self, "_use_s1", None),
            "pixel_zscore":       getattr(self, "_pixel_zscore", None),
            "feature_cols":       getattr(self, "_feature_cols", None),
            "s1_feature_cols":    getattr(self, "_s1_feature_cols", None),
        }

    @classmethod
    def from_config(cls, cfg: TAMConfig) -> "TAMClassifier":
        return cls(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            n_bands=cfg.n_bands,
            use_n_obs=cfg.use_n_obs,
            n_global_features=cfg.n_global_features,
            doy_density_norm=cfg.doy_density_norm,
        )
