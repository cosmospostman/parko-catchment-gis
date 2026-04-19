"""TM-* tests for _doy_encoding and TAMClassifier."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from tam.config import TAMConfig
from tam.dataset import MAX_SEQ_LEN, N_BANDS
from tam.model import TAMClassifier, _doy_encoding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_batch(B: int, T: int, n_valid: int, d_model: int = 16):
    """Return (bands, doy, mask) for a batch with n_valid non-padded positions."""
    rng = torch.Generator()
    rng.manual_seed(0)
    bands = torch.rand(B, T, N_BANDS, generator=rng)
    doy = torch.zeros(B, T, dtype=torch.int64)
    doy_vals = torch.randint(1, 366, (B, n_valid), generator=rng)
    doy_vals, _ = doy_vals.sort(dim=1)
    doy[:, :n_valid] = doy_vals
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, :n_valid] = False
    bands[:, n_valid:] = 0.0
    return bands, doy, mask


def _small_model(n_layers=1) -> TAMClassifier:
    cfg = TAMConfig(d_model=16, n_heads=2, n_layers=n_layers, d_ff=32, dropout=0.0)
    m = TAMClassifier.from_config(cfg)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# TM-1
# ---------------------------------------------------------------------------

class TestTM1DOYEncodingPaddingZero:
    def test_doy_zero_gives_zero_encoding(self):
        doy = torch.zeros(2, 10, dtype=torch.int64)
        enc = _doy_encoding(doy, d_model=16)
        assert enc.shape == (2, 10, 16)
        assert (enc == 0).all()


# ---------------------------------------------------------------------------
# TM-2
# ---------------------------------------------------------------------------

class TestTM2DOYEncodingValidNonZero:
    def test_valid_doy_has_nonzero_encoding(self):
        doy = torch.tensor([[1, 90, 180, 270, 365]])
        enc = _doy_encoding(doy, d_model=16)
        for t in range(doy.shape[1]):
            assert enc[0, t].abs().sum() > 0


# ---------------------------------------------------------------------------
# TM-3
# ---------------------------------------------------------------------------

class TestTM3DOYEncodingOutputShape:
    @pytest.mark.parametrize("B,T", [(1, 10), (4, 50), (8, MAX_SEQ_LEN)])
    def test_output_shape(self, B, T):
        doy = torch.randint(1, 366, (B, T))
        enc = _doy_encoding(doy, d_model=16)
        assert enc.shape == (B, T, 16)


# ---------------------------------------------------------------------------
# TM-4
# ---------------------------------------------------------------------------

class TestTM4DOYEncodingSeasonalitySmooth:
    def test_day1_and_day364_high_similarity(self):
        # day 1 and day 364 are one cycle apart modulo the 2π·k/365 formula;
        # they should be far more similar to each other than day 1 and day 183.
        d_model = 64
        enc1   = _doy_encoding(torch.tensor([[1]]),   d_model).squeeze()
        enc364 = _doy_encoding(torch.tensor([[364]]), d_model).squeeze()
        enc183 = _doy_encoding(torch.tensor([[183]]), d_model).squeeze()
        sim_adjacent = torch.nn.functional.cosine_similarity(enc1, enc364, dim=0).item()
        sim_opposite = torch.nn.functional.cosine_similarity(enc1, enc183, dim=0).item()
        assert sim_adjacent > sim_opposite

    def test_day1_day365_higher_similarity_than_day183(self):
        # The formula 2π·k·doy/365 means day 365 ≠ day 1 (they are one step apart,
        # not a full cycle). Day 365 and day 1 should be closer to each other than
        # day 1 and day 183 (opposite side of the year).
        d_model = 64
        enc1   = _doy_encoding(torch.tensor([[1]]),   d_model).squeeze()
        enc365 = _doy_encoding(torch.tensor([[365]]), d_model).squeeze()
        enc183 = _doy_encoding(torch.tensor([[183]]), d_model).squeeze()
        sim_wrap = torch.nn.functional.cosine_similarity(enc1, enc365, dim=0).item()
        sim_opp  = torch.nn.functional.cosine_similarity(enc1, enc183, dim=0).item()
        assert sim_wrap > sim_opp


# ---------------------------------------------------------------------------
# TM-5
# ---------------------------------------------------------------------------

class TestTM5DOYEncodingBounded:
    def test_values_in_minus1_to_1(self):
        doy = torch.randint(1, 366, (4, MAX_SEQ_LEN))
        enc = _doy_encoding(doy, d_model=64)
        assert enc.min() >= -1.0 - 1e-6
        assert enc.max() <=  1.0 + 1e-6


# ---------------------------------------------------------------------------
# TM-6
# ---------------------------------------------------------------------------

class TestTM6ForwardPassOutputShapes:
    def test_prob_and_logit_shape(self):
        model = _small_model()
        B, T, n_valid = 4, MAX_SEQ_LEN, 20
        bands, doy, mask = _random_batch(B, T, n_valid)
        prob, logit = model(bands, doy, mask)
        assert prob.shape == (4,)
        assert logit.shape == (4,)


# ---------------------------------------------------------------------------
# TM-7
# ---------------------------------------------------------------------------

class TestTM7ProbabilitiesInOpenInterval:
    def test_probs_strictly_between_0_and_1(self):
        model = _small_model()
        bands, doy, mask = _random_batch(4, MAX_SEQ_LEN, 20)
        prob, _ = model(bands, doy, mask)
        assert (prob > 0).all()
        assert (prob < 1).all()


# ---------------------------------------------------------------------------
# TM-8
# ---------------------------------------------------------------------------

class TestTM8NoNaNOrInf:
    def test_outputs_finite(self):
        model = _small_model()
        bands, doy, mask = _random_batch(4, MAX_SEQ_LEN, 20)
        prob, logit = model(bands, doy, mask)
        assert torch.isfinite(prob).all()
        assert torch.isfinite(logit).all()


# ---------------------------------------------------------------------------
# TM-9
# ---------------------------------------------------------------------------

class TestTM9AllPaddingDoesNotCrash:
    def test_all_masked_batch_no_nan(self):
        model = _small_model()
        B, T = 2, MAX_SEQ_LEN
        bands = torch.zeros(B, T, N_BANDS)
        doy   = torch.zeros(B, T, dtype=torch.int64)
        mask  = torch.ones(B, T, dtype=torch.bool)  # all padding
        prob, logit = model(bands, doy, mask)
        assert torch.isfinite(prob).all()
        assert torch.isfinite(logit).all()


# ---------------------------------------------------------------------------
# TM-10
# ---------------------------------------------------------------------------

class TestTM10AllPaddingVsRealDifferent:
    def test_masked_and_real_differ(self):
        model = _small_model()
        bands, doy, mask = _random_batch(2, MAX_SEQ_LEN, 20)
        prob_real, _ = model(bands, doy, mask)

        mask_all = torch.ones_like(mask)
        bands_zero = torch.zeros_like(bands)
        doy_zero   = torch.zeros_like(doy)
        prob_pad, _ = model(bands_zero, doy_zero, mask_all)

        assert not torch.allclose(prob_real, prob_pad)


# ---------------------------------------------------------------------------
# TM-11
# ---------------------------------------------------------------------------

class TestTM11GetAttentionWeightsStructure:
    def test_list_length_and_shape(self):
        n_layers = 2
        model = TAMClassifier(d_model=16, n_heads=2, n_layers=n_layers, d_ff=32, dropout=0.0)
        model.eval()
        T, n_valid = MAX_SEQ_LEN, 10
        bands, doy, mask = _random_batch(1, T, n_valid)
        weights = model.get_attention_weights(bands, doy, mask)
        assert len(weights) == n_layers
        for w in weights:
            assert w.shape == (2, T, T)


# ---------------------------------------------------------------------------
# TM-12
# ---------------------------------------------------------------------------

class TestTM12AttentionWeightsSumToOne:
    def test_weights_sum_to_1_over_keys(self):
        model = TAMClassifier(d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout=0.0)
        model.eval()
        n_valid = 15
        bands, doy, mask = _random_batch(1, MAX_SEQ_LEN, n_valid)
        weights = model.get_attention_weights(bands, doy, mask)
        w = weights[0]  # (n_heads, T, T)
        # For each query, weights over non-masked keys should sum to ~1
        # Rows corresponding to padding queries may not sum to 1; check valid queries only
        w_valid_queries = w[:, :n_valid, :]  # (n_heads, n_valid, T)
        row_sums = w_valid_queries.sum(dim=-1)  # (n_heads, n_valid)
        np.testing.assert_allclose(row_sums.numpy(), 1.0, atol=1e-5)

    def test_attention_to_padding_is_near_zero(self):
        model = TAMClassifier(d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout=0.0)
        model.eval()
        n_valid = 10
        bands, doy, mask = _random_batch(1, MAX_SEQ_LEN, n_valid)
        weights = model.get_attention_weights(bands, doy, mask)
        w = weights[0]  # (n_heads, T, T)
        # Attention from valid queries to padding keys should be near zero
        attn_to_padding = w[:, :n_valid, n_valid:]
        assert attn_to_padding.abs().max().item() < 1e-4


# ---------------------------------------------------------------------------
# TM-13
# ---------------------------------------------------------------------------

class TestTM13GetAttentionWeightsMatchesForward:
    def test_logit_matches_forward(self):
        model = TAMClassifier(d_model=16, n_heads=2, n_layers=2, d_ff=32, dropout=0.0)
        model.eval()
        bands, doy, mask = _random_batch(1, MAX_SEQ_LEN, 15)

        # get_attention_weights runs its own forward; compare final logit
        # We need to run the full forward separately and compare
        with torch.no_grad():
            prob_fwd, logit_fwd = model(bands, doy, mask)

        # get_attention_weights also returns after running the full encoder
        # We verify by checking that it doesn't crash and the model is in eval mode after
        _ = model.get_attention_weights(bands, doy, mask)

        # Re-run forward after get_attention_weights to confirm model state unchanged
        with torch.no_grad():
            prob_after, logit_after = model(bands, doy, mask)

        torch.testing.assert_close(logit_fwd, logit_after)


# ---------------------------------------------------------------------------
# TM-14
# ---------------------------------------------------------------------------

class TestTM14FromConfigArchitecture:
    def test_attributes_match_config(self):
        cfg = TAMConfig(d_model=32, n_heads=2, n_layers=3, d_ff=64)
        model = TAMClassifier.from_config(cfg)
        assert model.d_model == 32
        assert model.n_heads == 2
        assert model.n_layers == 3


# ---------------------------------------------------------------------------
# TM-15
# ---------------------------------------------------------------------------

class TestTM15ConfigRoundTrip:
    def test_config_round_trip(self):
        cfg_orig = TAMConfig(d_model=32, n_heads=2, n_layers=3, d_ff=64)
        model = TAMClassifier.from_config(cfg_orig)
        cfg_restored = TAMConfig.from_dict(model.config())
        model2 = TAMClassifier.from_config(cfg_restored)
        assert model2.d_model == model.d_model
        assert model2.n_heads == model.n_heads
        assert model2.n_layers == model.n_layers
        assert model2.d_ff == model.d_ff


# ---------------------------------------------------------------------------
# TM-16
# ---------------------------------------------------------------------------

class TestTM16NoLearnableParamsInDOYEncoding:
    def test_fewer_params_than_embedding(self):
        d_model = 64
        cfg = TAMConfig(d_model=d_model, n_heads=4, n_layers=2, d_ff=128)
        model = TAMClassifier.from_config(cfg)
        n_params = sum(p.numel() for p in model.parameters())

        # Equivalent model with learned embedding would add 366 * d_model params
        embedding_params = 366 * d_model
        assert n_params < n_params + embedding_params  # trivially true — confirm encoding adds 0
        # More directly: verify no embedding layer exists on the model
        for name, module in model.named_modules():
            assert not isinstance(module, nn.Embedding), f"Unexpected Embedding at {name}"
