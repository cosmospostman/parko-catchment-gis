"""TC-* tests for TAMConfig."""

from __future__ import annotations

import pytest

from tam.config import TAMConfig


class TestTC1DefaultValuesConsistent:
    def test_d_model_divisible_by_n_heads(self):
        cfg = TAMConfig()
        assert cfg.d_model % cfg.n_heads == 0

    def test_d_model_even(self):
        cfg = TAMConfig()
        assert cfg.d_model % 2 == 0


class TestTC2RoundTripSerialisation:
    def test_round_trip_equality(self):
        cfg = TAMConfig()
        assert TAMConfig.from_dict(cfg.to_dict()) == cfg

    def test_falsy_numeric_fields_preserved(self):
        # from_dict must not drop fields whose value is 0 or 0.0
        cfg = TAMConfig(d_model=64, n_heads=4, n_layers=2, d_ff=128,
                        dropout=0.0, weight_decay=0.0)
        restored = TAMConfig.from_dict(cfg.to_dict())
        assert restored.dropout == 0.0
        assert restored.weight_decay == 0.0


class TestTC3UnknownKeysIgnored:
    def test_unknown_key_does_not_raise(self):
        TAMConfig.from_dict({"d_model": 32, "unknown_future_key": 999})


class TestTC4PartialDictUsesDefaults:
    def test_partial_dict_produces_valid_config(self):
        cfg = TAMConfig.from_dict({"d_model": 32})
        assert cfg.d_model == 32
        # All other fields should equal the defaults
        defaults = TAMConfig()
        assert cfg.n_heads == defaults.n_heads
        assert cfg.n_layers == defaults.n_layers
        assert cfg.d_ff == defaults.d_ff
        assert cfg.dropout == defaults.dropout
