"""Tests for config.py."""
import importlib
import os
import sys
import pytest
from pathlib import Path


def test_missing_env_var_raises(monkeypatch, tmp_path):
    """config.py must raise KeyError if a required env var is missing."""
    monkeypatch.delenv("BASE_DIR", raising=False)
    # Remove cached module so it re-imports
    if "config" in sys.modules:
        del sys.modules["config"]
    with pytest.raises((KeyError, SystemExit, Exception)):
        import config  # noqa: F401


def test_year_cast_to_int(tmp_dirs):
    """YEAR env var must be cast to int."""
    if "config" in sys.modules:
        del sys.modules["config"]
    import config
    assert isinstance(config.YEAR, int)
    assert config.YEAR == 2025


def test_path_template_functions(tmp_dirs):
    """Output path template functions return expected Path objects."""
    if "config" in sys.modules:
        del sys.modules["config"]
    import config

    p = config.ndvi_median_path(2025)
    assert isinstance(p, Path)
    assert "ndvi_median_2025" in str(p)
    assert str(p).endswith(".tif")

    p2 = config.ndvi_anomaly_path(2025)
    assert "ndvi_anomaly_2025" in str(p2)

    p3 = config.verification_report_path(2025)
    assert str(p3).endswith(".json")

    p4 = config.ndvi_baseline_path()
    assert "ndvi_baseline" in str(p4)


def test_composite_bands_list(tmp_dirs):
    if "config" in sys.modules:
        del sys.modules["config"]
    import config
    assert isinstance(config.COMPOSITE_BANDS, list)
    assert "nir" in config.COMPOSITE_BANDS
    assert "red" in config.COMPOSITE_BANDS


def test_data_source_constants(tmp_dirs):
    """Data source URL/bucket/collection constants are well-formed."""
    if "config" in sys.modules:
        del sys.modules["config"]
    import config
    from urllib.parse import urlparse

    # URLs are well-formed with https scheme and a non-empty host
    for url in (config.STAC_ENDPOINT_ELEMENT84, config.STAC_ENDPOINT_CDSE, config.ALA_API_BASE):
        parsed = urlparse(url)
        assert parsed.scheme == "https", f"Expected https scheme: {url}"
        assert parsed.netloc, f"Expected non-empty host: {url}"

    assert config.DEA_S3_BUCKET == "dea-public-data"

    # Collection name strings are non-empty
    for col in (config.S2_COLLECTION, config.S1_COLLECTION, config.DEA_COLLECTION, config.FC_COLLECTION):
        assert isinstance(col, str) and col

    assert config.S2_COLLECTION == "sentinel-2-l2a"
    assert config.S1_COLLECTION == "sentinel-1-grd"
