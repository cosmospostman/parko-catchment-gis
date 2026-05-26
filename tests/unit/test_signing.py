"""Unit tests for utils/signing.py — MPC asset href signing.

Tests
-----
SG-1. _is_mpc_href: recognises Azure blob hosts and rejects non-MPC URLs.
SG-2. _strip_sas: removes SAS query parameters from a signed URL.
SG-3. _MpcSignCache.sign: calls the sign API and returns the signed href.
SG-4. _MpcSignCache.sign: caches the result; second call does not hit the API.
SG-5. _MpcSignCache.sign: re-fetches when cached token is near expiry.
SG-6. _MpcSignCache.sign: passes non-MPC hrefs through unchanged (no API call).
SG-7. _MpcSignCache.sign: falls back to the unsigned href on API failure.
SG-8. _MpcSignCache.sign_item: signs all MPC assets on the item copy.
SG-9. _MpcSignCache.sign_item: leaves non-MPC assets unchanged.
SG-10. make_mpc_signer: returns a callable that wraps the module cache.
SG-11. _strip_sas: leaves non-SAS query params intact.
SG-12. _MpcSignCache.sign: treats an already-signed href as the same key
       as its unsigned counterpart (stable cache key).
SG-13. _MpcSignCache.sign: retries on 429 with backoff, succeeds on retry.
SG-14. _MpcSignCache.sign: falls back to unsigned href after exhausting 429 retries.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_item(assets: dict[str, str]) -> object:
    """Minimal pystac.Item-like object with .assets dict of Asset(href=...)."""
    item = SimpleNamespace()
    item.assets = {
        band: SimpleNamespace(href=href)
        for band, href in assets.items()
    }
    return item


_MPC_HREF = "https://sentinel1euwestrtc.blob.core.windows.net/sentinel1-grd-rtc/GRD/2025/1/1/IW/DV/S1A_IW_GRDH_1SDV/measurement/iw-vh.rtc.tiff"
_MPC_SIGNED_HREF = _MPC_HREF + "?sig=abc123&se=2025-01-01T00%3A00%3A00Z&sv=2020-08-04&sp=r&sr=b"
_PUBLIC_HREF = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/55/H/BU/2023/6/S2B_55HBU_20230601_0_L2A/B04.tif"


def _fake_sign_response(signed_href: str, expiry_offset_s: int = 3600) -> bytes:
    expiry = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(time.time() + expiry_offset_s),
    )
    return json.dumps({"href": signed_href, "msft:expiry": expiry}).encode()


# ---------------------------------------------------------------------------
# SG-1. _is_mpc_href
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("href,expected", [
    (_MPC_HREF, True),
    ("https://ai4edataeuwest.blob.core.windows.net/foo/bar.tif", True),
    ("https://anythingelse.blob.core.windows.net/x/y.tif", True),
    (_PUBLIC_HREF, False),
    ("s3://sentinel-cogs/foo/bar.tif", False),
    ("/local/path/file.tif", False),
    ("", False),
])
def test_is_mpc_href(href, expected):
    from utils.signing import _is_mpc_href
    assert _is_mpc_href(href) is expected


# ---------------------------------------------------------------------------
# SG-2. _strip_sas: removes SAS params
# ---------------------------------------------------------------------------

def test_strip_sas_removes_sas_params():
    from utils.signing import _strip_sas
    signed = _MPC_HREF + "?sig=abc&se=2025-01-01T00:00:00Z&sv=2020-08-04&sp=r&sr=b"
    result = _strip_sas(signed)
    assert "sig=" not in result
    assert "se=" not in result
    assert "sv=" not in result
    assert _MPC_HREF in result  # base URL preserved


# ---------------------------------------------------------------------------
# SG-11. _strip_sas: preserves non-SAS params
# ---------------------------------------------------------------------------

def test_strip_sas_preserves_non_sas_params():
    from utils.signing import _strip_sas
    href = _MPC_HREF + "?sig=abc&custom_param=keep"
    result = _strip_sas(href)
    assert "custom_param=keep" in result
    assert "sig=" not in result


# ---------------------------------------------------------------------------
# SG-3. sign: calls API and returns signed href
# ---------------------------------------------------------------------------

def test_sign_calls_api_and_returns_signed_href():
    from utils.signing import _MpcSignCache

    cache = _MpcSignCache()
    signed = _MPC_HREF + "?sig=freshtoken"
    response_body = _fake_sign_response(signed)

    mock_resp = MagicMock()
    mock_resp.read.return_value = response_body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        result = cache.sign(_MPC_HREF)

    assert result == signed
    assert mock_open.call_count == 1
    call_url = mock_open.call_args[0][0]
    assert "sign?" in call_url
    assert "sentinel1euwestrtc" in call_url


# ---------------------------------------------------------------------------
# SG-4. sign: second call hits cache (no second API call)
# ---------------------------------------------------------------------------

def test_sign_caches_result():
    from utils.signing import _MpcSignCache

    cache = _MpcSignCache()
    signed = _MPC_HREF + "?sig=cached"
    response_body = _fake_sign_response(signed, expiry_offset_s=3600)

    mock_resp = MagicMock()
    mock_resp.read.return_value = response_body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        r1 = cache.sign(_MPC_HREF)
        r2 = cache.sign(_MPC_HREF)

    assert r1 == signed
    assert r2 == signed
    assert mock_open.call_count == 1  # second call served from cache


# ---------------------------------------------------------------------------
# SG-5. sign: re-fetches when token is near expiry
# ---------------------------------------------------------------------------

def test_sign_refetches_near_expiry():
    from utils.signing import _MpcSignCache, _CACHE_MARGIN_S

    cache = _MpcSignCache()
    signed_v1 = _MPC_HREF + "?sig=old"
    signed_v2 = _MPC_HREF + "?sig=fresh"

    responses = [
        _fake_sign_response(signed_v1, expiry_offset_s=_CACHE_MARGIN_S - 10),  # near expiry
        _fake_sign_response(signed_v2, expiry_offset_s=3600),
    ]
    call_idx = [0]

    def urlopen_side_effect(url, timeout=15):
        mock_resp = MagicMock()
        mock_resp.read.return_value = responses[call_idx[0]]
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        call_idx[0] += 1
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=urlopen_side_effect) as mock_open:
        r1 = cache.sign(_MPC_HREF)  # fills cache with near-expiry token
        r2 = cache.sign(_MPC_HREF)  # cache miss (near expiry) → re-fetch

    assert r1 == signed_v1
    assert r2 == signed_v2
    assert mock_open.call_count == 2


# ---------------------------------------------------------------------------
# SG-6. sign: non-MPC hrefs pass through unchanged
# ---------------------------------------------------------------------------

def test_sign_passthrough_non_mpc():
    from utils.signing import _MpcSignCache

    cache = _MpcSignCache()
    with patch("urllib.request.urlopen") as mock_open:
        result = cache.sign(_PUBLIC_HREF)

    assert result == _PUBLIC_HREF
    mock_open.assert_not_called()


# ---------------------------------------------------------------------------
# SG-7. sign: falls back to unsigned href on API error
# ---------------------------------------------------------------------------

def test_sign_fallback_on_api_error():
    from utils.signing import _MpcSignCache

    cache = _MpcSignCache()

    with patch("urllib.request.urlopen", side_effect=OSError("network error")):
        result = cache.sign(_MPC_HREF)

    assert result == _MPC_HREF  # falls back to unsigned


# ---------------------------------------------------------------------------
# SG-12. sign: signed href uses same cache key as unsigned
# ---------------------------------------------------------------------------

def test_sign_stable_cache_key_for_signed_href():
    from utils.signing import _MpcSignCache

    cache = _MpcSignCache()
    signed_input = _MPC_HREF + "?sig=old&se=2020-01-01T00:00:00Z&sv=x&sp=r&sr=b"
    signed_output = _MPC_HREF + "?sig=new"
    response_body = _fake_sign_response(signed_output, expiry_offset_s=3600)

    mock_resp = MagicMock()
    mock_resp.read.return_value = response_body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        # First call with unsigned href — fills cache
        cache.sign(_MPC_HREF)
        # Second call with an already-signed (stale) href — must hit cache
        result = cache.sign(signed_input)

    assert result == signed_output
    assert mock_open.call_count == 1  # second call served from cache


# ---------------------------------------------------------------------------
# SG-8. sign_item: signs MPC assets
# ---------------------------------------------------------------------------

def test_sign_item_signs_mpc_assets():
    from utils.signing import _MpcSignCache

    cache = _MpcSignCache()
    item = _make_item({"vh": _MPC_HREF, "vv": _MPC_HREF + "_vv"})

    signed_vh = _MPC_HREF + "?sig=x"
    signed_vv = _MPC_HREF + "_vv?sig=y"

    call_map = {
        _MPC_HREF: signed_vh,
        _MPC_HREF + "_vv": signed_vv,
    }

    def fake_sign(href, **kw):
        return call_map.get(href, href)

    cache.sign = fake_sign
    result = cache.sign_item(item)

    assert result is not item  # returns a copy
    assert result.assets["vh"].href == signed_vh
    assert result.assets["vv"].href == signed_vv
    # Original item unchanged
    assert item.assets["vh"].href == _MPC_HREF


# ---------------------------------------------------------------------------
# SG-9. sign_item: leaves non-MPC assets unchanged
# ---------------------------------------------------------------------------

def test_sign_item_leaves_non_mpc_unchanged():
    from utils.signing import _MpcSignCache

    cache = _MpcSignCache()
    item = _make_item({"B04": _PUBLIC_HREF})

    called_with = []
    real_sign = cache.sign

    def tracking_sign(href, **kw):
        called_with.append(href)
        return real_sign(href, **kw)

    cache.sign = tracking_sign
    result = cache.sign_item(item)

    assert result.assets["B04"].href == _PUBLIC_HREF
    # sign() is called but returns unchanged (non-MPC passthrough)
    assert all(h == _PUBLIC_HREF for h in called_with)


# ---------------------------------------------------------------------------
# SG-10. make_mpc_signer: returns a callable
# ---------------------------------------------------------------------------

def test_make_mpc_signer_returns_callable():
    from utils.signing import make_mpc_signer

    signer = make_mpc_signer()
    assert callable(signer)


# ---------------------------------------------------------------------------
# SG-13. sign: retries on 429, succeeds on the retry
# ---------------------------------------------------------------------------

def test_sign_retries_on_429_and_succeeds():
    import urllib.error
    from utils.signing import _MpcSignCache

    cache = _MpcSignCache()
    signed = _MPC_HREF + "?sig=retried"
    response_body = _fake_sign_response(signed, expiry_offset_s=3600)

    call_count = [0]

    def urlopen_side_effect(url, timeout=15):
        call_count[0] += 1
        if call_count[0] == 1:
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=urlopen_side_effect), \
         patch("time.sleep"):  # don't actually sleep in tests
        result = cache.sign(_MPC_HREF)

    assert result == signed
    assert call_count[0] == 2  # one 429 then one success


# ---------------------------------------------------------------------------
# SG-14. sign: falls back to unsigned after exhausting all 429 retries
# ---------------------------------------------------------------------------

def test_sign_fallback_after_exhausting_429_retries():
    import urllib.error
    from utils.signing import _MpcSignCache, _SIGN_MAX_RETRIES

    cache = _MpcSignCache()

    def urlopen_side_effect(url, timeout=15):
        raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)

    with patch("urllib.request.urlopen", side_effect=urlopen_side_effect), \
         patch("time.sleep"):
        result = cache.sign(_MPC_HREF)

    assert result == _MPC_HREF  # falls back to unsigned


def test_make_mpc_signer_signs_item():
    from utils.signing import make_mpc_signer, _mpc_cache

    item = _make_item({"vh": _MPC_HREF})
    signed_href = _MPC_HREF + "?sig=moduletoken"
    response_body = _fake_sign_response(signed_href, expiry_offset_s=3600)

    mock_resp = MagicMock()
    mock_resp.read.return_value = response_body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    # Clear the module-level cache entry for this href so we get a fresh API call
    unsigned = _MPC_HREF
    _mpc_cache._cache.pop(unsigned, None)

    signer = make_mpc_signer()
    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = signer(item)

    assert result.assets["vh"].href == signed_href
