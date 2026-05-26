"""utils/signing.py — Per-item STAC asset href signing.

Provides item-signer callables suitable for passing to fetch_patches()
as the ``item_signer`` parameter.  Each signer takes a pystac.Item and
returns a copy with signed asset hrefs, leaving the original unchanged.

Currently supported:
  make_mpc_signer()  — Microsoft Planetary Computer (Azure Blob Storage)
                       Uses the public PC SAS signing API; no
                       planetary_computer package required.

Token caching
-------------
MPC SAS tokens are valid for ~1 hour.  _MpcSignCache reuses a cached
signed URL until CACHE_MARGIN_S seconds before its expiry, then fetches
a fresh one.  The cache is per-process (module-level singleton) and
thread-safe.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Callable
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

logger = logging.getLogger(__name__)

_MPC_SIGN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
_MPC_HOSTS = frozenset([
    "planetarycomputer.microsoft.com",
    "ai4edataeuwest.blob.core.windows.net",
    "sentinel1euwestrtc.blob.core.windows.net",
    "sentinel2l2a01.blob.core.windows.net",
    "modiseuwest.blob.core.windows.net",
    "nasaeuwest.blob.core.windows.net",
])
# MPC blob hostnames share a *.blob.core.windows.net pattern
_MPC_BLOB_SUFFIX = ".blob.core.windows.net"

# Refresh a cached SAS URL this many seconds before its stated expiry.
_CACHE_MARGIN_S = 300  # 5 minutes


def _is_mpc_href(href: str) -> bool:
    """Return True if href is an MPC-hosted asset that needs SAS signing."""
    try:
        host = urlparse(href).netloc
    except Exception:
        return False
    return host in _MPC_HOSTS or host.endswith(_MPC_BLOB_SUFFIX)


def _parse_expiry(expiry_str: str) -> float:
    """Return expiry as a POSIX timestamp, or 0.0 on parse failure."""
    try:
        # MPC returns ISO 8601 like "2024-05-26T14:30:00Z"
        dt = datetime.fromisoformat(expiry_str.rstrip("Z")).replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return 0.0


# Maximum concurrent signing API calls.  The MPC sign endpoint rate-limits
# at a low threshold (observed 429s at 32 concurrent callers); serialising
# through a small semaphore keeps us well under it.  Signing is fast (~100 ms)
# so the queue drains quickly and callers find a warm cache on the second request.
_SIGN_CONCURRENCY = 4
_SIGN_MAX_RETRIES = 4


class _MpcSignCache:
    """Thread-safe in-process cache for MPC-signed hrefs.

    Signed hrefs are reused until _CACHE_MARGIN_S before their stated
    expiry.  On cache miss (or near-expiry), the MPC sign API is called
    and the result stored.

    Concurrent signing API calls are limited to _SIGN_CONCURRENCY so the
    MPC rate limiter is not triggered when many threads start simultaneously.
    429 responses are retried with exponential backoff.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # unsigned_href → (signed_href, expiry_posix)
        self._cache: dict[str, tuple[str, float]] = {}
        self._api_sem = threading.Semaphore(_SIGN_CONCURRENCY)

    def _call_sign_api(self, unsigned: str, timeout: int) -> tuple[str, float]:
        """Call the MPC sign API with retry on 429. Returns (signed_href, expiry)."""
        import json
        import urllib.error
        import urllib.request
        req_url = f"{_MPC_SIGN_URL}?href={urllib.request.quote(unsigned, safe='')}"
        for attempt in range(_SIGN_MAX_RETRIES + 1):
            try:
                with self._api_sem:
                    with urllib.request.urlopen(req_url, timeout=timeout) as resp:
                        data = json.loads(resp.read())
                signed = data["href"]
                expiry = _parse_expiry(data.get("msft:expiry", ""))
                if expiry == 0.0:
                    expiry = time.time() + 55 * 60
                return signed, expiry
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < _SIGN_MAX_RETRIES:
                    wait = 2 ** attempt  # 1, 2, 4, 8 s
                    logger.debug("MPC sign API 429 (attempt %d/%d) — retrying in %ds",
                                 attempt + 1, _SIGN_MAX_RETRIES, wait)
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("unreachable")

    def sign(self, href: str, timeout: int = 15) -> str:
        """Return a fresh signed href, using the cache where possible."""
        if not _is_mpc_href(href):
            return href

        # Strip any existing SAS token from the key so stale-token hrefs
        # hit the same cache entry as the unsigned version.
        unsigned = _strip_sas(href)

        now = time.time()
        with self._lock:
            entry = self._cache.get(unsigned)
            if entry is not None:
                signed, expiry = entry
                if expiry - now > _CACHE_MARGIN_S:
                    return signed

        try:
            signed, expiry = self._call_sign_api(unsigned, timeout)
        except Exception as exc:
            logger.warning("MPC sign API failed for %s: %s — using unsigned href", unsigned, exc)
            return href

        with self._lock:
            self._cache[unsigned] = (signed, expiry)

        logger.debug("MPC signed href (expires in %.0fs): %s", expiry - now, unsigned)
        return signed

    def sign_item(self, item) -> object:
        """Return a shallow-copied item with all MPC asset hrefs signed."""
        item = copy.copy(item)
        new_assets = {}
        for band, asset in item.assets.items():
            if _is_mpc_href(asset.href):
                asset = copy.copy(asset)
                asset.href = self.sign(asset.href)
            new_assets[band] = asset
        item.assets = new_assets
        return item


def _strip_sas(href: str) -> str:
    """Remove SAS query parameters from an Azure Blob Storage URL.

    Strips ``se``, ``sig``, ``sp``, ``spr``, ``sr``, ``sv`` — the standard
    SAS parameters — so the unsigned URL can serve as a stable cache key.
    Returns href unchanged if it is not an Azure blob URL or has no SAS params.
    """
    _SAS_PARAMS = frozenset(["se", "sig", "sp", "spr", "sr", "sv", "skoid",
                              "sktid", "skt", "ske", "sks", "skv"])
    try:
        parsed = urlparse(href)
        qs = parse_qs(parsed.query, keep_blank_values=True)
        stripped = {k: v for k, v in qs.items() if k not in _SAS_PARAMS}
        new_query = urlencode(stripped, doseq=True)
        return urlunparse(parsed._replace(query=new_query))
    except Exception:
        return href


# Module-level singleton — shared across all callers in the same process.
_mpc_cache = _MpcSignCache()


def make_mpc_signer() -> Callable:
    """Return an item-signer callable for Microsoft Planetary Computer assets.

    The returned callable has signature ``(item) -> item``, suitable for
    passing to ``fetch_patches(item_signer=...)``.  It signs all MPC asset
    hrefs using the public PC SAS API (no ``planetary_computer`` package
    required) and caches tokens for their full validity window.

    The callable is safe to call from multiple threads concurrently.
    """
    return _mpc_cache.sign_item
