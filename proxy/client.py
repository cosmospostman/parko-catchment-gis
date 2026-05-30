"""proxy/client.py — Workstation-side client for the fetch proxy.

fetch_tiles() posts one /run/tile request per (tile_id, year), reads the
length-prefixed frame stream, writes strip shards atomically to a tmp
directory, then calls merge_strips() to produce the final tile parquet.

Frame wire format (mirrors proxy/server.py):
  [TYPE 1 byte][LENGTH 4 bytes big-endian][PAYLOAD LENGTH bytes]
  0x01  UTF-8 JSON progress line
  0x02  Raw parquet bytes for one completed strip shard
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import httpx

from proxy._pipeline import read_frame, StreamBuffer as _StreamBuffer  # noqa: F401

logger = logging.getLogger("proxy.client")


# ---------------------------------------------------------------------------
# Core client
# ---------------------------------------------------------------------------

def fetch_tiles(
    proxy_url: str,
    tile_ids: list[str],
    years: list[int],
    polygon_wkb_b64: str,
    out_dir: Path,
    tmp_dir: Path,
    cloud_max: int = 20,
    apply_nbar: bool = True,
    strip_height_px: int = 1024,
    max_concurrent: int = 32,
    n_workers: int | None = None,
    timeout_s: float = 7200.0,
) -> list[Path]:
    """Fetch one tile×year at a time from the proxy and write merged parquets.

    Returns list of written output paths.

    Strip-level resume: for each (tile, year), scans tmp_dir for already-
    complete <tile_id>_strip_NN.parquet files and resumes from the first gap.

    Tile-level resume: skips if <out_dir>/<year>/<tile_id>.parquet exists and
    has a sibling .done sentinel.
    """
    from utils.parquet_utils import merge_strips

    written: list[Path] = []

    for year in years:
        for tile_id in tile_ids:
            out_path = out_dir / str(year) / f"{tile_id}.parquet"
            done_sentinel = out_path.with_suffix(".done")
            if done_sentinel.exists() and out_path.exists():
                logger.info("[%s %d] already done — skipping", tile_id, year)
                written.append(out_path)
                continue

            tile_tmp = tmp_dir / tile_id / str(year)
            tile_tmp.mkdir(parents=True, exist_ok=True)

            # Determine resume strip.
            # Strip indices are absolute COG block rows so the first strip on
            # disk may not be strip_00 — find the lowest present and walk forward.
            complete_strips = sorted(tile_tmp.glob(f"{tile_id}_strip_??.parquet"),
                                     key=lambda p: int(p.stem.split("_")[-1]))
            resume_from = 0
            if complete_strips:
                indices = [int(p.stem.split("_")[-1]) for p in complete_strips]
                expected = indices[0]
                for idx in indices:
                    if idx == expected:
                        expected += 1
                    else:
                        break
                resume_from = expected
                logger.info(
                    "[%s %d] resuming from strip %d (%d already complete)",
                    tile_id, year, resume_from, len(complete_strips),
                )

            # Delete any leftover .tmp files (incomplete from prior run)
            for p in tile_tmp.glob(f"{tile_id}_strip_??.tmp"):
                p.unlink(missing_ok=True)

            request_body = {
                "tile_id": tile_id,
                "year": year,
                "polygon_wkb_b64": polygon_wkb_b64,
                "cloud_max": cloud_max,
                "apply_nbar": apply_nbar,
                "strip_height_px": strip_height_px,
                "max_concurrent": max_concurrent,
                "resume_from_strip": resume_from,
            }
            if n_workers is not None:
                request_body["n_workers"] = n_workers

            url = proxy_url.rstrip("/") + "/run/tile"
            logger.info("[%s %d] POST %s (resume_from_strip=%d)", tile_id, year, url, resume_from)
            t0 = time.monotonic()

            received_strips: list[Path] = list(complete_strips)
            pending_strip_idx: int | None = None  # set by 0x01 frame, consumed by 0x02

            with httpx.stream(
                "POST", url,
                json=request_body,
                timeout=httpx.Timeout(timeout_s, connect=30.0),
            ) as resp:
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"proxy returned HTTP {resp.status_code}: {resp.text[:200]}"
                    )

                buf = _StreamBuffer(resp.iter_raw(chunk_size=65536))
                while True:
                    frame = read_frame(buf)
                    if frame is None:
                        break
                    frame_type, payload = frame

                    if frame_type == 0x01:
                        try:
                            msg = json.loads(payload.decode())
                            pending_strip_idx = int(msg["strip"])
                            logger.info(
                                "[%s %d] strip %s  stage=%s  t=%.1fs",
                                tile_id, year, msg.get("strip"), msg.get("stage"), msg.get("t", 0),
                            )
                        except Exception:
                            pass

                    elif frame_type == 0x02:
                        if pending_strip_idx is None:
                            raise RuntimeError("received 0x02 data frame without preceding 0x01 index frame")
                        strip_idx  = pending_strip_idx
                        pending_strip_idx = None
                        strip_path = tile_tmp / f"{tile_id}_strip_{strip_idx:02d}.parquet"
                        tmp_path   = tile_tmp / f"{tile_id}_strip_{strip_idx:02d}.tmp"

                        tmp_path.write_bytes(payload)
                        # Verify byte count matches frame LENGTH field
                        written_bytes = tmp_path.stat().st_size
                        if written_bytes != len(payload):
                            tmp_path.unlink(missing_ok=True)
                            raise RuntimeError(
                                f"strip {strip_idx}: byte count mismatch "
                                f"(frame={len(payload)}, disk={written_bytes})"
                            )
                        tmp_path.replace(strip_path)
                        received_strips.append(strip_path)
                        mb = len(payload) / 1e6
                        elapsed = time.monotonic() - t0
                        logger.info(
                            "[%s %d] strip %02d received — %.0f MB  (%.0f s elapsed)",
                            tile_id, year, strip_idx, mb, elapsed,
                        )

                    else:
                        logger.warning("unknown frame type 0x%02x — ignoring", frame_type)

            if not received_strips:
                logger.warning("[%s %d] no strips received — skipping merge", tile_id, year)
                continue

            # Sort strips by index for merge
            received_strips.sort(key=lambda p: int(p.stem.split("_")[-1]))

            out_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("[%s %d] merging %d strips → %s", tile_id, year, len(received_strips), out_path)
            merge_strips(received_strips, out_path)

            done_sentinel.touch()
            logger.info("[%s %d] done — %s", tile_id, year, out_path)
            written.append(out_path)

            # Clean up strip shards
            for p in received_strips:
                p.unlink(missing_ok=True)
            try:
                tile_tmp.rmdir()
            except OSError:
                pass

    return written
