"""proxy/client.py — Workstation-side client for the fetch proxy.

fetch_tiles() posts one /run/tile request per (tile_id, year), reads the
length-prefixed frame stream, writes chunk shards atomically to a tmp
directory, then calls merge_chunks() to produce the final tile parquet.

Frame wire format (mirrors proxy/server.py):
  [TYPE 1 byte][LENGTH 4 bytes big-endian][PAYLOAD LENGTH bytes]
  0x01  UTF-8 JSON progress line
  0x02  Raw parquet bytes for one completed chunk shard
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import httpx

from proxy._pipeline import read_frame, StreamBuffer as _StreamBuffer  # noqa: F401

logger = logging.getLogger("proxy.client")

_CHUNK_RE = re.compile(r"_r(\d{3})_c(\d{3})$")


def _chunk_key(stem: str) -> tuple[int, int]:
    m = _CHUNK_RE.search(stem)
    return (int(m.group(1)), int(m.group(2))) if m else (9999, 9999)


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
    cloud_max: int = 80,
    apply_nbar: bool = True,
    chunk_height_px: int = 1024,
    chunk_width_px: int = 1024,
    max_concurrent: int = 32,
    n_workers: int | None = None,
    timeout_s: float = 7200.0,
) -> list[Path]:
    """Fetch one tile×year at a time from the proxy and write merged parquets.

    Returns list of written output paths.

    Chunk-level resume: for each (tile, year), scans tmp_dir for already-
    complete <tile_id>_r??_c??.parquet files and resumes from the chunk
    after the last complete one (row-major order).

    Tile-level resume: skips if <out_dir>/<year>/<tile_id>.parquet exists and
    has a sibling .done sentinel.
    """
    from utils.parquet_utils import merge_chunks

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

            # Determine resume chunk (row-major: resume after the last complete chunk).
            complete_chunks = sorted(tile_tmp.glob(f"{tile_id}_r??_c??.parquet"),
                                     key=lambda p: _chunk_key(p.stem))
            resume_from_chunk = [0, 0]
            if complete_chunks:
                last_row, last_col = _chunk_key(complete_chunks[-1].stem)
                resume_from_chunk = [last_row, last_col + 1]
                logger.info(
                    "[%s %d] resuming from chunk (%d,%d) (%d already complete)",
                    tile_id, year, resume_from_chunk[0], resume_from_chunk[1], len(complete_chunks),
                )

            # Delete any leftover .tmp files (incomplete from prior run)
            for p in tile_tmp.glob(f"{tile_id}_r??_c??.tmp"):
                p.unlink(missing_ok=True)

            request_body = {
                "tile_id": tile_id,
                "year": year,
                "polygon_wkb_b64": polygon_wkb_b64,
                "cloud_max": cloud_max,
                "apply_nbar": apply_nbar,
                "chunk_height_px": chunk_height_px,
                "chunk_width_px": chunk_width_px,
                "max_concurrent": max_concurrent,
                "resume_from_chunk": resume_from_chunk,
            }
            if n_workers is not None:
                request_body["n_workers"] = n_workers

            url = proxy_url.rstrip("/") + "/run/tile"
            logger.info("[%s %d] POST %s (resume_from_chunk=%s)", tile_id, year, url, resume_from_chunk)
            t0 = time.monotonic()

            received_chunks: list[Path] = list(complete_chunks)
            pending_chunk_row: int | None = None
            pending_chunk_col: int | None = None

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
                            pending_chunk_row = int(msg["chunk_row"])
                            pending_chunk_col = int(msg["chunk_col"])
                            logger.info(
                                "[%s %d] chunk (%d,%d)  stage=%s  t=%.1fs",
                                tile_id, year,
                                msg.get("chunk_row"), msg.get("chunk_col"),
                                msg.get("stage"), msg.get("t", 0),
                            )
                        except Exception:
                            pass

                    elif frame_type == 0x02:
                        if pending_chunk_row is None or pending_chunk_col is None:
                            raise RuntimeError("received 0x02 data frame without preceding 0x01 index frame")
                        crow = pending_chunk_row
                        ccol = pending_chunk_col
                        pending_chunk_row = None
                        pending_chunk_col = None

                        chunk_path = tile_tmp / f"{tile_id}_r{crow:02d}_c{ccol:02d}.parquet"
                        tmp_path   = tile_tmp / f"{tile_id}_r{crow:02d}_c{ccol:02d}.tmp"

                        tmp_path.write_bytes(payload)
                        written_bytes = tmp_path.stat().st_size
                        if written_bytes != len(payload):
                            tmp_path.unlink(missing_ok=True)
                            raise RuntimeError(
                                f"chunk ({crow},{ccol}): byte count mismatch "
                                f"(frame={len(payload)}, disk={written_bytes})"
                            )
                        tmp_path.replace(chunk_path)
                        received_chunks.append(chunk_path)
                        mb = len(payload) / 1e6
                        elapsed = time.monotonic() - t0
                        logger.info(
                            "[%s %d] chunk (%d,%d) received — %.0f MB  (%.0f s elapsed)",
                            tile_id, year, crow, ccol, mb, elapsed,
                        )

                    else:
                        logger.warning("unknown frame type 0x%02x — ignoring", frame_type)

            if not received_chunks:
                logger.warning("[%s %d] no chunks received — skipping merge", tile_id, year)
                continue

            # Sort chunks in row-major order before merge
            received_chunks.sort(key=lambda p: _chunk_key(p.stem))

            out_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("[%s %d] merging %d chunks → %s", tile_id, year, len(received_chunks), out_path)
            merge_chunks(received_chunks, out_path)

            done_sentinel.touch()
            logger.info("[%s %d] done — %s", tile_id, year, out_path)
            written.append(out_path)

            # Clean up chunk shards
            for p in received_chunks:
                p.unlink(missing_ok=True)
            try:
                tile_tmp.rmdir()
            except OSError:
                pass

    return written
