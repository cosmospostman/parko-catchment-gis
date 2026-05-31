"""proxy/server.py — Fetch-proxy VM server.

FastAPI app that runs the full S2+S1 extraction pipeline on the VM (co-located
with Element84 S3 COGs) and streams sorted strip-shard parquets back to the
workstation over a length-prefixed frame protocol.

Listens on localhost:8765 (SSH tunnel only — no auth token needed).
Single-process, one job at a time.  Second POST while one is running → 409.

Frame wire format (from proxy/_pipeline.py):
  [TYPE 1 byte][LENGTH 4 bytes big-endian][PAYLOAD LENGTH bytes]
  0x01  UTF-8 JSON progress line
  0x02  Raw parquet bytes for one completed strip shard
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from threading import Lock
from typing import AsyncGenerator

# Ensure project root is on sys.path so utils/ imports work on the VM.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from proxy._pipeline import write_frame, progress_frame

logger = logging.getLogger("proxy.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI()
_job_lock = Lock()
_job_running = False


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class TileRequest(BaseModel):
    tile_id: str
    year: int
    polygon_wkb_b64: str
    cloud_max: int = 80
    apply_nbar: bool = True
    chunk_height_px: int = 1024
    chunk_width_px: int = 1024
    max_concurrent: int = int(os.environ.get("PROXY_MAX_CONCURRENT", "32"))
    n_workers: int | None = None
    resume_from_chunk: list[int] = [0, 0]  # [row, col]; JSON has no tuples


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------

@app.post("/run/tile")
async def run_tile(req: TileRequest):
    global _job_running
    with _job_lock:
        if _job_running:
            raise HTTPException(status_code=409, detail="A job is already running")
        _job_running = True

    return StreamingResponse(
        _produce(req),
        media_type="application/octet-stream",
    )


async def _produce(req: TileRequest) -> AsyncGenerator[bytes, None]:
    global _job_running
    loop = asyncio.get_event_loop()
    try:
        async for chunk in _run_pipeline(req, loop):
            yield chunk
    finally:
        with _job_lock:
            _job_running = False


async def _run_pipeline(req: TileRequest, loop: asyncio.AbstractEventLoop) -> AsyncGenerator[bytes, None]:
    import base64
    import queue as _queue
    import threading as _threading
    from shapely import wkb as shapely_wkb
    from proxy._pipeline import run_tile_pipeline_v2 as run_tile_pipeline

    t_start = time.monotonic()
    polygon_geometry = shapely_wkb.loads(base64.b64decode(req.polygon_wkb_b64))
    n_workers = req.n_workers or int(os.environ.get("PROXY_N_WORKERS", "0")) or None

    with tempfile.TemporaryDirectory(prefix=f"proxy_{req.tile_id}_{req.year}_") as _tmpdir:
        tmp = Path(_tmpdir)

        # Run the synchronous generator in a thread so yielded strips can be
        # streamed back to the client as they complete, not all at once.
        q: _queue.Queue = _queue.Queue()

        def _run_gen():
            try:
                for item in run_tile_pipeline(
                    tile_id=req.tile_id,
                    year=req.year,
                    polygon_geometry=polygon_geometry,
                    tmp=tmp,
                    cloud_max=req.cloud_max,
                    apply_nbar=req.apply_nbar,
                    chunk_height_px=req.chunk_height_px,
                    chunk_width_px=req.chunk_width_px,
                    max_concurrent=req.max_concurrent,
                    n_workers=n_workers,
                    resume_from_chunk=tuple(req.resume_from_chunk),
                ):
                    q.put(item)
            except Exception as exc:
                q.put(exc)
            finally:
                q.put(None)  # sentinel

        t = _threading.Thread(target=_run_gen, daemon=True)
        t.start()

        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            chunk_row, chunk_col, chunk_path = item
            yield progress_frame(chunk_row, chunk_col, "stream", time.monotonic() - t_start)
            logger.info("[chunk %02d_%02d] streaming %s ...", chunk_row, chunk_col, chunk_path.name)
            yield write_frame(0x02, chunk_path.read_bytes())
            chunk_path.unlink(missing_ok=True)

        t.join()

    logger.info("[tile %s %d] all chunks complete (%.0f s total)",
                req.tile_id, req.year, time.monotonic() - t_start)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PROXY_PORT", "8765"))
    uvicorn.run("proxy.server:app", host="127.0.0.1", port=port, workers=1)
