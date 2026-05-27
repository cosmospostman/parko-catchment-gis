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

from proxy._pipeline import write_frame, progress_frame, merge_scenes, compute_strips

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
    cloud_max: int = 20
    apply_nbar: bool = True
    strip_height_px: int = 1024
    max_concurrent: int = int(os.environ.get("PROXY_MAX_CONCURRENT", "32"))
    n_workers: int | None = None
    resume_from_strip: int = 0


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
    import shutil
    from shapely import wkb as shapely_wkb
    from utils.stac import search_sentinel2
    from utils.pixel_collector import collect
    from utils.s1_collector import collect_s1_for_tile

    t_start = time.monotonic()

    polygon_geometry = shapely_wkb.loads(base64.b64decode(req.polygon_wkb_b64))
    bbox_wgs84 = list(polygon_geometry.bounds)

    start_date = f"{req.year}-01-01"
    end_date   = f"{req.year}-12-31"

    logger.info("[tile %s %d] STAC search ...", req.tile_id, req.year)
    items = await loop.run_in_executor(None, lambda: search_sentinel2(
        bbox=bbox_wgs84,
        start=start_date,
        end=end_date,
        cloud_cover_max=req.cloud_max,
    ))
    logger.info("[tile %s %d] %d STAC items", req.tile_id, req.year, len(items))

    if not items:
        return

    strips = await loop.run_in_executor(
        None, lambda: compute_strips(bbox_wgs84, req.strip_height_px, polygon_geometry)
    )
    logger.info("[tile %s %d] %d strips of %d px", req.tile_id, req.year, len(strips), req.strip_height_px)

    n_workers = req.n_workers or int(os.environ.get("PROXY_N_WORKERS", "0")) or None

    with tempfile.TemporaryDirectory(prefix=f"proxy_{req.tile_id}_{req.year}_") as _tmpdir:
        tmp = Path(_tmpdir)

        for strip in strips:
            strip_idx  = strip["strip_idx"]
            strip_bbox = strip["bbox"]
            strip_pts  = strip["points"]

            if strip_idx < req.resume_from_strip:
                logger.info("[strip %04d] skipping (resume_from_strip=%d)", strip_idx, req.resume_from_strip)
                continue

            t_strip = time.monotonic()
            scene_dir = tmp / f"strip_{strip_idx:04d}_scenes"
            scene_dir.mkdir(parents=True, exist_ok=True)

            yield progress_frame(strip_idx, "fetch", time.monotonic() - t_start)
            logger.info("[strip %04d] fetch+extract %d points, %d items ...", strip_idx, len(strip_pts), len(items))

            scene_paths: list[Path] = []
            for scene_id, scene_path in await loop.run_in_executor(
                None,
                lambda sd=scene_dir, sb=strip_bbox, sp=strip_pts: list(collect(
                    bbox_wgs84=sb,
                    start=start_date,
                    end=end_date,
                    out_dir=sd,
                    cloud_max=req.cloud_max,
                    apply_nbar=req.apply_nbar,
                    max_concurrent=req.max_concurrent,
                    items=items,
                    geometry=polygon_geometry,
                    n_workers=n_workers,
                    per_scene=True,
                    cache_dir=sd / "cache",
                )),
            ):
                scene_paths.append(scene_path)

            yield progress_frame(strip_idx, "extract", time.monotonic() - t_start)

            s1_path: Path | None = await loop.run_in_executor(
                None,
                lambda sb=strip_bbox, sd=scene_dir, sp=strip_pts: collect_s1_for_tile(
                    s2_path=None,
                    bbox_wgs84=sb,
                    start=start_date,
                    end=end_date,
                    out_path=sd / "s1_strip.parquet",
                    cache_dir=sd / "cache",
                    max_concurrent=req.max_concurrent,
                    points=sp,
                ),
            )

            yield progress_frame(strip_idx, "merge", time.monotonic() - t_start)
            strip_out = tmp / f"strip_{strip_idx:04d}_sorted.parquet"

            if not scene_paths:
                logger.info("[strip %04d] no scene data — skipping", strip_idx)
                shutil.rmtree(scene_dir, ignore_errors=True)
                continue

            await loop.run_in_executor(
                None,
                lambda sp=scene_paths, s1=s1_path, so=strip_out: merge_scenes(sp, s1, so),
            )

            shutil.rmtree(scene_dir, ignore_errors=True)

            yield progress_frame(strip_idx, "stream", time.monotonic() - t_start)
            logger.info("[strip %04d] streaming %s ...", strip_idx, strip_out.name)

            strip_bytes = strip_out.read_bytes()
            yield write_frame(0x02, strip_bytes)
            strip_out.unlink(missing_ok=True)

            logger.info("[strip %04d] done in %.1f s", strip_idx, time.monotonic() - t_strip)

    logger.info("[tile %s %d] all strips complete (%.0f s total)", req.tile_id, req.year, time.monotonic() - t_start)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PROXY_PORT", "8765"))
    uvicorn.run("proxy.server:app", host="127.0.0.1", port=port, workers=1)
