"""utils/chunk_log.py — Per-chunk file logging for the fetch pipeline.

Each (tile, year, chunk_row, chunk_col) combination gets its own log file at:
    out_dir/<year>/<tile_id>/fetchlogs/r{row:02d}_c{col:02d}.log

The pipeline runs each stage in its own thread pool (via Pipeline from
pipeline_types.py).  contextvars.ContextVar tokens are NOT inherited across
stage-thread boundaries — each stage thread starts with a fresh copy of the
context.  To work around this, ChunkLogger stores the underlying
logging.Logger on the work item itself and each stage calls
`work.clog.activate()` to install it in the current thread's ContextVar.

Typical usage inside a stage function:

    def _stage_fetch_tiffs(work):
        clog = work.clog
        if clog:
            clog.activate()          # install in this thread
            clog.info("fetching %d items", n)
        ...

get_chunk_logger() returns the ContextVar value for the current thread, used
by helpers (e.g. _extract_one) that share the same thread as the stage caller.
"""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from pathlib import Path

_CHUNK_LOGGER_VAR: ContextVar[logging.Logger | None] = ContextVar(
    "_CHUNK_LOGGER_VAR", default=None
)

_FMT = logging.Formatter(
    fmt="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)


def get_chunk_logger() -> logging.Logger | None:
    """Return the chunk logger installed in the current thread, or None."""
    return _CHUNK_LOGGER_VAR.get()


class ChunkLogger:
    """File-backed logger for one (tile, year, chunk_row, chunk_col) unit.

    Lifecycle
    ---------
    - Created in fetch_tile_local before the pipeline starts.
    - open() / close() open and close the file handler.
    - activate() installs the logger into the current thread's ContextVar so
      that get_chunk_logger() calls in the same thread resolve to this logger.
      Must be called at the top of each stage function because stage workers
      run in different thread-pool threads.
    - Used as a context manager: __enter__ calls open() + activate(),
      __exit__ calls close().  Suitable for single-threaded use; in the
      pipeline, call open() once before the pipeline and close() after.
    """

    def __init__(
        self,
        tile_id: str,
        year: int,
        chunk_row: int,
        chunk_col: int,
        log_dir: Path,
    ) -> None:
        self._tile_id = tile_id
        self._year = year
        self._row = chunk_row
        self._col = chunk_col
        self._log_dir = log_dir
        self._handler: logging.FileHandler | None = None
        self._logger: logging.Logger | None = None

    # ------------------------------------------------------------------ public

    @property
    def logger(self) -> logging.Logger | None:
        return self._logger

    def activate(self) -> None:
        """Install this logger into the current thread's ContextVar."""
        _CHUNK_LOGGER_VAR.set(self._logger)

    def open(self) -> None:
        """Open the log file and prepare the handler.  Idempotent."""
        if self._logger is not None:
            return
        self._log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._log_dir / f"r{self._row:02d}_c{self._col:02d}.log"

        name = f"chunk.{self._tile_id}.{self._year}.{self._row:02d}.{self._col:02d}"
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        lg.propagate = False  # don't also write to root logger

        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(_FMT)
        # Remove stale handlers left over from a previous run in the same process.
        for old in list(lg.handlers):
            lg.removeHandler(old)
            old.close()
        lg.addHandler(fh)
        self._handler = fh
        self._logger = lg

        lg.info(
            "========== chunk r%02d_c%02d  tile=%s  year=%d ==========",
            self._row, self._col, self._tile_id, self._year,
        )
        self.activate()

    def close(self, success: bool = True) -> None:
        """Flush and close the file handler."""
        if self._logger is None:
            return
        if success:
            self._logger.info(
                "========== chunk r%02d_c%02d DONE ==========",
                self._row, self._col,
            )
        else:
            self._logger.error(
                "========== chunk r%02d_c%02d FAILED ==========",
                self._row, self._col,
            )
        if self._handler:
            self._handler.flush()
            self._handler.close()
            self._logger.removeHandler(self._handler)
        self._handler = None
        self._logger = None

    def info(self, msg: str, *args, **kwargs) -> None:
        if self._logger:
            self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        if self._logger:
            self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        if self._logger:
            self._logger.error(msg, *args, **kwargs)

    # ----------------------------------------------------------------- context manager (single-threaded convenience)

    def __enter__(self) -> "ChunkLogger":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self._logger and self._logger.error(
                "chunk r%02d_c%02d FAILED: %s: %s",
                self._row, self._col, exc_type.__name__, exc_val,
            )
        self.close(success=exc_type is None)


def make_chunk_logger(
    tile_id: str,
    year: int,
    chunk_row: int,
    chunk_col: int,
    log_dir: Path,
) -> ChunkLogger:
    """Create and open a ChunkLogger.  Caller must call .close() when done."""
    cl = ChunkLogger(tile_id, year, chunk_row, chunk_col, log_dir)
    cl.open()
    return cl
