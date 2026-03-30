"""utils/progress.py — Dask progress callback for pipeline logging."""

import logging
import threading

from dask.callbacks import Callback

logger = logging.getLogger(__name__)


class LogProgressCallback(Callback):
    """Log dask task completion progress at a fixed interval.

    Emits a log line every ``log_every`` completed tasks, plus a final
    summary line when the computation finishes.

    Usage::

        with LogProgressCallback(label="median composite", log_every=100):
            result = da.compute()
    """

    def __init__(self, label: str = "compute", log_every: int = 100) -> None:
        self._label = label
        self._log_every = log_every
        self._total = 0
        self._completed = 0
        self._lock = threading.Lock()

    def _start(self, dsk) -> None:
        self._total = len(dsk)
        self._completed = 0
        logger.info("%s — starting (%d tasks)", self._label, self._total)

    def _posttask(self, key, result, dsk, state, worker_id) -> None:
        with self._lock:
            self._completed += 1
            completed = self._completed
            total = self._total
        if total > 0 and completed % self._log_every == 0:
            pct = 100 * completed // total
            logger.info("%s — %d/%d tasks (%d%%)", self._label, completed, total, pct)

    def _finish(self, dsk, state, errored) -> None:
        if errored:
            logger.error("%s — failed after %d/%d tasks", self._label, self._completed, self._total)
        else:
            logger.info("%s — complete (%d tasks)", self._label, self._total)
