"""utils/pipeline_types.py — Beam-style pipeline vocabulary.

Provides StageSpec and Pipeline: a minimal language for expressing linear
stage pipelines with bounded concurrency and a startup memory-budget check.

Usage::

    pipeline = Pipeline([
        StageSpec("fetch",   fn=_fetch,   concurrency=2, ram_gb=0.4),
        StageSpec("extract", fn=_extract, concurrency=1, ram_gb=4.0),
        StageSpec("merge",   fn=_merge,   concurrency=1, ram_gb=0.1),
    ], ram_budget_gb=12.0)

    for result in pipeline.run(inputs):
        consume(result)

Stage functions have the signature ``fn(item) -> item | list[item] | None``.
Returning a list is a flat_map (one input yields multiple outputs).
Returning None drops the item.

The runner uses one daemon thread per concurrency slot, connected by bounded
Queues.  Backpressure is natural: Queue.put() blocks when the downstream
queue is full (maxsize = concurrency * 2).

Exceptions in any stage thread are re-raised in the main thread on the next
``next()`` call, with the original traceback attached.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from queue import Queue
from typing import Any

logger = logging.getLogger(__name__)

_SENTINEL = object()  # end-of-stream marker


@dataclass(frozen=True)
class StageSpec:
    """Declare one stage of a Pipeline.

    Parameters
    ----------
    name:
        Human-readable label used in logs and budget reports.
    fn:
        Callable ``(item) -> item | list[item] | None``.
        Returning a list produces multiple outputs (flat_map).
        Returning None drops the item silently.
    concurrency:
        Maximum number of parallel calls to *fn*.  Controls both the thread
        count and the memory budget contribution.
    ram_gb:
        Estimated peak RAM per concurrent instance.  Multiplied by
        *concurrency* for the budget check.
    """

    name: str
    fn: Callable[..., Any]
    concurrency: int
    ram_gb: float


class Pipeline:
    """Linear stage pipeline with per-stage concurrency bounds.

    Parameters
    ----------
    stages:
        Ordered list of StageSpec.  Data flows left to right.
    ram_budget_gb:
        Optional hard cap.  Raises ValueError at construction time if
        ``sum(s.concurrency * s.ram_gb for s in stages) > ram_budget_gb``.
    name:
        Optional label for log messages.
    """

    def __init__(
        self,
        stages: list[StageSpec],
        ram_budget_gb: float | None = None,
        name: str = "pipeline",
    ) -> None:
        if not stages:
            raise ValueError("Pipeline requires at least one stage")
        self.stages = stages
        self.name = name
        self._peak_gb = sum(s.concurrency * s.ram_gb for s in stages)
        if ram_budget_gb is not None and self._peak_gb > ram_budget_gb:
            lines = "\n".join(
                f"  {s.name}: {s.concurrency} × {s.ram_gb:.2f} GB = {s.concurrency * s.ram_gb:.2f} GB"
                for s in stages
            )
            raise ValueError(
                f"[{name}] peak RAM {self._peak_gb:.1f} GB exceeds budget "
                f"{ram_budget_gb:.1f} GB\n{lines}"
            )
        logger.info(
            "[%s] stages: %s | peak RAM: %.1f GB%s",
            name,
            " → ".join(f"{s.name}(×{s.concurrency})" for s in stages),
            self._peak_gb,
            f" / {ram_budget_gb:.1f} GB budget" if ram_budget_gb is not None else "",
        )

    # ------------------------------------------------------------------

    def run(self, inputs: Iterable[Any]) -> Iterator[Any]:
        """Feed *inputs* through all stages; yield outputs of the last stage.

        Exceptions raised in any stage worker are propagated to the caller on
        the next iteration.  All worker threads are daemon threads so they do
        not prevent interpreter exit.
        """
        n = len(self.stages)

        # One queue per stage boundary: queues[0] feeds stage 0, queues[n]
        # is the output queue read by the caller.
        # maxsize = concurrency * 2 provides a one-strip look-ahead buffer
        # without allowing unbounded accumulation.
        queues: list[Queue] = [
            Queue(maxsize=s.concurrency * 2) for s in self.stages
        ] + [Queue()]

        # Errors from worker threads land here; re-raised in the main thread.
        error_q: Queue = Queue()

        def _worker(
            stage: StageSpec,
            in_q: Queue,
            out_q: Queue,
            # Shared counter so the last worker to finish sends exactly one
            # sentinel downstream (avoids N sentinels accumulating when
            # concurrency > 1).
            active_count: list[int],
            count_lock: threading.Lock,
        ) -> None:
            def _exit(exc: Exception | None = None) -> None:
                if exc is not None:
                    error_q.put(exc)
                in_q.put(_SENTINEL)  # wake remaining sibling workers
                with count_lock:
                    active_count[0] -= 1
                    last = active_count[0] == 0
                if last:
                    out_q.put(_SENTINEL)

            try:
                while True:
                    item = in_q.get()
                    if item is _SENTINEL:
                        _exit()
                        return
                    try:
                        result = stage.fn(item)
                    except Exception as exc:
                        _exit(exc)
                        return
                    if isinstance(result, list):
                        for r in result:
                            out_q.put(r)
                    elif result is not None:
                        out_q.put(result)
            except Exception as exc:
                _exit(exc)

        # Start worker threads for each stage.
        for i, stage in enumerate(self.stages):
            active_count = [stage.concurrency]
            count_lock = threading.Lock()
            for _ in range(stage.concurrency):
                t = threading.Thread(
                    target=_worker,
                    args=(stage, queues[i], queues[i + 1], active_count, count_lock),
                    daemon=True,
                    name=f"{self.name}.{stage.name}",
                )
                t.start()

        # Feed inputs into the first queue.
        def _feed() -> None:
            try:
                for item in inputs:
                    queues[0].put(item)
            except Exception as exc:
                error_q.put(exc)
            finally:
                queues[0].put(_SENTINEL)

        threading.Thread(target=_feed, daemon=True, name=f"{self.name}.feed").start()

        out_q = queues[n]
        while True:
            # Check for errors before blocking on get().
            if not error_q.empty():
                raise error_q.get()
            item = out_q.get()
            if item is _SENTINEL:
                # Drain any remaining errors.
                if not error_q.empty():
                    raise error_q.get()
                return
            yield item
