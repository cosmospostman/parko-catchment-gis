# Plan: Beam-style pipeline vocabulary for fetch (train/score later)

## Context

The fetch pipeline (`proxy/_pipeline.py::run_tile_pipeline_v2`) is a 250-line function
with closures inside closures, a manual prefetch deque, and implicit stage ordering.
The train and score pipelines have the same structural problem.

The goal is a small, shared vocabulary — types that make the stage structure of each
pipeline legible at a glance — without imposing a generic runner that would fight the
memory and concurrency constraints of each pipeline.

The scope of this plan is: **express the fetch pipeline using the new vocabulary.**
Train and score can adopt the same vocabulary later.

The key insight from reviewing all three pipelines: the score pipeline is already a
carefully tuned 6-stage thread pipeline with CUDA streams and gate thresholds. Forcing
it through a shared runner would destroy that. The abstraction should be a **language**
(shared types, shared names) not a **framework** (shared executor).

---

## Design

### Core types — `utils/pipeline_types.py` (new file, ~80 lines)

```python
@dataclass(frozen=True)
class StageSpec:
    name: str
    concurrency: int   # max parallel instances of this stage
    ram_gb: float      # peak RAM per instance (advisory, checked at startup)
```

```python
class Pipeline:
    """Linear stage pipeline with bounded concurrency per stage.

    Expresses a read → map/flat_map → write DAG as a sequence of StageSpecs.
    The runner uses one ThreadPoolExecutor per stage, passing outputs directly
    to the next stage as futures. Semaphore(concurrency) gates each stage.

    Memory budget: checked at __init__ time.
      assert sum(s.concurrency * s.ram_gb for s in stages) <= ram_budget_gb
    """

    def __init__(
        self,
        stages: list[StageSpec],
        ram_budget_gb: float | None = None,
    ): ...

    def run(self, inputs: Iterable[T]) -> Iterator[U]: ...
```

The runner is ~60 lines: one `Queue` per stage boundary, one `ThreadPoolExecutor`
per stage, a sentinel to signal completion. Each stage's worker acquires a
`Semaphore(spec.concurrency)` before calling the user function and releases on exit.

### Rewriting `run_tile_pipeline_v2` using the vocabulary

The function stays in `proxy/_pipeline.py`. The 250-line monolith becomes:

```python
_FETCH_PIPELINE = Pipeline([
    StageSpec("fetch_tiffs",    concurrency=2,  ram_gb=0.4),
    StageSpec("extract_scenes", concurrency=1,  ram_gb=4.0),
    StageSpec("collect_s1",     concurrency=1,  ram_gb=0.5),
    StageSpec("merge_scenes",   concurrency=2,  ram_gb=0.05),
], ram_budget_gb=12.0)
```

The stage functions (`_fetch_strip_to_tiff`, `_extract_strip`, `collect_s1_for_tile`,
`merge_scenes`) are lifted out of the closure and become module-level functions with
clear signatures. The pipeline table replaces the manual deque + prefetch_pool logic.

The `flat_map` from tile → strips and the strip-level resume logic stay in
`fetch_tile_local` (caller); `run_tile_pipeline_v2` just runs the per-strip stages.

### What the rewritten `run_tile_pipeline_v2` looks like

Before (implicit, buried in closures):
```python
with _TPE(max_workers=prefetch_depth) as prefetch_pool:
    pts_queue: _deque = _deque()
    fetch_futs: _deque = _deque()
    for k in range(min(prefetch_depth, ...)):
        ...
    for i, strip in enumerate(active_strips):
        tiff_dir = _await(fetch_futs.popleft())
        scene_paths = _extract_strip(strip, tiff_dir, strip_pts)
        shutil.rmtree(tiff_dir, ...)
        next_k = i + prefetch_depth
        if next_k < len(active_strips):
            ...
        s1_path = collect_s1_for_tile(...)
        merge_scenes(scene_paths, s1_path, strip_out)
        yield strip_idx, strip_out
```

After (explicit, readable):
```python
for strip_idx, strip_out in _FETCH_PIPELINE.run(active_strips):
    yield strip_idx, strip_out
```

The concurrency model is the same — it's just declared in `_FETCH_PIPELINE` rather than
embedded in the loop.

---

## Files

| File | Change |
|---|---|
| `utils/pipeline_types.py` | New — `StageSpec`, `Pipeline` (~140 lines total) |
| `proxy/_pipeline.py` | Lift stage closures to module-level fns; replace manual deque loop with `_FETCH_PIPELINE.run()` |
| `tests/unit/test_pipeline_types.py` | New — budget check, stage ordering, semaphore enforcement |

`utils/tile_pipeline.py` and `utils/location.py` are unchanged — they call
`run_tile_pipeline_v2` by the same interface.

---

## Runner implementation sketch

```python
# utils/pipeline_types.py

from __future__ import annotations
import threading
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue
from typing import Any, TypeVar

_SENTINEL = object()
T = TypeVar("T")

@dataclass(frozen=True)
class StageSpec:
    name: str
    fn: Callable[..., Any]     # (item, **ctx) → item | list[item]
    concurrency: int
    ram_gb: float

class Pipeline:
    def __init__(self, stages: list[StageSpec], ram_budget_gb: float | None = None):
        self.stages = stages
        if ram_budget_gb is not None:
            peak = sum(s.concurrency * s.ram_gb for s in stages)
            if peak > ram_budget_gb:
                raise ValueError(
                    f"Pipeline peak RAM {peak:.1f} GB exceeds budget {ram_budget_gb:.1f} GB\n"
                    + "\n".join(f"  {s.name}: {s.concurrency} × {s.ram_gb} GB" for s in stages)
                )

    def run(self, inputs: Iterable[Any], **ctx) -> Iterator[Any]:
        """Run inputs through all stages; yield final-stage outputs."""
        n = len(self.stages)
        queues = [Queue(maxsize=s.concurrency * 2) for s in self.stages] + [Queue()]

        def _worker(stage: StageSpec, in_q: Queue, out_q: Queue, sem: threading.Semaphore):
            while True:
                item = in_q.get()
                if item is _SENTINEL:
                    out_q.put(_SENTINEL)
                    return
                with sem:
                    result = stage.fn(item, **ctx)
                    # flat_map: fn may return a list
                    if isinstance(result, list):
                        for r in result:
                            out_q.put(r)
                    elif result is not None:
                        out_q.put(result)

        threads = []
        for i, stage in enumerate(self.stages):
            sem = threading.Semaphore(stage.concurrency)
            for _ in range(stage.concurrency):
                t = threading.Thread(
                    target=_worker, args=(stage, queues[i], queues[i + 1], sem), daemon=True
                )
                t.start()
                threads.append(t)

        # Feed inputs into first queue
        def _feed():
            for item in inputs:
                queues[0].put(item)
            queues[0].put(_SENTINEL)

        feed_t = threading.Thread(target=_feed, daemon=True)
        feed_t.start()

        out_q = queues[-1]
        while True:
            item = out_q.get()
            if item is _SENTINEL:
                break
            yield item
```

**Note:** The sentinel propagation above handles single-worker stages. For
`concurrency > 1`, a counter-based sentinel (one per worker, drain N sentinels
before forwarding one) is needed — that's the main nuance in the full implementation.

---

## Memory budget — fetch pipeline

```
Stage           concurrency   ram_gb   total
──────────────────────────────────────────────
fetch_tiffs         2          0.40    0.80 GB
extract_scenes      1          4.00    4.00 GB
collect_s1          1          0.50    0.50 GB
merge_scenes        2          0.05    0.10 GB
──────────────────────────────────────────────
                                       5.40 GB  ← checked at startup
```

On a 12 GB machine this leaves ~6.6 GB for the OS, Python runtime, and strip_pts
(~0.8 GB per strip, held briefly by each extract call).

---

## Verification

```bash
# Unit tests
python -m pytest tests/unit/test_pipeline_types.py -v

# Integration: run fetch for one small tile, verify strips appear
python cli/location.py fetch mitchell --years 2025 --output-dir /mnt/external/chunkstore

# Confirm memory budget is logged at startup:
# "[pipeline] fetch_tile peak budget: 5.4 GB (budget: 12.0 GB)"
```
