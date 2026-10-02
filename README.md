# Deadline Ctx

Propagate a monotonic deadline through nested calls via a `ContextVar`, with no framework and no third-party dependencies.

```python
from deadline_ctx import with_deadline, current_deadline, DeadlineExceeded
import time

def worker():
    ctx = current_deadline()
    if ctx.expired():
        raise DeadlineExceeded("bail before doing more work")
    # ... do a natural unit of work ...
    ctx.check()  # raises DeadlineExceeded if the deadline has passed

def run():
    with with_deadline(0.5) as ctx:  # 0.5 s against time.monotonic()
        print("remaining:", ctx.remaining())
        worker()
```

Exported names: `Deadline`, `DeadlineContext`, `DeadlineExceeded`, `current_deadline`, `with_deadline`.

## Why this exists

The problem: you have a call tree — request handler calling a client calling a retry loop — and you want every layer to know how much time is left, without threading a `timeout` argument through every signature and without depending on asyncio, a web framework, or a feature flag client.

The trade-off: this library only **records and reports** a deadline. It does not interrupt running code. Python has no portable, safe way to kill a thread or cancel a sync function from the outside, and `signal.raise_signal` only works on the main thread. So the contract is cooperative: long-running code calls `ctx.check()` between natural units of work, and `ctx.expired()` / `ctx.remaining()` are available for cheap reads. If you need hard preemption, you need a different tool.

A deadline is an absolute `time.monotonic()` timestamp, not a duration, so nesting is just `min(outer, inner)`. The inner block can only tighten a deadline, never extend it. The clock is a constructor argument (`with_deadline(timeout, clock=fn)`), so tests pass a fake clock and never depend on wall-clock time.

## The awkward edge

`ContextVar` propagates across `asyncio` tasks and is per-thread for synchronous code. It does **not** cross into worker threads spawned by a `ThreadPoolExecutor` — each worker starts with no deadline in scope. If you hand work to a thread pool, re-establish the deadline inside each task:

```python
from concurrent.futures import ThreadPoolExecutor

def thread_task(timeout):
    with with_deadline(timeout):
        worker()

with ThreadPoolExecutor() as pool:
    fut = pool.submit(thread_task, 0.2)
```

There is no implicit copy-on-spawn. This is deliberate: silently inheriting a deadline into a thread whose lifetime you don't control tends to mask expiry bugs.

## Running the tests

```
PYTHONPATH=src python -m unittest discover -s tests
```
