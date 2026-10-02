"""Propagate a monotonic deadline through nested calls without a framework.

Design decisions, because the brief is ambiguous on several points and this
library picks one reading of each:

1. Single, monotonic clock. A deadline carries an absolute timestamp in the
   units of ``time.monotonic()``. We never call the wall clock internally, so
   tests can drive the library by passing a fake monotonic clock and no test
   ever depends on real elapsed time. The clock is a constructor argument so
   the library remains pure with respect to time.

2. ContextVar, not thread-local. ``ContextVar`` propagates across ``asyncio``
   tasks AND is per-thread for synchronous code. A thread-local would silently
   break the first time someone awaited. The trade-off is that manual thread
   pools need to re-establish the deadline in each worker thread; that is the
   documented awkward edge.

3. No automatic cancellation. Python has no safe way to interrupt arbitrary
   running code from the outside without ``signal.raise_signal`` or
   thread-killing, both of which are fragile. The library's contract is
   narrower: it records the deadline and exposes a cheap check
   (``DeadlineContext.exceeded()``). Callers check it between natural units of
   work. This is the compromise — we trade enforced deadlines for safety and
   portability.

4. Only monotonic timestamps are stored and compared. The deadline is an
   absolute ``time.monotonic()`` value, not a duration, so nesting is trivial:
   the inner deadline is simply ``min(outer, inner)``.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Iterator, Optional


class DeadlineExceeded(Exception):
    """Raised by ``check()`` when the deadline has passed.

    We subclass ``Exception`` rather than ``BaseException`` so it interacts
    correctly with ordinary ``except Exception`` handlers and does not bypass
    generators the way ``KeyboardInterrupt`` would.
    """


@dataclass(frozen=True)
class Deadline:
    """An absolute monotonic deadline and the clock it is meaningful against.

    Carrying the clock alongside the timestamp means a ``Deadline`` is
    self-contained: anyone holding one can ask ``remaining()`` without needing
    to know which clock produced it. This matters because a deadline created in
    one module with one fake clock must not be checked against another.
    """

    expires_at: float
    clock: Callable[[], float]

    def remaining(self) -> float:
        """Seconds left until expiry. May be negative if already past."""
        return self.expires_at - self.clock()

    def expired(self) -> bool:
        """True iff the deadline has passed."""
        return self.clock() >= self.expires_at


# The context variable holds the *innermost* active Deadline, or None when no
# deadline is in scope. We store the Deadline rather than just a timestamp so
# that nested callers can reuse the originating clock without re-deriving it.
_current: ContextVar[Optional[Deadline]] = ContextVar("deadline_ctx.current", default=None)


class DeadlineContext:
    """User-facing handle for the currently-active deadline.

    Returned by ``current_deadline()`` and bound by ``with_deadline()``. The
    methods are the only supported way to interact with the active deadline;
    reading the ``ContextVar`` directly is an implementation detail.
    """

    __slots__ = ("_deadline",)

    def __init__(self, deadline: Optional[Deadline]) -> None:
        self._deadline = deadline

    @property
    def deadline(self) -> Optional[Deadline]:
        """The underlying ``Deadline``, or ``None`` when nothing is bound."""
        return self._deadline

    def is_set(self) -> bool:
        """Whether a deadline is currently in scope."""
        return self._deadline is not None

    def remaining(self) -> Optional[float]:
        """Seconds remaining, or ``None`` when no deadline is set."""
        if self._deadline is None:
            return None
        return self._deadline.remaining()

    def expired(self) -> bool:
        """True if a deadline is set AND has passed.

        Returns ``False`` when no deadline is set — absence of a deadline is
        not the same as an expired deadline, and conflating them forces callers
        to add a redundant ``is_set()`` check.
        """
        if self._deadline is None:
            return False
        return self._deadline.expired()

    def check(self) -> None:
        """Raise ``DeadlineExceeded`` if the deadline has passed.

        No-op when no deadline is set. This is the cooperative interrupt point:
        long-running code should call it between natural units of work.
        """
        if self._deadline is not None and self._deadline.expired():
            raise DeadlineExceeded(
                f"deadline expired; expired {(-self._deadline.remaining()):.6f}s ago"
            )

    def __repr__(self) -> str:
        if self._deadline is None:
            return "DeadlineContext(None)"
        return f"DeadlineContext(expires_at={self._deadline.expires_at!r})"


def current_deadline() -> DeadlineContext:
    """Return a handle on the innermost active deadline.

    Returns a context with ``is_set() == False`` when nothing is bound, so
    callers can use the same API regardless of whether a deadline is in scope.
    """
    return DeadlineContext(_current.get())


@contextmanager
def with_deadline(
    timeout: float,
    *,
    clock: Optional[Callable[[], float]] = None,
) -> Iterator[DeadlineContext]:
    """Bind a deadline for the duration of the ``with`` block.

    The deadline is computed as ``now + timeout`` against ``clock`` (defaulting
    to ``time.monotonic``). When nested inside another ``with_deadline``, the
    effective expiry is the earlier of the two, so an inner block can only
    tighten a deadline, never loosen it. The inner block inherits the outer
    block's clock so both ends of the comparison share a time basis.

    Yields a ``DeadlineContext`` describing the *effective* deadline that will
    be in scope inside the block.

    Raises ``ValueError`` if ``timeout`` is not a finite, non-negative number.
    """
    if not isinstance(timeout, (int, float)):
        raise TypeError(f"timeout must be a number, got {type(timeout).__name__}")
    # bool is a subclass of int in Python; reject it explicitly so that
    # with_deadline(True) doesn't silently become with_deadline(1).
    if isinstance(timeout, bool):
        raise TypeError("timeout must be a number, not a bool")
    if timeout != timeout:  # NaN
        raise ValueError("timeout must not be NaN")
    if timeout == float("inf"):
        raise ValueError("timeout must be finite")
    if timeout < 0:
        raise ValueError(f"timeout must be non-negative, got {timeout}")

    # Resolve the clock: inherit from the active deadline if one is set so a
    # nested block stays on the same time basis; otherwise use the caller's
    # clock or the real monotonic clock.
    parent = _current.get()
    if clock is None:
        clock = parent.clock if parent is not None else time.monotonic

    requested = clock() + timeout
    if parent is not None and parent.expires_at < requested:
        effective = parent
    else:
        effective = Deadline(expires_at=requested, clock=clock)

    token = _current.set(effective)
    try:
        yield DeadlineContext(effective)
    finally:
        _current.reset(token)
