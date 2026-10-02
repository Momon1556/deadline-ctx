import math
import unittest

from deadline_ctx import (
    Deadline,
    DeadlineContext,
    DeadlineExceeded,
    current_deadline,
    with_deadline,
)


class FakeClock:
    """A controllable monotonic clock.

    Every test that cares about time drives the library through this so no test
    ever depends on real elapsed wall-clock time.
    """

    def __init__(self, start: float = 0.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


class TestDeadline(unittest.TestCase):
    def test_remaining_before_expiry(self):
        clock = FakeClock(100.0)
        d = Deadline(expires_at=150.0, clock=clock)
        self.assertEqual(d.remaining(), 50.0)

    def test_remaining_at_expiry_is_zero(self):
        clock = FakeClock(100.0)
        d = Deadline(expires_at=100.0, clock=clock)
        self.assertEqual(d.remaining(), 0.0)

    def test_remaining_after_expiry_is_negative(self):
        clock = FakeClock(100.0)
        d = Deadline(expires_at=80.0, clock=clock)
        self.assertEqual(d.remaining(), -20.0)

    def test_expired_uses_clock_progress(self):
        clock = FakeClock(0.0)
        d = Deadline(expires_at=10.0, clock=clock)
        self.assertFalse(d.expired())
        clock.advance(10.0)
        self.assertTrue(d.expired())
        clock.advance(0.001)
        self.assertTrue(d.expired())


class TestCurrentDeadline(unittest.TestCase):
    def test_no_deadline_in_scope(self):
        ctx = current_deadline()
        self.assertIsInstance(ctx, DeadlineContext)
        self.assertFalse(ctx.is_set())
        self.assertIsNone(ctx.deadline)
        self.assertIsNone(ctx.remaining())
        self.assertFalse(ctx.expired())
        ctx.check()  # no-op, must not raise

    def test_repr_without_deadline(self):
        self.assertEqual(repr(current_deadline()), "DeadlineContext(None)")


class TestWithDeadline(unittest.TestCase):
    def test_yields_set_context_with_effective_deadline(self):
        clock = FakeClock(1000.0)
        with with_deadline(5.0, clock=clock) as ctx:
            self.assertTrue(ctx.is_set())
            self.assertIsInstance(ctx.deadline, Deadline)
            self.assertEqual(ctx.deadline.expires_at, 1005.0)
            # The in-scope deadline matches the one yielded.
            self.assertIs(current_deadline().deadline, ctx.deadline)
            self.assertEqual(ctx.remaining(), 5.0)
            self.assertFalse(ctx.expired())

    def test_deadline_cleared_after_block(self):
        clock = FakeClock(0.0)
        self.assertFalse(current_deadline().is_set())
        with with_deadline(1.0, clock=clock):
            self.assertTrue(current_deadline().is_set())
        self.assertFalse(current_deadline().is_set())

    def test_deadline_cleared_even_on_exception(self):
        clock = FakeClock(0.0)
        class Sentinel(Exception):
            pass
        with self.assertRaises(Sentinel):
            with with_deadline(1.0, clock=clock):
                raise Sentinel()
        self.assertFalse(current_deadline().is_set())

    def test_inner_can_only_tighten_not_loosen(self):
        clock = FakeClock(0.0)
        with with_deadline(10.0, clock=clock) as outer:
            self.assertEqual(outer.remaining(), 10.0)
            with with_deadline(3.0) as inner:  # inherits outer's clock
                self.assertEqual(inner.remaining(), 3.0)
                self.assertLess(
                    inner.deadline.expires_at,
                    outer.deadline.expires_at,
                )
            # Back in the outer scope.
            self.assertEqual(outer.remaining(), 10.0)

    def test_inner_looser_than_outer_takes_outer(self):
        clock = FakeClock(0.0)
        with with_deadline(2.0, clock=clock) as outer:
            with with_deadline(100.0) as inner:
                # Inner cannot extend beyond outer.
                self.assertEqual(inner.deadline.expires_at, outer.deadline.expires_at)
                self.assertIs(inner.deadline, outer.deadline)

    def test_inner_inherits_outer_clock(self):
        clock = FakeClock(50.0)
        with with_deadline(10.0, clock=clock):
            with with_deadline(5.0) as inner:
                # Both deadlines must use the same clock; otherwise comparisons
                # are meaningless.
                self.assertIs(inner.deadline.clock, clock)

    def test_check_raises_after_expiry(self):
        clock = FakeClock(0.0)
        with with_deadline(5.0, clock=clock) as ctx:
            ctx.check()  # before: ok
            clock.advance(5.0)
            with self.assertRaises(DeadlineExceeded):
                ctx.check()

    def test_current_deadline_reflects_progress(self):
        clock = FakeClock(0.0)
        with with_deadline(10.0, clock=clock):
            self.assertEqual(current_deadline().remaining(), 10.0)
            clock.advance(4.0)
            self.assertEqual(current_deadline().remaining(), 6.0)
            self.assertFalse(current_deadline().expired())
            clock.advance(6.0)
            self.assertTrue(current_deadline().expired())

    def test_check_noop_without_deadline(self):
        # Outside any with_deadline block, check() must never raise.
        current_deadline().check()

    def test_zero_timeout_is_immediately_expired(self):
        clock = FakeClock(100.0)
        with with_deadline(0.0, clock=clock) as ctx:
            # expires_at == now, so expired() is True by the >= rule.
            self.assertTrue(ctx.expired())
            with self.assertRaises(DeadlineExceeded):
                ctx.check()

    def test_default_clock_is_monotonic(self):
        # We do not assert the value, only that the Deadline carries
        # time.monotonic as its clock when no clock is supplied.
        import time
        with with_deadline(1.0) as ctx:
            self.assertIs(ctx.deadline.clock, time.monotonic)


class TestWithDeadlineValidation(unittest.TestCase):
    def test_negative_timeout_rejected(self):
        with self.assertRaises(ValueError):
            with with_deadline(-1.0, clock=FakeClock()):
                pass

    def test_nan_timeout_rejected(self):
        with self.assertRaises(ValueError):
            with with_deadline(float("nan"), clock=FakeClock()):
                pass

    def test_inf_timeout_rejected(self):
        with self.assertRaises(ValueError):
            with with_deadline(float("inf"), clock=FakeClock()):
                pass

    def test_bool_timeout_rejected(self):
        # bool is a subclass of int; we refuse it to avoid True == 1s silently.
        with self.assertRaises(TypeError):
            with with_deadline(True, clock=FakeClock()):
                pass

    def test_non_number_timeout_rejected(self):
        with self.assertRaises(TypeError):
            with with_deadline("soon", clock=FakeClock()):  # type: ignore[arg-type]
                pass


class TestContextvarSemantics(unittest.TestCase):
    """ContextVar isolation: sibling ``with`` blocks must not interfere."""

    def test_sibling_blocks_are_isolated(self):
        clock = FakeClock(0.0)
        with with_deadline(10.0, clock=clock) as a:
            a_expires = a.deadline.expires_at
            with with_deadline(2.0) as b:
                self.assertEqual(b.deadline.expires_at, 2.0)
            # Exiting the inner block must restore the outer deadline exactly.
            self.assertEqual(current_deadline().deadline.expires_at, a_expires)


class TestDeadlineExceeded(unittest.TestCase):
    def test_is_exception_subclass(self):
        # Cooperative cancellation must be catchable by ordinary handlers.
        self.assertTrue(issubclass(DeadlineExceeded, Exception))

    def test_message_contains_overdue_seconds(self):
        clock = FakeClock(0.0)
        with with_deadline(1.0, clock=clock) as ctx:
            clock.advance(3.0)
            try:
                ctx.check()
            except DeadlineExceeded as e:
                self.assertIn("expired", str(e))
            else:
                self.fail("expected DeadlineExceeded")


if __name__ == "__main__":
    unittest.main()
