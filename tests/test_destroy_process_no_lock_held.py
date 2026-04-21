"""Verify that `_destroy_process` does NOT hold `self._lock` during its
`_run_async` calls (A2 in the fix plan).

Before: `_destroy_process` acquired `self._lock` and then called
`_run_async(stop_async_safe)` + `_run_async(_release_slot)` inside the
same `with self._lock:` block.  If either coroutine stalled on the loop,
the lock was held the whole time — every subsequent handler that needed
it (status, get_process_id, return_process, ...) blocked for the entire
duration of the stall.  That was failure mode #2 in the plan.

After A2, the lock is released before the `_run_async` calls.  This test
monkey-patches `_run_async` to spin for 5 s, kicks off `_destroy_process`
from one thread, and from another thread asserts that a call which
*also* needs `self._lock` (`_get_process_id`) returns within ~100 ms —
i.e. is NOT blocked behind the slow `_run_async`.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from leantree.repl_adapter.server import LeanServer


class _FakePool:
    """Minimum LeanProcessPool shim (same as test_run_async_timeout)."""

    def __init__(self, max_processes: int = 2):
        self.max_processes = max_processes
        self.available_processes = []
        self._num_used_processes = 1  # so _release_slot decrement is non-negative
        self._num_starting_processes = 0
        self._lock = None
        self._process_available_event = None
        # _destroy_process's _release_slot pops this dict; without it the
        # coroutine raises AttributeError (silently, in a background thread).
        self.checkpoints: dict = {}

    @property
    def lock(self):
        import asyncio
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @property
    def process_available_event(self):
        import asyncio
        if self._process_available_event is None:
            self._process_available_event = asyncio.Event()
        return self._process_available_event

    async def shutdown_async(self):
        return None


class _FakeProcess:
    """Stand-in for LeanProcess with a stop_async_safe that is safe to call."""

    async def stop_async_safe(self):
        return None


@pytest.fixture
def running_server():
    pool = _FakePool()
    server = LeanServer(pool, address="127.0.0.1", port=0, log_level="ERROR")
    server.start()
    try:
        yield server
    finally:
        server.stop()


def test_destroy_process_does_not_hold_self_lock_across_run_async(running_server):
    """Slow-poke `_run_async`, call `_destroy_process` from one thread,
    and concurrently call `_get_process_id` from another.  The second
    call must return promptly — it would hang if A2 regressed."""
    server = running_server

    # Register a fake process so _destroy_process has something to destroy.
    process = _FakeProcess()
    process_id = server._get_process_id(process)

    original_run_async = server._run_async
    entered = threading.Event()
    release = threading.Event()

    def _slow_run_async(coro, timeout=None):
        entered.set()
        # Don't hang forever — block only until the test signals release.
        release.wait(timeout=10.0)
        # Still invoke the original so the coroutine doesn't leak.
        return original_run_async(coro, timeout=1.0)

    with patch.object(server, "_run_async", side_effect=_slow_run_async):
        destroy_thread = threading.Thread(
            target=server._destroy_process, args=(process_id,), daemon=True
        )
        destroy_thread.start()

        # Wait until _destroy_process is parked inside the slow _run_async.
        assert entered.wait(timeout=2.0), "destroy_process never reached _run_async"

        # NOW assert another lock-requiring path returns promptly.  If the
        # old lock-holding behavior regressed, this call would block the
        # whole 10 s release-timeout above.
        other_process = _FakeProcess()
        t0 = time.monotonic()
        other_id = server._get_process_id(other_process)
        elapsed = time.monotonic() - t0

        assert elapsed < 0.5, (
            f"_get_process_id blocked for {elapsed:.2f}s while _destroy_process was in _run_async — "
            "self._lock is held across the loop boundary (A2 regression)"
        )
        assert isinstance(other_id, int)

        # Release the slow _run_async so the destroy thread can finish.
        release.set()
        destroy_thread.join(timeout=5.0)
        assert not destroy_thread.is_alive(), "destroy thread stuck after release"


if __name__ == "__main__":
    pool = _FakePool()
    server = LeanServer(pool, address="127.0.0.1", port=0, log_level="ERROR")
    server.start()
    try:
        test_destroy_process_does_not_hold_self_lock_across_run_async(server)
        print("OK: _destroy_process releases self._lock before crossing _run_async")
    finally:
        server.stop()
