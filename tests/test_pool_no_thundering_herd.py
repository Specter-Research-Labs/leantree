"""Regression test for the get_process_async thundering herd.

Before: ``LeanProcessPool`` used ``asyncio.Event`` to signal "a process
became available". ``event.set()`` woke ALL waiters at once. With ~64
in-flight ``get_process_async`` calls (the steady state when a prover
saturates the pool), every return triggered all 64 to be scheduled in a
single event-loop tick, all contending for ``pool.lock``. 63 lost the
race and re-armed ``event.wait()`` - cycle repeats. The loop spent
seconds-to-minutes processing this herd per return, surfacing as
``get_process_async exceeded 40.0s on the event loop`` errors and
watchdog dumps showing the loop frozen for 65+ seconds.

After: ``LeanProcessPool`` uses ``asyncio.Semaphore`` (count =
``max_processes - _num_used_processes``). Each ``release()`` wakes
EXACTLY ONE waiter, eliminating the herd entirely.

This test drives the pool with a mocked ``_create_process_async`` so we
don't need real Lean: drain the pool, spawn N waiters, return ONE
process, assert exactly ONE waiter woke up.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from leantree.repl_adapter.process_pool import LeanProcessPool


def _mock_process() -> MagicMock:
    """Stand-in LeanProcess that's safe to put through return_process_async."""
    p = MagicMock()
    p.drain_repl_output_async = AsyncMock()
    p.rollback_to = MagicMock()
    p.stop_async = AsyncMock()
    return p


def _make_pool(max_processes: int = 4) -> LeanProcessPool:
    pool = LeanProcessPool(
        repl_exe=Path("/nonexistent/repl"),
        project_path=Path("/nonexistent/project"),
        max_processes=max_processes,
        # Disable PSS recycling so return_process_async takes the simple path.
        pss_recycle_limit=None,
        rss_hard_limit=None,
        logger=logging.getLogger("test_pool_herd"),
    )
    # _create_process_async would otherwise spawn lake/repl.  Replace with
    # a mock that hands out fresh fake processes on demand.
    async def fake_create():
        return _mock_process()
    pool._create_process_async = fake_create
    return pool


@pytest.mark.asyncio
async def test_one_return_wakes_only_one_waiter():
    """The core regression: a single return must wake exactly one waiter,
    not all of them. This is what makes the loop responsive under
    saturated load."""
    pool = _make_pool(max_processes=2)

    # Drain the pool: 2 acquires, both succeed immediately (sem starts at max).
    p1 = await pool.get_process_async()
    p2 = await pool.get_process_async()
    assert p1 is not None and p2 is not None
    assert pool._num_used_processes == 2

    # Spawn 5 waiters; all should park on sem.acquire() because the pool
    # is at capacity (more waiters than max_processes is exactly the
    # production scenario - many HTTP request handlers in flight).
    waiters = [asyncio.create_task(pool.get_process_async()) for _ in range(5)]
    # Give the loop a few ticks to schedule everyone onto the sem queue.
    for _ in range(10):
        await asyncio.sleep(0)
    assert all(not t.done() for t in waiters), (
        "all waiters should be parked on slots.acquire(), but some completed early"
    )

    # Return one process.  With the OLD asyncio.Event broadcast, all 5
    # waiters would have been scheduled in the next tick.  With the new
    # asyncio.Semaphore, exactly ONE wakes (FIFO).
    await pool.return_process_async(p1)
    for _ in range(10):
        await asyncio.sleep(0)

    done = [t for t in waiters if t.done()]
    assert len(done) == 1, (
        f"expected exactly 1 waiter to wake on a single return, got {len(done)}; "
        "thundering herd has regressed (asyncio.Event-style broadcast)"
    )

    # Cleanup: cancel the rest, await them so the test doesn't leak tasks.
    for t in waiters:
        if not t.done():
            t.cancel()
    for t in waiters:
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_n_returns_wake_n_waiters():
    """Sanity: N returns wake N waiters, in FIFO order. Confirms the
    semaphore really does propagate each release to exactly one waiter."""
    pool = _make_pool(max_processes=3)

    # Drain.
    held = [await pool.get_process_async() for _ in range(3)]
    assert pool._num_used_processes == 3

    # 6 waiters parked.
    waiters = [asyncio.create_task(pool.get_process_async()) for _ in range(6)]
    for _ in range(10):
        await asyncio.sleep(0)
    assert all(not t.done() for t in waiters)

    # Return 3 processes one at a time.  After each return, exactly one
    # additional waiter should have completed.
    for i, p in enumerate(held, start=1):
        await pool.return_process_async(p)
        for _ in range(10):
            await asyncio.sleep(0)
        done = sum(1 for t in waiters if t.done())
        assert done == i, f"after {i} returns, expected {i} waiters done, got {done}"

    # Cleanup.
    for t in waiters:
        if not t.done():
            t.cancel()
    for t in waiters:
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_get_process_timeout_returns_none_without_leaking_slot():
    """If a get times out, the slot count must be unchanged."""
    pool = _make_pool(max_processes=1)

    # Drain.
    held = await pool.get_process_async()
    assert pool._num_used_processes == 1
    assert pool.slots._value == 0  # one slot used

    # Try to get with a short timeout - no return is coming, so this must
    # time out and return None, leaving the slot count untouched.
    result = await pool.get_process_async(blocking=True, timeout=0.1)
    assert result is None
    assert pool._num_used_processes == 1
    assert pool.slots._value == 0, (
        f"slot leaked on timeout: _value={pool.slots._value} (expected 0)"
    )

    # Returning still works, and a subsequent acquire still succeeds.
    await pool.return_process_async(held)
    again = await pool.get_process_async()
    assert again is not None


@pytest.mark.asyncio
async def test_non_blocking_returns_none_when_full_without_leaking_slot():
    """Non-blocking get must not consume a slot when the pool is full."""
    pool = _make_pool(max_processes=1)
    held = await pool.get_process_async()
    assert pool.slots._value == 0

    result = await pool.get_process_async(blocking=False)
    assert result is None
    assert pool.slots._value == 0, (
        f"non-blocking get leaked a slot: _value={pool.slots._value}"
    )

    # Sanity: after return, non-blocking get succeeds.
    await pool.return_process_async(held)
    p = await pool.get_process_async(blocking=False)
    assert p is not None


@pytest.mark.asyncio
async def test_release_slot_async_decrements_used_and_wakes_waiter():
    """release_slot_async (used by server.py's destroy paths) must
    decrement the used count and wake one waiter, just like a normal
    return."""
    pool = _make_pool(max_processes=1)
    held = await pool.get_process_async()
    pool.checkpoints[held] = "sentinel"

    # One waiter parks.
    waiter = asyncio.create_task(pool.get_process_async())
    for _ in range(10):
        await asyncio.sleep(0)
    assert not waiter.done()

    # Destroy-path release: pops checkpoint, decrements used, wakes waiter.
    await pool.release_slot_async(held)
    for _ in range(10):
        await asyncio.sleep(0)

    assert held not in pool.checkpoints, "release_slot_async must pop checkpoint"
    assert waiter.done(), "release_slot_async must wake one waiter"
    p = await waiter
    assert p is not None
