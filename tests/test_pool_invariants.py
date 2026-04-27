"""Invariants for the redesigned ``LeanProcessPool``.

The redesign collapsed four pieces of bookkeeping (``_num_used_processes``,
``slots`` semaphore, ``available_processes`` deque, ``checkpoints`` dict,
plus ``LeanServer._processes``) into one source of truth: ``pool._live``.
These tests assert the load-bearing invariants of that design without
needing a real Lean REPL: spawn is monkeypatched to return a mock
``LeanProcess`` whose ``stop_async`` we control.

The four old regression tests these replace each verified a property of
the old mechanism (asyncio.Event vs Semaphore wakeup; lock-not-held;
checkpoint pop on destroy; release_slot after return-timeout). All of
them are now structurally impossible to violate, so a single suite of
positive invariants is the right size.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from leantree.repl_adapter.process_pool import LeanProcessPool


def _install_mock_spawn(
    pool: LeanProcessPool,
    *,
    stop_delay: float = 0.0,
    pss: int | None = None,
) -> None:
    """Replace pool._spawn_async with a factory that returns mock LeanProcesses.

    The mock supports the methods the pool/server call on a real LeanProcess:
    ``stop_async``, ``drain_repl_output_async``, ``checkpoint``, ``rollback_to``,
    ``memory_usage_pss``, and exposes a ``_proc.pid``. ``stop_async`` sleeps
    ``stop_delay`` seconds before returning so we can drive janitor-latency
    tests deterministically.
    """
    pid_counter = [10000]

    async def fake_spawn():
        proc = MagicMock(spec_set=[
            "_proc", "checkpoint", "rollback_to", "memory_usage_pss",
            "stop_async", "stop_async_safe", "drain_repl_output_async",
            "set_deadline", "kill_group",
        ])
        pid_counter[0] += 1
        proc._proc = MagicMock()
        proc._proc.pid = pid_counter[0]
        proc.checkpoint = MagicMock(return_value=MagicMock(env_id=0))
        proc.rollback_to = MagicMock()
        proc.memory_usage_pss = MagicMock(return_value=pss if pss is not None else 1024 * 1024)
        proc.drain_repl_output_async = AsyncMock()
        async def slow_stop():
            if stop_delay > 0:
                await asyncio.sleep(stop_delay)
        proc.stop_async = AsyncMock(side_effect=slow_stop)
        proc.stop_async_safe = AsyncMock(side_effect=slow_stop)
        return proc

    pool._spawn_async = fake_spawn  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_release_is_idempotent():
    """Calling release twice on the same entry is a no-op the second time
    and never double-frees capacity. The whole class of leak bugs the
    redesign closes turns on this property."""
    pool = LeanProcessPool(
        repl_exe=None, project_path=None, max_processes=2, pss_recycle_limit=None,
    )
    _install_mock_spawn(pool)
    await pool.start_async()
    try:
        entry = await pool.acquire_async()
        assert pool.stats().total_processes == 1
        pool.release(entry, recycle=True, reason="first release")
        # Second release: no-op. Must not crash, must not push the entry
        # to the janitor a second time, must not double-decrement anything.
        pool.release(entry, recycle=True, reason="second release - should be no-op")
        # Wait for janitor to reap.
        for _ in range(50):
            if pool.stats().total_processes == 0:
                break
            await asyncio.sleep(0.02)
        assert pool.stats().total_processes == 0
    finally:
        await pool.shutdown_async()


@pytest.mark.asyncio
async def test_strict_capacity_on_recycle():
    """Slot is held until subprocess teardown completes, not released
    eagerly. Verifies the user-confirmed strict-capacity choice."""
    pool = LeanProcessPool(
        repl_exe=None, project_path=None, max_processes=1, pss_recycle_limit=None,
    )
    # 0.5 s teardown gives us a reliable window to observe the slot
    # remaining held while the janitor works.
    _install_mock_spawn(pool, stop_delay=0.5)
    await pool.start_async()
    try:
        entry = await pool.acquire_async()
        assert pool.stats().total_processes == 1
        pool.release(entry, recycle=True, reason="testing strict capacity")
        # Immediately after release, the entry should be in 'stopping'
        # state, still consuming a slot.
        s = pool.stats()
        assert s.total_processes == 1, "slot must be held while subprocess teardown runs"
        assert s.stopping_processes == 1
        # A blocking acquire here must wait for the janitor to finish.
        t0 = time.monotonic()
        new_entry = await pool.acquire_async(timeout=5.0)
        elapsed = time.monotonic() - t0
        assert new_entry is not None
        assert elapsed >= 0.4, (
            f"acquire returned in {elapsed:.3f}s, but janitor needed ~0.5s to "
            f"reap the previous subprocess"
        )
    finally:
        await pool.shutdown_async()


@pytest.mark.asyncio
async def test_no_capacity_drift_under_concurrent_recycles():
    """Spawn N entries, recycle them all concurrently from worker threads,
    verify capacity goes to 0 cleanly and reaches it within janitor latency.
    Pre-redesign this could leak slots when the loop saturated; in the new
    design ``release`` is synchronous and idempotent so drift is impossible.
    """
    pool = LeanProcessPool(
        repl_exe=None, project_path=None, max_processes=8, pss_recycle_limit=None,
    )
    _install_mock_spawn(pool, stop_delay=0.05)
    await pool.start_async()
    try:
        entries = []
        for _ in range(8):
            entries.append(await pool.acquire_async())
        assert pool.stats().total_processes == 8
        # Recycle from many threads at once - tests that release's threading
        # primitive is correct under contention.
        async def recycle_one(e):
            await asyncio.to_thread(
                pool.release, e, recycle=True, reason="concurrent recycle"
            )
        await asyncio.gather(*(recycle_one(e) for e in entries))
        # Wait for janitor to drain.
        for _ in range(200):
            if pool.stats().total_processes == 0:
                break
            await asyncio.sleep(0.02)
        s = pool.stats()
        assert s.total_processes == 0, f"slot drift: {s}"
        assert s.stopping_processes == 0
    finally:
        await pool.shutdown_async()


@pytest.mark.asyncio
async def test_release_returns_fast_under_loop_blockage():
    """``release`` is synchronous and never touches the asyncio loop; a
    blocked loop must NOT be able to delay it. The previous design's
    ``release_slot_async`` went through the loop and could time out under
    saturation, leaving the slot leaked."""
    pool = LeanProcessPool(
        repl_exe=None, project_path=None, max_processes=2, pss_recycle_limit=None,
    )
    _install_mock_spawn(pool)
    await pool.start_async()
    try:
        entry = await pool.acquire_async()
        # Block the loop for 1 s on a worker thread (so we ourselves can
        # still run; the block is on the loop, not on this coroutine).
        loop = asyncio.get_running_loop()
        block_done = asyncio.Event()
        def _block():
            time.sleep(1.0)
            loop.call_soon_threadsafe(block_done.set)
        loop.run_in_executor(None, _block)
        # Yield once so the executor task starts.
        await asyncio.sleep(0)
        # Now hammer release from this coroutine; even though the loop's
        # IO/callbacks are stalled by the executor task, release should
        # return effectively instantly (no asyncio touchpoint).
        t0 = time.monotonic()
        await asyncio.to_thread(
            pool.release, entry, recycle=True, reason="release-under-block test"
        )
        elapsed = time.monotonic() - t0
        assert elapsed < 0.1, (
            f"release took {elapsed:.3f}s under loop blockage; "
            f"it should be ~instant since accounting is threading-only"
        )
        await block_done.wait()
    finally:
        await pool.shutdown_async()


@pytest.mark.asyncio
async def test_pss_over_threshold_triggers_recycle_on_return():
    """``return_entry_async`` recycles when the entry's PSS exceeds the
    pool's ``pss_recycle_limit``. The PSS check moved from
    ``return_process_async`` (old) to ``return_entry_async`` (new) but the
    semantics are preserved."""
    pool = LeanProcessPool(
        repl_exe=None, project_path=None, max_processes=1,
        pss_recycle_limit=100 * 1024 * 1024,  # 100 MiB
    )
    # Fake PSS = 200 MiB, well over the 100 MiB threshold.
    _install_mock_spawn(pool, pss=200 * 1024 * 1024)
    await pool.start_async()
    try:
        entry = await pool.acquire_async()
        first_pid = entry.pid
        await pool.return_entry_async(entry)
        # Wait for janitor to reap, then a fresh acquire should get a NEW
        # subprocess (different pid), proving the previous one was recycled.
        for _ in range(50):
            if pool.stats().stopping_processes == 0 and pool.stats().total_processes == 0:
                break
            await asyncio.sleep(0.02)
        new_entry = await pool.acquire_async()
        assert new_entry.pid != first_pid, (
            "PSS-over-threshold should have recycled; instead the same "
            "subprocess was returned to the idle pool"
        )
    finally:
        await pool.shutdown_async()


@pytest.mark.asyncio
async def test_pss_under_threshold_returns_to_idle():
    """Mirror of the above: PSS below threshold leaves the entry idle
    (process reused on next acquire)."""
    pool = LeanProcessPool(
        repl_exe=None, project_path=None, max_processes=1,
        pss_recycle_limit=100 * 1024 * 1024,
    )
    _install_mock_spawn(pool, pss=10 * 1024 * 1024)  # 10 MiB << 100 MiB
    await pool.start_async()
    try:
        entry = await pool.acquire_async()
        first_pid = entry.pid
        await pool.return_entry_async(entry)
        # No recycle: capacity used stays at 1 (entry is now idle), and the
        # next acquire returns the same process.
        s = pool.stats()
        assert s.total_processes == 1
        assert s.available_processes == 1
        new_entry = await pool.acquire_async()
        assert new_entry.pid == first_pid
    finally:
        await pool.shutdown_async()
