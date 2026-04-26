"""Verify `_destroy_process` evicts the LeanProcess from `pool.checkpoints`.

Without this eviction, every poisoned Lean process keeps its full
``LeanProcess`` Python object alive in the dict forever - including the
underlying ``asyncio.subprocess.Process``'s ~16 MiB-per-stream buffers.

Production root cause of the 110/137/204 GiB leanserver leak: ~30
poisoned processes/min (from RLIMIT_AS hits, Lean stack overflows,
etc.) * hours = thousands of retained LeanProcess objects, each
holding tens of MiB. The fix (commit following this test) pops
``pool.checkpoints[process]`` inside the ``_release_slot`` coroutine
that ``_destroy_process`` already schedules.

This test uses the same fake-pool + fake-process shims as
``test_destroy_process_no_lock_held.py`` so it doesn't need a real Lean
REPL to exercise the leanserver's bookkeeping.
"""

from __future__ import annotations

import asyncio

import pytest

from leantree.repl_adapter.server import LeanServer


class _FakePool:
    """Minimum LeanProcessPool shim with the checkpoints dict we care about."""

    def __init__(self, max_processes: int = 2):
        self.max_processes = max_processes
        self.available_processes = []
        self._num_used_processes = 1  # something for release_slot_async to decrement
        self._num_starting_processes = 0
        self._lock = None
        # The dict we're asserting gets popped.
        self.checkpoints: dict = {}

    @property
    def lock(self):
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def release_slot_async(self, process):
        async with self.lock:
            self.checkpoints.pop(process, None)
            if self._num_used_processes > 0:
                self._num_used_processes -= 1

    async def shutdown_async(self):
        return None


class _FakeProcess:
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


def test_destroy_process_evicts_checkpoint(running_server):
    """Register a process in pool.checkpoints, then destroy it; assert the
    dict entry is gone.  Without the fix, the entry stays forever, which
    is the production leak root cause."""
    server = running_server
    process = _FakeProcess()
    # The pool simulates having created a process and recorded its checkpoint.
    server.pool.checkpoints[process] = "sentinel-checkpoint"
    # Register it with the server as checked-out (how a real request flow
    # would have it - process_id_to_process + process_to_id populated).
    process_id = server._get_process_id(process)

    assert process in server.pool.checkpoints
    assert len(server.pool.checkpoints) == 1

    server._destroy_process(process_id, reason="test")

    assert process not in server.pool.checkpoints, (
        "LeanProcess remained in pool.checkpoints after _destroy_process - "
        "production leak regression (see commit message)"
    )
    assert len(server.pool.checkpoints) == 0


if __name__ == "__main__":
    pool = _FakePool()
    server = LeanServer(pool, address="127.0.0.1", port=0, log_level="ERROR")
    server.start()
    try:
        test_destroy_process_evicts_checkpoint(server)
        print("OK: _destroy_process evicts the process from pool.checkpoints")
    finally:
        server.stop()
