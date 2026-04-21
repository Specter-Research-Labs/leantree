"""Regression test for the pool-checkpoint leak when ``_handle_return_process``
times out on ``return_process_async``.

Production symptom (2026-04-21 h5):
  - ``/status`` reported ``total_tracked_processes=0`` but ``used_processes=62``
  - 82 live lake/repl children on a box configured for 62 (~20 leaked)
  - 181 GB RAM in repl children + 68 GB in the parent Python process
  - ``pool.checkpoints`` was retaining LeanProcess objects with their ~16 MiB-
    per-stream StreamReader buffers for every leaked process

Root cause: ``_handle_return_process`` removes the process from server
tracking BEFORE calling ``return_process_async``.  If that call times out
(30s), the handler logs and returns HTTP 503 but leaves the process in
``pool.checkpoints``, the pool slot stuck as "used", and the lake/repl
subprocess running.  The reaper can't help because ``_process_last_used``
was already popped.

Fix: on both the timeout and generic-exception paths in the three affected
handlers (``_handle_return_process``, ``_handle_get_process``'s
BrokenPipeError branch, ``_reap_leaked_processes``), call
``_destroy_untracked_process`` to stop the subprocess, pop from
``pool.checkpoints``, and decrement ``_num_used_processes``.

This test: monkey-patch ``pool.return_process_async`` to sleep for longer
than RETURN_PROCESS_TIMEOUT, then fire ``_handle_return_process`` from a
separate thread and assert that after the handler gives up, the process has
been evicted from ``pool.checkpoints`` and the slot is released.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from http.client import HTTPConnection
from unittest.mock import patch

try:
    import pytest
except ImportError:  # local dev boxes without pytest installed
    pytest = None

from leantree.repl_adapter.server import LeanServer, RETURN_PROCESS_TIMEOUT


class _FakePool:
    """Minimum LeanProcessPool shim compatible with the server.

    We reuse the same shim shape as existing tests in this dir.
    """

    def __init__(self, max_processes: int = 2):
        self.max_processes = max_processes
        self.available_processes = []
        self._num_used_processes = 1  # a process has been "checked out"
        self._num_starting_processes = 0
        self._lock = None
        self._process_available_event = None
        self.checkpoints: dict = {}

    @property
    def lock(self):
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @property
    def process_available_event(self):
        if self._process_available_event is None:
            self._process_available_event = asyncio.Event()
        return self._process_available_event

    async def shutdown_async(self):
        return None

    # Default: just add the process back, decrement the slot.  The test
    # monkey-patches this with a slow version to trigger the timeout.
    async def return_process_async(self, process):
        async with self.lock:
            if self._num_used_processes > 0:
                self._num_used_processes -= 1
            self.available_processes.append(process)
            self.process_available_event.set()


class _FakeProcess:
    """Stand-in for LeanProcess with a safe stop_async."""

    async def stop_async_safe(self):
        return None


if pytest is not None:

    @pytest.fixture
    def running_server():
        pool = _FakePool()
        server = LeanServer(pool, address="127.0.0.1", port=0, log_level="ERROR")
        server.start()
        try:
            yield server
        finally:
            server.stop()


def test_return_process_timeout_destroys_orphan_not_leaks(running_server):
    """If ``return_process_async`` takes longer than RETURN_PROCESS_TIMEOUT,
    the handler must destroy the orphaned process rather than leak it."""
    server = running_server

    # Register a fake process with the server so /return can find it.
    process = _FakeProcess()
    process_id = server._get_process_id(process)
    # Simulate the pool state: checkpoint exists, slot counted as used.
    server.pool.checkpoints[process] = "sentinel-checkpoint"
    assert process in server.pool.checkpoints
    assert server.pool._num_used_processes == 1

    # Monkey-patch the pool's return_process_async to sleep past the handler
    # timeout.  Sleeping 2x the handler timeout is plenty.
    async def slow_return(_proc):
        await asyncio.sleep(RETURN_PROCESS_TIMEOUT * 2.0)

    with patch.object(server.pool, "return_process_async", side_effect=slow_return):
        # Fire /process/<id>/return.  Must respond within ~30s (the handler
        # timeout) + a little slack, and must NOT wait for slow_return to
        # finish.
        t0 = time.monotonic()
        # port=0 was requested; the actual bound port is on the HTTPServer's socket
        actual_port = server.server.server_address[1]
        conn = HTTPConnection(server.address, actual_port, timeout=RETURN_PROCESS_TIMEOUT + 15.0)
        conn.request(
            "POST",
            f"/process/{process_id}/return",
            body=json.dumps({}),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        _body = resp.read()
        elapsed = time.monotonic() - t0

    # Handler should have given up and returned 503 by now.
    assert resp.status == 503, f"expected 503 after handler timeout, got {resp.status}"
    assert elapsed < RETURN_PROCESS_TIMEOUT + 10.0, (
        f"handler took {elapsed:.1f}s, should be bounded by RETURN_PROCESS_TIMEOUT={RETURN_PROCESS_TIMEOUT}"
    )

    # Critical: after handler gave up, the orphaned process must be destroyed:
    #   - evicted from pool.checkpoints
    #   - slot released (_num_used_processes decremented)
    # Give the destroy helper's _release_slot coroutine a moment to run.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if process not in server.pool.checkpoints and server.pool._num_used_processes == 0:
            break
        time.sleep(0.1)

    assert process not in server.pool.checkpoints, (
        "process still in pool.checkpoints after handler timeout — leak regression. "
        "This is the production bug from 2026-04-21 h5 (82 live repl children, 181 GB RAM)."
    )
    assert server.pool._num_used_processes == 0, (
        f"slot not released: _num_used_processes={server.pool._num_used_processes}. "
        "Handler timeout must also decrement the pool's used-count, not just log."
    )


if __name__ == "__main__":
    pool = _FakePool()
    server = LeanServer(pool, address="127.0.0.1", port=0, log_level="ERROR")
    server.start()
    try:
        test_return_process_timeout_destroys_orphan_not_leaks(server)
        print("OK: handler timeout destroys orphaned process instead of leaking")
    finally:
        server.stop()
