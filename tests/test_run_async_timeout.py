"""Verify that LeanServer handler threads never wait forever on a stuck
event loop.

Before the fix, every ``_run_async(coro)`` call in server.py passed
``timeout=None`` and so the HTTP handler thread would block on
``future.result()`` indefinitely if any coroutine on the loop didn't
complete.  That was the architectural root cause of the "leanserver
looks alive but /status hangs" failure mode — one stuck coro froze every
handler behind it.

These tests exercise the fix by blocking the event loop *on purpose*
(via ``loop.call_soon_threadsafe`` + ``time.sleep`` on the loop thread,
which really parks the loop — not just an async sleep that yields) and
asserting that handler threads still return promptly with HTTP 503.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import http.client
import json
import threading
import time
import urllib.error

import pytest

from leantree.repl_adapter.server import (
    LeanServer,
    LeanClient,
    RUN_ASYNC_HEADROOM,
    DEFAULT_IS_VALID_SOURCE_TIMEOUT,
)


class _FakePool:
    """Minimum shape of LeanProcessPool that LeanServer needs.

    Only ``available_processes``, ``_num_used_processes``,
    ``_num_starting_processes``, ``max_processes``, and the async
    ``get_process_async`` / ``return_process_async`` methods are exercised
    here, so we stub them out rather than spinning up real Lean.
    """

    def __init__(self, max_processes: int = 2):
        self.max_processes = max_processes
        self.available_processes = []
        self._num_used_processes = 0
        self._num_starting_processes = 0
        self._lock = None
        self._process_available_event = None

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

    async def get_process_async(self, blocking: bool = True, timeout: float | None = None):
        # Never returns a process — simulates a fully-occupied pool so the
        # get_process_async coro has to wait (and we can time it out).
        if blocking:
            await asyncio.sleep(timeout if timeout is not None else 999)
        return None

    async def return_process_async(self, process):
        return None

    async def shutdown_async(self):
        return None


@pytest.fixture
def running_server():
    """Start a LeanServer on an ephemeral port, yield (server, port), tear
    down.  Uses a fake pool so no Lean subprocess is involved."""
    pool = _FakePool(max_processes=2)
    server = LeanServer(pool, address="127.0.0.1", port=0, log_level="ERROR")
    server.start()
    port = server.server.server_address[1]
    try:
        yield server, port
    finally:
        server.stop()


def _post_json(port: int, path: str, body: dict, timeout: float = 120.0) -> tuple[int, dict]:
    """POST JSON over the stdlib HTTP client (not urllib) so we can tune
    our own socket timeout independently from any urllib global."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        payload = resp.read()
        try:
            return resp.status, json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError:
            return resp.status, {"raw": payload}
    finally:
        conn.close()


def _block_loop(server: LeanServer, seconds: float) -> threading.Event:
    """Make the event loop unresponsive for ``seconds`` seconds.

    We schedule a callback that calls ``time.sleep`` directly on the loop
    thread — not ``asyncio.sleep``, which would yield.  Returns an event
    set once the block has actually started so the caller can synchronize.
    """
    started = threading.Event()

    def _park():
        started.set()
        time.sleep(seconds)

    server._event_loop.call_soon_threadsafe(_park)
    # Wait for the block to actually land on the loop thread.
    assert started.wait(timeout=5.0), "loop never scheduled the blocker"
    return started


def test_handler_returns_503_when_loop_is_blocked(running_server):
    """A fully-blocked event loop should surface to the client as HTTP 503
    within the configured server-side timeout, not as a hung connection."""
    server, port = running_server
    # Block the loop for 60 s.  The get_process_async handler uses
    # timeout + RUN_ASYNC_HEADROOM (=30 s) as its _run_async deadline, and
    # the inner coro's own timeout is what the caller passes as
    # ``timeout`` in the request body.  We pass timeout=1.0 so the total
    # server-side deadline is ~31 s — well under the 60 s loop block.
    _block_loop(server, seconds=60.0)

    # Allow a socket timeout generous enough to catch a real 503 but short
    # enough that we fail fast if the fix regresses.
    t0 = time.monotonic()
    status, body = _post_json(port, "/process/get", {"blocking": True, "timeout": 1.0}, timeout=90.0)
    elapsed = time.monotonic() - t0
    assert status == 503, f"expected 503 on stuck loop, got {status} body={body}"
    # Client-visible latency should match the server-side deadline
    # (1.0 + RUN_ASYNC_HEADROOM=30.0), not the full 60 s loop block.
    assert elapsed < 1.0 + RUN_ASYNC_HEADROOM + 5.0, (
        f"handler took {elapsed:.1f}s — server didn't honor the _run_async deadline"
    )


def test_status_still_responds_while_loop_slow(running_server):
    """/status does not cross into the event loop, so it must stay
    responsive even when the loop is blocked.  This is a regression guard
    for future refactors that might accidentally route /status through
    _run_async."""
    server, port = running_server
    _block_loop(server, seconds=10.0)
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
    try:
        conn.request("GET", "/status")
        resp = conn.getresponse()
        assert resp.status == 200
    finally:
        conn.close()


if __name__ == "__main__":
    pool = _FakePool()
    server = LeanServer(pool, address="127.0.0.1", port=0, log_level="ERROR")
    server.start()
    port = server.server.server_address[1]
    try:
        test_handler_returns_503_when_loop_is_blocked((server, port))
        test_status_still_responds_while_loop_slow((server, port))
        print("OK: stuck loop → HTTP 503, /status stays up")
    finally:
        server.stop()
