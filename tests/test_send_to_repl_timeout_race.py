"""Regression test for CPython issue 30289 in `_send_to_repl_async`.

Background - when the original implementation used ``asyncio.wait_for`` to
bound a ``readline()`` call, a timeout would cancel the inner coroutine
mid-await but leave ``StreamReader._waiter`` set.  The next read on the
same stream then raised ``RuntimeError: readuntil() called while another
coroutine is already waiting for incoming data``.  Under load that
RuntimeError propagated into the leanserver's asyncio event loop and
deadlocked the whole HTTP server.

This test forces a deadline-induced read abort and asserts that the
follow-up call fails with ``LeanProcessException`` (poisoned process -
expected after we kill it) rather than the RuntimeError race.

Run with::

    cd ~/repos/leantree
    PATH=/lnet/troja/work/people/kripner/.elan_home/bin:$PATH \\
    pytest -xvs tests/test_send_to_repl_timeout_race.py

Requires a working ``lake`` and a built REPL binary at
``lean-repl/.lake/build/bin/repl``.
"""

import asyncio
import os
from pathlib import Path

import pytest

from leantree.repl_adapter.interaction import LeanProcess, LeanProcessException


REPL_EXE = Path(
    os.environ.get("LEANTREE_REPL", "/home/kripner/repos/leantree/lean-repl/.lake/build/bin/repl")
)
PROJECT_PATH = Path(
    os.environ.get(
        "LEANTREE_PROJECT", "/home/kripner/troja/nanoproof/leantree_project_v4.27"
    )
)


def _have_repl_setup() -> bool:
    return REPL_EXE.exists() and PROJECT_PATH.exists()


pytestmark = pytest.mark.skipif(
    not _have_repl_setup(),
    reason=f"REPL binary {REPL_EXE} or project {PROJECT_PATH} not available",
)


async def _force_timeout_then_retry():
    """Trigger the timeout path on a long-running command, then issue a
    second call on the same `LeanProcess` instance.

    With the bug present (pre-fix code that used ``asyncio.wait_for`` on
    the inner read coroutine), the second call raises
    ``RuntimeError: readuntil() called while another coroutine is already
    waiting for incoming data``.

    With the fix in place, the timeout SIGKILLs the subprocess, the
    inner read returns ``b""`` (EOF), and both calls surface as
    ``LeanProcessException`` - no RuntimeError, no stream corruption.
    """
    proc = LeanProcess(REPL_EXE, PROJECT_PATH)
    await proc.start_async()
    try:
        # Importing Mathlib is expensive (multi-second).  Asking for it
        # with a 0.2 s timeout is a guaranteed timeout in the readline
        # await of `_send_to_repl_async`.
        with pytest.raises(LeanProcessException) as first_exc_info:
            await proc._send_to_repl_async({"cmd": "import Mathlib"}, timeout=0.2)
        assert "did not respond within" in str(first_exc_info.value), (
            f"first call should have surfaced as a timeout, got: {first_exc_info.value!r}"
        )

        # The bug - if present - would manifest here as a RuntimeError
        # bubbling up from `StreamReader._waiter`.  Any LeanProcessException
        # is acceptable (the process is dead, so EOF / pipe-closed are both
        # fine), as long as it is NOT a RuntimeError.
        with pytest.raises(LeanProcessException) as second_exc_info:
            await proc._send_to_repl_async({"cmd": "def x := 1"}, timeout=5.0)
        # Defensive: the type pytest.raises is enforcing (LeanProcessException)
        # already excludes RuntimeError, but spell the regression intent out
        # loud in the assertion message for whoever reads test failures next.
        assert "readuntil" not in str(second_exc_info.value), (
            f"second call leaked the StreamReader race: {second_exc_info.value!r}"
        )
    finally:
        await proc.stop_async_safe()


async def _concurrent_reads_must_not_race():
    """Two coroutines calling ``_send_to_repl_async`` concurrently on the
    same `LeanProcess` instance.

    In the buggy pre-fix code, this is the production trigger: the
    leanserver's asyncio event loop runs many handler-coroutines, and a
    timeout in one of them (cancelled mid-``readline()``) corrupts the
    shared ``StreamReader`` so the *next* read raises
    ``RuntimeError: readuntil() called while another coroutine is
    already waiting for incoming data``.

    With the fix, the first timeout SIGKILLs the subprocess.  The second
    coroutine sees EOF / pipe-closed and surfaces as
    ``LeanProcessException`` - never as ``RuntimeError``.
    """
    proc = LeanProcess(REPL_EXE, PROJECT_PATH)
    await proc.start_async()
    try:
        # Coroutine A: tight timeout that will fire while readline is
        # parked.  Coroutine B: launched 50 ms later - under the buggy
        # code its readline await lands while A's _waiter is still set.
        async def call(cmd, timeout, delay):
            await asyncio.sleep(delay)
            try:
                await proc._send_to_repl_async({"cmd": cmd}, timeout=timeout)
            except LeanProcessException as e:
                return ("ok", str(e))
            except RuntimeError as e:
                return ("race", str(e))
            return ("returned-normally", None)

        results = await asyncio.gather(
            call("import Mathlib", 0.2, 0.0),
            call("def y := 2", 5.0, 0.05),
            return_exceptions=False,
        )
        for tag, msg in results:
            assert tag != "race", (
                f"StreamReader race regression - got RuntimeError: {msg}"
            )
    finally:
        await proc.stop_async_safe()


def test_send_to_repl_timeout_does_not_corrupt_stream():
    asyncio.run(_force_timeout_then_retry())


def test_concurrent_send_to_repl_does_not_race_after_timeout():
    asyncio.run(_concurrent_reads_must_not_race())


if __name__ == "__main__":
    # Run standalone without pytest, so it's easy to spot-check on a host
    # that doesn't have pytest installed in the active env.
    asyncio.run(_force_timeout_then_retry())
    asyncio.run(_concurrent_reads_must_not_race())
    print("OK: timeout did not corrupt the stream (single + concurrent)")
