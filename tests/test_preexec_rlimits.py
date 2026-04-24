"""Verify the subprocess preexec_fn applies RLIMIT_AS and PR_SET_PDEATHSIG.

These two limits are the load-bearing mechanism for keeping the leanserver
alive under pathological tactics:

- ``RLIMIT_AS`` makes the kernel SIGKILL a Lean subprocess that grows past
  the configured address-space ceiling, instead of letting it eat all of
  host RAM and trigger the OOM killer (which historically took the whole
  leanserver down with it).

- ``PR_SET_PDEATHSIG = SIGKILL`` makes each lake/repl subprocess die
  immediately when its parent (the leanserver Python process) exits.
  Eliminates the orphaned-grandchild problem we kept cleaning up by hand.

We don't need a real Lean REPL to verify either property - the preexec_fn
runs in any subprocess.create_subprocess_exec call.  These tests use
``sleep`` and a tiny Python one-liner so they finish in <2 s.

Run with::

    cd ~/repos/leantree
    pytest -xvs tests/test_preexec_rlimits.py
"""

import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from leantree.repl_adapter.interaction import _make_subprocess_preexec_fn


# --- RLIMIT_AS ---------------------------------------------------------------

def _run_with_preexec(cmd: list[str], max_memory_bytes: int | None = None) -> subprocess.Popen:
    """Spawn ``cmd`` with the same preexec_fn LeanProcess uses."""
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=_make_subprocess_preexec_fn(max_memory_bytes),
    )


def test_rlimit_as_is_applied_to_subprocess():
    """`/proc/<pid>/limits` should reflect the address-space ceiling we asked
    for when the subprocess was spawned via the preexec_fn."""
    target_bytes = 2 * 1024 * 1024 * 1024  # 2 GiB
    proc = _run_with_preexec(["sleep", "10"], max_memory_bytes=target_bytes)
    try:
        # The kernel updates /proc/<pid>/limits as soon as the child execs.
        # `sleep` is small enough that this is ~immediate.
        for _ in range(20):
            text = Path(f"/proc/{proc.pid}/limits").read_text()
            if "Max address space" in text:
                break
            time.sleep(0.05)
        else:
            pytest.fail("/proc/<pid>/limits never appeared")
        # Format is e.g. "Max address space     2147483648  2147483648  bytes"
        lines = [l for l in text.splitlines() if l.startswith("Max address space")]
        assert len(lines) == 1, f"unexpected limits text: {text!r}"
        parts = lines[0].split()
        soft, hard = int(parts[3]), int(parts[4])
        assert soft == target_bytes, f"soft={soft} vs requested {target_bytes}"
        assert hard == target_bytes, f"hard={hard} vs requested {target_bytes}"
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_rlimit_as_kills_oversize_allocation():
    """A subprocess that tries to allocate more than its RLIMIT_AS ceiling
    should die with a non-zero exit (kernel SIGKILL on `mmap` /
    `mprotect`).  This is the actual production property: a runaway Lean
    tactic gets killed by the kernel cleanly instead of taking the host
    down."""
    # Limit must be high enough for Python + libc to start at all.  256 MiB
    # is plenty for `import sys` but far below the 1 GiB allocation below.
    limit = 256 * 1024 * 1024
    # bytearray of 1 GiB will request ~1 GiB of address space - easily
    # past the 256 MiB ceiling.
    code = "x = bytearray(1024*1024*1024); print(len(x))"
    proc = _run_with_preexec(
        [sys.executable, "-c", code],
        max_memory_bytes=limit,
    )
    proc.wait(timeout=10)
    # Either MemoryError (Python catches the failed alloc) returns nonzero,
    # or the kernel kills the process with SIGKILL.  Both are acceptable -
    # what we care about is that the giant allocation did NOT succeed.
    assert proc.returncode != 0, (
        f"subprocess returned 0 despite {limit}-byte RLIMIT_AS - alloc was not blocked"
    )


def test_no_rlimit_as_when_disabled():
    """Passing ``max_memory_bytes=None`` should leave the subprocess with
    its inherited (effectively unlimited) RLIMIT_AS, so this is the
    opt-out for users who don't want the ceiling."""
    proc = _run_with_preexec(["sleep", "10"], max_memory_bytes=None)
    try:
        for _ in range(20):
            text = Path(f"/proc/{proc.pid}/limits").read_text()
            if "Max address space" in text:
                break
            time.sleep(0.05)
        lines = [l for l in text.splitlines() if l.startswith("Max address space")]
        soft, hard = lines[0].split()[3], lines[0].split()[4]
        # When unset we expect "unlimited" - the proc text uses that literal
        # (it's what the kernel reports when RLIM_INFINITY is in effect).
        assert soft == "unlimited", f"soft={soft!r} (expected unlimited)"
        assert hard == "unlimited", f"hard={hard!r} (expected unlimited)"
    finally:
        proc.kill()
        proc.wait(timeout=5)


# --- PR_SET_PDEATHSIG --------------------------------------------------------

def test_pdeathsig_kills_child_when_parent_dies():
    """A grandchild spawned via _make_subprocess_preexec_fn should die
    immediately when its (intermediate) parent exits - even if we never
    explicitly kill it.  Historically these grandchildren were orphaned
    when the leanserver crashed, accumulating until manual cleanup."""
    # Pattern: this test process spawns a Python child; the child spawns a
    # grandchild via subprocess.Popen(preexec_fn=_make_subprocess_preexec_fn).
    # We capture the grandchild's pid via stdout, then SIGKILL the child and
    # check that the grandchild also dies (PDEATHSIG=SIGKILL on parent exit).
    parent_code = """
import os, subprocess, sys, time
sys.path.insert(0, %r)
from leantree.repl_adapter.interaction import _make_subprocess_preexec_fn
gc = subprocess.Popen(
    ["sleep", "60"],
    preexec_fn=_make_subprocess_preexec_fn(None),
)
print(gc.pid, flush=True)
time.sleep(60)
""" % str(Path(__file__).resolve().parents[1])
    child = subprocess.Popen(
        [sys.executable, "-c", parent_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # Read the grandchild pid the child printed.
        line = child.stdout.readline()
        assert line, f"child produced no stdout; stderr={child.stderr.read()!r}"
        grandchild_pid = int(line.strip())
        assert _pid_alive(grandchild_pid), "grandchild not alive after spawn"
        # Now SIGKILL the intermediate parent (the child).  PR_SET_PDEATHSIG
        # should fire and the kernel should send SIGKILL to the grandchild.
        child.kill()
        child.wait(timeout=5)
        # Allow up to 2 s for the kernel to reap the grandchild.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if not _pid_alive(grandchild_pid):
                return
            time.sleep(0.05)
        # Cleanup if the test failed
        try:
            os.kill(grandchild_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        pytest.fail(f"grandchild pid={grandchild_pid} survived parent death - PDEATHSIG didn't fire")
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we don't own it; for our test this means alive.
        return True


if __name__ == "__main__":
    test_rlimit_as_is_applied_to_subprocess()
    test_rlimit_as_kills_oversize_allocation()
    test_no_rlimit_as_when_disabled()
    test_pdeathsig_kills_child_when_parent_dies()
    print("OK: preexec rlimits + pdeathsig all working")
