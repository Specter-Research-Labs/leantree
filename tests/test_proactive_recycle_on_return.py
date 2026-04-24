"""Regression test for proactive PSS-based recycling on ``return_process_async``.

Context: with ``RLIMIT_AS`` enabled (the production default), the original
``return_process_async`` skipped the PSS check entirely and trusted the kernel
to SIGKILL runaway processes at the absolute ceiling. That was cheap on paper
but expensive in practice: the kernel kill fires mid-tactic, surfaces as
``GNU MP: Cannot allocate memory``, and forces the actor into a retry cycle.

Fix (option 2 from the 2026-04-21 discussion): always run the PSS check on
return; when RLIMIT_AS is on, set the threshold to a fraction (default 0.5)
of ``max_process_memory_bytes``. A calm return-to-pool moment is a better
place to terminate a bloated process than mid-tactic. Each recycle is logged
with PSS + threshold + RLIMIT_AS cap + percentage so tuning is informed.

This test: drive ``return_process_async`` with a mock LeanProcess whose
``memory_usage()`` returns a value above the computed threshold, assert the
process is terminated (not returned to the pool), asserted that the log line
contains the measured PSS + RLIMIT_AS details, and asserted the pool slot is
released + checkpoint evicted.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

try:
    import pytest
except ImportError:
    pytest = None

from leantree.repl_adapter.process_pool import LeanProcessPool
from leantree.repl_adapter.interaction import LeanEnvironmentCheckpoint


def _make_pool(max_process_memory_bytes: int | None, recycle_ratio: float = 0.5):
    pool = LeanProcessPool(
        repl_exe=Path("/nonexistent/repl"),  # not used by return_process_async
        project_path=Path("/nonexistent/project"),
        max_processes=4,
        max_process_memory_bytes=max_process_memory_bytes,
        recycle_memory_ratio=recycle_ratio,
        logger=logging.getLogger("test_proactive_recycle"),
    )
    return pool


def _make_mock_process(pss_bytes: int):
    """Mock just the methods return_process_async touches."""
    process = MagicMock()
    process.memory_usage.return_value = pss_bytes
    process.stop_async = AsyncMock()
    process.drain_repl_output_async = AsyncMock()
    process.rollback_to = MagicMock()
    return process


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_recycle_triggers_when_pss_exceeds_threshold_under_rlimit_as():
    """Under RLIMIT_AS, a process whose PSS is above threshold is terminated."""
    rlimit = 32 * 1024**3  # 32 GiB
    pool = _make_pool(max_process_memory_bytes=rlimit, recycle_ratio=0.5)
    # Threshold should be 16 GiB.
    assert pool.memory_threshold_per_process == rlimit // 2

    # Process is "bloated" - PSS at 20 GiB, above the 16 GiB recycle threshold.
    process = _make_mock_process(pss_bytes=20 * 1024**3)
    pool.checkpoints[process] = LeanEnvironmentCheckpoint(env_id=1)
    pool._num_used_processes = 1

    _run(pool.return_process_async(process))

    # Must have been terminated, not returned to the pool
    process.stop_async.assert_awaited_once()
    assert process not in pool.checkpoints, "checkpoints not evicted on recycle"
    assert pool._num_used_processes == 0, "slot not released on recycle"
    assert process not in pool.available_processes, "bloated process was re-pooled"


def test_no_recycle_when_pss_below_threshold_under_rlimit_as():
    """A healthy process (below threshold) is returned to the pool normally."""
    rlimit = 32 * 1024**3
    pool = _make_pool(max_process_memory_bytes=rlimit, recycle_ratio=0.5)

    process = _make_mock_process(pss_bytes=2 * 1024**3)  # 2 GiB, well below 16 GiB
    pool.checkpoints[process] = LeanEnvironmentCheckpoint(env_id=1)
    pool._num_used_processes = 1

    _run(pool.return_process_async(process))

    process.stop_async.assert_not_awaited()
    assert process in pool.available_processes, "healthy process was not re-pooled"
    assert process in pool.checkpoints, "checkpoint should survive for the next checkout"
    assert pool._num_used_processes == 0


def test_recycle_log_contains_pss_threshold_and_rlimit():
    """The recycle log line must report measured PSS, threshold, and RLIMIT_AS %."""
    rlimit = 32 * 1024**3
    pool = _make_pool(max_process_memory_bytes=rlimit, recycle_ratio=0.5)

    # Replace pool.logger.info with a capturing shim
    messages: list[str] = []
    pool.logger = logging.getLogger("test_proactive_recycle_capture")
    pool.logger.info = messages.append

    process = _make_mock_process(pss_bytes=24 * 1024**3)  # 24 GiB = 75% of 32 GiB cap
    pool.checkpoints[process] = LeanEnvironmentCheckpoint(env_id=1)
    pool._num_used_processes = 1

    _run(pool.return_process_async(process))

    recycle_msgs = [m for m in messages if "Recycling Lean process" in m]
    assert len(recycle_msgs) == 1, f"expected exactly one recycle log line, got: {messages}"
    msg = recycle_msgs[0]
    assert "PSS=24576.0 MiB" in msg, msg  # 24 GiB in MiB
    assert "RLIMIT_AS 32768 MiB" in msg, msg
    assert "75.0% of cap" in msg, msg
    assert "PSS exceeds recycle threshold" in msg, msg


def test_fallback_threshold_without_rlimit_as():
    """Without RLIMIT_AS, the legacy system-memory-fraction threshold is used."""
    pool = _make_pool(max_process_memory_bytes=None)
    assert pool.memory_threshold_per_process > 0, "legacy threshold must still be set"
    # And it should be derived from system memory, not RLIMIT_AS
    # (can't assert an exact value since it depends on the host)


if __name__ == "__main__":
    test_recycle_triggers_when_pss_exceeds_threshold_under_rlimit_as()
    print("PASS: bloated process is recycled under RLIMIT_AS")
    test_no_recycle_when_pss_below_threshold_under_rlimit_as()
    print("PASS: healthy process is re-pooled")
    test_recycle_log_contains_pss_threshold_and_rlimit()
    print("PASS: recycle log carries PSS + RLIMIT_AS + percentage")
    test_fallback_threshold_without_rlimit_as()
    print("PASS: legacy threshold still active when RLIMIT_AS is off")
