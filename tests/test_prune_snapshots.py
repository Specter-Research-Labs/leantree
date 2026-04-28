import asyncio
import pytest


def test_prune_snapshots_noop_when_no_bounds():
    from leantree.repl_adapter.interaction import LeanProcess

    p = LeanProcess.__new__(LeanProcess)
    p._env_id = None
    p._assert_started = lambda: None

    called = False

    async def _send_to_repl_async(_data):
        nonlocal called
        called = True
        return "OK"

    p._send_to_repl_async = _send_to_repl_async

    asyncio.run(p.prune_snapshots_async())
    assert called is False


def test_prune_snapshots_refuses_to_prune_current_env():
    from leantree.repl_adapter.interaction import LeanProcess

    p = LeanProcess.__new__(LeanProcess)
    p._env_id = 5
    p._assert_started = lambda: None

    async def _send_to_repl_async(_data):
        raise AssertionError("should not be called")

    p._send_to_repl_async = _send_to_repl_async

    with pytest.raises(ValueError, match="Refusing to prune"):
        asyncio.run(p.prune_snapshots_async(cmd_from_id=5))


def test_prune_snapshots_sends_payload_and_requires_ok():
    from leantree.repl_adapter.interaction import LeanProcess

    p = LeanProcess.__new__(LeanProcess)
    p._env_id = 3
    p._assert_started = lambda: None

    sent = []

    async def _send_to_repl_async(data):
        sent.append(data)
        return "OK"

    p._send_to_repl_async = _send_to_repl_async

    asyncio.run(p.prune_snapshots_async(cmd_from_id=4, proof_from_id=0))
    assert sent == [{"cmdFromId": 4, "proofFromId": 0}]


def test_pool_prunes_on_return_and_terminates_on_prune_error():
    from pathlib import Path

    from leantree.repl_adapter.process_pool import LeanProcessPool
    from leantree.repl_adapter.interaction import LeanEnvironmentCheckpoint

    class DummyProcess:
        def __init__(self):
            self.prune_calls = []
            self.stop_calls = 0
            self.rollback_calls = []

        def virtual_memory_usage(self) -> int:
            return 0

        async def drain_repl_output_async(self):
            return None

        def rollback_to(self, checkpoint: LeanEnvironmentCheckpoint):
            self.rollback_calls.append(checkpoint.env_id)

        async def prune_snapshots_async(self, *, cmd_from_id=None, proof_from_id=None):
            self.prune_calls.append((cmd_from_id, proof_from_id))
            raise RuntimeError("boom")

        async def stop_async(self):
            self.stop_calls += 1

    pool = LeanProcessPool(
        repl_exe=Path("repl"),
        project_path=Path("proj"),
        max_processes=1,
    )
    pool.memory_threshold_per_process = 0
    pool._num_used_processes = 1

    proc = DummyProcess()
    pool.checkpoints[proc] = LeanEnvironmentCheckpoint(env_id=7)

    asyncio.run(pool.return_process_async(proc))

    assert proc.rollback_calls == [7]
    assert proc.prune_calls == [(8, 0)]
    assert proc.stop_calls == 1
    assert proc not in pool.available_processes
