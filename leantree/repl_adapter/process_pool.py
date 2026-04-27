import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Coroutine, Literal, Self

from leantree.repl_adapter.interaction import LeanProcess, LeanEnvironmentCheckpoint
from leantree.utils import to_sync


# Cap how long we'll wait for a single PSS measurement. memory_usage_pss reads
# /proc/<pid>/smaps_rollup; on calm processes that's 50-300 ms, but when the
# Lean child is CPU-saturated (pathological tactic, GC storm) the kernel walk
# stretches to many seconds. Without a bound, the awaiting coroutine waits
# forever and the to_thread worker stays wedged - on a thundering return-burst
# the executor (default 32 workers) fills with stuck reads and no further
# return path can complete.
PSS_MEASUREMENT_TIMEOUT = 5.0

# Bound on subprocess teardown. After SIGKILL the kernel should reap within
# milliseconds, but NFS hangs / disk thrash have been seen to delay reaps.
# Cap so the janitor can't stall the rest of the queue on one wedged subprocess.
JANITOR_STOP_TIMEOUT = 5.0

# Max time shutdown() will wait for the janitor to drain `_live` before giving
# up and leaking remaining subprocesses (zombies). Long enough that a backlog
# of janitor work after a server stop usually completes; short enough that
# the host shutdown isn't blocked indefinitely.
SHUTDOWN_DRAIN_TIMEOUT = 30.0

# Cadence for the idle-lease reaper and liveness reconciler. 60 s matches the
# previous _reap_leaked_processes cadence in server.py, which has been
# production-tuned to balance reclamation latency against probe overhead.
#
# (Deadline-on-tactic kills live in interaction.py's module-level watchdog,
# not here - one watchdog covers bare LeanProcess and pool-managed alike.)
REAPER_INTERVAL = 60.0
LIVENESS_INTERVAL = 60.0


PoolEntryState = Literal["starting", "idle", "checked_out", "stopping"]


@dataclass
class PoolEntry:
    """A single managed Lean process and its lifecycle state.

    Carries everything that used to be split across the four pieces of
    bookkeeping (`_num_used_processes` int, `available_processes` deque,
    `checkpoints` dict, `_processes` registry). Membership in `pool._live`
    IS the source of truth; capacity used = `len(pool._live)`. An entry is
    removed from `_live` only after its subprocess is fully reaped, which
    is the strict-capacity guarantee.

    The dataclass is also an async context manager: an `async with` body
    exiting cleanly calls `pool.return_entry_async`, while an exception
    forces `pool.release(recycle=True)`. Direct use of `pool.acquire_async`
    + `pool.release` is also fine; release is idempotent.
    """
    id: int
    pool: "LeanProcessPool"
    process: LeanProcess | None
    pid: int | None
    state: PoolEntryState
    checkpoint: LeanEnvironmentCheckpoint | None = None
    last_used: float = 0.0

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.id not in self.pool._live:
            return  # already released; idempotent
        if exc is not None:
            self.pool.release(
                self,
                recycle=True,
                reason=f"context exit raised: {exc_type.__name__}",
            )
        else:
            await self.pool.return_entry_async(self)


@dataclass
class PoolStats:
    """Snapshot of pool state for /status.  Wire DTO; field names use the
    operator-facing vocabulary (``used`` / ``available``) rather than the
    internal state-machine literals (``checked_out`` / ``idle``).  The
    ``used`` semantics match the pre-redesign field of the same name -
    "actor is currently holding this process" - so dashboards consuming
    /status need no rewrite.

    Invariant:
        total_processes ==
            available_processes + used_processes
            + starting_processes + stopping_processes

    Reading guide:
    - ``used_processes``: actors actively holding a process. Primary
      utilization signal.
    - ``available_processes``: idle, immediately acquirable.
    - ``starting_processes``: mid-spawn (warmup or on-demand).
    - ``stopping_processes``: janitor backlog. Transiently >0 after a
      recycle; persistently >0 means subprocess teardown is wedged.
    - ``total_processes``: alive in pool = capacity used. New summary
      field; sum of the four state counts.
    """
    max_processes: int
    total_processes: int
    available_processes: int
    starting_processes: int
    used_processes: int
    stopping_processes: int


class LeanProcessPool:
    """Process pool with single-source-of-truth accounting.

    The set of managed processes lives in `self._live: dict[int, PoolEntry]`.
    Capacity used is `len(self._live)`. There is no separate counter that
    can drift: every transition mutates `_live` under a `threading.RLock`.
    All accounting is synchronous and never crosses the asyncio event loop,
    so loop saturation cannot leave the pool in an inconsistent state - the
    failure mode that motivated this rewrite (production: 64 used / 49
    tracked, 15 unrecoverable slots after RL training).

    Subprocess I/O (spawn, stop, drain, PSS read) is async and runs on a
    caller-supplied event loop, but is decoupled from accounting: the
    janitor task drains `stopping` entries off the critical path and the
    deadline watchdog runs on its own thread.
    """

    def __init__(
        self,
        repl_exe: Path,
        project_path: Path,
        max_processes: int,
        env_setup_async: Callable[[LeanProcess], Coroutine] | None = None,
        logger: logging.Logger | None = None,
        rss_hard_limit: int | None = 32 * 1024 ** 3,
        pss_recycle_limit: int | None = 4 * 1024 ** 3,
        lease_timeout: float = 600.0,
    ):
        self.repl_exe = repl_exe
        self.project_path = project_path
        self.max_processes = max_processes
        self.env_setup_async = env_setup_async
        self.rss_hard_limit = rss_hard_limit
        self.pss_recycle_limit = pss_recycle_limit
        self.logger = logger if logger else logging.getLogger(__name__)

        # Single source of truth.
        self._live: dict[int, PoolEntry] = {}
        # threading primitives, NOT asyncio: callable from any thread,
        # immune to event-loop saturation. The whole point of the redesign.
        self._lock = threading.RLock()
        self._capacity_changed = threading.Condition(self._lock)
        self._id_counter = 0
        self._shutdown = False

        # Idle reaper threshold: how long a checked-out entry can sit
        # untouched before we assume the client is gone and reclaim it.
        self._lease_timeout = lease_timeout

        # Background workers. Set in start().
        self._loop: asyncio.AbstractEventLoop | None = None
        self._janitor_inbox: asyncio.Queue[PoolEntry] | None = None
        self._janitor_task: asyncio.Task | None = None
        self._workers_stop = threading.Event()
        self._idle_reaper_thread: threading.Thread | None = None
        self._liveness_thread: threading.Thread | None = None

        # External hook the server installs so the idle reaper can avoid
        # reclaiming entries whose HTTP request is still in flight (e.g. a
        # long-running tactic on a checked-out process). Returns True iff
        # the entry currently has an active request. Default no-op for
        # tests / standalone usage.
        self.has_active_request: Callable[[int], bool] = lambda entry_id: False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the pool to a running asyncio loop in *another* thread
        (the leanserver pattern). For in-loop callers (pytest-asyncio,
        single-loop scripts) use ``await start_async()`` instead.
        Idempotent within a single pool lifetime.
        """
        if self._loop is not None:
            return
        # Sanity-check: callers using start() must NOT be on the target
        # loop themselves; otherwise run_coroutine_threadsafe would deadlock.
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            raise RuntimeError(
                "pool.start(loop) called from within the loop itself; "
                "use `await pool.start_async()` instead"
            )

        self._loop = loop

        async def _create() -> tuple[asyncio.Queue, asyncio.Task]:
            q: asyncio.Queue = asyncio.Queue()
            task = asyncio.create_task(
                self._janitor_loop(q), name="LeanPool-janitor"
            )
            return q, task
        fut = asyncio.run_coroutine_threadsafe(_create(), loop)
        self._janitor_inbox, self._janitor_task = fut.result(timeout=5.0)

        self._start_worker_threads()

    async def start_async(self) -> None:
        """In-loop variant of ``start``. Pool binds to the currently-running
        loop and creates the janitor task directly. Idempotent.
        """
        if self._loop is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._janitor_inbox = asyncio.Queue()
        self._janitor_task = asyncio.create_task(
            self._janitor_loop(self._janitor_inbox), name="LeanPool-janitor"
        )
        self._start_worker_threads()

    def _start_worker_threads(self) -> None:
        self._idle_reaper_thread = threading.Thread(
            target=self._idle_reaper_loop,
            name="LeanPool-idle-reaper",
            daemon=True,
        )
        self._idle_reaper_thread.start()

        self._liveness_thread = threading.Thread(
            target=self._liveness_reconciler_loop,
            name="LeanPool-liveness-reconciler",
            daemon=True,
        )
        self._liveness_thread.start()

    def shutdown(self) -> None:
        """Stop all processes and join workers. Blocks until the janitor
        drains `_live` (or `SHUTDOWN_DRAIN_TIMEOUT` elapses, in which case
        remaining subprocesses leak as zombies - the kernel will reap them
        when our process exits).

        Cross-thread: this assumes the loop is running in another thread.
        For in-loop callers, use ``await shutdown_async()`` to avoid
        deadlocking the janitor task on its own loop.
        """
        entries_to_stop = self._mark_shutdown_and_collect()
        if entries_to_stop and self._janitor_inbox is not None and self._loop is not None:
            for entry in entries_to_stop:
                self._loop.call_soon_threadsafe(
                    self._janitor_inbox.put_nowait, entry
                )

        deadline = time.monotonic() + SHUTDOWN_DRAIN_TIMEOUT
        while True:
            with self._lock:
                if not self._live:
                    break
            if time.monotonic() > deadline:
                self.logger.warning(
                    f"Pool shutdown: janitor did not drain {len(self._live)} "
                    f"entries within {SHUTDOWN_DRAIN_TIMEOUT} s; leaking the rest"
                )
                break
            time.sleep(0.05)

        # Cancel the janitor task on its own loop and wait for it to
        # actually exit; otherwise the upcoming ``loop.stop()`` in the
        # server's ``stop()`` would leave the task parked on
        # ``inbox.get()`` and asyncio would log
        # "Task was destroyed but it is pending!".
        if (
            self._janitor_task is not None
            and self._loop is not None
            and not self._janitor_task.done()
        ):
            cancelled = threading.Event()
            def _cancel_and_wait():
                async def _cancel():
                    self._janitor_task.cancel()
                    try:
                        await self._janitor_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    cancelled.set()
                asyncio.ensure_future(_cancel(), loop=self._loop)
            try:
                self._loop.call_soon_threadsafe(_cancel_and_wait)
                cancelled.wait(timeout=5.0)
            except RuntimeError:
                # Loop closed already; nothing to do.
                pass

        self._join_worker_threads()

    async def shutdown_async(self) -> None:
        """In-loop variant of ``shutdown``. Awaits the janitor draining
        rather than blocking on a sync sleep, so the janitor task on the
        same loop can actually run."""
        entries_to_stop = self._mark_shutdown_and_collect()
        if entries_to_stop and self._janitor_inbox is not None:
            for entry in entries_to_stop:
                self._janitor_inbox.put_nowait(entry)

        deadline = time.monotonic() + SHUTDOWN_DRAIN_TIMEOUT
        while True:
            with self._lock:
                if not self._live:
                    break
            if time.monotonic() > deadline:
                self.logger.warning(
                    f"Pool shutdown: janitor did not drain {len(self._live)} "
                    f"entries within {SHUTDOWN_DRAIN_TIMEOUT} s; leaking the rest"
                )
                break
            await asyncio.sleep(0.05)

        # Cancel the janitor task explicitly; otherwise it would wait
        # forever for the next inbox item and leak as a pending coroutine.
        if self._janitor_task is not None and not self._janitor_task.done():
            self._janitor_task.cancel()
            try:
                await self._janitor_task
            except (asyncio.CancelledError, Exception):
                pass

        self._join_worker_threads()

    def _mark_shutdown_and_collect(self) -> list[PoolEntry]:
        """Mark the pool as shut down, wake all acquirers, and return the
        list of live entries that need teardown.  Shared by
        ``shutdown`` and ``shutdown_async``.
        """
        with self._capacity_changed:
            if self._shutdown:
                return []
            self._shutdown = True
            self._capacity_changed.notify_all()
            entries_to_stop = [
                e for e in self._live.values() if e.state != "stopping"
            ]
            for entry in entries_to_stop:
                entry.state = "stopping"
        self._workers_stop.set()
        return entries_to_stop

    def _join_worker_threads(self) -> None:
        for thread in (self._idle_reaper_thread, self._liveness_thread):
            if thread is not None:
                thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Acquire / release (synchronous accounting; never touches the loop)
    # ------------------------------------------------------------------

    def acquire(self, timeout: float | None = None) -> PoolEntry | None:
        """Reserve capacity. Returns either an idle entry (state:
        checked_out) or a 'starting' placeholder whose subprocess the
        caller must spawn. Returns None iff timeout expired with the pool
        at capacity. Raises RuntimeError if the pool has been shut down.

        Synchronous; safe to call from any thread. Loop saturation cannot
        affect it - that's the whole reason this exists.
        """
        deadline = time.monotonic() + timeout if timeout is not None else None
        with self._capacity_changed:
            while True:
                if self._shutdown:
                    raise RuntimeError("Process pool has been shut down")
                # Reuse idle if any. Iteration order is insertion order in
                # Python 3.7+, so this is roughly LRU-by-creation.
                for entry in self._live.values():
                    if entry.state == "idle":
                        entry.state = "checked_out"
                        entry.last_used = time.monotonic()
                        return entry
                # Else: do we have capacity to start a new one?
                if len(self._live) < self.max_processes:
                    return self._new_placeholder_locked()
                # Capacity full; wait for janitor to free a slot or for
                # an in-flight return to publish an idle entry.
                if deadline is None:
                    self._capacity_changed.wait()
                else:
                    rem = deadline - time.monotonic()
                    if rem <= 0:
                        return None
                    self._capacity_changed.wait(timeout=rem)

    async def acquire_async(self, timeout: float | None = None) -> PoolEntry | None:
        """Async wrapper around `acquire`. On placeholder return, spawns
        the subprocess outside any lock and transitions the entry from
        ``starting`` to ``checked_out``.  On spawn failure, recycles the
        placeholder (frees the capacity slot) and re-raises.
        """
        entry = await asyncio.to_thread(self.acquire, timeout)
        if entry is None:
            return None
        if entry.process is None:
            try:
                process = await self._spawn_async()
            except BaseException:
                self.release(entry, recycle=True, reason="spawn failed")
                raise
            with self._lock:
                entry.process = process
                entry.pid = process._proc.pid
                entry.checkpoint = process.checkpoint()
                # Promote out of "starting": this entry is now in the
                # caller's hands.  Without this, /status would report it
                # as a starting_processes count forever - including the
                # used_processes wire field, which would mis-report 0.
                entry.state = "checked_out"
                entry.last_used = time.monotonic()
        return entry

    def release(self, entry: PoolEntry, *, recycle: bool, reason: str = "") -> None:
        """Idempotent release. With ``recycle=True`` the entry is queued
        for teardown (capacity slot freed only after the subprocess is
        reaped - the strict-capacity choice). With ``recycle=False`` the
        entry returns to the idle pool for the next acquirer.

        Calling release on an already-released entry is a no-op. This is
        the property that eliminates the bookkeeping-drift class of bugs:
        every cleanup path can call ``release`` without coordinating with
        every other cleanup path.

        Synchronous; never touches the asyncio loop directly. The handoff
        to the janitor (when ``recycle=True``) is via a thread-safe
        ``call_soon_threadsafe`` onto the loop.
        """
        with self._capacity_changed:
            if entry.id not in self._live:
                return
            if recycle or self._shutdown:
                if entry.state == "stopping":
                    return  # already queued for teardown
                entry.state = "stopping"
                if reason:
                    self.logger.info(
                        f"Recycling Lean entry {entry.id} (pid={entry.pid}): {reason}"
                    )
                if self._janitor_inbox is None or self._loop is None:
                    # Pre-start path: pool was never started, so there's
                    # no janitor. Drop directly. Subprocess (if any) leaks
                    # to the OS - acceptable; this branch is for tests
                    # that use the pool without start().
                    self._live.pop(entry.id, None)
                    self._capacity_changed.notify()
                    return
                self._loop.call_soon_threadsafe(
                    self._janitor_inbox.put_nowait, entry
                )
            else:
                entry.state = "idle"
                entry.last_used = time.monotonic()
                if entry.process is not None:
                    entry.process.set_deadline(None)
                # Wake exactly one acquirer; mirroring asyncio.Semaphore's
                # FIFO release behaviour. notify_all would re-create the
                # thundering herd that the asyncio.Semaphore originally
                # fixed (see git history of process_pool.py).
                self._capacity_changed.notify()

    async def return_entry_async(self, entry: PoolEntry) -> None:
        """Client-driven return path. Measures PSS (bounded), drains REPL
        output, rolls back to checkpoint, and releases the entry to idle -
        OR recycles if PSS exceeds the threshold or the measurement hangs.

        This is the only path that ever ends in ``release(recycle=False)``.
        Every brutal-recycle source (deadline watchdog, idle reaper,
        liveness reconciler, poisoned process from ``LeanProcessException``)
        goes through ``release(recycle=True)`` directly and skips PSS - PSS
        is meaningless once the recycle decision is made and the read can
        wedge against a CPU-saturated REPL (commit f49398d).
        """
        if entry.process is None:
            self.release(entry, recycle=True, reason="return of un-spawned placeholder")
            return

        if self.pss_recycle_limit:
            try:
                pss = await asyncio.wait_for(
                    asyncio.to_thread(entry.process.memory_usage_pss),
                    timeout=PSS_MEASUREMENT_TIMEOUT,
                )
            except asyncio.TimeoutError:
                self.release(
                    entry,
                    recycle=True,
                    reason=f"PSS measurement timed out (>{PSS_MEASUREMENT_TIMEOUT}s)",
                )
                return
            except Exception as e:
                self.release(
                    entry,
                    recycle=True,
                    reason=f"PSS measurement raised {type(e).__name__}: {e}",
                )
                return
            if pss > self.pss_recycle_limit:
                mib = 1024 * 1024
                self.release(
                    entry,
                    recycle=True,
                    reason=(
                        f"PSS={pss / mib:.1f} MiB > "
                        f"recycle limit {self.pss_recycle_limit / mib:.1f} MiB"
                    ),
                )
                return

        try:
            await entry.process.drain_repl_output_async()
            if entry.checkpoint is not None:
                entry.process.rollback_to(entry.checkpoint)
        except Exception as e:
            self.release(
                entry,
                recycle=True,
                reason=f"drain/rollback raised {type(e).__name__}: {e}",
            )
            return
        self.release(entry, recycle=False)

    # ------------------------------------------------------------------
    # Per-entry attributes
    # ------------------------------------------------------------------

    def touch(self, entry: PoolEntry) -> None:
        """Refresh last_used so the idle reaper doesn't reclaim an entry
        that's actively serving requests on a long-lived branch."""
        with self._lock:
            entry.last_used = time.monotonic()

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get_entry(self, entry_id: int) -> PoolEntry | None:
        with self._lock:
            return self._live.get(entry_id)

    def stats(self) -> PoolStats:
        with self._lock:
            counts = {"idle": 0, "starting": 0, "checked_out": 0, "stopping": 0}
            for e in self._live.values():
                counts[e.state] += 1
            # Translate state-machine names -> operator-facing names at
            # the wire boundary.  ``used`` == checked_out (matches the
            # pre-redesign meaning of /status's ``used_processes``).
            return PoolStats(
                max_processes=self.max_processes,
                total_processes=len(self._live),
                available_processes=counts["idle"],
                starting_processes=counts["starting"],
                used_processes=counts["checked_out"],
                stopping_processes=counts["stopping"],
            )

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    async def warmup_async(self, batch_size: int | None = None) -> None:
        """Pre-spawn processes until ``len(_live) == max_processes``.
        Spawns in batches to limit resource contention - 128 concurrent
        Mathlib imports thrash I/O and blow past the per-import timeout.

        Allocates placeholders synchronously up-front so capacity is
        accurately reflected even mid-warmup; concurrent ``acquire``
        callers will either reuse a just-published idle entry or wait
        for capacity, never overcommit.
        """
        with self._lock:
            n = self.max_processes - len(self._live)
        if n <= 0:
            return
        effective_batch = batch_size if batch_size else n
        if effective_batch < n:
            self.logger.info(
                f"Starting {n} processes in batches of {effective_batch}"
            )
        else:
            self.logger.info(f"Starting {n} processes in parallel")

        done = 0
        while done < n:
            cur = min(effective_batch, n - done)
            if effective_batch < n:
                self.logger.info(
                    f"Starting batch of {cur} processes ({done}/{n} done)"
                )
            placeholders: list[PoolEntry] = []
            with self._capacity_changed:
                for _ in range(cur):
                    if len(self._live) >= self.max_processes:
                        break
                    placeholders.append(self._new_placeholder_locked())
            if not placeholders:
                break
            await asyncio.gather(
                *(self._fill_placeholder_async(p, into_idle=True) for p in placeholders)
            )
            done += len(placeholders)

        s = self.stats()
        self.logger.info(
            f"Started {done} processes. Live: {s.total_processes}, "
            f"available: {s.available_processes}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_placeholder_locked(self) -> PoolEntry:
        # Caller must hold self._lock (or self._capacity_changed, which
        # is the same RLock).
        self._id_counter += 1
        entry = PoolEntry(
            id=self._id_counter,
            pool=self,
            process=None,
            pid=None,
            state="starting",
            checkpoint=None,
            last_used=time.monotonic(),
        )
        self._live[entry.id] = entry
        return entry

    async def _spawn_async(self) -> LeanProcess:
        """Create + start + env-setup a Lean subprocess. Cleans up partial
        state on any failure between start and env-setup (otherwise the
        ~48 MiB-of-StreamReader-buffers leak documented in the old code's
        _create_process_async would resurface).
        """
        process = LeanProcess(
            self.repl_exe,
            self.project_path,
            self.logger,
            rss_hard_limit=self.rss_hard_limit,
        )
        started = False
        try:
            await process.start_async()
            started = True
            if self.env_setup_async:
                await self.env_setup_async(process)
            return process
        except BaseException:
            if started:
                try:
                    await process.stop_async_safe()
                except Exception as e:
                    self.logger.warning(
                        f"_spawn_async cleanup: stop_async_safe failed: {e}"
                    )
            raise

    async def _fill_placeholder_async(
        self, entry: PoolEntry, *, into_idle: bool
    ) -> None:
        """Spawn the subprocess for a placeholder. Used only by warmup
        right now (acquire_async inlines this for a simpler error path).
        """
        try:
            process = await self._spawn_async()
        except BaseException:
            self.release(entry, recycle=True, reason="warmup spawn failed")
            raise
        with self._capacity_changed:
            entry.process = process
            entry.pid = process._proc.pid
            entry.checkpoint = process.checkpoint()
            entry.state = "idle" if into_idle else "checked_out"
            entry.last_used = time.monotonic()
            if into_idle:
                self._capacity_changed.notify()

    # ------------------------------------------------------------------
    # Background workers
    # ------------------------------------------------------------------

    async def _janitor_loop(self, inbox: asyncio.Queue) -> None:
        """Drain entries marked 'stopping': stop the subprocess (5 s
        bounded), then remove from `_live` and notify capacity waiters.
        Lives on the server's event loop; never holds `_lock` across an
        `await`.

        Strict-capacity guarantee: capacity[i.e. `len(_live)`] decrements
        only here, after the subprocess is fully reaped. Combined with
        synchronous `release` adding entries to this queue, every
        `release(recycle=True)` eventually frees exactly one slot.
        """
        while True:
            entry = await inbox.get()
            if entry.process is not None:
                try:
                    await asyncio.wait_for(
                        entry.process.stop_async(), timeout=JANITOR_STOP_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    self.logger.warning(
                        f"Janitor: stop_async for entry {entry.id} "
                        f"(pid={entry.pid}) did not complete in "
                        f"{JANITOR_STOP_TIMEOUT} s; removing from pool anyway "
                        f"(subprocess leaked as zombie)"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Janitor: stop_async for entry {entry.id} "
                        f"(pid={entry.pid}) raised {type(e).__name__}: {e}"
                    )
            with self._capacity_changed:
                self._live.pop(entry.id, None)
                # notify (not notify_all): one freed slot wakes one waiter,
                # mirroring the FIFO semantics that asyncio.Semaphore gave us
                # before. notify_all here would re-create the thundering-herd
                # bug from the original pool design.
                self._capacity_changed.notify()

    def _idle_reaper_loop(self) -> None:
        """Reclaim entries whose client checked them out and went away.
        Skips entries with active HTTP requests (server installs
        ``has_active_request``). Replaces the old
        `LeanServer._reap_leaked_processes` thread.
        """
        while not self._workers_stop.wait(REAPER_INTERVAL):
            now = time.monotonic()
            stale: list[PoolEntry] = []
            with self._lock:
                for entry in self._live.values():
                    if entry.state != "checked_out":
                        continue
                    if now - entry.last_used <= self._lease_timeout:
                        continue
                    if self.has_active_request(entry.id):
                        continue
                    stale.append(entry)
            for entry in stale:
                self.logger.warning(
                    f"Idle reaper: reclaiming entry {entry.id} "
                    f"(pid={entry.pid}); unused for >{self._lease_timeout} s"
                )
                self.release(entry, recycle=True, reason="idle lease timeout")

    def _liveness_reconciler_loop(self) -> None:
        """Detect subprocesses whose kernel entry has vanished entirely
        (reaped without our knowledge) and recycle the pool entry. After
        commit e2a7390 made every kill use ``killpg``, the lake parent
        and repl grandchild always die together, so a missing PID is
        unambiguous: the whole REPL tree is gone.

        Note: ``os.kill(pid, 0)`` returns success for zombies, so this
        won't catch "subprocess died but kernel still has zombie entry";
        that case is already handled at the next REPL I/O attempt (EOF
        on stdout -> LeanProcessException -> recycle). This reconciler
        is the safety net for the rarer "fully reaped without us
        noticing" case.
        """
        while not self._workers_stop.wait(LIVENESS_INTERVAL):
            probes: list[tuple[PoolEntry, int]] = []
            with self._lock:
                for entry in self._live.values():
                    if entry.state not in ("idle", "checked_out"):
                        continue
                    if entry.pid is None:
                        continue
                    probes.append((entry, entry.pid))
            for entry, pid in probes:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    self.logger.warning(
                        f"Liveness reconciler: entry {entry.id} (pid={pid}) "
                        f"subprocess gone; recycling"
                    )
                    self.release(entry, recycle=True, reason="subprocess vanished")
                except PermissionError:
                    # PID recycled to a different process owned by someone
                    # else. Extremely unlikely (we're the parent, kernel
                    # reserves PIDs until wait()) but treat conservatively.
                    self.logger.warning(
                        f"Liveness reconciler: entry {entry.id} (pid={pid}) "
                        f"got EPERM; assuming PID was recycled, recycling entry"
                    )
                    self.release(
                        entry, recycle=True, reason="subprocess pid recycled"
                    )

    # ------------------------------------------------------------------
    # Sync wrappers (for non-async callers, mainly tests)
    # ------------------------------------------------------------------

    return_entry = to_sync(return_entry_async)
    warmup = to_sync(warmup_async)
