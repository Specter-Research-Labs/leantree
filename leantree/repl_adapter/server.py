import asyncio
import concurrent.futures
import contextlib
import json
import logging
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Self
import urllib.request
import urllib.error

import psutil

from leantree.repl_adapter.interaction import LeanProcess, LeanProcessException, LeanProofBranch, LeanInteractionException
from leantree.repl_adapter.process_pool import LeanProcessPool
from leantree.core.lean import LeanProofState, LeanTactic, LeanGoal
from leantree.utils import serialize_exception, deserialize_exception, ValueOrError


# Hard deadlines for every cross-thread await-on-the-event-loop call.
# Without these, one stuck coroutine can block an HTTP handler thread forever,
# queueing every subsequent request behind it until the server appears dead
# (observed in production, see plan `the-fix-2-is-streamed-valiant.md`).
# These wrap the *inner* async operation's own timeout with a small headroom,
# so the server-side `_run_async` never waits longer than the coroutine itself
# promised to take.
RUN_ASYNC_HEADROOM = 30.0  # seconds — wraps caller-supplied operation timeouts
STOP_ASYNC_TIMEOUT = 10.0  # bounded wait in stop_async is already 5s (D1); +5s headroom
RELEASE_SLOT_TIMEOUT = 5.0  # trivial bookkeeping coro
RETURN_PROCESS_TIMEOUT = 30.0  # drain_repl_output is near-instant; 30s is generous
POOL_SHUTDOWN_TIMEOUT = 30.0

# Default per-operation deadlines used when the caller didn't specify one.
# Each of these is an inner (coro-level) deadline; add RUN_ASYNC_HEADROOM
# when passing to _run_async.
DEFAULT_COMMAND_TIMEOUT = 300.0  # matches _send_to_repl_async default
DEFAULT_PROOF_FROM_SORRY_TIMEOUT = 300.0
DEFAULT_IS_VALID_SOURCE_TIMEOUT = 60.0
DEFAULT_TRY_APPLY_TACTIC_MS = 1000  # matches existing handler default


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTP server that handles each request in a separate thread."""
    daemon_threads = True
    # Increase listen backlog for high-load scenarios (default is usually 5)
    request_queue_size = 128


class _ProcessRegistry:
    """Thread-safe bidirectional map: process_id <-> LeanProcess, plus last-used timestamps.

    Replaces three dicts + one lock + one id counter that were previously
    sprinkled around ``LeanServer`` and hand-juggled in at least seven call
    sites (``_destroy_process``, ``_handle_return_process``, ``stop``,
    ``_reap_leaked_processes``, ...).  Every atomic update has to touch all
    three dicts consistently — forgetting to pop one was how we leaked
    ``LeanProcess`` objects into ``pool.checkpoints`` (fixed in
    commit 013014b).  Encapsulating the invariant here means each call site
    reads as one operation instead of four, and new call sites can't
    accidentally leave the dicts out of sync.
    """

    def __init__(self):
        self._id_to_process: dict[int, LeanProcess] = {}
        self._process_to_id: dict[LeanProcess, int] = {}
        self._last_used: dict[int, float] = {}
        self._id_counter: int = 0
        self._lock = threading.Lock()

    def register(self, process: LeanProcess) -> int:
        """Get-or-create a stable integer id for ``process``. Touches last-used."""
        with self._lock:
            now = time.time()
            if process in self._process_to_id:
                pid = self._process_to_id[process]
                self._last_used[pid] = now
                return pid
            self._id_counter += 1
            pid = self._id_counter
            self._process_to_id[process] = pid
            self._id_to_process[pid] = process
            self._last_used[pid] = now
            return pid

    def get(self, process_id: int) -> LeanProcess:
        """Look up a process by id. Raises ``ValueError`` if not registered. Touches last-used."""
        with self._lock:
            if process_id not in self._id_to_process:
                raise ValueError(f"Process {process_id} not found")
            self._last_used[process_id] = time.time()
            return self._id_to_process[process_id]

    def remove(self, process_id: int) -> LeanProcess | None:
        """Atomically unregister ``process_id`` and return its LeanProcess (or None if unknown)."""
        with self._lock:
            process = self._id_to_process.pop(process_id, None)
            if process is not None:
                self._process_to_id.pop(process, None)
            self._last_used.pop(process_id, None)
            return process

    def touch(self, process_id: int) -> None:
        """Refresh last-used for ``process_id`` if tracked. No-op otherwise.

        Needed because branch-only handlers (``try_apply_tactic``,
        ``branch_state``) access the process indirectly via ``_get_branch``
        and would otherwise never update the process's ``_last_used``,
        letting the reaper reclaim a process that is being actively
        hammered with tactics (e.g. under ``--tactic-delay``).
        """
        with self._lock:
            if process_id in self._last_used:
                self._last_used[process_id] = time.time()

    def __contains__(self, process_id: int) -> bool:
        with self._lock:
            return process_id in self._id_to_process

    def __len__(self) -> int:
        with self._lock:
            return len(self._id_to_process)

    def inactive_count(self, inactive_threshold_s: float) -> tuple[int, int]:
        """Return (inactive_count, total_count) where inactive = last_used older than threshold."""
        now = time.time()
        with self._lock:
            inactive = sum(
                1 for last in self._last_used.values() if now - last > inactive_threshold_s
            )
            return inactive, len(self._last_used)

    def find_stale(self, stale_threshold_s: float) -> list[int]:
        """Return process_ids whose last_used is older than ``stale_threshold_s`` (used by the reaper)."""
        now = time.time()
        with self._lock:
            return [
                pid for pid, last in list(self._last_used.items())
                if now - last > stale_threshold_s
            ]

    def drain(self) -> list[LeanProcess]:
        """Atomically remove and return every registered process. Used during shutdown."""
        with self._lock:
            processes = list(self._id_to_process.values())
            self._id_to_process.clear()
            self._process_to_id.clear()
            self._last_used.clear()
            return processes


class LeanServer:
    """Manages a LeanProcessPool and exposes it over a HTTP port."""

    def __init__(self, pool: LeanProcessPool, address: str = "localhost", port: int = 8000, log_level: str = "INFO"):
        self.pool = pool
        self.address = address
        self.port = port
        self.log_level = log_level
        self.server = None
        self.server_thread = None

        # Root handler + formatter come from leantree.utils.setup_default_logging;
        # just lower this named logger's threshold if requested.
        self.logger = logging.getLogger("LeanServer")
        self.logger.setLevel(log_level)

        # Process tracking: encapsulates id<->process bidirectional map and
        # last-used timestamps in one lock; see _ProcessRegistry.
        self._processes = _ProcessRegistry()
        self._event_loop = None
        self._loop_thread = None

        # Warmup gating. When a warmup_coro is passed to start(), this flips to
        # True only after it completes; /process/get rejects with 503 until then.
        # Default True so servers started without warmup immediately accept requests.
        self._warmup_complete: bool = True

        # Branch tracking: maps (process_id, branch_id) to LeanProofBranch
        self._branch_id_counter = 0
        self._branches: dict[tuple[int, int], LeanProofBranch] = {}
        self._branches_lock = threading.Lock()

        # Last used tracking for inactivity detection (branches only; processes
        # live in self._processes)
        self._branch_last_used: dict[tuple[int, int], float] = {}  # (process_id, branch_id) -> timestamp
        
        # Request tracking for monitoring
        self._active_requests: dict[int, dict] = {}  # request_id -> {path, start_time, thread_name}
        self._request_id_counter = 0
        self._requests_lock = threading.Lock()

        # Lease timeout: processes not used for this long are assumed leaked
        # and returned to the pool automatically.
        self._process_lease_timeout = 600.0  # 10 minutes
        self._reaper_thread = None

    def _get_process_id(self, process: LeanProcess) -> int:
        """Get or create a process ID for a LeanProcess."""
        return self._processes.register(process)

    def _get_process(self, process_id: int) -> LeanProcess:
        """Get a LeanProcess by its ID."""
        return self._processes.get(process_id)

    def _remove_process(self, process_id: int):
        """Remove a process from tracking and its branches."""
        self._processes.remove(process_id)
        self._remove_branches_for_process(process_id)

    def _destroy_process(self, process_id: int, reason: str | None = None):
        """Kill a process and remove it from tracking.

        Used when a REPL timeout or crash leaves the process in an
        undefined state and it cannot be safely returned to the pool.

        Args:
            process_id: the ID of the poisoned process.
            reason: human-readable explanation (usually ``str(exception)``).
                Surfaced in the warning log so we can tell at a glance
                whether processes are dying from RLIMIT_AS, timeouts,
                Lean segfaults, etc. — without this, every destruction
                looked identical in the log.

        Note (A2): the slow `_run_async` calls below MUST happen *outside*
        the process-registry lock.  Holding a server-wide lock while waiting
        on the asyncio loop would freeze every other handler that needs it
        (status, get_process_id, return_process, ...) if the loop ever
        slowed down — exactly the cascading-deadlock failure mode we're
        fixing.  ``_ProcessRegistry.remove`` scopes its lock to the atomic
        dict pop and releases before we return.
        """
        # Step 1: atomic unregister; releases the registry lock immediately.
        process = self._processes.remove(process_id)
        self._remove_branches_for_process(process_id)
        if process is None:
            return

        # Step 2: cross the loop boundary with the lock released.
        try:
            self._run_async(process.stop_async_safe(), timeout=STOP_ASYNC_TIMEOUT)
        except concurrent.futures.TimeoutError:
            self.logger.error(
                f"stop_async_safe for poisoned process {process_id} did not complete in "
                f"{STOP_ASYNC_TIMEOUT}s — leaking the subprocess to avoid blocking the loop"
            )
        except Exception as e:
            self.logger.warning(f"Error stopping poisoned process {process_id}: {e}")
        # Decrement pool's used count so a replacement can be created.
        # Critically, also evict the LeanProcess from self.pool.checkpoints —
        # otherwise every poisoned process leaks the full LeanProcess object
        # (including its ~16 MiB-per-stream asyncio buffers for stdin /
        # stdout / stderr) forever.  Observed in production: ~30 poisoned
        # processes/min × hours → 200+ GiB leanserver RSS growth with
        # ~350 live branches (i.e. the leak is in retained *dead* processes,
        # not live branches).  return_process_async pops checkpoints on the
        # terminate path, but _destroy_process bypasses return_process_async.
        async def _release_slot():
            async with self.pool.lock:
                self.pool.checkpoints.pop(process, None)
                if self.pool._num_used_processes > 0:
                    self.pool._num_used_processes -= 1
                self.pool.process_available_event.set()
        try:
            self._run_async(_release_slot(), timeout=RELEASE_SLOT_TIMEOUT)
        except concurrent.futures.TimeoutError:
            self.logger.error(
                f"_release_slot for poisoned process {process_id} did not complete in "
                f"{RELEASE_SLOT_TIMEOUT}s — pool slot accounting may drift"
            )
        if reason:
            self.logger.warning(f"Destroyed poisoned process {process_id}: {reason}")
        else:
            self.logger.warning(f"Destroyed poisoned process {process_id}")

    def _destroy_untracked_process(self, process: LeanProcess, reason: str | None = None):
        """Clean up a LeanProcess that has already been removed from server tracking.

        Three code paths do "remove-from-tracking first, then return-to-pool":
        ``_handle_return_process``, ``_handle_get_process``'s BrokenPipeError
        branch, and ``_reap_leaked_processes``.  If the subsequent
        ``return_process_async`` times out or raises, the process is orphaned:
          - ``pool.checkpoints`` still holds the LeanProcess Python object
            (retaining its ~16 MiB-per-stream asyncio StreamReader buffers),
          - ``pool._num_used_processes`` is never decremented (slot leak),
          - the lake/repl subprocess keeps running.
        The reaper can't help because the registry entry was already
        removed.  This helper is the destroy-by-object counterpart to
        ``_destroy_process``, usable when the id is no longer valid.

        Observed in production: under event-loop saturation, many
        ``return_process_async`` calls exceeded RETURN_PROCESS_TIMEOUT, each
        one leaked a ~3-6 GiB lake/repl child + its Python-side buffers.
        Over hours the leanserver accumulated 20+ extra subprocesses and
        grew to hundreds of GiB of RAM.
        """
        try:
            self._run_async(process.stop_async_safe(), timeout=STOP_ASYNC_TIMEOUT)
        except concurrent.futures.TimeoutError:
            self.logger.error(
                f"stop_async_safe on orphaned process did not complete in "
                f"{STOP_ASYNC_TIMEOUT}s — leaking the subprocess to avoid blocking the loop"
            )
        except Exception as e:
            self.logger.warning(f"Error stopping orphaned process: {e}")

        async def _release_slot():
            async with self.pool.lock:
                self.pool.checkpoints.pop(process, None)
                if self.pool._num_used_processes > 0:
                    self.pool._num_used_processes -= 1
                self.pool.process_available_event.set()

        try:
            self._run_async(_release_slot(), timeout=RELEASE_SLOT_TIMEOUT)
        except concurrent.futures.TimeoutError:
            self.logger.error(
                f"_release_slot for orphaned process did not complete in "
                f"{RELEASE_SLOT_TIMEOUT}s — pool slot accounting may drift"
            )
        except Exception as e:
            self.logger.warning(f"Error releasing slot for orphaned process: {e}")

        if reason:
            self.logger.warning(f"Destroyed orphaned process: {reason}")

    def _register_branch(self, process_id: int, branch: LeanProofBranch) -> int:
        """Register a branch and return its ID."""
        with self._branches_lock:
            self._branch_id_counter += 1
            branch_id = self._branch_id_counter
            key = (process_id, branch_id)
            self._branches[key] = branch
            self._branch_last_used[key] = time.time()
            return branch_id

    def _get_branch(self, process_id: int, branch_id: int) -> LeanProofBranch:
        """Get a branch by its ID."""
        with self._branches_lock:
            key = (process_id, branch_id)
            if key not in self._branches:
                raise ValueError(f"Branch {branch_id} not found for process {process_id}")
            self._branch_last_used[key] = time.time()
            branch = self._branches[key]
        # Refresh the process's _last_used too — otherwise a worker that only
        # hits try_apply_tactic / branch/state (never _get_process) can have
        # its process reaped out from under it.
        self._processes.touch(process_id)
        return branch

    def _remove_branches_for_process(self, process_id: int):
        """Remove all branches associated with a process."""
        with self._branches_lock:
            keys_to_remove = [k for k in self._branches if k[0] == process_id]
            for key in keys_to_remove:
                del self._branches[key]
                self._branch_last_used.pop(key, None)

    def _run_async(self, coro, timeout: float | None = None):
        """Run an async coroutine in the event loop.

        Args:
            coro: The coroutine to run.
            timeout: Raw wait-for-future timeout in seconds. Use this when the
                coroutine does NOT have its own inner deadline (bookkeeping,
                ``stop_async_safe``, etc.). For coroutines that DO have their
                own deadline, prefer ``_run_async_op`` which adds the
                ``RUN_ASYNC_HEADROOM`` for you.
        """
        if self._event_loop is None:
            raise RuntimeError("Server not started")
        future = asyncio.run_coroutine_threadsafe(coro, self._event_loop)
        return future.result(timeout=timeout)

    def _run_async_op(self, coro, op_timeout: float):
        """Run a coroutine that has its own inner deadline (``op_timeout``).

        Adds ``RUN_ASYNC_HEADROOM`` so the coroutine has a chance to enforce
        its own deadline (and emit a meaningful ``LeanProcessException``)
        before the outer threadsafe wait gives up with a generic TimeoutError.
        Forgetting this headroom used to be a latent-deadlock footgun — the
        outer wait would fire first, cancel observation of the coroutine, and
        the inner operation would never surface its real error.
        """
        return self._run_async(coro, timeout=op_timeout + RUN_ASYNC_HEADROOM)
    
    def _start_request(self, path: str) -> int:
        """Register start of a request. Returns request_id."""
        with self._requests_lock:
            self._request_id_counter += 1
            request_id = self._request_id_counter
            self._active_requests[request_id] = {
                "path": path,
                "start_time": time.time(),
                "thread_name": threading.current_thread().name,
            }
            return request_id
    
    def _end_request(self, request_id: int):
        """Register end of a request."""
        with self._requests_lock:
            self._active_requests.pop(request_id, None)
    
    def get_active_requests(self) -> list[dict]:
        """Get list of currently active requests with their duration."""
        now = time.time()
        with self._requests_lock:
            return [
                {
                    "request_id": rid,
                    "path": info["path"],
                    "duration_seconds": round(now - info["start_time"], 2),
                    "thread": info["thread_name"],
                }
                for rid, info in self._active_requests.items()
            ]

    def _reap_leaked_processes(self):
        """Periodically reclaim processes whose clients have disappeared.

        A process is considered leaked when it has been checked out (tracked in
        ``self._processes``) but not touched for longer than
        ``_process_lease_timeout`` seconds AND there is no active HTTP request
        referencing it.
        """
        while True:
            time.sleep(60)
            for pid in self._processes.find_stale(self._process_lease_timeout):
                # Check there is no active request for this process (e.g.
                # a long-running tactic application).
                with self._requests_lock:
                    active_for_pid = any(
                        f"/process/{pid}/" in req["path"]
                        for req in self._active_requests.values()
                    )
                if active_for_pid:
                    continue
                self.logger.warning(
                    f"Reclaiming leaked process {pid} (unused for "
                    f">{self._process_lease_timeout}s)"
                )
                process = self._processes.remove(pid)
                self._remove_branches_for_process(pid)
                if process is not None:
                    try:
                        self._run_async(
                            self.pool.return_process_async(process),
                            timeout=RETURN_PROCESS_TIMEOUT,
                        )
                    except concurrent.futures.TimeoutError:
                        self.logger.error(
                            f"return_process_async for leaked process {pid} exceeded "
                            f"{RETURN_PROCESS_TIMEOUT}s — destroying orphaned process"
                        )
                        self._destroy_untracked_process(
                            process, reason=f"reaper return timed out for process_id={pid}"
                        )
                    except Exception as e:
                        self.logger.error(f"Error returning leaked process {pid}: {e} — destroying orphaned process")
                        self._destroy_untracked_process(
                            process, reason=f"reaper return raised: {type(e).__name__}: {e}"
                        )

    def start(self, warmup_coro=None):
        """Start the HTTP server.

        If ``warmup_coro`` is given, the HTTP listener comes up immediately
        (so ``/status`` is queryable during warmup) but ``/process/get``
        returns 503 until the coroutine completes. This prevents clients from
        racing warmup and triggering extra ``_create_process_async`` calls on
        top of the warmup spawns, while keeping the server introspectable.
        """
        if self.server is not None:
            raise RuntimeError("Server already started")

        # Start event loop in a separate thread
        def run_event_loop():
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
            self._event_loop.run_forever()

        self._loop_thread = threading.Thread(target=run_event_loop, daemon=True)
        self._loop_thread.start()

        # Wait for event loop to be ready
        while self._event_loop is None:
            time.sleep(0.01)

        if warmup_coro is not None:
            self._warmup_complete = False

        handler = self._create_handler()
        self.server = ThreadingHTTPServer((self.address, self.port), handler)

        def run_server():
            self.server.serve_forever()

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

        # Run warmup after the HTTP listener is up so /status stays reachable
        # while it runs. Block start() until it completes so callers can log
        # "ready" in the obvious order.
        if warmup_coro is not None:
            try:
                self._run_async(warmup_coro)
            finally:
                self._warmup_complete = True

        # Start the lease-timeout reaper to reclaim leaked processes
        self._reaper_thread = threading.Thread(
            target=self._reap_leaked_processes, daemon=True
        )
        self._reaper_thread.start()

    def stop(self):
        """Stop the HTTP server and all Lean processes (both idle and checked-out)."""
        # First, stop checked-out processes that the pool doesn't track.
        # Must happen while the event loop is still running.
        if self._event_loop is not None:
            checked_out = self._processes.drain()
            for process in checked_out:
                try:
                    self._run_async(process.stop_async_safe(), timeout=STOP_ASYNC_TIMEOUT)
                except concurrent.futures.TimeoutError:
                    self.logger.warning(
                        f"stop_async_safe did not complete in {STOP_ASYNC_TIMEOUT}s during shutdown"
                    )
                except Exception as e:
                    self.logger.warning(f"Error stopping checked-out process: {e}")
            # Shut down the pool (stops idle/available processes)
            try:
                self._run_async(self.pool.shutdown_async(), timeout=POOL_SHUTDOWN_TIMEOUT)
            except concurrent.futures.TimeoutError:
                self.logger.warning(f"pool.shutdown_async did not complete in {POOL_SHUTDOWN_TIMEOUT}s")
            except Exception as e:
                self.logger.warning(f"Error shutting down pool: {e}")
        if self.server is not None:
            self.server.shutdown()
            self.server = None
        if self._event_loop is not None:
            self._event_loop.call_soon_threadsafe(self._event_loop.stop)
            self._event_loop = None

    def _create_handler(self):
        server = self

        class LeanServerHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                server.logger.debug(f"GET request to {self.path} from {self.client_address}")
                request_id = server._start_request(self.path)
                try:
                    if self.path == "/status":
                        self._handle_status()
                    elif self.path == "/heap":
                        self._handle_heap()
                    else:
                        self._send_error(404, "Not Found")
                except BrokenPipeError:
                    # Client disconnected before we could send response - this is normal
                    server.logger.debug(f"Client {self.client_address} disconnected during GET {self.path}")
                finally:
                    server._end_request(request_id)

            def do_POST(self):
                server.logger.debug(f"POST request to {self.path} from {self.client_address}")
                request_id = server._start_request(self.path)
                try:
                    self._do_POST_inner()
                except BrokenPipeError:
                    # Client disconnected before we could send response - this is normal
                    server.logger.debug(f"Client {self.client_address} disconnected during POST {self.path}")
                finally:
                    server._end_request(request_id)

            def _do_POST_inner(self):
                if self.path == "/process/get":
                    self._handle_get_process()
                elif self.path.startswith("/process/"):
                    parts = self.path.split("/")
                    if len(parts) >= 4:
                        process_id = int(parts[2])
                        action = parts[3]
                        if action == "command":
                            self._handle_command(process_id)
                        elif action == "return":
                            self._handle_return_process(process_id)
                        elif action == "is_valid_source":
                            self._handle_is_valid_source(process_id)
                        elif action == "proof_from_sorry":
                            self._handle_proof_from_sorry(process_id)
                        elif action == "branch" and len(parts) >= 6:
                            branch_id = int(parts[4])
                            if parts[5] == "try_apply_tactic":
                                self._handle_try_apply_tactic(process_id, branch_id)
                            elif parts[5] == "state":
                                self._handle_branch_state(process_id, branch_id)
                            else:
                                self._send_error(404, "Not Found")
                        else:
                            self._send_error(404, "Not Found")
                    else:
                        self._send_error(404, "Not Found")
                else:
                    self._send_error(404, "Not Found")

            @contextlib.contextmanager
            def _process_op_errors(self, process_id: int, op_name: str, op_timeout: float):
                """Uniform error handling for handlers that run a per-process coroutine.

                The body runs inside the context and is expected to call
                ``server._run_async_op(...)`` with ``op_timeout`` as its
                inner deadline (the context manager only dispatches on the
                outer exception type — it doesn't run the coroutine itself).

                On ``concurrent.futures.TimeoutError`` or ``LeanProcessException``
                the process is destroyed (slot released, subprocess stopped,
                ``pool.checkpoints`` evicted) and the appropriate HTTP error is
                sent.  On any other ``Exception`` the process is left alone
                (the failure may be unrelated to the Lean subprocess — e.g.
                malformed request JSON) and a generic 500 is sent.

                Collapses ~160 lines of near-identical try/except blocks
                across ``_handle_command``, ``_handle_is_valid_source``,
                ``_handle_proof_from_sorry``, and ``_handle_try_apply_tactic``.
                Centralising the policy means future handlers inherit the
                "destroy on timeout" reflex for free, closing a category of
                leak-path bugs.
                """
                total_timeout = op_timeout + RUN_ASYNC_HEADROOM
                try:
                    yield
                except concurrent.futures.TimeoutError:
                    server.logger.error(
                        f"{op_name} on process {process_id} did not complete in "
                        f"{total_timeout}s — destroying process"
                    )
                    server._destroy_process(
                        process_id,
                        reason=f"{op_name} exceeded {total_timeout}s deadline",
                    )
                    self._send_error(503, "leanserver event loop did not respond in time")
                except LeanProcessException as e:
                    server._destroy_process(process_id, reason=f"{op_name}: {e}")
                    self._send_error(500, str(e), exception=e)
                except Exception as e:
                    self._send_error(500, str(e), exception=e)

            def _handle_heap(self):
                """Live Python-object census for diagnosing memory leaks.

                Returns the top 30 object types by (a) count and (b) total
                size (sys.getsizeof), plus a few named leanserver-specific
                counts (LeanProofBranch, LeanGoal, LeanProcess) and the
                process's current RSS.

                This uses `gc.get_objects()` — no external profiling tools
                required, no tracemalloc overhead.  sys.getsizeof only
                accounts for the object's own header + immediate payload
                (not referenced objects), so a 10 MB string shows as 10 MB
                but a dict of 100 big strings only shows as the dict
                overhead.  Good enough to identify the leak fingerprint
                when RSS is climbing.

                Walking the object graph briefly pauses Python execution
                (GIL held) — typically <1 s for millions of objects.
                """
                import gc
                import sys
                from collections import Counter

                # gc.collect() first so we don't count soon-to-be-dead objects.
                gc.collect()
                objs = gc.get_objects()
                type_counts: Counter[str] = Counter()
                type_sizes: Counter[str] = Counter()
                for o in objs:
                    tn = type(o).__name__
                    type_counts[tn] += 1
                    try:
                        type_sizes[tn] += sys.getsizeof(o)
                    except (TypeError, AttributeError):
                        pass
                try:
                    rss_bytes = psutil.Process().memory_info().rss
                except Exception:
                    rss_bytes = None
                self._send_json(200, {
                    "total_objects": len(objs),
                    "rss_bytes": rss_bytes,
                    "top_by_count": type_counts.most_common(30),
                    "top_by_size_bytes": type_sizes.most_common(30),
                    "server_branches": len(server._branches),
                    "server_tracked_processes": len(server._processes),
                    "pool_available_processes": len(server.pool.available_processes),
                    "pool_used_processes": server.pool._num_used_processes,
                })

            def _handle_status(self):
                pool = server.pool

                # Get per-CPU core utilization
                cpu_percent_per_core = psutil.cpu_percent(percpu=True)

                # Get RAM utilization
                memory = psutil.virtual_memory()

                # Leanserver Python-process RSS. A sudden growth here is a memory leak in our own code.
                try:
                    self_rss_bytes = psutil.Process().memory_info().rss
                except Exception:
                    self_rss_bytes = None

                available = len(pool.available_processes)
                used = pool._num_used_processes
                starting = pool._num_starting_processes
                
                # Get active requests (excluding this status request itself)
                active_requests = [
                    req for req in server.get_active_requests()
                    if req["path"] != "/status"
                ]
                
                # Count inactive processes and branches (not used in last 60 seconds)
                inactive_threshold = 60.0
                inactive_processes, total_tracked_processes = server._processes.inactive_count(
                    inactive_threshold
                )

                now = time.time()
                with server._branches_lock:
                    inactive_branches = sum(
                        1 for key, last_used in server._branch_last_used.items()
                        if now - last_used > inactive_threshold
                    )
                    total_branches = len(server._branch_last_used)
                
                status = {
                    "status": "ready" if server._warmup_complete else "warming_up",
                    "available_processes": available,
                    "used_processes": used,
                    "starting_processes": starting,
                    "max_processes": pool.max_processes,
                    "active_requests": active_requests,
                    "cpu_percent_per_core": cpu_percent_per_core,
                    "ram": {
                        "total_bytes": memory.total,
                        "available_bytes": memory.available,
                        "used_bytes": memory.used,
                        "percent": memory.percent,
                    },
                    # Python-process-level RSS for the leanserver itself.
                    # Grow-over-time here signals a leak in our own code;
                    # see the note in _handle_status. None if unavailable.
                    "leanserver_rss_bytes": self_rss_bytes,
                    "inactive_processes": inactive_processes,
                    "total_tracked_processes": total_tracked_processes,
                    "inactive_branches": inactive_branches,
                    "total_branches": total_branches,
                }
                self._send_json(200, status)

            def _handle_get_process(self):
                # Reject while warmup is running. Otherwise these requests race
                # warmup: available_processes is empty (gather only publishes at
                # the end) and _num_used_processes is 0, so _get_or_create_process_async
                # spawns an extra REPL on top of the warmup N, piling onto an
                # already saturated system.
                if not server._warmup_complete:
                    self._send_error(503, "server is warming up")
                    return

                data = self._read_json()
                blocking = data.get("blocking", True)
                # Default timeout of 1 minute to prevent indefinite blocking
                timeout = data.get("timeout", 60.0)

                process = None
                process_id = None
                try:
                    process = server._run_async_op(
                        server.pool.get_process_async(blocking=blocking, timeout=timeout),
                        op_timeout=timeout,
                    )
                    if process is None:
                        self._send_json(200, {"process_id": None})
                    else:
                        process_id = server._get_process_id(process)
                        self._send_json(200, {"process_id": process_id})
                except concurrent.futures.TimeoutError:
                    server.logger.error(
                        f"get_process_async exceeded {timeout + RUN_ASYNC_HEADROOM}s on the event loop — "
                        f"loop is unresponsive (send SIGUSR1 for a thread dump)"
                    )
                    self._send_error(503, "leanserver event loop is unresponsive")
                except BrokenPipeError:
                    # Client disconnected - return the process to the pool if we got one
                    if process is not None:
                        server.logger.warning(
                            f"Client disconnected during /process/get, returning process {process_id} to pool"
                        )
                        server._remove_process(process_id)
                        try:
                            server._run_async(
                                server.pool.return_process_async(process),
                                timeout=RETURN_PROCESS_TIMEOUT,
                            )
                        except concurrent.futures.TimeoutError:
                            server.logger.error(
                                f"return_process_async after client disconnect did not complete in "
                                f"{RETURN_PROCESS_TIMEOUT}s — destroying orphaned process"
                            )
                            server._destroy_untracked_process(
                                process, reason="return timed out after client disconnect"
                            )
                        except Exception as e:
                            server.logger.error(f"Error returning process after client disconnect: {e} — destroying orphaned process")
                            server._destroy_untracked_process(
                                process, reason=f"return raised after client disconnect: {type(e).__name__}: {e}"
                            )
                    raise  # Re-raise to let the server handle the broken connection
                except Exception as e:
                    self._send_error(500, str(e), exception=e)

            def _handle_command(self, process_id: int):
                with self._process_op_errors(process_id, "send_command_async", DEFAULT_COMMAND_TIMEOUT):
                    process = server._get_process(process_id)
                    command = self._read_json()["command"]
                    response = server._run_async_op(
                        process.send_command_async(command),
                        op_timeout=DEFAULT_COMMAND_TIMEOUT,
                    )
                    self._send_json(200, response)

            def _handle_is_valid_source(self, process_id: int):
                with self._process_op_errors(process_id, "is_valid_source_async", DEFAULT_IS_VALID_SOURCE_TIMEOUT):
                    process = server._get_process(process_id)
                    source = self._read_json()["source"]
                    is_valid = server._run_async_op(
                        process.is_valid_source_async(source),
                        op_timeout=DEFAULT_IS_VALID_SOURCE_TIMEOUT,
                    )
                    self._send_json(200, {"is_valid": is_valid})

            def _handle_return_process(self, process_id: int):
                # A2: snapshot + drop tracking BEFORE crossing into the event loop.
                # Holding server-wide state through `_run_async` froze every other
                # handler whenever the loop slowed down.
                # Removing from tracking before returning to pool also closes a race
                # where another client could get the same process back under its old id.
                process = server._processes.remove(process_id)
                if process is None:
                    # Already returned - this is idempotent
                    self._send_json(200, {"status": "ok", "already_returned": True})
                    return
                server._remove_branches_for_process(process_id)
                # Keep the try narrow around return_process_async only. If we
                # also wrapped _send_json, a BrokenPipeError from writing the
                # success response to a disconnected client (common when a
                # client hard-exits, e.g. stress_test Ctrl+C with daemon
                # threads) would be mis-attributed to return_process_async
                # and destroy a process that's already healthy in the pool —
                # leaving a dead LeanProcess in pool.available_processes for
                # the next get_process to hand out.
                try:
                    server._run_async(
                        server.pool.return_process_async(process),
                        timeout=RETURN_PROCESS_TIMEOUT,
                    )
                except concurrent.futures.TimeoutError:
                    server.logger.error(
                        f"return_process_async for process {process_id} exceeded "
                        f"{RETURN_PROCESS_TIMEOUT}s — destroying orphaned process to avoid leak"
                    )
                    server._destroy_untracked_process(
                        process, reason=f"return_process_async timed out for process_id={process_id}"
                    )
                    self._send_error(503, "leanserver event loop did not respond in time")
                    return
                except Exception as e:
                    server._destroy_untracked_process(
                        process, reason=f"return_process_async raised {type(e).__name__}: {e}"
                    )
                    self._send_error(500, str(e), exception=e)
                    return
                # Process is safely back in the pool. A BrokenPipeError here
                # (client gone) is handled by the outer do_POST wrapper.
                self._send_json(200, {"status": "ok"})

            def _handle_proof_from_sorry(self, process_id: int):
                with self._process_op_errors(process_id, "proof_from_sorry", DEFAULT_PROOF_FROM_SORRY_TIMEOUT):
                    process = server._get_process(process_id)
                    theorem_with_sorry = self._read_json()["theorem_with_sorry"]

                    # proofs_from_sorries_async is an AsyncIterator; collect into a list
                    async def collect_proof_branches():
                        return [branch async for branch in process.proofs_from_sorries_async(theorem_with_sorry)]

                    # LeanInteractionException is a business-level error (bad theorem
                    # input, not a process failure) — return 200 with an error payload
                    # rather than destroying the process.  Intercept before it escapes
                    # to the context manager.
                    try:
                        proof_branches = server._run_async_op(
                            collect_proof_branches(),
                            op_timeout=DEFAULT_PROOF_FROM_SORRY_TIMEOUT,
                        )
                    except LeanInteractionException as e:
                        self._send_json(200, {"error": str(e)})
                        return

                    if len(proof_branches) == 0:
                        self._send_json(200, {"error": "No sorries found in theorem"})
                        return
                    if len(proof_branches) > 1:
                        self._send_json(200, {"error": f"Expected 1 sorry, found {len(proof_branches)}"})
                        return

                    proof_branch = proof_branches[0]
                    branch_id = server._register_branch(process_id, proof_branch)
                    goals = [goal.serialize() for goal in proof_branch.state.goals]
                    self._send_json(200, {"value": {"branch_id": branch_id, "goals": goals}})

            def _handle_try_apply_tactic(self, process_id: int, branch_id: int):
                data = self._read_json()
                tactic = data["tactic"]
                tactic_timeout_ms = data.get("timeout", DEFAULT_TRY_APPLY_TACTIC_MS)
                # tactic timeout is ms in the payload; _run_async_op takes seconds.
                op_timeout_s = tactic_timeout_ms / 1000.0
                op_name = f"try_apply_tactic (branch {branch_id})"
                with self._process_op_errors(process_id, op_name, op_timeout_s):
                    branch = server._get_branch(process_id, branch_id)
                    result = server._run_async_op(
                        branch.try_apply_tactic_async(tactic, timeout=tactic_timeout_ms),
                        op_timeout=op_timeout_s,
                    )
                    if not result.is_success():
                        self._send_json(200, {"error": str(result.error)})
                        return
                    new_branches = result.value
                    branches_data = []
                    for new_branch in new_branches:
                        new_branch_id = server._register_branch(process_id, new_branch)
                        branches_data.append({
                            "branch_id": new_branch_id,
                            "goals": [goal.serialize() for goal in new_branch.state.goals],
                        })
                    self._send_json(200, {"value": branches_data})

            def _handle_branch_state(self, process_id: int, branch_id: int):
                try:
                    branch = server._get_branch(process_id, branch_id)
                    self._send_json(200, {
                        "branch_id": branch_id,
                        "goals": [goal.serialize() for goal in branch.state.goals],
                        "is_solved": branch.is_solved,
                    })
                except Exception as e:
                    self._send_error(500, str(e), exception=e)

            def _read_json(self) -> dict:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                return json.loads(body.decode("utf-8"))

            def _send_json(self, status_code: int, data: dict):
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode("utf-8"))

            def _send_error(self, status_code: int, message: str, exception: Exception = None):
                """Send an error response, optionally including a pickled exception."""
                error_data = {"error": message}

                if exception is not None:
                    # Serialize the exception using the utility function
                    exception_data = serialize_exception(exception)
                    error_data.update(exception_data)

                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(error_data).encode("utf-8"))

            def log_message(self, format, *args):
                # Suppress default logging
                pass

        return LeanServerHandler


class LeanClient:
    """Connects to a LeanServer."""

    def __init__(self, address: str, port: int):
        self.address = address
        self.port = port
        self.base_url = f"http://{address}:{port}"

    def _request(self, method: str, path: str, data: dict = None, timeout: float | None = 360) -> dict:
        """Make an HTTP request to the server.

        Args:
            timeout: Socket timeout in seconds.  Defaults to 360s (slightly
                     above the server-side 300s REPL timeout so the server has
                     a chance to respond with an error first).  Pass ``None``
                     to wait indefinitely.
        """
        url = f"{self.base_url}{path}"
        if data is not None:
            data_bytes = json.dumps(data).encode("utf-8")
            req = urllib.request.Request(url, data=data_bytes, method=method)
            req.add_header("Content-Type", "application/json")
        else:
            req = urllib.request.Request(url, method=method)

        try:
            kwargs = {"timeout": timeout} if timeout is not None else {}
            with urllib.request.urlopen(req, **kwargs) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            try:
                error_data = json.loads(error_body)
                error_message = error_data.get("error", str(e))

                # Deserialize the exception using the utility function
                raise deserialize_exception(error_data, f"Error from LeanServer: {error_message}")
            except json.JSONDecodeError:
                raise RuntimeError(f"Error from LeanServer: {str(e)}")
        except urllib.error.URLError as e:
            raise ConnectionError(f"LeanServer request to {path} failed: {e}") from e
        except TimeoutError as e:
            # Socket-level timeouts during response reading may bypass URLError
            raise ConnectionError(f"LeanServer request to {path} timed out: {e}") from e

    def check_status(self) -> dict:
        """Check the status of the server."""
        return self._request("GET", "/status")

    def get_process(self, blocking: bool = True, timeout: float | None = 300.0) -> "LeanRemoteProcess | None":
        """Get a process from the server.
        
        Args:
            blocking: If True, wait until a process is available. If False, return None immediately if unavailable.
            timeout: Maximum time to wait for a process (in seconds). Only used if blocking=True.
                     Default is 300 seconds (5 minutes). Set to None for no timeout.
        
        Returns:
            A LeanRemoteProcess if available, None if not available (non-blocking) or timeout expired.
        """
        data = {"blocking": blocking}
        if timeout is not None:
            data["timeout"] = timeout
        # Socket timeout slightly exceeds server-side timeout to allow the
        # server to respond with "no process available" rather than the socket
        # timing out first.  +30s gives headroom for server-side bookkeeping.
        socket_timeout = timeout + 30.0 if timeout is not None else None
        response = self._request("POST", "/process/get", data, timeout=socket_timeout)
        process_id = response.get("process_id")
        if process_id is None:
            return None
        return LeanRemoteProcess(self, process_id)


class LeanRemoteProcess:
    """A remote Lean process managed by a LeanServer."""

    def __init__(self, client: LeanClient, process_id: int):
        self.client = client
        self.process_id = process_id
        self._returned = False
        self._lock = threading.Lock()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args, **kwargs):
        """Return the process to the pool when exiting context."""
        try:
            self.return_process()
        except Exception:
            # Don't mask the original exception (if any) with a return failure.
            # return_process already retries internally, so if we get here the
            # server is likely unreachable.
            logging.getLogger("LeanClient").warning(
                f"Failed to return process {self.process_id} in __exit__"
            )

    def __del__(self):
        """Best-effort cleanup: try to return the process if not already returned."""
        if not self._returned:
            try:
                self.return_process()
            except Exception:
                pass

    def _check_not_returned(self):
        """Check that the process hasn't been returned. Must be called with lock held."""
        if self._returned:
            raise RuntimeError("Process has already been returned to the pool")

    def send_command(self, command: str) -> dict:
        """Send a command to the remote process."""
        with self._lock:
            self._check_not_returned()
            return self.client._request(
                "POST",
                f"/process/{self.process_id}/command",
                {"command": command},
                timeout=360,
            )

    def is_valid_source(self, source: str) -> bool:
        """Check if the source is valid Lean code."""
        with self._lock:
            self._check_not_returned()
            response = self.client._request(
                "POST",
                f"/process/{self.process_id}/is_valid_source",
                {"source": source},
                timeout=60,
            )
            return response["is_valid"]

    def return_process(self):
        """Return the process to the pool. Safe to call multiple times.

        Retries up to 3 times to avoid leaking processes on the server when
        the network is transiently unreachable.
        """
        with self._lock:
            if self._returned:
                return  # Already returned, nothing to do
        # Send the return request outside the lock to avoid holding it during I/O.
        # Only mark _returned=True on success to allow retries from __exit__/__del__.
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                self.client._request("POST", f"/process/{self.process_id}/return", timeout=10)
                with self._lock:
                    self._returned = True
                return
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
        # All retries exhausted - mark as returned to prevent infinite retries,
        # but log the leak.  The process will be stuck on the server.
        with self._lock:
            self._returned = True
        logging.getLogger("LeanClient").error(
            f"Failed to return process {self.process_id} after {max_retries} attempts: {last_error}"
        )

    def proof_from_sorry(self, theorem_with_sorry: str) -> ValueOrError["RemoteLeanProofBranch"]:
        """Create a proof branch from a theorem with sorry."""
        with self._lock:
            self._check_not_returned()
            response = self.client._request(
                "POST",
                f"/process/{self.process_id}/proof_from_sorry",
                {"theorem_with_sorry": theorem_with_sorry},
                timeout=60,
            )

        if "error" in response:
            return ValueOrError.from_error(response["error"])

        value = response["value"]
        return ValueOrError.from_success(RemoteLeanProofBranch(
            self,  # Pass the LeanRemoteProcess to keep it alive
            value["branch_id"],
            value["goals"]
        ))


class RemoteLeanProofBranch:
    """A remote proof branch managed by a LeanServer.
    
    This is a thin client-side proxy that maps 1:1 to a LeanProofBranch on the server.
    All operations are delegated to the server-side branch.
    """

    def __init__(self, remote_process: LeanRemoteProcess, branch_id: int, goals: list[dict]):
        # Hold a reference to the LeanRemoteProcess to prevent it from being
        # garbage collected (and returning the process to the pool) while this
        # proof branch is still in use.
        self._remote_process = remote_process
        self._branch_id = branch_id
        self._goals = [LeanGoal.deserialize(g) for g in goals]

    @property
    def client(self) -> LeanClient:
        return self._remote_process.client

    @property
    def process_id(self) -> int:
        return self._remote_process.process_id

    @property
    def state(self) -> LeanProofState:
        """Get the current proof state."""
        return LeanProofState(self._goals)

    @property
    def is_solved(self) -> bool:
        """Check if the proof is solved."""
        return self.state.is_solved()

    def try_apply_tactic(
            self,
            tactic: LeanTactic | str,
            timeout: int | None = 1000,
    ) -> ValueOrError[list["RemoteLeanProofBranch"]]:
        """Apply a tactic to the proof branch.
        
        This delegates to LeanProofBranch.try_apply_tactic_async on the server,
        ensuring identical behavior to local execution.
        """
        if isinstance(tactic, LeanTactic):
            tactic = tactic.tactic

        data = {
            "tactic": tactic,
        }
        if timeout is not None:
            data["timeout"] = timeout

        # Socket timeout: tactic timeout (ms->s) + 30s headroom for server overhead.
        # Prevents actors from blocking for 360s on an unresponsive server.
        socket_timeout = (timeout / 1000 if timeout else 300) + 30
        response = self.client._request(
            "POST",
            f"/process/{self.process_id}/branch/{self._branch_id}/try_apply_tactic",
            data,
            timeout=socket_timeout,
        )

        if "error" in response:
            return ValueOrError.from_error(response["error"])

        branches_data = response["value"]
        branches = []
        for branch_data in branches_data:
            branches.append(RemoteLeanProofBranch(
                self._remote_process,  # Pass the LeanRemoteProcess to keep it alive
                branch_data["branch_id"],
                branch_data["goals"]
            ))

        return ValueOrError.from_success(branches)


def start_server(
        pool: LeanProcessPool,
        address: str = "localhost",
        port: int = 8000,
        log_level: str = "INFO",
        warmup: bool = False,
) -> LeanServer:
    """Start a LeanServer with the given pool.

    If ``warmup`` is True, pre-start processes up to ``pool.max_processes``
    and block until they are ready *before* the HTTP server accepts requests.
    """
    server = LeanServer(pool, address, port, log_level)
    warmup_coro = pool.max_out_processes_async() if warmup else None
    server.start(warmup_coro=warmup_coro)
    return server


if __name__ == "__main__":
    # Example usage
    repl_exe = Path("../lean-repl/.lake/build/bin/repl")
    project_path = Path("../leantree_project")

    pool = LeanProcessPool(
        repl_exe=repl_exe,
        project_path=project_path,
        max_processes=2,
    )

    server = start_server(pool, address="localhost", port=8000)
    print(f"Server started on http://localhost:8000")
    print("Press Ctrl+C to stop")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
        pool.shutdown()
