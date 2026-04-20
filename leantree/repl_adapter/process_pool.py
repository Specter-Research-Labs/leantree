import time
import asyncio
from pathlib import Path
from typing import Callable, Coroutine
import psutil
from leantree.repl_adapter.interaction import LeanProcess, LeanEnvironmentCheckpoint
from leantree.utils import Logger, NullLogger, to_sync


class LeanProcessPool:
    """
    A pool of LeanProcess instances for parallel processing.

    This class manages a pool of LeanProcess instances, handling their creation,
    allocation, and recycling. It also monitors memory usage and restarts processes
    that exceed memory thresholds.
    """

    def __init__(
            self,
            repl_exe: Path,
            project_path: Path,
            max_processes: int,
            max_memory_utilization: float = 80.0,  # percentage
            env_setup_async: Callable[[LeanProcess], Coroutine] | None = None,
            logger: Logger | None = None,
            max_process_memory_bytes: int | None = None,
    ):
        """
        Initialize the process pool.

        Args:
            repl_exe: Path to the Lean REPL executable
            project_path: Path to the Lean project
            max_processes: Maximum number of parallel processes
            max_memory_utilization: Maximum memory utilization as a percentage,
                used by the legacy on-return memory check.  Largely superseded
                by ``max_process_memory_bytes`` which is enforced by the kernel
                via RLIMIT_AS at subprocess creation, but kept as a soft check
                for hosts that don't enforce the kernel limit.
            max_process_memory_bytes: Hard per-Lean-subprocess address-space
                ceiling enforced by ``RLIMIT_AS`` set in a ``preexec_fn``.
                A pathological tactic that exceeds this gets SIGKILLed by the
                kernel and surfaces as a clean ``LeanProcessException`` —
                which the leanserver's poisoned-process path then handles by
                replacing the dead process.  ``None`` disables the limit.
            logger: Optional logger
        """
        self.repl_exe = repl_exe
        self.project_path = project_path
        self.max_processes = max_processes
        self.max_memory_utilization = max_memory_utilization
        self.max_process_memory_bytes = max_process_memory_bytes
        self.logger = logger if logger else NullLogger()

        # Pool state
        self.available_processes: list[LeanProcess] = []
        self.checkpoints: dict[LeanProcess, LeanEnvironmentCheckpoint] = {}
        self._num_used_processes: int = 0
        self._num_starting_processes: int = 0
        # Lazily initialized asyncio primitives (to bind to correct event loop)
        self._lock: asyncio.Lock | None = None
        self._process_available_event: asyncio.Event | None = None
        self.env_setup_async = env_setup_async
        # Calculate memory threshold per server based on total system memory.
        # This is the *legacy* on-return PSS check; the new RLIMIT_AS
        # enforcement (max_process_memory_bytes) is the primary safety net.
        total_memory = psutil.virtual_memory().total
        self.memory_threshold_per_process = int(total_memory * (self.max_memory_utilization / 100) / self.max_processes)

        self._was_shutdown = False

    @property
    def lock(self) -> asyncio.Lock:
        """Lazily create the lock to bind to the correct event loop."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @property
    def process_available_event(self) -> asyncio.Event:
        """Lazily create the event to bind to the correct event loop."""
        if self._process_available_event is None:
            self._process_available_event = asyncio.Event()
        return self._process_available_event

    async def _create_process_async(self, track_starting: bool = True) -> LeanProcess:
        """Create a new LeanProcess instance.
        
        Args:
            track_starting: If True, increment/decrement _num_starting_processes counter.
        """
        if track_starting:
            self._num_starting_processes += 1
        try:
            process = LeanProcess(
                self.repl_exe,
                self.project_path,
                self.logger,
                pool=self,
                max_memory_bytes=self.max_process_memory_bytes,
            )
            await process.start_async()
            if self.env_setup_async:
                await self.env_setup_async(process)
            self.checkpoints[process] = process.checkpoint()
            return process
        finally:
            if track_starting:
                self._num_starting_processes -= 1

    async def max_out_processes_async(self, batch_size: int | None = None):
        """
        Start processes in parallel until we reach max_processes capacity.

        This method ensures that len(self.available_processes) + self._num_used_processes
        equals self.max_processes by starting new processes in parallel.

        Args:
            batch_size: If set, start processes in batches of this size to limit
                        resource contention (useful when env_setup is expensive).
                        ``None`` means start all at once.
        """
        async with self.lock:
            processes_to_start = self.max_processes - (len(self.available_processes) + self._num_used_processes)
            if processes_to_start <= 0:
                return
            self.logger.info(f"Starting {processes_to_start} processes in parallel")

        if batch_size is not None and batch_size < processes_to_start:
            remaining = processes_to_start
            while remaining > 0:
                current_batch = min(batch_size, remaining)
                self.logger.info(f"Starting batch of {current_batch} processes ({processes_to_start - remaining}/{processes_to_start} done)")
                tasks = [self._create_process_async() for _ in range(current_batch)]
                new_processes = await asyncio.gather(*tasks)
                async with self.lock:
                    self.available_processes.extend(new_processes)
                    if self.available_processes:
                        self.process_available_event.set()
                remaining -= current_batch
            self.logger.info(
                f"Started {processes_to_start} processes. Available: {len(self.available_processes)}, Used: {self._num_used_processes}")
        else:
            # Create processes OUTSIDE the lock
            tasks = [self._create_process_async() for _ in range(processes_to_start)]
            new_processes = await asyncio.gather(*tasks)

            async with self.lock:
                self.available_processes.extend(new_processes)
                if self.available_processes:
                    self.process_available_event.set()
                self.logger.info(
                    f"Started {len(new_processes)} processes. Available: {len(self.available_processes)}, Used: {self._num_used_processes}")

    async def _get_or_create_process_async(self) -> LeanProcess | None:
        """Try to get an available process or reserve a slot and create one.

        Returns a process if one was available or successfully created,
        None if at capacity with no available processes.

        The lock is only held briefly to check/update counters; the slow
        process creation happens outside the lock.
        """
        async with self.lock:
            if self._was_shutdown:
                raise RuntimeError("Process pool has been shut down")

            if self.available_processes:
                process = self.available_processes.pop()
                if not self.available_processes:
                    self.process_available_event.clear()
                self._num_used_processes += 1
                return process

            # Reserve a slot for a new process (but create it outside the lock)
            if self._num_used_processes < self.max_processes:
                self._num_used_processes += 1
                need_create = True
            else:
                need_create = False

        if not need_create:
            return None

        # Create the process OUTSIDE the lock so returns aren't blocked
        try:
            process = await self._create_process_async()
            return process
        except Exception:
            # Release the reserved slot on failure
            async with self.lock:
                self._num_used_processes -= 1
                self.process_available_event.set()
            raise

    async def get_process_async(self, blocking: bool = True, timeout: float | None = None) -> LeanProcess | None:
        """
        Get a process from the pool asynchronously.

        Args:
            blocking: If True, wait until a process is available. If False, return None if no process is available.
            timeout: Maximum time to wait for a process (in seconds). Only used if blocking=True.
                     If None, waits indefinitely.

        Returns:
            A LeanProcess instance if available, None otherwise (only if blocking=False or timeout expires)

        Raises:
            RuntimeError: If the pool has been shut down
        """
        process = await self._get_or_create_process_async()
        if process is not None:
            return process

        # No processes available and at max capacity
        if not blocking:
            return None

        # Wait for a process to become available asynchronously
        start_time = time.monotonic()
        while True:
            # Calculate remaining time if timeout is set
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    return None  # Timeout expired
                wait_timeout = min(5.0, timeout - elapsed)
            else:
                wait_timeout = 5.0

            try:
                # Wait for the event to be set with a timeout
                await asyncio.wait_for(self.process_available_event.wait(), timeout=wait_timeout)
            except asyncio.TimeoutError:
                # Continue waiting if timeout occurs (unless total timeout expired)
                pass

            process = await self._get_or_create_process_async()
            if process is not None:
                return process

    async def return_process_async(self, process: LeanProcess):
        """
        Return a process to the pool.

        If the process's memory usage exceeds the threshold, it will be terminated
        instead of being returned to the pool.

        Args:
            process: The LeanProcess instance to return
        """

        # Check shutdown OUTSIDE the lock; if the host enforces RLIMIT_AS via
        # the preexec_fn (the new default), pathological-memory subprocesses
        # already get SIGKILLed by the kernel as soon as they balloon — no
        # need to re-measure PSS on return, and trying to do so for an already-
        # killed process raises spuriously.  When `max_process_memory_bytes`
        # is None (legacy/disabled) we still fall back to the PSS check.
        should_terminate = False
        async with self.lock:
            if self._was_shutdown:
                should_terminate = True

        kernel_enforced_memory_limit = self.max_process_memory_bytes is not None
        if not should_terminate and not kernel_enforced_memory_limit:
            try:
                memory_usage = process.memory_usage()
                if self.memory_threshold_per_process and memory_usage > self.memory_threshold_per_process:
                    self.logger.info(
                        f"Process memory usage ({memory_usage / (1024 * 1024):.2f} MB RSS) exceeds threshold "
                        f"({self.memory_threshold_per_process / (1024 * 1024):.2f} MB). Terminating and replacing."
                    )
                    should_terminate = True
            except Exception as e:
                self.logger.warning(f"Error checking process memory: {e}. Terminating process.")
                should_terminate = True

        if should_terminate:
            await process.stop_async()
            async with self.lock:
                self.checkpoints.pop(process, None)
                assert self._num_used_processes > 0, "No processes in use"
                self._num_used_processes -= 1
                self.process_available_event.set()
            return

        # Drain and rollback outside the lock (drain is near-instant,
        # rollback is just an int assignment)
        await process.drain_repl_output_async()
        if process in self.checkpoints:
            process.rollback_to(self.checkpoints[process])

        async with self.lock:
            assert self._num_used_processes > 0, "No processes in use"
            self._num_used_processes -= 1
            self.available_processes.append(process)
            # Notify waiting coroutines that a process is available
            self.process_available_event.set()

    return_process = to_sync(return_process_async)

    async def shutdown_async(self):
        """Shut down all processes in the pool asynchronously."""
        async with self.lock:
            if self._was_shutdown:
                return
            self._was_shutdown = True

            # Wake up any coroutines waiting for processes so they can see _was_shutdown
            self.process_available_event.set()

            # Shut down available processes
            for process in self.available_processes:
                try:
                    await process.stop_async()
                except Exception as e:
                    self.logger.warning(f"Error shutting down process: {e}")
            self.available_processes = []
            self.checkpoints.clear()

    shutdown = to_sync(shutdown_async)
