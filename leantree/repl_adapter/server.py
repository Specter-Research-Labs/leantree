import asyncio
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

from leantree.repl_adapter.interaction import LeanProcess, LeanProofBranch, LeanInteractionException
from leantree.repl_adapter.process_pool import LeanProcessPool
from leantree.core.lean import LeanProofState, LeanTactic, LeanGoal
from leantree.utils import serialize_exception, deserialize_exception, ValueOrError


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTP server that handles each request in a separate thread."""
    daemon_threads = True
    # Increase listen backlog for high-load scenarios (default is usually 5)
    request_queue_size = 128


class LeanServer:
    """Manages a LeanProcessPool and exposes it over a HTTP port."""

    def __init__(self, pool: LeanProcessPool, address: str = "localhost", port: int = 8000, log_level: str = "INFO"):
        self.pool = pool
        self.address = address
        self.port = port
        self.log_level = log_level
        self.server = None
        self.server_thread = None

        # Setup logger
        self.logger = logging.getLogger("LeanServer")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)

        self._process_id_counter = 0
        self._process_id_to_process = {}  # Maps process_id to LeanProcess
        self._process_to_id = {}  # Maps LeanProcess to process_id
        self._lock = threading.Lock()
        self._event_loop = None
        self._loop_thread = None
        
        # Branch tracking: maps (process_id, branch_id) to LeanProofBranch
        self._branch_id_counter = 0
        self._branches: dict[tuple[int, int], LeanProofBranch] = {}
        self._branches_lock = threading.Lock()
        
        # Last used tracking for inactivity detection
        self._process_last_used: dict[int, float] = {}  # process_id -> timestamp
        self._branch_last_used: dict[tuple[int, int], float] = {}  # (process_id, branch_id) -> timestamp
        
        # Request tracking for monitoring
        self._active_requests: dict[int, dict] = {}  # request_id -> {path, start_time, thread_name}
        self._request_id_counter = 0
        self._requests_lock = threading.Lock()

    def _get_process_id(self, process: LeanProcess) -> int:
        """Get or create a process ID for a LeanProcess."""
        with self._lock:
            if process in self._process_to_id:
                process_id = self._process_to_id[process]
                self._process_last_used[process_id] = time.time()
                return process_id
            self._process_id_counter += 1
            process_id = self._process_id_counter
            self._process_to_id[process] = process_id
            self._process_id_to_process[process_id] = process
            self._process_last_used[process_id] = time.time()
            return process_id

    def _get_process(self, process_id: int) -> LeanProcess:
        """Get a LeanProcess by its ID."""
        with self._lock:
            if process_id not in self._process_id_to_process:
                raise ValueError(f"Process {process_id} not found")
            self._process_last_used[process_id] = time.time()
            return self._process_id_to_process[process_id]

    def _remove_process(self, process_id: int):
        """Remove a process from tracking."""
        with self._lock:
            if process_id in self._process_id_to_process:
                process = self._process_id_to_process[process_id]
                del self._process_id_to_process[process_id]
                if process in self._process_to_id:
                    del self._process_to_id[process]
            self._process_last_used.pop(process_id, None)
        # Also clean up any branches associated with this process
        self._remove_branches_for_process(process_id)

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
            return self._branches[key]

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
            coro: The coroutine to run
            timeout: Optional timeout in seconds
        """
        if self._event_loop is None:
            raise RuntimeError("Server not started")
        future = asyncio.run_coroutine_threadsafe(coro, self._event_loop)
        return future.result(timeout=timeout)
    
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

    def start(self):
        """Start the HTTP server."""
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

        handler = self._create_handler()
        self.server = ThreadingHTTPServer((self.address, self.port), handler)

        def run_server():
            self.server.serve_forever()

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

    def stop(self):
        """Stop the HTTP server."""
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

            def _handle_status(self):
                pool = server.pool
                
                # Get per-CPU core utilization
                cpu_percent_per_core = psutil.cpu_percent(percpu=True)
                
                # Get RAM utilization
                memory = psutil.virtual_memory()
                
                available = len(pool.available_processes)
                used = pool._num_used_processes
                starting = pool._num_starting_processes
                
                # Get active requests (excluding this status request itself)
                active_requests = [
                    req for req in server.get_active_requests()
                    if req["path"] != "/status"
                ]
                
                # Count inactive processes and branches (not used in last 60 seconds)
                now = time.time()
                inactive_threshold = 60.0
                
                with server._lock:
                    inactive_processes = sum(
                        1 for pid, last_used in server._process_last_used.items()
                        if now - last_used > inactive_threshold
                    )
                    total_tracked_processes = len(server._process_last_used)
                
                with server._branches_lock:
                    inactive_branches = sum(
                        1 for key, last_used in server._branch_last_used.items()
                        if now - last_used > inactive_threshold
                    )
                    total_branches = len(server._branch_last_used)
                
                status = {
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
                    "inactive_processes": inactive_processes,
                    "total_tracked_processes": total_tracked_processes,
                    "inactive_branches": inactive_branches,
                    "total_branches": total_branches,
                }
                self._send_json(200, status)

            def _handle_get_process(self):
                data = self._read_json()
                blocking = data.get("blocking", True)
                # Default timeout of 1 minute to prevent indefinite blocking
                timeout = data.get("timeout", 60.0)

                process = None
                process_id = None
                try:
                    process = server._run_async(server.pool.get_process_async(blocking=blocking, timeout=timeout))
                    if process is None:
                        self._send_json(200, {"process_id": None})
                    else:
                        process_id = server._get_process_id(process)
                        self._send_json(200, {"process_id": process_id})
                except BrokenPipeError:
                    # Client disconnected - return the process to the pool if we got one
                    if process is not None:
                        server.logger.warning(
                            f"Client disconnected during /process/get, returning process {process_id} to pool"
                        )
                        server._remove_process(process_id)
                        try:
                            server._run_async(server.pool.return_process_async(process))
                        except Exception as e:
                            server.logger.error(f"Error returning process after client disconnect: {e}")
                    raise  # Re-raise to let the server handle the broken connection
                except Exception as e:
                    self._send_error(500, str(e), exception=e)

            def _handle_command(self, process_id: int):
                try:
                    process = server._get_process(process_id)
                    data = self._read_json()
                    command = data["command"]

                    response = server._run_async(process.send_command_async(command))
                    self._send_json(200, response)
                except Exception as e:
                    self._send_error(500, str(e), exception=e)

            def _handle_is_valid_source(self, process_id: int):
                try:
                    process = server._get_process(process_id)
                    data = self._read_json()
                    source = data["source"]

                    is_valid = server._run_async(process.is_valid_source_async(source))
                    self._send_json(200, {"is_valid": is_valid})
                except Exception as e:
                    self._send_error(500, str(e), exception=e)

            def _handle_return_process(self, process_id: int):
                try:
                    with server._lock:
                        if process_id not in server._process_id_to_process:
                            # Already returned - this is idempotent
                            self._send_json(200, {"status": "ok", "already_returned": True})
                            return
                        process = server._process_id_to_process[process_id]
                        # Remove from tracking BEFORE returning to pool to prevent
                        # a race where another client gets this process from the pool
                        # while it's still tracked with the old ID
                        del server._process_id_to_process[process_id]
                        if process in server._process_to_id:
                            del server._process_to_id[process]
                        server._process_last_used.pop(process_id, None)
                    # Clean up any branches associated with this process
                    server._remove_branches_for_process(process_id)
                    # Now return to pool - any new client will get a fresh ID
                    server._run_async(server.pool.return_process_async(process))
                    self._send_json(200, {"status": "ok"})
                except Exception as e:
                    self._send_error(500, str(e), exception=e)

            def _handle_proof_from_sorry(self, process_id: int):
                try:
                    process = server._get_process(process_id)
                    data = self._read_json()
                    theorem_with_sorry = data["theorem_with_sorry"]

                    # Get the first proof branch
                    # proofs_from_sorries_async returns an AsyncIterator, so we need to collect it
                    async def collect_proof_branches():
                        return [branch async for branch in process.proofs_from_sorries_async(theorem_with_sorry)]

                    try:
                        proof_branches = server._run_async(collect_proof_branches())
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
                    # Register the branch so we can use it later
                    branch_id = server._register_branch(process_id, proof_branch)
                    goals = [goal.serialize() for goal in proof_branch.state.goals]

                    response = {
                        "branch_id": branch_id,
                        "goals": goals,
                    }
                    self._send_json(200, {"value": response})
                except Exception as e:
                    self._send_error(500, str(e), exception=e)

            def _handle_try_apply_tactic(self, process_id: int, branch_id: int):
                try:
                    branch = server._get_branch(process_id, branch_id)
                    data = self._read_json()
                    tactic = data["tactic"]
                    timeout = data.get("timeout", 1000)

                    result = server._run_async(
                        branch.try_apply_tactic_async(tactic, timeout=timeout)
                    )

                    if not result.is_success():
                        self._send_json(200, {"error": str(result.error)})
                        return

                    # Register new branches and return their info
                    new_branches = result.value
                    branches_data = []
                    for new_branch in new_branches:
                        new_branch_id = server._register_branch(process_id, new_branch)
                        branches_data.append({
                            "branch_id": new_branch_id,
                            "goals": [goal.serialize() for goal in new_branch.state.goals],
                        })

                    self._send_json(200, {"value": branches_data})
                except Exception as e:
                    self._send_error(500, str(e), exception=e)

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

    def _request(self, method: str, path: str, data: dict = None, timeout: float | None = None) -> dict:
        """Make an HTTP request to the server."""
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
        self.return_process()
    
    def __del__(self):
        """Best-effort cleanup: try to return the process if not already returned.
        
        Note: This is a fallback. Users should prefer using the context manager
        or explicitly calling return_process() for reliable cleanup.
        """
        if not self._returned:
            try:
                self.return_process()
            except Exception:
                # Ignore errors during garbage collection - the server's lease
                # timeout will eventually reclaim the process anyway
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
                {"command": command}
            )

    def is_valid_source(self, source: str) -> bool:
        """Check if the source is valid Lean code."""
        with self._lock:
            self._check_not_returned()
            response = self.client._request(
                "POST",
                f"/process/{self.process_id}/is_valid_source",
                {"source": source}
            )
            return response["is_valid"]

    def return_process(self):
        """Return the process to the pool. Safe to call multiple times."""
        with self._lock:
            if self._returned:
                return  # Already returned, nothing to do
            self._returned = True
        # Send the return request outside the lock to avoid holding it during I/O
        self.client._request("POST", f"/process/{self.process_id}/return")

    def proof_from_sorry(self, theorem_with_sorry: str) -> ValueOrError["RemoteLeanProofBranch"]:
        """Create a proof branch from a theorem with sorry."""
        with self._lock:
            self._check_not_returned()
            response = self.client._request(
                "POST",
                f"/process/{self.process_id}/proof_from_sorry",
                {"theorem_with_sorry": theorem_with_sorry}
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

        response = self.client._request(
            "POST",
            f"/process/{self.process_id}/branch/{self._branch_id}/try_apply_tactic",
            data
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
        log_level: str = "INFO"
) -> LeanServer:
    """Start a LeanServer with the given pool."""
    server = LeanServer(pool, address, port, log_level)
    server.start()
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
