"""CLI script to start the Lean server."""

import argparse
import atexit
import faulthandler
import logging
import os
import resource
import signal
import sys
import termios
import threading
from pathlib import Path

from leantree.repl_adapter.server import start_server, LeanClient
from leantree.repl_adapter.process_pool import LeanProcessPool
from leantree.utils import Logger, LogLevel


# Default soft limit for open file descriptors.  The OS default of 1024 is too
# tight for ~62 lake/repl subprocesses + ~125 concurrent HTTP clients + their
# transient TIME_WAIT sockets.  We've observed FD exhaustion (Errno 24) at
# 1024 in production; 65536 leaves headroom for >5x the steady-state load.
DEFAULT_NOFILE_SOFT_LIMIT = 65536


def _raise_nofile_soft_limit(target: int = DEFAULT_NOFILE_SOFT_LIMIT) -> None:
    """Raise RLIMIT_NOFILE up to ``min(target, current_hard_limit)``.

    No-op on platforms without the resource module (e.g. Windows).
    Logs the before/after values so the actual limit is visible at startup.
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (AttributeError, ValueError, OSError) as e:
        print(f"Could not query RLIMIT_NOFILE: {e}", file=sys.stderr)
        return
    new_soft = min(target, hard)
    if new_soft <= soft:
        print(f"RLIMIT_NOFILE soft limit already at {soft} (>= requested {target}); leaving as-is")
        return
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    except (ValueError, OSError) as e:
        print(f"Could not raise RLIMIT_NOFILE from {soft} to {new_soft}: {e}", file=sys.stderr)
        return
    print(f"Raised RLIMIT_NOFILE soft limit from {soft} to {new_soft} (hard={hard})", flush=True)

# Terminal settings preservation
# ----------------------------
# The keyboard_monitor thread uses input() to read keypresses, which can modify
# terminal settings (e.g., disabling echo mode). If the process is killed via
# Ctrl+C while input() is active, these settings may not be restored, leaving
# the terminal in a broken state (characters you type are not displayed).
# We save the original settings at startup and restore them on exit.
_original_terminal_settings = None
try:
    _original_terminal_settings = termios.tcgetattr(sys.stdin)
except (termios.error, AttributeError):
    pass  # Not a terminal or termios not available


def _restore_terminal():
    """Restore terminal settings to their original state."""
    if _original_terminal_settings is not None:
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _original_terminal_settings)
        except (termios.error, AttributeError):
            pass


atexit.register(_restore_terminal)


def main():
    """CLI entry point for the Lean server."""
    # Line-buffer stdout so nohup-redirected log files show diagnostic prints
    # immediately (RLIMIT_NOFILE raise, RLIMIT_AS default, warmup progress,
    # SIGUSR1 registration) rather than waiting for block-buffer flush.
    # Without this, empty log files mid-run masked useful startup info.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="Start a Lean server")
    parser.add_argument(
        "--address",
        type=str,
        default="localhost",
        help="Server address (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)"
    )
    parser.add_argument(
        "--repl-exe",
        type=str,
        default=None,
        help="Path to Lean REPL executable (default: from LEAN_REPL_EXE)"
    )
    parser.add_argument(
        "--project-path",
        type=str,
        default=None,
        help="Path to Lean project (default: from LEAN_PROJECT_PATH env or ./leantree_project)"
    )
    parser.add_argument(
        "--max-processes",
        type=int,
        default=8,
        help="Maximum number of parallel processes (default: 2)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--imports",
        type=str,
        nargs="*",
        help="List of Lean packages to import"
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Pre-start all processes to max capacity before accepting requests"
    )
    parser.add_argument(
        "--warmup-batch-size",
        type=int,
        default=None,
        help="Start warmup processes in batches of this size to limit resource contention (default: all at once)"
    )
    parser.add_argument(
        "--max-process-memory-gb",
        type=float,
        default=8.0,
        help=(
            "Per-Lean-subprocess RLIMIT_AS in GiB (default: 8). When a tactic's "
            "address space exceeds this, the kernel SIGKILLs the subprocess "
            "cleanly; the pool's existing poisoned-process logic swaps in a "
            "fresh one. 0 disables the limit (not recommended in production)."
        )
    )

    args = parser.parse_args()

    # FD limit must be raised BEFORE the HTTP socket is opened or any subprocess
    # is spawned, so do it as early as possible after argparse.
    _raise_nofile_soft_limit()

    # Make `kill -USR1 <pid>` dump a Python traceback for every thread to
    # stderr.  Pure diagnostic surface — does not affect normal operation.  The
    # bounded `_run_async` timeouts in server.py mean the leanserver shouldn't
    # actually wedge in production, but if it ever does this is how we learn
    # what's stuck without having to attach gdb.
    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True)
        print("Registered SIGUSR1 handler: kill -USR1 <pid> dumps all-thread tracebacks to stderr")
    except (AttributeError, ValueError, OSError) as e:
        print(f"Could not register SIGUSR1 faulthandler: {e}", file=sys.stderr)

    # Determine repl_exe path
    if args.repl_exe:
        repl_exe = Path(args.repl_exe)
    elif os.getenv("LEAN_REPL_EXE"):
        repl_exe = Path(os.getenv("LEAN_REPL_EXE"))
    else:
        raise ValueError("REPL executable not specified")

    if not repl_exe.exists():
        print(f"Error: REPL executable not found at {repl_exe}", file=sys.stderr)
        print("Please specify --repl-exe or set LEAN_REPL_EXE environment variable", file=sys.stderr)
        sys.exit(1)

    # Determine project_path
    if args.project_path:
        project_path = Path(args.project_path)
    elif os.getenv("LEAN_PROJECT_PATH"):
        project_path = Path(os.getenv("LEAN_PROJECT_PATH"))
    else:
        # Default relative to current working directory
        project_path = Path("leantree_project").resolve()

    if not project_path.exists():
        print(f"Error: Project path not found at {project_path}", file=sys.stderr)
        print("Please specify --project-path or set LEAN_PROJECT_PATH environment variable", file=sys.stderr)
        sys.exit(1)

    # Create process pool
    env_setup_async = None
    if args.imports:
        async def setup_imports(process):
            imports_str = "\n".join(f"import {imp}" for imp in args.imports)
            await process.send_command_async(imports_str)
        env_setup_async = setup_imports

    max_process_memory_bytes = (
        int(args.max_process_memory_gb * 1024 ** 3) if args.max_process_memory_gb > 0 else None
    )
    if max_process_memory_bytes is not None:
        print(
            f"Per-process RLIMIT_AS: {args.max_process_memory_gb} GiB "
            f"({max_process_memory_bytes} bytes) — runaway tactics will be SIGKILLed by the kernel"
        )
    else:
        print("Per-process RLIMIT_AS: DISABLED (--max-process-memory-gb=0)")

    pool = LeanProcessPool(
        repl_exe=repl_exe,
        project_path=project_path,
        max_processes=args.max_processes,
        logger=Logger(LogLevel.DEBUG) if args.log_level == "DEBUG" else None,
        env_setup_async=env_setup_async,
        max_process_memory_bytes=max_process_memory_bytes,
    )

    # Start server
    server = start_server(
        pool,
        address=args.address,
        port=args.port,
        log_level=args.log_level
    )

    # Warmup: pre-start all processes if requested (must be after server starts to use its event loop)
    if args.warmup:
        batch_size = args.warmup_batch_size
        if batch_size:
            print(f"Warming up {args.max_processes} processes in batches of {batch_size}...")
        else:
            print(f"Warming up {args.max_processes} processes...")
        server._run_async(pool.max_out_processes_async(batch_size=batch_size))
        print("Warmup complete.")
    print(f"Lean project: {project_path}")
    print(f"REPL executable: {repl_exe}")
    if args.imports:
        print(f"Importing packages: {", ".join(args.imports)}")
    print(f"Server started on http://{args.address}:{args.port} with log level {args.log_level}")

    # Handle shutdown gracefully
    _shutting_down = False

    def signal_handler(sig, frame):
        nonlocal _shutting_down
        if _shutting_down:
            print("\nForced shutdown.")
            _restore_terminal()
            os._exit(1)
        _shutting_down = True
        print("\nShutting down server... (press Ctrl+C again to force quit)")
        # server.stop() handles both checked-out and idle processes
        server.stop()
        # Restore terminal settings (input() in keyboard_monitor can leave terminal in bad state)
        _restore_terminal()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start keyboard monitoring thread
    def keyboard_monitor():
        client = LeanClient(args.address, args.port)
        while True:
            try:
                input()  # Wait for Enter key
                try:
                    status = client.check_status()
                    print(f"\n=== Server Status ===")
                    print(f"Processes: {status['available_processes']} available, "
                          f"{status['used_processes']} used, "
                          f"{status['starting_processes']} starting, "
                          f"{status['max_processes']} max")
                    print(f"RAM: {status['ram']['percent']:.1f}% used "
                          f"({status['ram']['used_bytes'] / (1024**3):.1f}GB / "
                          f"{status['ram']['total_bytes'] / (1024**3):.1f}GB)")
                    avg_cpu = sum(status['cpu_percent_per_core']) / len(status['cpu_percent_per_core'])
                    print(f"CPU: {avg_cpu:.1f}% average across {len(status['cpu_percent_per_core'])} cores")
                    
                    # Show inactive processes and branches
                    inactive_proc = status.get('inactive_processes', 0)
                    total_proc = status.get('total_tracked_processes', 0)
                    inactive_br = status.get('inactive_branches', 0)
                    total_br = status.get('total_branches', 0)
                    print(f"Inactive (>60s): {inactive_proc}/{total_proc} processes, "
                          f"{inactive_br}/{total_br} branches")
                    
                    # Show active requests
                    active_requests = status.get('active_requests', [])
                    if active_requests:
                        print(f"Active requests ({len(active_requests)}):")
                        for req in active_requests:
                            print(f"  - {req['path']} ({req['duration_seconds']}s, thread: {req['thread']})")
                    else:
                        print("Active requests: none")
                    print()
                except Exception as e:
                    print(f"Error getting status: {e}")
            except EOFError:
                # stdin closed, exit the thread
                break

    keyboard_thread = threading.Thread(target=keyboard_monitor, daemon=True)
    keyboard_thread.start()
    print("Press Enter to show server status")

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()
