"""CLI script to start the Lean server."""

import argparse
import atexit
import os
import signal
import sys
import termios
import threading
from pathlib import Path

from leantree.repl_adapter.server import start_server, LeanClient
from leantree.repl_adapter.process_pool import LeanProcessPool
from leantree.utils import Logger, LogLevel

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

    args = parser.parse_args()

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

    pool = LeanProcessPool(
        repl_exe=repl_exe,
        project_path=project_path,
        max_processes=args.max_processes,
        logger=Logger(LogLevel.DEBUG) if args.log_level == "DEBUG" else None,
        env_setup_async=env_setup_async,
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
        print(f"Warming up {args.max_processes} processes...")
        server._run_async(pool.max_out_processes_async())
        print("Warmup complete.")
    print(f"Lean project: {project_path}")
    print(f"REPL executable: {repl_exe}")
    if args.imports:
        print(f"Importing packages: {", ".join(args.imports)}")
    print(f"Server started on http://{args.address}:{args.port} with log level {args.log_level}")

    # Handle shutdown gracefully
    def signal_handler(sig, frame):
        print("\nShutting down server...")
        # Shut down pool on the server's event loop before stopping it
        # (pool's asyncio primitives are bound to that loop)
        server._run_async(pool.shutdown_async())
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
