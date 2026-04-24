"""Benchmark leanserver warmup time across fresh subprocess starts.

Each iteration: launch a fresh `leanserver --warmup ...` subprocess, measure
the wall-clock time until it logs that it is accepting requests, then kill it.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


WARMUP_MARKER = "Server accepting requests"


def run_one_iter(
    repl_exe: Path,
    project_path: Path,
    num_processes: int,
    port: int,
    imports: list[str],
) -> float:
    cmd = [
        "leanserver",
        "--project-path", str(project_path),
        "--repl-exe", str(repl_exe),
        "--imports", *imports,
        "--max-processes", str(num_processes),
        "--port", str(port),
        "--warmup",
    ]
    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        # New session so we can SIGKILL the whole group (leanserver spawns REPL children).
        start_new_session=True,
    )
    elapsed: float | None = None
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(f"  > {line}")
            sys.stdout.flush()
            if WARMUP_MARKER in line:
                elapsed = time.monotonic() - t0
                break
        else:
            rc = proc.wait()
            raise RuntimeError(
                f"leanserver exited (rc={rc}) before printing {WARMUP_MARKER!r}"
            )
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    assert elapsed is not None
    return elapsed


def summarize(label: str, times: list[float]) -> None:
    if not times:
        print(f"{label}: (none)")
        return
    avg = sum(times) / len(times)
    items = ", ".join(f"{t:.2f}s" for t in times)
    print(f"{label}: avg={avg:.2f}s  times=[{items}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-processes", type=int, required=True,
                        help="Value passed to leanserver --max-processes.")
    parser.add_argument("--warmup-iters", type=int, default=1,
                        help="Iterations excluded from the measured average (default: 1).")
    parser.add_argument("--n-iters", type=int, default=3,
                        help="Measured iterations (default: 3).")
    parser.add_argument("--project-path", type=str,
                        default=os.getenv("LEAN_PROJECT_PATH"),
                        help="Lean project path (default: $LEAN_PROJECT_PATH).")
    parser.add_argument("--repl-exe", type=str,
                        default=os.getenv("LEAN_REPL_EXE"),
                        help="Lean REPL executable (default: $LEAN_REPL_EXE).")
    parser.add_argument("--port", type=int, default=8765,
                        help="Base port; each iter uses port+i to avoid TIME_WAIT collisions.")
    parser.add_argument("--imports", type=str, nargs="*", default=["Mathlib"],
                        help="Lean packages to import (default: Mathlib).")
    args = parser.parse_args()

    if not args.project_path:
        parser.error("--project-path or $LEAN_PROJECT_PATH must be set")
    if not args.repl_exe:
        parser.error("--repl-exe or $LEAN_REPL_EXE must be set")

    project_path = Path(args.project_path)
    repl_exe = Path(args.repl_exe)

    warmup_times: list[float] = []
    measured_times: list[float] = []
    total = args.warmup_iters + args.n_iters
    for i in range(total):
        is_warmup = i < args.warmup_iters
        phase = "warmup" if is_warmup else "measure"
        print(f"\n=== iter {i + 1}/{total} ({phase}) ===", flush=True)
        t = run_one_iter(
            repl_exe=repl_exe,
            project_path=project_path,
            num_processes=args.num_processes,
            port=args.port + i,
            imports=args.imports,
        )
        print(f"  -> warmup took {t:.2f}s", flush=True)
        (warmup_times if is_warmup else measured_times).append(t)

    print()
    summarize("Warmup iters ", warmup_times)
    summarize("Measured iters", measured_times)


if __name__ == "__main__":
    main()
