"""Stress-test a running Lean server.

Two actions:
  nice: workers grab a process, run a few tactics, return it, and repeat.
  hold: workers grab a process once and apply tactics on it forever.
"""

import argparse
import itertools
import sys
import threading
import time
import traceback

from leantree.repl_adapter.server import LeanClient
from leantree.utils import RemoteException


THEOREM = "example : 1 + 1 = 2 := by sorry"

# Rotation for the `nice` worker: same shape as TACTICS_HOLD below but with
# list sizes / exponents scaled ~10x down, so each acquire/tactics/return
# cycle peaks in the low-GB range instead of hold's multi-GB.
TACTICS_NICE = [
    "skip",
    # ~1.2 GB transient: 50M-cons list, length
    "have _h : (List.range 50000000).length = 50000000 := by native_decide",
    # ~2.4 GB transient: 100M-element list + fold
    "have _h : (List.range 100000000).foldl (· + ·) 0 = 4999999950000000 := by native_decide",
    "skip",
    # ~3.6 GB transient: 150M-element replicate + fold
    "have _h : (List.replicate 150000000 1).foldl (· + ·) 0 = 150000000 := by native_decide",
    # kernel bigint work, lighter than hold's 2^100
    "have _h : 2 ^ 33 = 8589934592 := by decide",
    "rfl",
]

# Rotation for the `hold` worker: each native_decide materializes a huge list
# at elab time, driving multi-GB transient allocations in the LeanServer.
# With many workers hitting heavy steps concurrently, peak RSS easily exceeds
# physical RAM — tune sizes down (or worker count) if the box OOMs.
TACTICS_HOLD = [
    "skip",
    # ~12 GB transient: 500M-cons list, traverse to length
    "have _h : (List.range 500000000).length = 500000000 := by native_decide",
    # ~24 GB transient: 1B-cons list + fold
    "have _h : (List.range 1000000000).foldl (· + ·) 0 = 499999999500000000 := by native_decide",
    "skip",
    # ~36 GB transient: 1.5B-element replicate + fold
    "have _h : (List.replicate 1500000000 1).foldl (· + ·) 0 = 1500000000 := by native_decide",
    # kernel bigint work on ~100-bit Nat
    "have _h : 2 ^ 100 = 1267650600228229401496703205376 := by decide",
    "skip",
    "rfl",
]


class Counters:
    def __init__(self):
        self._lock = threading.Lock()
        self.tactics = 0
        self.processes_acquired = 0
        self.proofs_started = 0
        self.remote_exceptions = 0
        self.other_errors = 0
        self.workers_alive = 0

    def inc(self, field: str, n: int = 1):
        with self._lock:
            setattr(self, field, getattr(self, field) + n)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "tactics": self.tactics,
                "processes_acquired": self.processes_acquired,
                "proofs_started": self.proofs_started,
                "remote_exceptions": self.remote_exceptions,
                "other_errors": self.other_errors,
                "workers_alive": self.workers_alive,
            }


def _report_exception(worker_id: int, where: str, exc: BaseException, counters: Counters):
    if isinstance(exc, RemoteException):
        counters.inc("remote_exceptions")
        print(
            f"[worker {worker_id}] RemoteException in {where}: {exc}",
            flush=True,
        )
    else:
        counters.inc("other_errors")
        print(
            f"[worker {worker_id}] {type(exc).__name__} in {where}: {exc}",
            flush=True,
        )


def nice_worker(
    worker_id: int,
    client: LeanClient,
    stop_event: threading.Event,
    counters: Counters,
    tactic_delay: float,
):
    counters.inc("workers_alive")
    try:
        while not stop_event.is_set():
            try:
                with client.get_process(blocking=True, timeout=60.0) as process:
                    if process is None:
                        continue
                    counters.inc("processes_acquired")

                    result = process.proof_from_sorry(THEOREM)
                    if not result.is_success():
                        counters.inc("other_errors")
                        print(
                            f"[worker {worker_id}] proof_from_sorry failed: {result.error}",
                            flush=True,
                        )
                        continue
                    counters.inc("proofs_started")
                    branch = result.value

                    for tactic in TACTICS_NICE:
                        if stop_event.is_set():
                            break
                        tac_result = branch.try_apply_tactic(tactic)
                        counters.inc("tactics")
                        if tactic_delay > 0 and stop_event.wait(tactic_delay):
                            break
                        if not tac_result.is_success():
                            # tactic failure is fine; just move to next one
                            continue
                        next_branches = tac_result.value
                        if not next_branches:
                            break
                        branch = next_branches[0]
                        if branch.is_solved:
                            break
            except Exception as e:
                _report_exception(worker_id, "nice loop", e, counters)
                time.sleep(0.5)
    finally:
        counters.inc("workers_alive", -1)


def hold_worker(
    worker_id: int,
    client: LeanClient,
    stop_event: threading.Event,
    counters: Counters,
    tactic_delay: float,
):
    counters.inc("workers_alive")
    try:
        while not stop_event.is_set():
            process = None
            try:
                process = client.get_process(blocking=True, timeout=60.0)
                if process is None:
                    continue
                counters.inc("processes_acquired")

                result = process.proof_from_sorry(THEOREM)
                if not result.is_success():
                    counters.inc("other_errors")
                    print(
                        f"[worker {worker_id}] proof_from_sorry failed: {result.error}",
                        flush=True,
                    )
                    process.return_process()
                    continue
                counters.inc("proofs_started")
                branch = result.value

                # Hammer tactics on this process forever (or until failure).
                for step in itertools.count():
                    if stop_event.is_set():
                        break
                    tactic = TACTICS_HOLD[step % len(TACTICS_HOLD)]
                    tac_result = branch.try_apply_tactic(tactic)
                    counters.inc("tactics")
                    if tactic_delay > 0 and stop_event.wait(tactic_delay):
                        break
                    if tac_result.is_success() and tac_result.value:
                        new_branch = tac_result.value[0]
                        if new_branch.is_solved:
                            # Start a fresh proof on the same process.
                            fresh = process.proof_from_sorry(THEOREM)
                            if not fresh.is_success():
                                break
                            counters.inc("proofs_started")
                            branch = fresh.value
                        else:
                            branch = new_branch
            except Exception as e:
                _report_exception(worker_id, "hold loop", e, counters)
                # Process is likely dead; drop it and acquire a new one.
                if process is not None:
                    try:
                        process.return_process()
                    except Exception:
                        pass
                time.sleep(0.5)
    finally:
        counters.inc("workers_alive", -1)


def status_printer(
    client: LeanClient,
    stop_event: threading.Event,
    counters: Counters,
    interval: float,
):
    started_at = time.time()
    while not stop_event.wait(interval):
        elapsed = time.time() - started_at
        snap = counters.snapshot()
        try:
            status = client.check_status()
        except Exception as e:
            print(f"[status] could not fetch /status: {e}", flush=True)
            continue

        ram = status["ram"]
        leanserver_rss = status.get("leanserver_rss_bytes")
        rss_str = (
            f"{leanserver_rss / (1024 ** 3):.2f}GB"
            if isinstance(leanserver_rss, (int, float))
            else "n/a"
        )
        tactics_per_sec = snap["tactics"] / elapsed if elapsed > 0 else 0.0

        print(
            f"[t={elapsed:6.1f}s] "
            f"procs {status['used_processes']}u/{status['available_processes']}a/"
            f"{status['starting_processes']}s/{status['max_processes']}m | "
            f"branches {status.get('total_branches', 0)} | "
            f"tactics {snap['tactics']} ({tactics_per_sec:.1f}/s) | "
            f"acquired {snap['processes_acquired']} | "
            f"proofs {snap['proofs_started']} | "
            f"RemoteExc {snap['remote_exceptions']} | "
            f"otherErr {snap['other_errors']} | "
            f"workers {snap['workers_alive']} | "
            f"RAM {ram['percent']:.1f}% | "
            f"leanserver_rss {rss_str}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=["nice", "hold"],
        help="nice: request/return processes continuously; "
             "hold: hold processes forever and hammer tactics",
    )
    parser.add_argument("--max-processes", type=int, default=8)
    parser.add_argument("--address", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--status-interval",
        type=float,
        default=1.0,
        help="Seconds between status lines.",
    )
    parser.add_argument(
        "--tactic-delay",
        type=float,
        default=0.0,
        help="Seconds to wait between tactic executions per worker.",
    )
    args = parser.parse_args()

    client = LeanClient(args.address, args.port)

    # Fail fast if the server isn't there.
    try:
        status = client.check_status()
    except Exception as e:
        print(f"Could not reach Lean server at {args.address}:{args.port}: {e}", file=sys.stderr)
        sys.exit(1)
    print(
        f"Connected. Server: {status['max_processes']} max processes, "
        f"{status['available_processes']} available, {status['used_processes']} used.",
        flush=True,
    )

    counters = Counters()
    stop_event = threading.Event()

    worker_fn = nice_worker if args.action == "nice" else hold_worker
    workers = [
        threading.Thread(
            target=worker_fn,
            args=(i, client, stop_event, counters, args.tactic_delay),
            name=f"{args.action}-worker-{i}",
            daemon=True,
        )
        for i in range(args.max_processes)
    ]
    status_thread = threading.Thread(
        target=status_printer,
        args=(client, stop_event, counters, args.status_interval),
        name="status-printer",
        daemon=True,
    )

    print(
        f"Starting {args.max_processes} '{args.action}' workers. Ctrl+C to stop.",
        flush=True,
    )
    for t in workers:
        t.start()
    status_thread.start()

    try:
        while any(t.is_alive() for t in workers):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
        stop_event.set()
        for t in workers:
            t.join(timeout=5.0)
        status_thread.join(timeout=2.0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
