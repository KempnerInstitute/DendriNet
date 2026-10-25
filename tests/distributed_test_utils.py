"""Bound distributed test workers and reap them even if pytest interrupts."""

import faulthandler
import time

import torch.multiprocessing as mp


def _run_with_tracebacks(rank, worker, args):
    # A collective timeout cannot diagnose a hang outside a collective.
    faulthandler.dump_traceback_later(120, repeat=True)
    try:
        worker(rank, *args)
    finally:
        faulthandler.cancel_dump_traceback_later()


def join_workers(context, *, timeout=300):
    """Propagate worker errors and stop every child on timeout/interruption."""
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Distributed test workers exceeded {timeout}s")
            if context.join(timeout=min(1, remaining)):
                return
    finally:
        # pytest-timeout raises in the parent; mp.spawn(join=True) does not
        # then clean up its children, which can keep pytest alive at exit.
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
        for process in context.processes:
            if process.is_alive():
                process.kill()
        for process in context.processes:
            process.join(timeout=5)


def spawn_workers(worker, *, args=(), nprocs=2, timeout=300):
    """Run spawned workers with child stack dumps and a parent deadline."""
    context = mp.spawn(
        _run_with_tracebacks, args=(worker, args), nprocs=nprocs, join=False
    )
    join_workers(context, timeout=timeout)
