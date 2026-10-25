"""The distributed test harness must not leave live children after failure."""

import time

import pytest
import torch.multiprocessing as mp
from distributed_test_utils import join_workers

pytestmark = [pytest.mark.slow, pytest.mark.multiprocess]


def _wait_forever(rank):
    while True:
        time.sleep(1)


def _fail(rank):
    raise RuntimeError("intentional worker failure")


def test_timeout_reaps_workers():
    context = mp.spawn(_wait_forever, nprocs=2, join=False)
    with pytest.raises(TimeoutError, match="Distributed test workers exceeded"):
        join_workers(context, timeout=0.1)
    assert all(not process.is_alive() for process in context.processes)
    assert all(process.exitcode is not None for process in context.processes)


def test_parent_interruption_reaps_workers(monkeypatch):
    context = mp.spawn(_wait_forever, nprocs=2, join=False)

    def interrupt(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(context, "join", interrupt)
    with pytest.raises(KeyboardInterrupt):
        join_workers(context)
    assert all(not process.is_alive() for process in context.processes)
    assert all(process.exitcode is not None for process in context.processes)


def test_worker_exception_is_preserved():
    context = mp.spawn(_fail, nprocs=2, join=False)
    with pytest.raises(mp.ProcessRaisedException, match="intentional worker failure"):
        join_workers(context)
    assert all(not process.is_alive() for process in context.processes)
