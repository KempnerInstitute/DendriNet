"""Opt-in performance profiling for the training loop (Rung 1: torch.profiler).

Everything in this module is gated by the ``DENDRINET_PROFILE`` environment
variable. When it is unset / ``0`` / ``false``, every helper here is a no-op and
training behaves byte-for-byte as before. This keeps profiling strictly opt-in
and guarantees it cannot change numerics or control flow.

Enable with::

    DENDRINET_PROFILE=1 python -m dendritic_modeling.scripts.training.train_experiments <cfg>
    # or pass --profile to the training entrypoint

Optional knobs (env vars):
    DENDRINET_PROFILE_DIR    output dir for traces        (default ./profiler_logs)
    DENDRINET_PROFILE_WAIT   schedule wait steps          (default 1)
    DENDRINET_PROFILE_WARMUP schedule warmup steps        (default 1)
    DENDRINET_PROFILE_ACTIVE schedule active steps        (default 3)

The schedule profiles ``active`` steps after ``wait + warmup`` warmup steps,
then the trace is flushed and the surrounding training run is asked to stop
early (see ``Trainer._epoch_train``) so a profiling job stays short.

Outputs (under ``DENDRINET_PROFILE_DIR``):
    * a TensorBoard trace (``tensorboard --logdir <dir>``)
    * ``chrome_trace.json`` (open at ``chrome://tracing`` / ``edge://tracing``)
    * a per-operator CPU/GPU summary table printed to the log
"""

import contextlib
import logging
import os

import torch

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}


def profiling_enabled() -> bool:
    """True iff DENDRINET_PROFILE is set to a truthy value."""
    return os.environ.get("DENDRINET_PROFILE", "").strip().lower() in _TRUTHY


def profile_backend() -> str:
    """Profiling backend: 'torch' (default, Rung 1) or 'nsys'/'nvtx' (Rung 2).

    Both backends emit NVTX ranges (via record_section) and stop after a few
    steps. Only the 'torch' backend additionally starts torch.profiler. Under
    Nsight Systems we use 'nsys' so torch.profiler's CUPTI use does not conflict
    with nsys's own CUPTI capture.
    """
    return os.environ.get("DENDRINET_PROFILE_BACKEND", "torch").strip().lower()


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _schedule_steps() -> tuple[int, int, int]:
    wait = _env_int("DENDRINET_PROFILE_WAIT", 1)
    warmup = _env_int("DENDRINET_PROFILE_WARMUP", 1)
    active = max(1, _env_int("DENDRINET_PROFILE_ACTIVE", 3))
    return wait, warmup, active


def total_profiled_steps() -> int:
    """Number of training steps to run before stopping the capture.

    One extra ``step()`` past the active window is needed for the schedule to
    fire ``on_trace_ready`` and flush the trace.
    """
    wait, warmup, active = _schedule_steps()
    return wait + warmup + active + 1


def _profile_dir() -> str:
    d = os.environ.get("DENDRINET_PROFILE_DIR", "./profiler_logs")
    os.makedirs(d, exist_ok=True)
    return d


def record_section(name: str):
    """Annotate a region of the step for the profiler timeline.

    Emits both a ``torch.profiler.record_function`` range (shows up in the
    torch.profiler / TensorBoard trace, Rung 1) and an NVTX range (shows up in
    Nsight Systems, Rung 2). Returns a no-op context manager when profiling is
    disabled, so it is free when off.
    """
    if not profiling_enabled():
        return contextlib.nullcontext()
    return _Section(name)


class _Section:
    """Paired torch.profiler + NVTX range for a named step phase."""

    def __init__(self, name: str):
        self.name = name
        self._use_nvtx = torch.cuda.is_available()
        self._rf = torch.profiler.record_function(name)

    def __enter__(self):
        if self._use_nvtx:
            torch.cuda.nvtx.range_push(self.name)
        self._rf.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._rf.__exit__(exc_type, exc, tb)
        if self._use_nvtx:
            torch.cuda.nvtx.range_pop()
        return False


def make_profiler():
    """Return a started ``torch.profiler.profile`` instance, or ``None``.

    Returns ``None`` when profiling is disabled. The caller is responsible for
    calling ``prof.step()`` once per training step and ``finish_profiler(prof)``
    when done.
    """
    if not profiling_enabled():
        return None
    if profile_backend() != "torch":
        # nsys/nvtx mode: NVTX ranges + early-stop stay active (driven by the
        # caller), but do NOT start torch.profiler — its CUPTI use conflicts
        # with Nsight Systems' own capture.
        logger.info(
            "[profiler] backend=%s -> torch.profiler disabled; NVTX ranges + "
            "early-stop remain active for Nsight Systems.",
            profile_backend(),
        )
        return None

    wait, warmup, active = _schedule_steps()
    logdir = _profile_dir()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    tb_handler = torch.profiler.tensorboard_trace_handler(logdir)

    def _on_ready(prof):
        tb_handler(prof)
        chrome_path = os.path.join(logdir, "chrome_trace.json")
        try:
            prof.export_chrome_trace(chrome_path)
            logger.info("[profiler] wrote chrome trace -> %s", chrome_path)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("[profiler] chrome trace export failed: %s", exc)

    prof = torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=1
        ),
        on_trace_ready=_on_ready,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )
    prof.start()
    logger.info(
        "[profiler] torch.profiler ENABLED -> %s "
        "(schedule wait=%d warmup=%d active=%d, activities=%s)",
        logdir,
        wait,
        warmup,
        active,
        [a.name for a in activities],
    )
    return prof


def finish_profiler(prof) -> None:
    """Stop the profiler and log a per-operator CPU/GPU summary table."""
    if prof is None:
        return
    try:
        prof.stop()
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("[profiler] stop() failed: %s", exc)
        return
    sort_key = (
        "self_cuda_time_total"
        if torch.cuda.is_available()
        else "self_cpu_time_total"
    )
    try:
        table = prof.key_averages().table(sort_by=sort_key, row_limit=25)
        logger.info("[profiler] top ops by %s:\n%s", sort_key, table)
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("[profiler] could not build summary table: %s", exc)
