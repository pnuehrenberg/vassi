import multiprocessing
import os
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import AbstractContextManager, contextmanager, nullcontext
from multiprocessing.context import SpawnContext

# from multiprocessing.managers import SyncManager
from typing import final

from .warnings import warn

_ctx = None
_semaphore = None
_n_jobs = None
_manager = None


@final
class Environment:
    def __init__(self) -> None:
        self._comm = None
        self._rank = 0
        self._size = 1

        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            if comm.Get_size() > 1:
                self._comm = comm
                self._rank = comm.Get_rank()
                self._size = comm.Get_size()
        except ImportError:
            pass

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_root(self) -> bool:
        return self._rank == 0

    def barrier(self) -> None:
        if self._comm is not None:
            self._comm.Barrier()

    def bcast[T](self, data: T, root: int = 0) -> T:
        if self._comm is not None:
            return self._comm.bcast(data, root=root)
        return data


def set_process_state(
    context: SpawnContext,
    semaphore: AbstractContextManager[object],
    n_jobs: int,
):
    global _ctx, _semaphore, _n_jobs
    _ctx = context
    _semaphore = semaphore
    _n_jobs = n_jobs


def get_process_state() -> AbstractContextManager[object]:
    # only semaphore is relevant for the worker
    global _semaphore
    return _semaphore if _semaphore else nullcontext()


def set_or_load_context(
    n_jobs: int,
) -> tuple[SpawnContext, AbstractContextManager[object], int]:
    global _ctx, _semaphore, _n_jobs

    if n_jobs == -1:
        if _n_jobs is None:
            n_jobs = cpu_count if (cpu_count := os.cpu_count()) else 1
        else:
            n_jobs = _n_jobs
    if _semaphore is not None:
        if _ctx is None or _n_jobs is None:
            raise RuntimeError("Cannot use context when context or n_jobs is not set.")
        if n_jobs != _n_jobs:
            warn(
                f"Reusing existing context with n_jobs={_n_jobs}. Ignoring request for new limit of n_jobs={n_jobs}."
            )
        return _ctx, _semaphore, _n_jobs
    if _ctx is not None or _n_jobs is not None:
        raise RuntimeError(
            "Cannot set context when context, manager, semaphore or n_jobs is already set."
        )
    _ctx = multiprocessing.get_context("spawn")
    _manager = _ctx.Manager()
    _semaphore = _manager.Semaphore(n_jobs)
    _n_jobs = n_jobs

    return _ctx, _semaphore, _n_jobs


@contextmanager
def limited_process_pool(n_jobs: int | None = None) -> Iterator[ProcessPoolExecutor]:
    """
    A robust Context Manager for nested ProcessPools.

    1. Detects if it's the Root process or a Nested process.
    2. Creates or Reuses a Global Semaphore.
    3. Configures the Pool to inject this semaphore into children.

    Usage:
        with limited_process_pool(n_jobs=4) as executor:
            executor.submit(func, args)
    """
    if n_jobs is None:
        n_jobs = -1
    # 1. Resolve Infrastructure
    context, semaphore, n_jobs = set_or_load_context(n_jobs)

    # 2. Setup the Executor
    # We pass the semaphore down to children via '_initializer'
    executor = ProcessPoolExecutor(
        max_workers=n_jobs,
        mp_context=context,
        initializer=set_process_state,
        initargs=(context, semaphore, n_jobs),
    )

    try:
        yield executor
    except Exception as e:
        executor.shutdown(wait=True)
        raise e
    finally:
        executor.shutdown(wait=True)
