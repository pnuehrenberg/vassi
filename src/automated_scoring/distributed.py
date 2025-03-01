from typing import Any, Optional

import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from .utils import Experiment, ensure_generator, to_int_seed


class DistributedExperiment(Experiment):
    def __init__(self, random_state: Optional[np.random.Generator | int] = None):
        self.seed = to_int_seed(ensure_generator(random_state))
        self.data = {}
        self.comm = None
        self.rank = 0
        self.size = 1
        if MPI is None:
            return
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def performs_run(self, run: int) -> bool:
        if self.comm is None:
            return True
        return run % self.size == self.rank

    def get_random_state(self, run: int, *, num_runs: int) -> np.random.Generator:
        random_state = ensure_generator(self.seed)
        return random_state.spawn(num_runs)[run]

    @property
    def is_root(self) -> bool:
        return self.rank == 0

    def add(self, run: int, data: Any) -> None:
        if self.comm is None or self.is_root:
            self.data[run] = data
            return
        self.comm.send(data, dest=0, tag=run)

    def collect(self, *, num_runs: int) -> dict[int, Any]:
        if self.comm is None:
            return self.data
        if not self.is_root:
            raise RuntimeError("collect should only be called from root mpi process")
        for run in range(1, num_runs):
            rank = run % self.size
            if rank == self.rank:
                continue
            self.data[run] = self.comm.recv(source=rank, tag=run)
        return {run: self.data[run] for run in sorted(self.data)}
