from typing import Any, Optional

import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from .utils import Experiment


class DistributedExperiment(Experiment):
    def __init__(
        self, num_runs: int, *, random_state: Optional[np.random.Generator | int] = None
    ):
        super().__init__(num_runs, random_state=random_state)
        self.data = {}
        self.comm = None
        self.rank = 0
        self.size = 1
        if MPI is None:
            return
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    @property
    def performs_run(self) -> bool:
        if self.comm is None:
            return True
        return self.run % self.size == self.rank

    @property
    def is_root(self) -> bool:
        return self.rank == 0

    def add(self, data: Any) -> None:
        if self.comm is None or self.is_root:
            self.data[self.run] = data
            return
        self.comm.send(data, dest=0, tag=self.run)

    def collect(self) -> dict[int, Any]:
        if self.comm is None:
            return self.data
        if not self.is_root:
            raise RuntimeError("collect should only be called from root mpi process")
        for run in range(1, self.num_runs):
            rank = run % self.size
            if rank == self.rank:
                continue
            self.data[run] = self.comm.recv(source=rank, tag=run)
        return dict(sorted(self.data.items()))
