from typing import TYPE_CHECKING, Any, Literal, Optional, overload

import numpy as np

try:
    from mpi4py import MPI  # type: ignore
except ImportError:
    MPI = None

from .io import from_cache, remove_cache, to_cache
from .utils import Experiment


class DistributedExperiment(Experiment):
    """
    To run experiments in parallel using MPI (via :mod:`~mpi4py`).

    Parameters:
        num_runs: Number of runs to perform.
        random_state: Random state to use for reproducibility.
    """

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
        if self.size > 1:
            self._is_distributed = True

    def broadcast[T](self, data: T) -> T:
        """
        Broadcast data to all processes.

        Ensures that all processes are synchronized before broadcasting, see also (:meth:`barrier`).

        Parameters:
            data: Data to broadcast.

        Returns:
            Broadcasted data.
        """
        if self.comm is None:
            raise RuntimeError("No MPI communicator available")
        temp_file = None
        if self.is_root:
            temp_file = to_cache(data)
        self.barrier()
        temp_file_broadcast: str = self.comm.bcast(temp_file, root=0)
        data = from_cache(temp_file_broadcast)
        self.barrier()
        remove_cache(temp_file_broadcast)
        return data

    def barrier(self) -> None:
        """Synchronize all MPI processes."""
        if self.comm is None:
            raise RuntimeError("No MPI communicator available")
        return self.comm.barrier()

    @property
    def performs_run(self) -> bool:
        """Property that checks if the current MPI process should perform a run when iterating over :class:`DistributedExperiment`."""
        if self.comm is None:
            return True
        return self.run % self.size == self.rank

    @property
    def is_root(self) -> bool:
        """Property that checks if the current MPI process is the root process."""
        return self.rank == 0

    def add(self, data: Any) -> None:
        """Add data of the current run to the distributed experiment. Note that this should be only used once per run."""
        if self.comm is None or self.is_root:
            self.data[self.run] = data
            return
        self.comm.send(data, dest=0, tag=self.run)

    @overload
    def collect(self, broadcast: Literal[True] = True) -> dict[int, Any]: ...

    @overload
    def collect(self, broadcast: bool) -> dict[int, Any] | None: ...

    def collect(self, broadcast: bool = True) -> dict[int, Any] | None:
        """
        Collect data from all MPI processes and return a sorted dictionary of run data.

        Parameters:
            broadcast (bool): Whether to broadcast the collected data to all processes.

        Returns:
            A sorted dictionary of run data if the current process is the root, otherwise None.
        """
        if self.comm is None:
            return self.data
        data = None
        if self.is_root:
            for run in range(1, self.num_runs):
                rank = run % self.size
                if rank == self.rank:
                    continue
                self.data[run] = self.comm.recv(source=rank, tag=run)
            data = dict(sorted(self.data.items()))
        if broadcast:
            data = self.broadcast(data)
            if TYPE_CHECKING:
                assert data is not None
        return data
