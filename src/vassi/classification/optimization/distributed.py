from typing import final


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
