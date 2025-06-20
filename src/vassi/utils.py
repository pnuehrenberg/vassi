from __future__ import annotations

import hashlib
import json
from multiprocessing import cpu_count
from typing import Any, Iterable, Optional, Protocol, Self

import numpy as np
from joblib.parallel import get_active_backend

Keypoint = int
KeypointPair = tuple[Keypoint, Keypoint]
Keypoints = Iterable[Keypoint]
KeypointPairs = Iterable[KeypointPair]


def to_scalars(values: Any) -> list:
    # https://stackoverflow.com/questions/9452775/converting-numpy-dtypes-to-native-python-types
    return getattr(values, "tolist", lambda: values)()


def available_resources(*, ensure_parallel_inner: bool = True) -> tuple[int, int]:
    nesting_level = get_active_backend()[0].nesting_level  # type: ignore
    num_cpus = cpu_count()
    if nesting_level > 0:
        num_cpus = num_cpus // (nesting_level * 4)
    num_inner_threads = max(1, num_cpus // 4)
    if ensure_parallel_inner and num_inner_threads == 1:
        num_jobs = 1
        num_inner_threads = num_cpus
    else:
        num_jobs = num_cpus // num_inner_threads
    return num_jobs, num_inner_threads


class Experiment:
    def __init__(
        self, num_runs: int, *, random_state: Optional[np.random.Generator | int] = None
    ):
        self._is_distributed = False
        self._num_runs = num_runs
        self._run = None
        self._seed = to_int_seed(np.random.default_rng(random_state))
        self._random_state = None
        self.data = {}

    def broadcast[T](self, data: T) -> T:
        return data

    def barrier(self) -> None:
        return

    @property
    def is_distributed(self) -> bool:
        return self._is_distributed

    @property
    def num_runs(self) -> int:
        return self._num_runs

    def get_random_state(self, run: int, *, num_runs: int) -> np.random.Generator:
        random_state = np.random.default_rng(self._seed)
        return random_state.spawn(num_runs)[run]

    def __iter__(self) -> Self:
        self._run = -1
        self._random_state = None
        return self

    def __next__(self) -> int:
        self._random_state = None
        if self._run is None:
            raise ValueError("not in experiment loop")
        self._run += 1
        if self._run >= self.num_runs:
            raise StopIteration
        while not self.performs_run:
            return next(self)
        return self._run

    @property
    def run(self) -> int:
        if self._run is None or self._run < 0 or self._run >= self.num_runs:
            raise ValueError("property only accessible within experiment loop")
        return self._run

    @property
    def random_state(self) -> np.random.Generator:
        if self._random_state is not None:
            return self._random_state
        self._random_state = self.get_random_state(self.run, num_runs=self.num_runs)
        return self._random_state

    @property
    def performs_run(self) -> bool:
        _ = self.run
        return True

    @property
    def is_root(self) -> bool:
        return True

    def add(self, data: Any) -> None:
        self.data[self.run] = data

    def collect(self) -> dict[int, Any]:
        return dict(sorted(self.data.items()))


def class_name(obj: object) -> str:
    if isinstance(obj, type):
        return obj.__name__
    return obj.__class__.__name__


def hash_dict(dictionary: dict) -> str:
    """
    Hashes a dictionary to a SHA1 hexadecimal string.

    Parameters
    ----------
    dictionary : dict
        The dictionary to hash. All keys and values must be json-serializable.

    Returns
    -------
    str
        The SHA1 hexadecimal hash of the dictionary.
    """
    return hashlib.sha1(
        json.dumps(dictionary, sort_keys=True).encode("utf-8")
    ).hexdigest()


class ToArray(Protocol):
    def __call__(self, *args, **kwargs) -> np.ndarray: ...


class ArrayToArray(Protocol):
    __name__: str  # named for sliding window aggregation (alternatively, use a dictionary naming scheme as implemented for the fature extractor)

    def __call__(self, array: np.ndarray, *args, **kwargs) -> np.ndarray: ...


class PairedArraysToArray(Protocol):
    def __call__(
        self, array_1: np.ndarray, array_2: np.ndarray, /, *args, **kwargs
    ) -> np.ndarray: ...


class SmoothingFunction(Protocol):
    def __call__(self, *args, array: np.ndarray, **kwargs) -> np.ndarray: ...


def flatten(array: np.ndarray, ensure_2d: bool = True) -> np.ndarray:
    if array.ndim < 2:
        raise ValueError("Can only flatten arrays with at least 2 dimensions.")
    if array.ndim == 2:
        return array
    if array.ndim == 3 or ensure_2d:
        return array.reshape(array.shape[0], -1)
    # keep last dim
    return array.reshape(array.shape[0], -1, array.shape[-1])


def perform_operation(
    operation_func: PairedArraysToArray,
    array_1: np.ndarray,
    array_2: np.ndarray,
    element_wise: bool,
    flat: bool,
    expand_dims_for_broadcasting: bool = True,
):
    if (
        not element_wise
        and expand_dims_for_broadcasting
        and array_1.ndim == array_2.ndim
    ):
        array_1 = np.expand_dims(array_1, axis=2)
        array_2 = np.expand_dims(array_2, axis=1)
    if not element_wise and array_1.ndim != array_2.ndim:
        raise NotImplementedError(
            f"Broadcasting not implemented for inputs of shapes {array_1.shape} and {array_2.shape}."
        )
    try:
        np.broadcast_shapes(array_1.shape, array_2.shape)
    except ValueError:
        raise ValueError(
            f"Can not broadcast inputs of shapes {array_1.shape} and {array_2.shape}."
        )
    result = operation_func(array_1, array_2)
    if flat:
        return flatten(result)
    return result


def pad_values(array: np.ndarray, step: int, value: int | float | str) -> np.ndarray:
    if value == "same":
        value = array[step]
    if step >= 0:
        array[:step] = value
    if step < 0:
        array[-step:] = value
    return array


def to_int_seed(random_state: np.random.Generator) -> int:
    # max int for sklearn random_state as int
    return int(random_state.integers(4294967295))


def closest_odd_divisible(number: float, divisor: int) -> int:
    if divisor % 2 == 0:
        raise ValueError("Closest odd number can not have an even divisor.")
    remainder = number % divisor
    closest_smaller_number = number - remainder
    while closest_smaller_number % 2 == 0:
        closest_smaller_number -= divisor
    closest_greater_number = (number + divisor) - remainder
    while closest_greater_number % 2 == 0:
        closest_greater_number += divisor
    if (number - closest_smaller_number) > (closest_greater_number - number):
        return int(closest_greater_number)
    return int(closest_smaller_number)
