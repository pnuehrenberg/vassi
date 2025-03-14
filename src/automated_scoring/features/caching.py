import os
import pickle
from typing import TYPE_CHECKING, Any, Callable

from ..data_structures import Trajectory
from ..utils import hash_dict

if TYPE_CHECKING:
    from .feature_extractor import BaseExtractor


def to_cache(obj: Any, cache_file: str) -> None:
    """
    Helper function to write an object to a cache file using pickle.

    Args:
        obj: The object to write.
        cache_file: The path to the cache file.
    """
    with open(cache_file, "wb") as cached:
        pickle.dump(obj, cached)


def from_cache(cache_file: str):
    """
    Helper function to read an object from a cache file using pickle.

    Args:
        cache_file: The path to the cache file.
    """
    with open(cache_file, "rb") as cached:
        return pickle.load(cached)


def hash_args(*args, **kwargs) -> str:
    """
    Helper function to hash the arguments of a function.
    The first argument should be a :class:`~automated_scoring.features.feature_extractor.BaseExtractor` instance.

    Args:
        *args: The positional arguments.
        **kwargs: The keyword arguments.

    Returns:
        str: The hash of the arguments.
    """
    def to_hash_string(arg):
        if arg is None:
            return "none"
        if isinstance(arg, str):
            return arg
        if isinstance(arg, Trajectory):
            return arg.sha1
        raise NotImplementedError("invalid argument type")

    extractor, args = args[0], args[1:]
    d = {"extractor": extractor.sha1}
    for idx, arg in enumerate(args):
        d[f"arg_{idx}"] = to_hash_string(arg)
    for key, value in kwargs.items():
        d[key] = to_hash_string(value)
    return hash_dict(d)


def cache[**P, T](func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator to cache the result of a method implemented by :class:`~automated_scoring.features.feature_extractor.BaseExtractor`.

    Args:
        func: The method to cache.
    """

    def _cache(*args: P.args, **kwargs: P.kwargs) -> T:
        extractor = args[0]
        if TYPE_CHECKING:
            assert isinstance(extractor, BaseExtractor)
        if not extractor.cache:
            return func(*args, **kwargs)
        hash_value = hash_args(*args, **kwargs)
        if TYPE_CHECKING:
            assert isinstance(extractor.cache_directory, str)
        cache_file = os.path.join(extractor.cache_directory, hash_value)
        if extractor.cache:
            try:
                return from_cache(cache_file)
            except FileNotFoundError:
                pass
        value = func(*args, **kwargs)
        if not extractor.cache:
            return value
        to_cache(value, cache_file)
        return value

    return _cache
