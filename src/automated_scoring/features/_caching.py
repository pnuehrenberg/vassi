import os
import pickle
from typing import TYPE_CHECKING, Any, Callable

from ..data_structures import Trajectory
from ..utils import hash_dict

if TYPE_CHECKING:
    from .feature_extractor import BaseExtractor


def _to_cache(obj: Any, cache_file: str) -> None:
    """
    Helper function to write an object to a cache file using pickle.

    Parameters
    ----------
    obj : Any
        The object to write.
    cache_file : str
        The path to the cache file.
    """
    # TODO writing should catch keyboardinterrupt to avoid corrupted files
    with open(cache_file, "wb") as cached:
        pickle.dump(obj, cached)


def _from_cache(cache_file: str):
    """
    Helper function to read an object from a cache file using pickle.

    Parameters
    ----------
    cache_file : str
        The path to the cache file.

    Returns
    -------
    Any
        The object read from the cache file.

    Raises
    ------
    FileNotFoundError
        If the cache file does not exist.
    """
    if not os.path.isfile(cache_file):
        raise FileNotFoundError
    # TODO or explicitly fail here and so that value gets recomputed
    with open(cache_file, "rb") as cached:
        return pickle.load(cached)


def _hash_args(*args, **kwargs) -> str:
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
    Decorator to cache the result of a method implemented by the BaseExtractor class.

    Parameters
    ----------
    func : Callable
        The method to cache.

    Returns
    -------
    Callable
        The decorated method.
    """

    def _cache(*args: P.args, **kwargs: P.kwargs) -> T:
        extractor = args[0]
        if TYPE_CHECKING:
            assert isinstance(extractor, BaseExtractor)
        if not extractor.cache:
            return func(*args, **kwargs)
        hash_value = _hash_args(*args, **kwargs)
        if TYPE_CHECKING:
            assert isinstance(extractor.cache_directory, str)
        cache_file = os.path.join(extractor.cache_directory, hash_value)
        if extractor.cache:
            try:
                return _from_cache(cache_file)
            except FileNotFoundError:
                pass
        value = func(*args, **kwargs)
        if not extractor.cache:
            return value
        _to_cache(value, cache_file)
        return value

    return _cache


# def _cache(extractor: "BaseExtractor", *args, func: Callable, **kwargs):
#     if not extractor.cache:
#         return func(extractor, *args, **kwargs)
#     hash_value = _hash_args(extractor, *args, **kwargs)
#     cache_file = os.path.join(extractor.cache_directory, hash_value)
#     try:
#         return _from_cache(cache_file)
#     except FileNotFoundError:
#         pass
#     value = func(extractor, *args, **kwargs)
#     _to_cache(value, cache_file)
#     return value


# def cache[**P, T](func: Callable[P, T]) -> Callable[P, T]:
#     """
#     Decorator to cache the result of a method implemented by the BaseExtractor class.

#     Parameters
#     ----------
#     func : Callable
#         The method to cache.

#     Returns
#     -------
#     Callable
#         The decorated method.
#     """

#     result_func = functools.partial(_cache, func=func)
#     decorated = functools.wraps(func)(result_func)
#     return decorated
