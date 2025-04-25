from __future__ import annotations

import functools
import sys
from collections.abc import Callable, Generator, Iterable, Sized
from contextlib import contextmanager
from copy import deepcopy
from time import perf_counter
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    cast,
    overload,
)

import loguru
from loguru import logger

Keypoint = int
KeypointPair = tuple[Keypoint, Keypoint]
Keypoints = Iterable[Keypoint]
KeypointPairs = Iterable[KeypointPair]


LOG_LEVEL = Literal["trace", "debug", "info", "success", "warning", "error"]


def _formatter(record):
    def get_loops_fmt():
        nonlocal record

        def get_loop_fmt(loop):
            nonlocal record
            info = f"extra[loops][{loop}]"
            values = record["extra"]["loops"][loop]
            has_name = isinstance(values["name"], str) and len(values["name"]) > 0
            has_total = values["total"] is not None
            if has_name and has_total:
                return f"[{{{info}[name]}}: {{{info}[step]}}/{{{info}[total]}}]"
            if has_name and not has_total:
                return f"[{{{info}[name]}}: {{{info}[step]}}]"
            if has_total:
                return f"[{{{info}[step]}}/{{{info}[total]}}]"
            return f"[{{{info}[step]}}]"

        if "loops" not in record["extra"]:
            return " "
        return (
            " "
            + " ".join([get_loop_fmt(loop) for loop in record["extra"]["loops"]])
            + " "
        )

    return f"<green>{{time:YYYY-MM-DD HH:mm:ss.SSS}}</green> <level>[{{level: <8}}]{get_loops_fmt()}{{message}}</level>\n"


def _create_log_in_subprocess(
    options: dict[str, Any],
    level: int,
    *,
    sink=None,
    format: str | Callable[..., str] = _formatter,
    enqueue: bool = True,
) -> loguru.Logger:
    log = set_logging_level(level, sink=sink, format=format, enqueue=enqueue)
    log._options = options  # type: ignore
    return log


def _prepare_log_for_subprocess(log: loguru.Logger) -> tuple[dict[str, Any], int]:
    return deepcopy(log._options), log._core.min_level  # type: ignore


def set_logging_level(
    level: LOG_LEVEL | int = "warning",
    *,
    sink=None,
    format: str | Callable[..., str] = _formatter,
    enqueue: bool = True,
) -> loguru.Logger:
    """Set the logging level (and sink, format and enqueue parameters of the loguru logger."""
    global logger
    if sink is None:
        sink = sys.stdout
    logger.remove()
    logger.add(
        sink=sink,
        level=level.upper() if isinstance(level, str) else level,
        format=format,
        enqueue=enqueue,
    )
    return logger


# def _mydeco[**P, T](func: Callable[P, T], *, additional_kwarg: int = 1) -> Callable[P, T]:
#     @functools.wraps(func)
#     def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
#         return func(*args, **kwargs)
#     return wrapper


# def _mydeco_factory[**P, T](func: Optional[Callable[P, T]] = None, *, additional_kwarg: int = 2) -> Callable[P, T]:
#     if func is None:
#         decorated = functools.partial(_mydeco, additional_kwarg=additional_kwarg)
#     else:
#         decorated = functools.partial(_mydeco, additional_kwarg=additional_kwarg)(func)
#     if TYPE_CHECKING:
#         decorated = cast(Callable[P, T], decorated)
#     return decorated


@contextmanager
def catch_time() -> Generator[Callable[[], float], None, None]:
    start = perf_counter()
    yield lambda: perf_counter() - start


def _log_time[T, **P](
    func: Callable[P, T],
    level_start: LOG_LEVEL,
    level_finish: LOG_LEVEL,
    description: str,
) -> Callable[P, T]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        nonlocal description
        description = description.strip()
        if len(description) > 0:
            description = " " + description
        if "log" not in kwargs:
            log = None
        else:
            log = kwargs.pop("log")
        if log is None:
            log = set_logging_level()
        if TYPE_CHECKING:
            assert isinstance(log, loguru.Logger)
        kwargs["log"] = log
        getattr(log, level_start)(f"started{description}")
        with catch_time() as get_time:
            result = func(*args, **kwargs)
        getattr(log, level_finish)(f"finished{description} in {get_time():.2f} seconds")
        return result

    return wrapper


def log_time[**P, T](
    func: Optional[Callable[P, T]] = None,
    *,
    level_start: LOG_LEVEL = "info",
    level_finish: LOG_LEVEL = "info",
    description: str = "",
) -> Callable[P, T]:
    if func is None:
        decorated = functools.partial(
            _log_time,
            level_start=level_start,
            level_finish=level_finish,
            description=description,
        )
    else:
        decorated = functools.partial(
            _log_time,
            level_start=level_start,
            level_finish=level_finish,
            description=description,
        )(func)
    if TYPE_CHECKING:
        # cast because functools.wraps returns _Wrapped
        decorated = cast(Callable[P, T], decorated)
    return decorated


def _get_extra(log: loguru.Logger) -> dict:
    extra = deepcopy(log._options[-1])  # type: ignore
    return extra


def increment_loop(log: loguru.Logger, *, name: int | str) -> loguru.Logger:
    extra = _get_extra(log)
    loops = extra["loops"]
    loops[name]["step"] += 1
    extra["loops"] = loops
    log._options = (*log._options[:-1], extra)  # type: ignore
    return log


@overload
def with_loop[T: int | str](
    log: loguru.Logger,
    *,
    name: Optional[T] = None,
    step: int,
    total: Optional[int] = None,
    prepare_for_subprocess: Literal[False] = False,
) -> tuple[loguru.Logger, T]: ...


@overload
def with_loop[T: int | str](
    log: loguru.Logger,
    *,
    name: Optional[T] = None,
    step: int,
    total: Optional[int] = None,
    prepare_for_subprocess: bool,
) -> tuple[tuple[dict[str, Any], int], T]: ...


def with_loop[T: int | str](
    log: loguru.Logger,
    *,
    name: Optional[T] = None,
    step: int,
    total: Optional[int] = None,
    prepare_for_subprocess: bool = False,
) -> tuple[loguru.Logger, T] | tuple[tuple[dict[str, Any], int], T]:
    if isinstance(name, str) and len(name) == 0:
        raise ValueError("name should be non-empty string (or int or None)")
    extra = _get_extra(log)
    if "loops" not in extra:
        loops = {}
    else:
        loops = extra["loops"]
    unnamed_loops = [level for level in loops if isinstance(level, int)]
    if name is None and len(unnamed_loops) > 0:
        loop = max(unnamed_loops) + 1
    elif name is None:
        loop = 0
    else:
        loop = name
    loops[loop] = {
        "name": loop if isinstance(loop, str) else "",
        "step": step,
        "total": total,
    }
    extra["loops"] = loops
    if TYPE_CHECKING:
        loop = cast(T, loop)
    log = log.bind(**extra)
    if not prepare_for_subprocess:
        return log, loop
    return _prepare_log_for_subprocess(log), loop


def log_loop[T](
    iterable: Iterable[T],
    *,
    level: LOG_LEVEL,
    message: str,
    name: Optional[str] = None,
    total: Optional[int] = None,
    log: Optional[loguru.Logger] = None,
) -> Generator[tuple[loguru.Logger, T], None, None]:
    if log is None:
        log = set_logging_level()
    if total is None and isinstance(iterable, Sized):
        total = len(iterable)
    _log, loop = with_loop(log, name=name, step=0, total=total)
    for idx, element in enumerate(iterable):
        if idx > 0:
            getattr(_log, level)(message)
        yield _log, element
        _log = increment_loop(_log, name=loop)
    getattr(_log, level)(message)
