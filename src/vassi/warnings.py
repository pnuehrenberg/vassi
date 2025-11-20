import warnings


def warning_on_one_line(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    *_: str | None,
) -> str:
    return "%s:%s: %s: %s\n" % (
        filename,
        lineno,
        category.__name__,
        message,
    )


warnings.formatwarning = warning_on_one_line
warn = warnings.warn

__all__ = ["warn"]
