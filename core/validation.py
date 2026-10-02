"""Actionable input validation. Never repair malformed configuration silently."""
import math
from numbers import Real


def number(value, label, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label}: expected a number, received {type(value).__name__}. "
                         "Check the connected config/reader socket; text is not a numeric setting.")
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{label}: expected a finite number from {minimum} to {maximum}; got {value!r}.")
    return value


def integer(value, label, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{label}: expected an integer from {minimum} to {maximum}; got {value!r}.")
    return value


def boolean(value, label):
    if not isinstance(value, bool):
        raise ValueError(f"{label}: expected a boolean, received {type(value).__name__}.")
    return value


def text(value, label):
    if not isinstance(value, str):
        raise ValueError(f"{label}: expected text, received {type(value).__name__}. "
                         "Connect a STRING reader, not a FLOAT/INT reader.")
    return value


def choice(value, label, choices):
    text(value, label)
    if value not in choices:
        raise ValueError(f"{label}: {value!r} is not supported. Choose one of: {', '.join(choices)}.")
    return value
