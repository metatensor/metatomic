import functools


def _identity_decorator(func):
    return func


try:
    import numba as _numba
except ImportError:  # pragma: no cover
    jit = _identity_decorator
else:
    jit = functools.partial(_numba.njit, cache=True)
