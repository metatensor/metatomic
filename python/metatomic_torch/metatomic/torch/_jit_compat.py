import functools


def _identity_decorator(*args, **kwargs):
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]

    def decorator(func):
        return func

    return decorator


try:
    import numba as _numba
except ImportError:  # pragma: no cover
    jit = _identity_decorator
else:
    jit = functools.partial(_numba.njit, cache=True)
