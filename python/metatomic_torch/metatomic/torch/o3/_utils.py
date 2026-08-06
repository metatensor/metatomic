"""Shared helpers for O(3) transformations."""

from numbers import Integral

import torch
from metatensor.torch import TensorMap


def copy_tensormap_info(source: TensorMap, result: TensorMap) -> TensorMap:
    """Copy global information from ``source`` to ``result``."""
    for info_name, info_value in source.info().items():
        result.set_info(info_name, info_value)
    return result


def validate_integer(name: str, value: int, minimum: int) -> int:
    """Check that ``value`` is an integer at least ``minimum``.

    Return it as a Python ``int``.
    """
    if torch.jit.is_scripting():
        integer_value = value
    else:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
        integer_value = int(value)
    if integer_value < minimum:
        if minimum == 0:
            qualifier = "non-negative"
        elif minimum == 1:
            qualifier = "positive"
        else:
            qualifier = f"larger or equal to {minimum}"
        raise ValueError(f"{name} must be {qualifier}, got {integer_value}")
    return integer_value
