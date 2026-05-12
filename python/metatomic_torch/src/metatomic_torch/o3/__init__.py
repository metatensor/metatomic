"""
This module provides utilities for O(3) symmetries (rotations and improper rotations)
management.

There are tools to augment training data with randomly rotated copies of
structures/targets, tools to convert :py:class:`metatensor.torch.TensorMap` between
Cartesian and spherical forms, and between uncoupled and coupled spherical forms.

See :ref:`o3-conventions` for the naming conventions used to identify Cartesian and
spherical components in a :py:class:`~metatensor.torch.TensorBlock`.
"""

from ._cartesian_spherical import cartesian_to_spherical, spherical_to_cartesian
from ._clebsch_gordan import calculate_cg_coefficients
from ._coupling import coupled_to_uncoupled, uncoupled_to_coupled
from ._symmetrized import SymmetrizedModel
from ._transformations import O3Transformations, random_transformations


__all__ = [
    "SymmetrizedModel",
    "O3Transformations",
    "random_transformations",
    "calculate_cg_coefficients",
    "cartesian_to_spherical",
    "spherical_to_cartesian",
    "uncoupled_to_coupled",
    "coupled_to_uncoupled",
]
