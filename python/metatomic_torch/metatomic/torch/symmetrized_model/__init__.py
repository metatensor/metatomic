"""
Symmetrization of atomistic model outputs over :math:`O(3)`, and equivariance
metrics. See :py:class:`SymmetrizedModel` for a description of the method.
"""

from ._model import SymmetrizedModel, per_system_equivariance_rmse
from ._projections import per_system_character_fractions
from ._quadrature import get_rotation_quadrature


__all__ = [
    "SymmetrizedModel",
    "get_rotation_quadrature",
    "per_system_character_fractions",
    "per_system_equivariance_rmse",
]
