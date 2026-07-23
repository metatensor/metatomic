"""
O(3) averaging and equivariance diagnostics for atomistic models.

See :py:class:`SymmetrizedModel` for the method and public output conventions.
"""

from ._model import SymmetrizedModel
from ._quadrature import get_rotation_quadrature


__all__ = [
    "SymmetrizedModel",
    "get_rotation_quadrature",
]
