"""
Apply O(3) transformations (rotations and improper rotations) to
:py:class:`metatomic.torch.System` and :py:class:`metatensor.torch.TensorMap`, for
example to augment training data with randomly rotated copies of a structure.

See :ref:`o3-conventions` for the naming conventions used to identify Cartesian and
spherical components in a :py:class:`~metatensor.torch.TensorBlock`.
"""

from ._transformations import O3Transformations, random_transformations


__all__ = [
    "O3Transformations",
    "random_transformations",
]
