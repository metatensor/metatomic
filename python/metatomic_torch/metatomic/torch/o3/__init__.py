"""
Apply O(3) transformations (rotations and improper rotations) to
:py:class:`metatomic.torch.System` and :py:class:`metatensor.torch.TensorMap`, for
example to augment training data with randomly rotated copies of a structure.

See :ref:`o3-conventions` for the naming conventions used to identify Cartesian and
spherical components in a :py:class:`~metatensor.torch.TensorBlock`.
"""

from ._tranformations import (
    O3Transformation,
    random_transformations,
    transform_block,
    transform_system,
    transform_tensor,
)


__all__ = [
    "O3Transformation",
    "random_transformations",
    "transform_system",
    "transform_tensor",
    "transform_block",
]
