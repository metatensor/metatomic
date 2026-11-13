O(3) symmetries
===============

The ``metatomic_torch.o3`` module provides utilities for handling O(3)
symmetries (rotations and improper rotations).

There are tools to augment training data with randomly rotated copies of
structures/targets, tools to convert :py:class:`metatensor.torch.TensorMap`
between Cartesian and spherical forms, and between uncoupled and coupled
spherical forms; and tools to explicitly symmetrize non-equivariant model
outputs by integration over a spherical grid.

See :ref:`o3-conventions` for the naming conventions used to identify Cartesian
and spherical components in a :py:class:`~metatensor_torch.TensorBlock`.

.. toctree::
    :maxdepth: 1

    conventions
    transformations
    tensors
    symmetrized
