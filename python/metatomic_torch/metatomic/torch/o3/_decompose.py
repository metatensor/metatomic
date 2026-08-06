"""
Decomposition of standard Cartesian outputs into O(3) irreducible components.

Standard scalars, vectors and matrices are re-expressed with explicit
``o3_lambda`` and ``o3_sigma`` keys, so that the variance and
character-projection machinery of the O(3)-symmetrized model treats them exactly
like natively spherical outputs.
"""

import math
from typing import List, Tuple

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from .._quantities import standard_quantity_categories
from ._utils import copy_tensormap_info, strip_placeholder_key


def o3_mu_labels(o3_lambda: int, device: torch.device) -> Labels:
    """Return ``o3_mu`` labels from ``-o3_lambda`` through ``o3_lambda``."""
    return Labels(
        "o3_mu",
        torch.arange(
            -o3_lambda,
            o3_lambda + 1,
            dtype=torch.int32,
            device=device,
        ).reshape(-1, 1),
    )


def _cartesian_vectors_to_spherical(
    values: torch.Tensor,
    component_axis: int,
) -> torch.Tensor:
    """Reorder ``(x, y, z)`` as ``(mu=-1, 0, 1) = (y, z, x)``."""
    return values.roll(-1, dims=component_axis)


def _matrices_to_spherical(
    values: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return orthonormal l=0, l=1, and l=2 components of a Cartesian matrix.

    Standard matrix quantities are symmetric, so their antisymmetric (l=1,
    pseudovector) part is zero by construction; it is still returned so that
    the diagnostics of a model producing a non-symmetric output (before any
    downstream symmetrization) capture the antisymmetric response as well.

    The three stacks together preserve the Frobenius norm of the matrix.
    """
    assert values.dim() == 4 and values.size(1) == 3 and values.size(2) == 3

    l0 = (values[:, 0, 0, :] + values[:, 1, 1, :] + values[:, 2, 2, :]).unsqueeze(
        1
    ) / math.sqrt(3.0)

    sqrt_two = math.sqrt(2.0)
    # the antisymmetric part packs into the axial pseudovector
    # v = ((M_21 - M_12) / 2, (M_02 - M_20) / 2, (M_10 - M_01) / 2), listed here
    # as sqrt(2) * v in the real spherical (mu=-1, 0, 1) = (y, z, x) order
    l1 = torch.stack(
        [
            (values[:, 0, 2, :] - values[:, 2, 0, :]) / sqrt_two,
            (values[:, 1, 0, :] - values[:, 0, 1, :]) / sqrt_two,
            (values[:, 2, 1, :] - values[:, 1, 2, :]) / sqrt_two,
        ],
        dim=1,
    )

    l2 = torch.stack(
        [
            (values[:, 0, 1, :] + values[:, 1, 0, :]) / sqrt_two,
            (values[:, 1, 2, :] + values[:, 2, 1, :]) / sqrt_two,
            (2.0 * values[:, 2, 2, :] - values[:, 0, 0, :] - values[:, 1, 1, :])
            / math.sqrt(6.0),
            (values[:, 0, 2, :] + values[:, 2, 0, :]) / sqrt_two,
            (values[:, 0, 0, :] - values[:, 1, 1, :]) / sqrt_two,
        ],
        dim=1,
    )

    return l0, l1, l2


def decompose_quantity(
    name: str,
    tensor: TensorMap,
) -> TensorMap:
    """Decompose standard quantities for variance and character projection.

    This takes the standard Cartesian or scalar quantities (inputs or outputs
    of a model) and re-expresses them in the usual O(3) spherical convention,
    i.e. as blocks labelled by ``o3_lambda``/``o3_sigma`` with ``o3_mu``
    components.

    ``feature`` is excluded from the decomposition table: features are not an
    irreducible representation of O(3), so they are passed through unchanged and
    their variance measures the deviation from invariance.
    """
    quantity = name.split("/", 1)[0]
    categories = standard_quantity_categories()
    if quantity not in categories:
        return tensor
    category = categories[quantity]

    if category == "scalar":
        scalar_blocks: List[TensorBlock] = []
        for block in tensor.blocks():
            assert len(block.components) == 0, (
                f"'{quantity}' outputs must not have components"
            )
            scalar_blocks.append(
                TensorBlock(
                    values=block.values.unsqueeze(1),
                    samples=block.samples,
                    components=[o3_mu_labels(0, block.values.device)],
                    properties=block.properties,
                )
            )
        result = TensorMap(
            _add_o3_irrep_to_keys(tensor.keys, 0, 1),
            scalar_blocks,
        )

    elif category == "cartesian_vector":
        vector_blocks: List[TensorBlock] = []
        for block in tensor.blocks():
            assert (
                len(block.components) == 1
                and block.components[0].names == ["xyz"]
                and len(block.components[0]) == 3
            ), f"'{quantity}' must have one 'xyz' component axis of size 3"
            vector_blocks.append(
                TensorBlock(
                    values=_cartesian_vectors_to_spherical(block.values, 1),
                    samples=block.samples,
                    components=[o3_mu_labels(1, block.values.device)],
                    properties=block.properties,
                )
            )
        result = TensorMap(
            _add_o3_irrep_to_keys(tensor.keys, 1, 1),
            vector_blocks,
        )

    else:
        assert category == "symmetric_matrix"
        blocks_l0: List[TensorBlock] = []
        blocks_l1: List[TensorBlock] = []
        blocks_l2: List[TensorBlock] = []
        for block in tensor.blocks():
            assert (
                len(block.components) == 2
                and block.components[0].names == ["xyz_1"]
                and block.components[1].names == ["xyz_2"]
                and len(block.components[0]) == 3
                and len(block.components[1]) == 3
            ), f"'{quantity}' must have 'xyz_1' and 'xyz_2' component axes of size 3"

            values_l0, values_l1, values_l2 = _matrices_to_spherical(block.values)
            blocks_l0.append(
                TensorBlock(
                    values=values_l0,
                    samples=block.samples,
                    components=[o3_mu_labels(0, block.values.device)],
                    properties=block.properties,
                )
            )
            blocks_l1.append(
                TensorBlock(
                    values=values_l1,
                    samples=block.samples,
                    components=[o3_mu_labels(1, block.values.device)],
                    properties=block.properties,
                )
            )
            blocks_l2.append(
                TensorBlock(
                    values=values_l2,
                    samples=block.samples,
                    components=[o3_mu_labels(2, block.values.device)],
                    properties=block.properties,
                )
            )

        keys_l0 = _add_o3_irrep_to_keys(tensor.keys, 0, 1)
        # a Cartesian rank-2 tensor is inversion-even, so its antisymmetric
        # (axial-vector) part is an l=1 pseudovector: o3_sigma = -1
        keys_l1 = _add_o3_irrep_to_keys(tensor.keys, 1, -1)
        keys_l2 = _add_o3_irrep_to_keys(tensor.keys, 2, 1)
        result = TensorMap(
            Labels(
                list(keys_l0.names),
                torch.cat([keys_l0.values, keys_l1.values, keys_l2.values], dim=0),
            ),
            blocks_l0 + blocks_l1 + blocks_l2,
        )

    return copy_tensormap_info(tensor, result)


def _add_o3_irrep_to_keys(
    keys: Labels,
    o3_lambda: int,
    o3_sigma: int,
) -> Labels:
    """Add or validate the ``o3_lambda`` and ``o3_sigma`` key columns."""
    names, values = strip_placeholder_key(list(keys.names), keys.values)

    for name, expected in (
        ("o3_lambda", o3_lambda),
        ("o3_sigma", o3_sigma),
    ):
        if name in names:
            column = values[:, names.index(name)]
            if not bool(torch.all(column == expected).item()):
                raise ValueError(
                    f"the existing '{name}' key column must contain only "
                    f"{expected} to assign O(3) irrep "
                    f"({o3_lambda}, {o3_sigma})"
                )
        else:
            names.append(name)
            values = torch.cat(
                [
                    values,
                    torch.full(
                        (len(keys), 1),
                        expected,
                        dtype=values.dtype,
                        device=values.device,
                    ),
                ],
                dim=1,
            )

    return Labels(names, values)
