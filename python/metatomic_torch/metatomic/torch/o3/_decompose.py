"""
Decomposition of standard Cartesian outputs into O(3) irreducible components.

Standard scalars, vectors and matrices are re-expressed with explicit
``o3_lambda`` and ``o3_sigma`` keys, so that the variance and
character-projection machinery of the O(3)-symmetrized model treats them exactly
like natively spherical outputs.
"""

import math
from typing import Dict, List

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


def _standard_quantity_categories() -> Dict[str, str]:
    """Return the Cartesian layout of every decomposable standard quantity.

    This is the single source of truth for which outputs and inputs are
    decomposed; it mirrors ``KNOWN_QUANTITIES`` in
    ``metatomic-torch/src/quantities.cpp``, minus ``feature``. Only the current
    (singular) spellings appear here: deprecated names are normalized before
    they reach this module.

    TorchScript cannot read a module-level dictionary from a compiled function,
    so the table is built by this function and bound to
    :py:data:`STANDARD_QUANTITY_CATEGORIES` for Python callers.
    """
    return {
        # scalars: l = 0
        "charge": "scalar",
        "energy": "scalar",
        "energy_ensemble": "scalar",
        "energy_uncertainty": "scalar",
        "mass": "scalar",
        "spin_multiplicity": "scalar",
        # Cartesian vectors: l = 1
        "heat_flux": "cartesian_vector",
        "momentum": "cartesian_vector",
        "non_conservative_force": "cartesian_vector",
        "position": "cartesian_vector",
        "velocity": "cartesian_vector",
        # symmetric 3x3 matrices: l = 0 and l = 2
        "non_conservative_stress": "symmetric_matrix",
    }


STANDARD_QUANTITY_CATEGORIES: Dict[str, str] = _standard_quantity_categories()

#: maximum angular momentum carried by each category above
MAX_O3_LAMBDA_PER_CATEGORY: Dict[str, int] = {
    "scalar": 0,
    "cartesian_vector": 1,
    "symmetric_matrix": 2,
}


def _o3_mu_labels(o3_lambda: int, device: torch.device) -> Labels:
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


def _symmetric_matrices_to_spherical(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return orthonormal l=0 and l=2 components of the symmetric matrix part.

    ``values`` must have shape ``(n_samples, 3, 3, n_properties)``.

    The antisymmetric (l=1) part is silently discarded.
    """
    l0 = (values[:, 0, 0, :] + values[:, 1, 1, :] + values[:, 2, 2, :]).unsqueeze(
        1
    ) / math.sqrt(3.0)

    sqrt_two = math.sqrt(2.0)
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

    return l0, l2


def decompose_output(
    source_name: str,
    tensor: TensorMap,
) -> TensorMap:
    """Decompose standard outputs for variance and character projection.

    This takes the standard Cartesian or scalar outputs of a model and
    re-expresses them in the usual O(3) spherical convention, i.e. as blocks
    labelled by ``o3_lambda``/``o3_sigma`` with ``o3_mu`` components.

    ``feature`` is excluded from the decomposition table: features are not an
    irreducible representation of O(3), so they are passed through unchanged and
    their variance measures the deviation from invariance.
    """
    quantity = source_name.split("/", 1)[0]
    categories = _standard_quantity_categories()
    if quantity not in categories:
        return tensor
    category = categories[quantity]

    if category == "scalar":
        scalar_blocks: List[TensorBlock] = []
        for block in tensor.blocks():
            if len(block.components) != 0:
                raise ValueError(f"'{quantity}' outputs must not have components")
            scalar_blocks.append(
                TensorBlock(
                    values=block.values.unsqueeze(1),
                    samples=block.samples,
                    components=[_o3_mu_labels(0, block.values.device)],
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
            if (
                len(block.components) != 1
                or block.components[0].names != ["xyz"]
                or len(block.components[0]) != 3
            ):
                raise ValueError(
                    f"'{quantity}' must have one 'xyz' component axis of size 3"
                )
            vector_blocks.append(
                TensorBlock(
                    values=_cartesian_vectors_to_spherical(block.values, 1),
                    samples=block.samples,
                    components=[_o3_mu_labels(1, block.values.device)],
                    properties=block.properties,
                )
            )
        result = TensorMap(
            _add_o3_irrep_to_keys(tensor.keys, 1, 1),
            vector_blocks,
        )

    else:
        blocks_l0: List[TensorBlock] = []
        blocks_l2: List[TensorBlock] = []
        for block in tensor.blocks():
            if (
                len(block.components) != 2
                or block.components[0].names != ["xyz_1"]
                or block.components[1].names != ["xyz_2"]
                or len(block.components[0]) != 3
                or len(block.components[1]) != 3
            ):
                raise ValueError(
                    f"'{quantity}' must have 'xyz_1' and 'xyz_2' component axes "
                    "of size 3"
                )

            values_l0, values_l2 = _symmetric_matrices_to_spherical(block.values)
            blocks_l0.append(
                TensorBlock(
                    values=values_l0,
                    samples=block.samples,
                    components=[_o3_mu_labels(0, block.values.device)],
                    properties=block.properties,
                )
            )
            blocks_l2.append(
                TensorBlock(
                    values=values_l2,
                    samples=block.samples,
                    components=[_o3_mu_labels(2, block.values.device)],
                    properties=block.properties,
                )
            )

        keys_l0 = _add_o3_irrep_to_keys(tensor.keys, 0, 1)
        keys_l2 = _add_o3_irrep_to_keys(tensor.keys, 2, 1)
        result = TensorMap(
            Labels(
                list(keys_l0.names),
                torch.cat([keys_l0.values, keys_l2.values], dim=0),
            ),
            blocks_l0 + blocks_l2,
        )

    for info_name, info_value in tensor.info().items():
        result.set_info(info_name, info_value)
    return result


def _add_o3_irrep_to_keys(
    keys: Labels,
    o3_lambda: int,
    o3_sigma: int,
) -> Labels:
    """Add or validate the ``o3_lambda`` and ``o3_sigma`` key columns."""
    names = list(keys.names)
    values = keys.values

    if names == ["_"]:
        if len(keys) != 1 or int(values[0, 0]) != 0:
            raise ValueError(
                "the '_' placeholder must contain exactly one key with value 0"
            )
        names = []
        values = values[:, :0]

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
