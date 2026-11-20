"""
Decomposition of standard Cartesian outputs into O(3) irreducible components.

Standard scalars, vectors and matrices are re-expressed with explicit
``o3_lambda`` and ``o3_sigma`` keys, so that the variance and
character-projection machinery of the O(3)-symmetrized model treats them exactly
like natively spherical outputs.
"""

from typing import List

from metatensor_torch import TensorMap

from .._quantities import current_quantity_name, standard_quantity_categories
from ._cartesian_spherical import cartesian_to_spherical


def decompose_quantity(
    name: str,
    tensor: TensorMap,
    cg_coefficients: TensorMap,
) -> TensorMap:
    """Decompose standard quantities for variance and character projection.

    This takes the standard Cartesian or scalar quantities (inputs or outputs of a
    model) and re-expresses them in the usual O(3) spherical convention, i.e. as blocks
    labelled by ``o3_lambda``/``o3_sigma`` with ``o3_mu`` components, using
    :py:func:`cartesian_to_spherical`.

    All the blocks are kept in the output, even if they are zero everywhere (for example
    the l=1 pseudovector part of a symmetric matrix), so the output metadata does not
    depend on the values. This also allows the diagnostics of a model producing a
    non-symmetric output (before any downstream symmetrization) to capture the
    antisymmetric response.

    ``feature`` is excluded from the decomposition table: features are not an
    irreducible representation of O(3), so they are passed through unchanged and their
    variance measures the deviation from invariance.

    Deprecated spellings are normalized before the lookup, so a model that only answers
    to e.g. ``non_conservative_forces`` still gets its output decomposed as the current
    ``non_conservative_force`` quantity, matching the aliasing that
    :py:class:`AtomisticModel` applies for engines.

    :param name: name of the quantity
    :param tensor: values of the quantity
    :param cg_coefficients: Clebsch-Gordan coefficients computed with
        :py:func:`calculate_cg_coefficients`, with ``lambda_max`` at least 2, and the
        same dtype and device as ``tensor``
    """
    quantity = current_quantity_name(name).split("/", 1)[0]
    categories = standard_quantity_categories()
    if quantity not in categories:
        return tensor
    category = categories[quantity]

    components: List[str] = []
    if category == "scalar":
        for block in tensor.blocks():
            assert len(block.components) == 0, (
                f"'{quantity}' outputs must not have components"
            )

    elif category == "cartesian_vector":
        for block in tensor.blocks():
            assert (
                len(block.components) == 1
                and block.components[0].names == ["xyz"]
                and len(block.components[0]) == 3
            ), f"'{quantity}' must have one 'xyz' component axis of size 3"
        components = ["xyz"]

    else:
        assert category == "symmetric_matrix"
        for block in tensor.blocks():
            assert (
                len(block.components) == 2
                and block.components[0].names == ["xyz_1"]
                and block.components[1].names == ["xyz_2"]
                and len(block.components[0]) == 3
                and len(block.components[1]) == 3
            ), f"'{quantity}' must have 'xyz_1' and 'xyz_2' component axes of size 3"
        components = ["xyz_1", "xyz_2"]

    return cartesian_to_spherical(
        tensor,
        components,
        remove_blocks_threshold=None,
        cg_coefficients=cg_coefficients,
    )
