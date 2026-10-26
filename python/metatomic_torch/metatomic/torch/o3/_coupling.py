"""
Coupling of the spherical components of a :py:class:`TensorMap`, going from an
uncoupled form (several ``o3_mu_1``, ``o3_mu_2``, ... components, transforming like a
product of spherical harmonics) to a coupled form (a single ``o3_mu`` component,
transforming like a single spherical harmonic), and back.

This module also contains the building blocks for these transformations, which are
shared with the conversion between Cartesian and spherical forms.
"""

from typing import Dict, List, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from ._clebsch_gordan import (
    cg_couple_array,
    cg_coupling_matrix,
    cg_uncouple_array,
    resolve_cg_coefficients,
)
from ._decompose import o3_mu_labels
from ._utils import copy_tensormap_info


class CouplingData:
    """
    Data of a :py:class:`TensorMap` while its components are coupled or uncoupled.
    During the transformation, we only keep track of the keys and the values of the
    blocks, and create the full metadata at the end.

    :param key_names: names of the keys
    :param keys_values: values of the keys, with one entry for each array in ``arrays``
    :param arrays: values of the blocks
    :param parents: index of the input block each array comes from, to get the rest of
        the metadata from
    """

    def __init__(
        self,
        key_names: List[str],
        keys_values: List[List[int]],
        arrays: List[torch.Tensor],
        parents: List[int],
    ):
        self.key_names = key_names
        self.keys_values = keys_values
        self.arrays = arrays
        self.parents = parents

    @staticmethod
    def create(keys: Labels, blocks: List[TensorBlock]) -> "CouplingData":
        keys_values: List[List[int]] = keys.values.to(torch.int64).tolist()
        arrays: List[torch.Tensor] = []
        parents: List[int] = []
        for block_index, block in enumerate(blocks):
            arrays.append(block.values)
            parents.append(block_index)

        return CouplingData(keys.names, keys_values, arrays, parents)


def uncoupled_to_coupled(
    tensor: TensorMap,
    components: List[str],
    remove_blocks_threshold: Optional[float] = 1e-9,
    cg_coefficients: Optional[TensorMap] = None,
) -> TensorMap:
    """
    Transform a ``tensor`` from an uncoupled spherical form to a coupled spherical form.

    In the uncoupled form, the blocks contain multiple spherical components (named
    ``o3_mu_1``, ``o3_mu_2``, ...), each transforming like a real spherical harmonic
    with degree and parity given by the corresponding ``o3_lambda_1``/``o3_sigma_1``,
    ``o3_lambda_2``/``o3_sigma_2``, ... keys. Such a block transforms like a product of
    spherical harmonics ``Y^{m_1}_{l_1} ⊗ Y^{m_2}_{l_2} ⊗ ...``. In the coupled form,
    these components are replaced by a single ``o3_mu`` component and each block
    transforms like a single spherical harmonic ``Y^M_L``, using Clebsch-Gordan
    coefficients to combine the spherical harmonics. For example, coupling two
    components with ``l_1 = 1`` and ``l_2 = 2`` gives three blocks, with ``L = 1, 2,
    3``.

    The output keys contain ``o3_lambda`` and ``o3_sigma`` (the degree and parity of the
    coupled spherical harmonic), followed by the keys of the input. The ``o3_sigma`` of
    the output is given by the product of the ``o3_sigma`` of the coupled components,
    times ``(-1)^(l_1 + l_2 + ... + L)``. The degree and parity of the coupled
    components are needed to go back to the uncoupled form with
    :py:func:`coupled_to_uncoupled`, so the ``o3_lambda_{i}`` and ``o3_sigma_{i}`` keys
    are kept in the output, renamed to ``l_{i}`` and ``s_{i}``. When coupling three or
    more components, the components are coupled two by two, starting from the last ones,
    and the degrees of the intermediate spherical harmonics are stored in ``k_{i}``
    keys.

    Depending on the symmetries of the input (for example when coupling a component with
    itself), some of the blocks will be zero everywhere. Any block with a squared norm
    below ``remove_blocks_threshold`` is not included in the output. To keep all blocks
    regardless of their norm, you can set ``remove_blocks_threshold=None``.

    Gradients are not supported yet, and this function raises an error if any of the
    input blocks has gradients.

    :param tensor: input tensor, in uncoupled form
    :param components: names of the spherical components to couple. These must be
        consecutive components named ``o3_mu_{suffix}``, with the corresponding
        ``o3_lambda_{suffix}`` and ``o3_sigma_{suffix}`` keys.
    :param remove_blocks_threshold: Numerical tolerance to use when determining if a
        block's norm is zero or not. Blocks with zero norm will be excluded from the
        output. Set this to ``None`` to keep all blocks in the output.
    :param cg_coefficients: Clebsch-Gordan coefficients computed with
        :py:func:`calculate_cg_coefficients`, with ``lambda_max`` at least equal to the
        sum of the degrees of the coupled components. This is optional except when using
        this function from TorchScript.

    :return: :py:class:`~metatensor.torch.TensorMap` in coupled form
    """
    if len(tensor) == 0 or len(components) == 0:
        # nothing to do
        return tensor

    check_no_gradients(tensor, "uncoupled_to_coupled")

    if not torch.jit.is_scripting():
        if not isinstance(components, list):
            raise TypeError(f"`components` should be a list, got {type(components)}")

    key_names = tensor.keys.names
    if "o3_lambda" in key_names or "o3_sigma" in key_names:
        raise ValueError(
            "this tensor already has `o3_lambda` or `o3_sigma` keys, "
            "is it already in coupled form?"
        )

    n_components = len(components)
    for i in range(1, n_components - 1):
        if f"k_{i}" in key_names:
            raise ValueError(
                f"this tensor already has a `k_{i}` key, is it already in coupled form?"
            )

    axes = _consecutive_axes(tensor, components)
    suffixes = _component_suffixes(components)
    lambda_names = ["o3_lambda" + suffix for suffix in suffixes]
    sigma_names = ["o3_sigma" + suffix for suffix in suffixes]
    _check_component_keys(key_names, components, lambda_names, sigma_names)

    # the degree and parity of the components are stored in the output keys as
    # `l_{suffix}` and `s_{suffix}`
    l_names = ["l" + suffix for suffix in suffixes]
    s_names = ["s" + suffix for suffix in suffixes]
    for name in l_names + s_names:
        if name in key_names:
            raise ValueError(
                f"this tensor already has a `{name}` key, "
                "is it already in coupled form?"
            )

    input_blocks = tensor.blocks()
    data = CouplingData.create(tensor.keys, input_blocks)

    # check the components of all blocks, and get the maximal degree we will need
    lambda_columns = [data.key_names.index(name) for name in lambda_names]
    sigma_columns = [data.key_names.index(name) for name in sigma_names]
    lambda_max = 0
    for block_index, block in enumerate(input_blocks):
        key = data.keys_values[block_index]
        block_components = block.components
        total_lambda = 0
        for i in range(n_components):
            _check_spherical_component(
                labels=block_components[axes[i] - 1],
                o3_lambda=key[lambda_columns[i]],
                o3_sigma=key[sigma_columns[i]],
                keys=tensor.keys,
                block_index=block_index,
            )
            total_lambda += key[lambda_columns[i]]

        if total_lambda > lambda_max:
            lambda_max = total_lambda

    data = _rename_key_dimensions(data, lambda_names + sigma_names, l_names + s_names)

    first_component = axes[0] - 1
    last_component = axes[-1] - 1

    if n_components == 1:
        # nothing to couple, the single component already transforms like a single
        # spherical harmonic
        keys_values: List[List[int]] = []
        for key in data.keys_values:
            keys_values.append([key[lambda_columns[0]], key[sigma_columns[0]]] + key)

        data = CouplingData(
            key_names=["o3_lambda", "o3_sigma"] + data.key_names,
            keys_values=keys_values,
            arrays=data.arrays,
            parents=data.parents,
        )

        coupled = build_spherical_tensor(
            tensor=tensor,
            data=data,
            first_component=first_component,
            last_component=last_component,
            remove_blocks_threshold=None,
        )
        return copy_tensormap_info(source=tensor, result=coupled)

    coefficients = resolve_cg_coefficients(
        cg_coefficients, lambda_max, tensor.block(0).values
    )

    # couple the components two by two, starting from the last ones. After the first
    # step, the right-hand side of the coupling is the result of the previous step: its
    # `o3_lambda` becomes `k_{iteration}` and its `o3_sigma` is removed from the keys,
    # since it can be computed from the `s_{i}`.
    for iteration in range(n_components - 1):
        left = n_components - 2 - iteration
        left_sigma_column = data.key_names.index(s_names[left])

        sigmas_1: List[int] = []
        sigmas_2: List[int] = []
        if iteration == 0:
            right_sigma_column = data.key_names.index(s_names[left + 1])
            for key in data.keys_values:
                sigmas_1.append(key[left_sigma_column])
                sigmas_2.append(key[right_sigma_column])
        else:
            new_keys_values: List[List[int]] = []
            for key in data.keys_values:
                sigmas_1.append(key[left_sigma_column])
                sigmas_2.append(key[1])
                new_keys_values.append([key[0]] + key[2:])

            data = CouplingData(
                key_names=[f"k_{iteration}"] + data.key_names[2:],
                keys_values=new_keys_values,
                arrays=data.arrays,
                parents=data.parents,
            )

        data = do_coupling(
            data=data,
            axis_1=axes[left],
            axis_2=axes[left + 1],
            sigmas_1=sigmas_1,
            sigmas_2=sigmas_2,
            cartesian_1=False,
            cartesian_2=False,
            cg_coefficients=coefficients,
        )

    coupled = build_spherical_tensor(
        tensor=tensor,
        data=data,
        first_component=first_component,
        last_component=last_component,
        remove_blocks_threshold=remove_blocks_threshold,
    )
    return copy_tensormap_info(source=tensor, result=coupled)


def coupled_to_uncoupled(
    tensor: TensorMap,
    components: List[str],
    cg_coefficients: Optional[TensorMap] = None,
) -> TensorMap:
    """
    Transform a ``tensor`` from a coupled spherical form back to an uncoupled spherical
    form.

    This is the inverse of :py:func:`uncoupled_to_coupled`. The single ``o3_mu``
    component of the input is replaced by the spherical ``components``, using the
    degrees stored in the corresponding ``l_{suffix}`` keys. The ``l_{suffix}`` and
    ``s_{suffix}`` keys are renamed back to ``o3_lambda_{suffix}`` and
    ``o3_sigma_{suffix}`` in the output, to match the components. All the blocks
    that only differ by their ``o3_lambda`` and ``o3_sigma`` (and ``k_{i}`` when
    uncoupling three or more components) are combined into a single block, and any
    block missing from the input (for example because its norm was zero, see the
    ``remove_blocks_threshold`` parameter of :py:func:`uncoupled_to_coupled`) is
    treated as a block full of zeros.

    Gradients are not supported yet, and this function raises an error if any of the
    input blocks has gradients.

    :param tensor: input tensor, in coupled form. The keys must contain ``o3_lambda``,
        ``o3_sigma``, the ``l_{suffix}`` and ``s_{suffix}`` for all ``components``, and
        the ``k_{i}`` describing the intermediate couplings when uncoupling three or
        more components.
    :param components: names of the spherical components to create, in the same order
        as they were given to :py:func:`uncoupled_to_coupled`. These must be named
        ``o3_mu_{suffix}``.
    :param cg_coefficients: Clebsch-Gordan coefficients computed with
        :py:func:`calculate_cg_coefficients`, with ``lambda_max`` at least equal to the
        sum of the degrees of the uncoupled components. This is optional except when
        using this function from TorchScript.

    :return: :py:class:`~metatensor.torch.TensorMap` in uncoupled form
    """
    if len(tensor) == 0 or len(components) == 0:
        # nothing to do
        return tensor

    check_no_gradients(tensor, "coupled_to_uncoupled")

    if not torch.jit.is_scripting():
        if not isinstance(components, list):
            raise TypeError(f"`components` should be a list, got {type(components)}")

    key_names = tensor.keys.names
    if "o3_lambda" not in key_names or "o3_sigma" not in key_names:
        raise ValueError(
            "this tensor is missing one of `o3_lambda` or `o3_sigma` in the keys, "
            "is it already in uncoupled form?"
        )

    if "o3_mu" not in tensor.component_names:
        raise ValueError(
            "this tensor does not have an `o3_mu` component, "
            "is it already in uncoupled form?"
        )

    n_components = len(components)
    for i in range(1, n_components - 1):
        if f"k_{i}" not in key_names:
            raise ValueError(f"expected a k_{i} dimension in the keys")

    suffixes = _component_suffixes(components)
    l_names = ["l" + suffix for suffix in suffixes]
    s_names = ["s" + suffix for suffix in suffixes]
    _check_component_keys(key_names, components, l_names, s_names)

    # the degree and parity of the components will be stored in the output keys as
    # `o3_lambda_{suffix}` and `o3_sigma_{suffix}`
    lambda_names = ["o3_lambda" + suffix for suffix in suffixes]
    sigma_names = ["o3_sigma" + suffix for suffix in suffixes]
    for name in lambda_names + sigma_names:
        if name in key_names:
            raise ValueError(
                f"this tensor already has a `{name}` key, "
                "is it already in uncoupled form?"
            )

    input_blocks = tensor.blocks()
    data = CouplingData.create(tensor.keys, input_blocks)

    l_columns = [data.key_names.index(name) for name in l_names]
    lambda_max = 0
    max_lambdas = [0 for _ in range(n_components)]
    for key in data.keys_values:
        total_lambda = 0
        for i in range(n_components):
            o3_lambda = key[l_columns[i]]
            if o3_lambda < 0:
                raise ValueError(f"expected all {l_names[i]} to be >= 0")
            total_lambda += o3_lambda
            if o3_lambda > max_lambdas[i]:
                max_lambdas[i] = o3_lambda

        if total_lambda > lambda_max:
            lambda_max = total_lambda

    # position of the `o3_mu` component that will be expanded into `components`
    mu_position = tensor.component_names.index("o3_mu")

    if n_components == 1:
        # nothing to uncouple, only remove `o3_lambda` and `o3_sigma` from the keys
        data = _remove_key_dimensions(data, ["o3_lambda", "o3_sigma"])
    else:
        coefficients = resolve_cg_coefficients(
            cg_coefficients, lambda_max, tensor.block(0).values
        )

        # Split off the first component from the coupled `o3_lambda`, leaving the
        # remaining (still coupled) intermediate to the right. The intermediate is
        # uncoupled further on the next iteration, until all components are uncoupled.
        lambda_name = "o3_lambda"
        sigma_name: Optional[str] = "o3_sigma"
        for i in range(n_components - 1):
            if i == n_components - 2:
                right_lambda_name = l_names[i + 1]
            else:
                right_lambda_name = f"k_{n_components - 2 - i}"

            data = do_decoupling(
                data=data,
                axis=mu_position + 1 + i,
                lambda_name=lambda_name,
                sigma_name=sigma_name,
                l1_name=l_names[i],
                l2_name=right_lambda_name,
                cartesian_1=False,
                cartesian_2=False,
                cg_coefficients=coefficients,
            )

            lambda_name = right_lambda_name
            sigma_name = None

    # create the output metadata
    data = _rename_key_dimensions(data, l_names + s_names, lambda_names + sigma_names)

    all_components_labels: List[List[Labels]] = []
    for i in range(n_components):
        labels: List[Labels] = []
        for o3_lambda in range(max_lambdas[i] + 1):
            mu_labels = o3_mu_labels(o3_lambda, tensor.device)
            labels.append(mu_labels.rename("o3_mu", components[i]))
        all_components_labels.append(labels)

    lambda_columns = [data.key_names.index(name) for name in lambda_names]
    new_blocks: List[TensorBlock] = []
    for i in range(len(data.arrays)):
        key = data.keys_values[i]
        new_components: List[Labels] = []
        for c in range(n_components):
            new_components.append(all_components_labels[c][key[lambda_columns[c]]])

        parent = input_blocks[data.parents[i]]
        parent_components = parent.components
        new_blocks.append(
            TensorBlock(
                values=data.arrays[i],
                samples=parent.samples,
                components=parent_components[:mu_position]
                + new_components
                + parent_components[mu_position + 1 :],
                properties=parent.properties,
            )
        )

    new_keys = Labels(
        data.key_names,
        torch.tensor(data.keys_values, dtype=torch.int32, device=tensor.keys.device),
        assume_unique=True,
    )
    return copy_tensormap_info(source=tensor, result=TensorMap(new_keys, new_blocks))


def check_no_gradients(tensor: TensorMap, function_name: str):
    """
    Check that none of the blocks in ``tensor`` has gradients, since they are not
    supported by the coupling and conversion functions yet.
    """
    for block_index, block in enumerate(tensor.blocks()):
        gradients = block.gradients_list()
        if len(gradients) != 0:
            raise ValueError(
                f"gradients are not supported yet by `{function_name}`, but the block "
                f"for {tensor.keys.entry(block_index).print()} has gradients with "
                f"respect to {gradients}"
            )


def _consecutive_axes(tensor: TensorMap, components: List[str]) -> List[int]:
    """
    Get the axes of the blocks values corresponding to ``components``, checking that
    they are consecutive.
    """
    axes: List[int] = []
    all_component_names = tensor.component_names
    for name in components:
        if name in all_component_names:
            axes.append(all_component_names.index(name) + 1)
        else:
            raise ValueError(f"'{name}' is not part of this tensor components")

    expected_axes = [axis for axis in range(axes[0], axes[-1] + 1)]
    if axes != expected_axes:
        raise ValueError(
            f"this function only supports consecutive components, {components} are not"
        )

    return axes


def _component_suffixes(components: List[str]) -> List[str]:
    """Get the suffixes of the spherical ``components``, named ``o3_mu_{suffix}``."""
    suffixes: List[str] = []
    for name in components:
        if not name.startswith("o3_mu_"):
            raise ValueError(
                f"spherical components should be named 'o3_mu_{{suffix}}', got '{name}'"
            )
        suffixes.append(name[len("o3_mu") :])

    return suffixes


def _check_component_keys(
    key_names: List[str],
    components: List[str],
    lambda_names: List[str],
    sigma_names: List[str],
):
    """
    Check that the keys holding the degree (``lambda_names``) and parity
    (``sigma_names``) of all ``components`` are part of ``key_names``.
    """
    for i in range(len(components)):
        if lambda_names[i] not in key_names or sigma_names[i] not in key_names:
            raise ValueError(
                f"expected `{lambda_names[i]}` and `{sigma_names[i]}` in the keys for "
                f"the '{components[i]}' component"
            )


def _check_spherical_component(
    labels: Labels,
    o3_lambda: int,
    o3_sigma: int,
    keys: Labels,
    block_index: int,
):
    """
    Check that a spherical component has the expected values for the given
    ``o3_lambda``, and that ``o3_sigma`` is valid.
    """
    if o3_lambda < 0:
        raise ValueError(
            f"expected all o3_lambda to be >= 0, got {o3_lambda} in block for "
            f"{keys.entry(block_index).print()}"
        )

    if o3_sigma != 1 and o3_sigma != -1:
        raise ValueError(
            f"expected all o3_sigma to be -1 or 1, got {o3_sigma} in block for "
            f"{keys.entry(block_index).print()}"
        )

    expected = torch.arange(
        -o3_lambda, o3_lambda + 1, dtype=labels.values.dtype, device=labels.device
    )
    if not torch.equal(labels.values[:, 0], expected):
        raise ValueError(
            f"component '{labels.names[0]}' in block for "
            f"{keys.entry(block_index).print()} should have values from {-o3_lambda} "
            f"to {o3_lambda}"
        )


def _rename_key_dimensions(
    data: CouplingData,
    old_names: List[str],
    new_names: List[str],
) -> CouplingData:
    """Rename the key dimensions ``old_names`` to ``new_names`` in ``data``."""
    key_names: List[str] = []
    for name in data.key_names:
        if name in old_names:
            key_names.append(new_names[old_names.index(name)])
        else:
            key_names.append(name)

    return CouplingData(key_names, data.keys_values, data.arrays, data.parents)


def _remove_key_dimensions(data: CouplingData, names: List[str]) -> CouplingData:
    """Remove the key dimensions with the given ``names`` from ``data``."""
    kept_columns: List[int] = []
    for column, name in enumerate(data.key_names):
        if name not in names:
            kept_columns.append(column)

    keys_values: List[List[int]] = []
    for key in data.keys_values:
        keys_values.append([key[column] for column in kept_columns])

    return CouplingData(
        key_names=[data.key_names[column] for column in kept_columns],
        keys_values=keys_values,
        arrays=data.arrays,
        parents=data.parents,
    )


def build_spherical_tensor(
    tensor: TensorMap,
    data: CouplingData,
    first_component: int,
    last_component: int,
    remove_blocks_threshold: Optional[float],
) -> TensorMap:
    """
    Create a spherical tensor, where the components from ``first_component`` to
    ``last_component`` of the blocks in ``tensor`` have been replaced by a single
    ``o3_mu`` component.

    :param tensor: input tensor, used to get the metadata of the blocks
    :param data: keys and values of the output blocks
    :param first_component: first converted component in the input blocks
    :param last_component: last converted component in the input blocks
    :param remove_blocks_threshold: arrays with a squared norm below this threshold
        are not included in the output
    """
    if remove_blocks_threshold is not None:
        if remove_blocks_threshold <= 0:
            raise ValueError("remove_blocks_threshold must be positive")

    o3_lambda_column = data.key_names.index("o3_lambda")

    lambda_max = 0
    for key in data.keys_values:
        if key[o3_lambda_column] > lambda_max:
            lambda_max = key[o3_lambda_column]

    all_mu_labels: List[Labels] = []
    for o3_lambda in range(lambda_max + 1):
        all_mu_labels.append(o3_mu_labels(o3_lambda, tensor.device))

    # get the metadata of the input blocks only once, since accessing it is not free
    input_samples: List[Labels] = []
    input_components: List[List[Labels]] = []
    input_properties: List[Labels] = []
    for block in tensor.blocks():
        input_samples.append(block.samples)
        input_components.append(block.components)
        input_properties.append(block.properties)

    new_keys_values: List[List[int]] = []
    new_blocks: List[TensorBlock] = []

    # For symmetry reasons, some of the blocks will be zero everywhere (for example
    # o3_sigma=-1 blocks if the input tensor is fully symmetric). If the user gave us a
    # threshold, we remove all blocks with a norm below this threshold.
    for i in range(len(data.arrays)):
        values = data.arrays[i]
        if remove_blocks_threshold is not None:
            norm = float(torch.linalg.vector_norm(values))
            if norm * norm <= remove_blocks_threshold:
                continue

        parent = data.parents[i]
        mu_labels = all_mu_labels[data.keys_values[i][o3_lambda_column]]
        new_blocks.append(
            TensorBlock(
                values=values,
                samples=input_samples[parent],
                components=input_components[parent][:first_component]
                + [mu_labels]
                + input_components[parent][last_component + 1 :],
                properties=input_properties[parent],
            )
        )
        new_keys_values.append(data.keys_values[i])

    if len(new_keys_values) == 0:
        new_keys = torch.zeros(
            (0, len(data.key_names)), dtype=torch.int32, device=tensor.keys.device
        )
    else:
        new_keys = torch.tensor(
            new_keys_values, dtype=torch.int32, device=tensor.keys.device
        )

    return TensorMap(Labels(data.key_names, new_keys, assume_unique=True), new_blocks)


def do_coupling(
    data: CouplingData,
    axis_1: int,
    axis_2: int,
    sigmas_1: List[int],
    sigmas_2: List[int],
    cartesian_1: bool,
    cartesian_2: bool,
    cg_coefficients: TensorMap,
) -> CouplingData:
    """
    Go from an uncoupled product basis that behave like a product of spherical harmonics
    to a coupled basis that behaves like a single spherical harmonic.

    This function takes in arrays where two of the axes (``axis_1`` and ``axis_2``)
    behave like spherical harmonics ``Y^m1_l1`` and ``Y^m2_l2``, and project them onto a
    single spherical harmonic ``Y^M_L``, using Clebsch-Gordan coefficients (see
    :py:func:`cg_couple_array`).

    The output will contain many arrays for each array in the input, matching all the
    different ``L`` (called ``o3_lambda`` in the code) required to do a full projection.

    This process can be iterated: a multi-dimensional array that is the product of many
    ``Y^m_l`` can be turned into a set of multiple terms transforming as a single
    ``Y^M_L``.

    :param data: keys and values of the blocks
    :param axis_1: first axis of the arrays behaving like spherical harmonics
    :param axis_2: second axis of the arrays behaving like spherical harmonics
    :param sigmas_1: ``o3_sigma`` of the first axis, for each array
    :param sigmas_2: ``o3_sigma`` of the second axis, for each array
    :param cartesian_1: is the first axis a Cartesian axis in the ``(x, y, z)`` order?
    :param cartesian_2: is the second axis a Cartesian axis in the ``(x, y, z)`` order?
    :param cg_coefficients: pre-computed set of Clebsch-Gordan coefficients

    :return: keys and values of the blocks using the coupled basis. The keys contain
        ``o3_lambda`` and ``o3_sigma``, followed by the input keys. The axes ``axis_1``
        and ``axis_2`` are replaced by a single axis.
    """
    assert axis_2 == axis_1 + 1
    assert "o3_lambda" not in data.key_names
    assert "o3_sigma" not in data.key_names

    # CG coefficients for each value of l1 and l2
    all_coefficients: Dict[str, torch.Tensor] = {}

    new_keys_values: List[List[int]] = []
    new_arrays: List[torch.Tensor] = []
    new_parents: List[int] = []
    for i in range(len(data.arrays)):
        # get l1, l2 from the array's shape
        values = data.arrays[i]
        shape = values.shape
        l1 = (shape[axis_1] - 1) // 2
        l2 = (shape[axis_2] - 1) // 2

        # generate the set of o3_lambda to compute
        o3_lambdas = [o3_lambda for o3_lambda in range(abs(l1 - l2), l1 + l2 + 1)]

        coefficients_key = f"{l1} {l2}"
        if coefficients_key not in all_coefficients:
            all_coefficients[coefficients_key] = cg_coupling_matrix(
                cg_coefficients,
                l1,
                l2,
                o3_lambdas,
                cartesian_1=cartesian_1,
                cartesian_2=cartesian_2,
            )

        # reshape the values to look like (n_s, 2*l1 + 1, 2*l2 + 1, n_p)
        shape_before = 1
        for axis in range(axis_1):
            shape_before *= shape[axis]

        shape_after = 1
        for axis in range(axis_2 + 1, len(shape)):
            shape_after *= shape[axis]

        array = values.reshape(shape_before, shape[axis_1], shape[axis_2], shape_after)

        # actual calculation
        outputs = cg_couple_array(array, o3_lambdas, all_coefficients[coefficients_key])

        # create one array for each output of `cg_couple_array`
        for output_index in range(len(o3_lambdas)):
            o3_lambda = o3_lambdas[output_index]
            parity = int((-1) ** (l1 + l2 + o3_lambda))
            o3_sigma = sigmas_1[i] * sigmas_2[i] * parity
            new_keys_values.append([o3_lambda, o3_sigma] + data.keys_values[i])

            new_shape: List[int] = []
            for axis in range(len(shape)):
                if axis == axis_1:
                    new_shape.append(2 * o3_lambda + 1)
                elif axis != axis_2:
                    new_shape.append(shape[axis])

            new_arrays.append(outputs[output_index].reshape(new_shape))
            new_parents.append(data.parents[i])

    return CouplingData(
        key_names=["o3_lambda", "o3_sigma"] + data.key_names,
        keys_values=new_keys_values,
        arrays=new_arrays,
        parents=new_parents,
    )


def do_decoupling(
    data: CouplingData,
    axis: int,
    lambda_name: str,
    sigma_name: Optional[str],
    l1_name: str,
    l2_name: str,
    cartesian_1: bool,
    cartesian_2: bool,
    cg_coefficients: TensorMap,
) -> CouplingData:
    """
    Undo a single coupling step performed by :py:func:`do_coupling`.

    The ``axis`` of the arrays (which transforms like a single spherical harmonic
    ``Y^M_L``, with ``L`` given by the ``lambda_name`` key) is split back into a product
    of two spherical harmonics, with degrees given by the ``l1_name`` and ``l2_name``
    keys. This is the inverse of coupling those two terms via Clebsch-Gordan
    coefficients.

    All arrays that share the same keys except for ``lambda_name`` and ``sigma_name``
    are grouped together and summed through :py:func:`cg_uncouple_array`, so the
    multiple ``L`` arrays created by the forward coupling collapse back into a single
    array.

    :param data: keys and values of the blocks, in coupled form
    :param axis: axis of the arrays to split
    :param lambda_name: name of the key holding the degree of the coupled spherical
        harmonic
    :param sigma_name: name of the key holding the parity of the coupled spherical
        harmonic, if it is part of the keys
    :param l1_name: name of the key holding the degree of the first spherical harmonic
        to recover
    :param l2_name: name of the key holding the degree of the second spherical harmonic
        to recover
    :param cartesian_1: should the first axis use the Cartesian ``(x, y, z)`` order?
    :param cartesian_2: should the second axis use the Cartesian ``(x, y, z)`` order?
    :param cg_coefficients: pre-computed Clebsch-Gordan coefficients

    :return: keys and values of the blocks, with ``lambda_name`` and ``sigma_name``
        removed from the keys, and ``axis`` replaced by two axes.
    """
    key_names = data.key_names
    keys_values = data.keys_values

    lambda_column = key_names.index(lambda_name)
    sigma_column = -1
    if sigma_name is not None:
        sigma_column = key_names.index(sigma_name)
    l1_column = key_names.index(l1_name)
    l2_column = key_names.index(l2_name)

    # group arrays by all key dimensions except `lambda_name` and `sigma_name`, which
    # are also the keys of the output
    kept_columns: List[int] = []
    for column in range(len(key_names)):
        if column != lambda_column and column != sigma_column:
            kept_columns.append(column)

    group_indexes: Dict[str, int] = {}
    group_members: List[List[int]] = []
    for i in range(len(keys_values)):
        signature = ",".join([str(keys_values[i][column]) for column in kept_columns])
        if signature in group_indexes:
            group_members[group_indexes[signature]].append(i)
        else:
            group_indexes[signature] = len(group_members)
            group_members.append([i])

    # CG coefficients for each value of l1, l2 and set of o3_lambda
    all_coefficients: Dict[str, torch.Tensor] = {}

    new_keys_values: List[List[int]] = []
    new_arrays: List[torch.Tensor] = []
    new_parents: List[int] = []
    for members in group_members:
        first = members[0]
        l1 = keys_values[first][l1_column]
        l2 = keys_values[first][l2_column]

        shape = data.arrays[first].shape
        n_before = 1
        for i in range(axis):
            n_before *= shape[i]
        n_after = 1
        for i in range(axis + 1, len(shape)):
            n_after *= shape[i]

        o3_lambdas: List[int] = []
        group_arrays: List[torch.Tensor] = []
        for i in members:
            o3_lambda = keys_values[i][lambda_column]
            o3_lambdas.append(o3_lambda)
            group_arrays.append(
                data.arrays[i].reshape(n_before, 2 * o3_lambda + 1, n_after)
            )

        coefficients_key = f"{l1} {l2} {o3_lambdas}"
        if coefficients_key not in all_coefficients:
            all_coefficients[coefficients_key] = cg_coupling_matrix(
                cg_coefficients,
                l1,
                l2,
                o3_lambdas,
                cartesian_1=cartesian_1,
                cartesian_2=cartesian_2,
            )

        uncoupled = cg_uncouple_array(
            group_arrays, l1, l2, all_coefficients[coefficients_key]
        )

        new_shape: List[int] = []
        for i in range(len(shape)):
            if i == axis:
                new_shape.append(2 * l1 + 1)
                new_shape.append(2 * l2 + 1)
            else:
                new_shape.append(shape[i])

        new_arrays.append(uncoupled.reshape(new_shape))
        new_parents.append(data.parents[first])
        new_keys_values.append([keys_values[first][column] for column in kept_columns])

    return CouplingData(
        key_names=[key_names[column] for column in kept_columns],
        keys_values=new_keys_values,
        arrays=new_arrays,
        parents=new_parents,
    )
