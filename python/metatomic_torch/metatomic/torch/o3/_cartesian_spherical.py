"""
Conversion of :py:class:`TensorMap` between a Cartesian product basis (``xyz_1``,
``xyz_2``, ... components) and a basis of real spherical harmonics (``o3_mu`` component,
with ``o3_lambda`` and ``o3_sigma`` keys).
"""

from typing import List, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from ._clebsch_gordan import resolve_cg_coefficients
from ._coupling import (
    CouplingData,
    build_spherical_tensor,
    check_no_gradients,
    do_coupling,
    do_decoupling,
)
from ._utils import copy_tensormap_info


def cartesian_to_spherical(
    tensor: TensorMap,
    components: List[str],
    keep_l_in_keys: Optional[bool] = None,
    remove_blocks_threshold: Optional[float] = 1e-9,
    cg_coefficients: Optional[TensorMap] = None,
) -> TensorMap:
    """
    Transform a ``tensor`` of arbitrary rank from Cartesian form to a spherical form.

    Starting from a tensor on a basis of product of Cartesian coordinates, this function
    computes the same tensor using a basis of real spherical harmonics ``Y^M_L``. For
    example, a rank 1 tensor with a single ``xyz`` component would be represented as a
    single L=1 spherical harmonic; while a rank 5 tensor using a product basis ``ℝ^3 ⊗
    ℝ^3 ⊗ ℝ^3 ⊗ ℝ^3 ⊗ ℝ^3`` would require multiple blocks up to L=5 spherical harmonics.

    A single :py:class:`~metatensor.torch.TensorBlock` in the input might correspond to
    multiple :py:class:`~metatensor.torch.TensorBlock` in the output. The output keys
    will contain all the dimensions of the input keys, plus ``o3_lambda`` (indicating
    the spherical harmonics degree) and ``o3_sigma`` (indicating that this block is a
    proper or improper tensor with ``+1`` and ``-1`` respectively). If
    ``keep_l_in_keys`` is ``True`` or if the input tensor is a tensor of rank 3 or more,
    the keys will also contain multiple ``l_{i}`` and ``k_{i}`` dimensions, which
    indicate which angular momenta have been coupled together in which order to get this
    block.

    ``components`` specifies which ones of the components of the input
    :py:class:`~metatensor.torch.TensorMap` should be transformed from Cartesian to
    spherical. All these components will be replaced in the output by a single ``o3_mu``
    component, corresponding to the spherical harmonics ``M``.

    By default, symmetric tensors will only contain blocks corresponding to
    ``o3_sigma=1``. This is achieved by checking the norm of the blocks after the full
    calculation; and dropping any block with a norm below ``remove_blocks_threshold``.
    To keep all blocks regardless of their norm, you can set
    ``remove_blocks_threshold=None``.

    Gradients are not supported yet, and this function raises an error if any of the
    input blocks has gradients.

    :param tensor: input tensor, using a Cartesian product basis
    :param components: components of the input tensor to transform into spherical
        components
    :param keep_l_in_keys: should the output contains the values of angular momenta that
        were combined together? This defaults to ``False`` for rank 1 and 2 tensors, and
        ``True`` for all other tensors.

        Keys named ``l_{i}`` correspond to the input ``components``, with ``l_1`` being
        the last entry in ``components`` and ``l_N`` the first one. Keys named ``k_{i}``
        correspond to intermediary spherical components created during the calculation,
        i.e. a ``k_{i}`` used to be ``o3_lambda``.
    :param remove_blocks_threshold: Numerical tolerance to use when determining if a
        block's norm is zero or not. Blocks with zero norm will be excluded from the
        output. Set this to ``None`` to keep all blocks in the output.
    :param cg_coefficients: Clebsch-Gordan coefficients computed with
        :py:func:`calculate_cg_coefficients`, with ``lambda_max`` at least equal to the
        number of ``components``. This is optional except when using this function from
        TorchScript.

    :return: :py:class:`~metatensor.torch.TensorMap` containing spherical components
        instead of Cartesian components.
    """
    if len(tensor) == 0 or len(components) == 0:
        # nothing to do
        return tensor

    check_no_gradients(tensor, "cartesian_to_spherical")

    if not torch.jit.is_scripting():
        if not isinstance(components, list):
            raise TypeError(f"`components` should be a list, got {type(components)}")

    tensor_rank = len(components)
    if keep_l_in_keys is None:
        keep_l = tensor_rank >= 3
    else:
        keep_l = keep_l_in_keys

    axes_to_convert: List[int] = []
    all_component_names = tensor.component_names
    for name in components:
        if name in all_component_names:
            axes_to_convert.append(all_component_names.index(name) + 1)
        else:
            raise ValueError(f"'{name}' is not part of this tensor components")

    xyz_values = torch.arange(3, dtype=torch.int32, device=tensor.device)
    input_blocks = tensor.blocks()
    for block_index, block in enumerate(input_blocks):
        block_components = block.components
        for axis in axes_to_convert:
            labels = block_components[axis - 1]
            if not torch.equal(labels.values[:, 0], xyz_values):
                labels_values: List[int] = labels.values[:, 0].to(torch.int64).tolist()
                key = tensor.keys.entry(block_index)
                raise ValueError(
                    f"component '{labels.names[0]}' in block for {key.print()} "
                    f"should have [0, 1, 2] as values, got {labels_values} instead"
                )

    # we need components to be consecutive
    expected_axes = [
        axis for axis in range(axes_to_convert[0], axes_to_convert[-1] + 1)
    ]
    if axes_to_convert != expected_axes:
        raise ValueError(
            f"this function only supports consecutive components, {components} are not"
        )

    key_names = tensor.keys.names
    if "o3_lambda" in key_names:
        raise ValueError(
            "this tensor already has an `o3_lambda` key, "
            "is it already in spherical form?"
        )

    for i in range(1, tensor_rank + 1):
        if f"l_{i}" in key_names:
            raise ValueError(
                f"this tensor already has an `l_{i}` key, "
                "is it already in spherical form?"
            )

    if tensor_rank > 2 and not keep_l:
        raise ValueError(
            "`keep_l_in_keys` must be `True` for tensors of rank 3 and above"
        )

    data = CouplingData.create(tensor.keys, input_blocks)

    first_component = axes_to_convert[0] - 1
    last_component = axes_to_convert[-1] - 1

    if tensor_rank == 1:
        # Transform xyz dimensions to o3_lambda=1 dimensions. This is done with `roll`,
        # since (y, z, x) is the same as mu = (-1, 0, 1)
        new_names = ["o3_lambda", "o3_sigma"]
        if keep_l:
            new_names.append("l_1")

        new_keys_values: List[List[int]] = []
        new_arrays: List[torch.Tensor] = []
        for i in range(len(data.arrays)):
            new_key = [1, 1]
            if keep_l:
                new_key.append(1)
            new_keys_values.append(new_key + data.keys_values[i])
            new_arrays.append(data.arrays[i].roll(-1, axes_to_convert[0]))

        data = CouplingData(
            key_names=new_names + data.key_names,
            keys_values=new_keys_values,
            arrays=new_arrays,
            parents=data.parents,
        )

        spherical = build_spherical_tensor(
            tensor=tensor,
            data=data,
            first_component=first_component,
            last_component=last_component,
            remove_blocks_threshold=None,
        )
        return copy_tensormap_info(source=tensor, result=spherical)

    # General case: if there is more than one dimension, couple them with CG.
    #
    # We start from an array of shape [..., 3, 3, 3, 3, 3, ...] with as many 3 as
    # `len(components)`. Then we iteratively combine the two rightmost components into
    # as many new lambda/mu entries as required, until there is only one component left.
    # Each step will create multiple blocks (corresponding to the different o3_lambda
    # created by combining two o3_lambda=1 terms), that might on their turn create more
    # blocks if more combinations are required.
    #
    # For example, with a rank 3 tensor we go through the following:
    #
    # - Step 1: [..., 3, 3, 3, ...] => [..., 3, 1, ...] (o3_lambda=0, o3_sigma=+1)
    #                               => [..., 3, 3, ...] (o3_lambda=1, o3_sigma=-1)
    #                               => [..., 3, 5, ...] (o3_lambda=2, o3_sigma=+1)
    #
    # - Step 2: [..., 3, 1, ...] => [..., 3, ...] (o3_lambda=1, o3_sigma=+1)
    #           [..., 3, 3, ...] => [..., 1, ...] (o3_lambda=0, o3_sigma=-1)
    #                            => [..., 3, ...] (o3_lambda=1, o3_sigma=+1)
    #                            => [..., 5, ...] (o3_lambda=2, o3_sigma=-1)
    #           [..., 3, 5, ...] => [..., 3, ...] (o3_lambda=1, o3_sigma=+1)
    #                            => [..., 5, ...] (o3_lambda=2, o3_sigma=-1)
    #                            => [..., 7, ...] (o3_lambda=3, o3_sigma=+1)
    #
    # The xyz dimensions are not re-ordered to (y, z, x) = (mu=-1, 0, 1) before the
    # coupling; instead the CG coefficients are re-ordered to act directly on (x, y, z).
    coefficients = resolve_cg_coefficients(
        cg_coefficients, tensor_rank, tensor.block(0).values
    )

    iteration_index = 0
    while len(axes_to_convert) > 1:
        # The Cartesian axes behave like o3_lambda=1, o3_sigma=1 spherical harmonics.
        # After the first step, the right-hand side of the coupling is the result of the
        # previous step: its `o3_lambda` becomes `k_{iteration_index}` and its
        # `o3_sigma` is removed from the keys.
        sigmas_1: List[int] = []
        sigmas_2: List[int] = []
        prepared_keys: List[List[int]] = []
        if iteration_index == 0:
            prepared_names: List[str] = []
            if keep_l:
                prepared_names = ["l_2", "l_1"]

            for key in data.keys_values:
                sigmas_1.append(1)
                sigmas_2.append(1)
                if keep_l:
                    prepared_keys.append([1, 1] + key)
                else:
                    prepared_keys.append(key)

            prepared_names = prepared_names + data.key_names
        else:
            for key in data.keys_values:
                sigmas_1.append(1)
                sigmas_2.append(key[1])
                prepared_keys.append([1, key[0]] + key[2:])

            prepared_names = [
                f"l_{iteration_index + 2}",
                f"k_{iteration_index}",
            ] + data.key_names[2:]

        data = CouplingData(prepared_names, prepared_keys, data.arrays, data.parents)
        data = do_coupling(
            data=data,
            axis_1=axes_to_convert[-2],
            axis_2=axes_to_convert[-1],
            sigmas_1=sigmas_1,
            sigmas_2=sigmas_2,
            cartesian_1=True,
            cartesian_2=iteration_index == 0,
            cg_coefficients=coefficients,
        )

        axes_to_convert.pop()
        iteration_index += 1

    spherical = build_spherical_tensor(
        tensor=tensor,
        data=data,
        first_component=first_component,
        last_component=last_component,
        remove_blocks_threshold=remove_blocks_threshold,
    )
    return copy_tensormap_info(source=tensor, result=spherical)


def spherical_to_cartesian(
    tensor: TensorMap,
    new_component_names: Optional[List[str]] = None,
    cg_coefficients: Optional[TensorMap] = None,
) -> TensorMap:
    """
    Transform a ``tensor`` from spherical form back to Cartesian form.

    This is the exact inverse of :py:func:`cartesian_to_spherical`. Starting from a
    tensor expressed on a basis of real spherical harmonics ``Y^M_L`` (as produced by
    :py:func:`cartesian_to_spherical`), this function reconstructs the corresponding
    Cartesian product-basis tensor. Blocks that were removed from the spherical tensor
    (because their norm was zero, see the ``remove_blocks_threshold`` parameter of
    :py:func:`cartesian_to_spherical`) are treated as blocks full of zeros.

    The input keys must contain ``o3_lambda`` and ``o3_sigma``, and the single ``o3_mu``
    component will be replaced by a number of Cartesian components (according to the
    rank of the tensor, each with values ``[0, 1, 2]``). For tensors of rank 3 and above
    the keys must also contain the ``l_{i}`` and ``k_{i}`` dimensions describing the
    coupling history (these are added automatically by
    :py:func:`cartesian_to_spherical`), and the rank of the tensor is given by the
    number of ``l_{i}`` dimensions. For rank 1 and 2 tensors, the ``l_{i}`` dimensions
    can be missing, and the rank of the tensor is then given by the largest
    ``o3_lambda``.

    Gradients are not supported yet, and this function raises an error if any of the
    input blocks has gradients.

    :param tensor: input tensor, using a spherical harmonics basis
    :param new_component_names: names to give to the reconstructed Cartesian components.
        There should be as many of them as is the rank of the tensor. If ``None``, the
        components are named ``xyz_1``, ``xyz_2``, ...
    :param cg_coefficients: Clebsch-Gordan coefficients computed with
        :py:func:`calculate_cg_coefficients`, with ``lambda_max`` at least equal to the
        rank of the tensor. This is optional except when using this function from
        TorchScript to convert tensors of rank 2 and above.

    :return: :py:class:`~metatensor.torch.TensorMap` containing Cartesian components
        instead of spherical components.
    """
    if len(tensor) == 0:
        # nothing to do
        return tensor

    check_no_gradients(tensor, "spherical_to_cartesian")

    keys = tensor.keys
    key_names = keys.names
    if "o3_lambda" not in key_names or "o3_sigma" not in key_names:
        raise ValueError(
            "this tensor is missing one of `o3_lambda` or `o3_sigma` in the keys, "
            "is it already in Cartesian form?"
        )

    if "o3_mu" not in tensor.component_names:
        raise ValueError(
            "this tensor does not have an `o3_mu` component, "
            "is it already in Cartesian form?"
        )

    lambda_min = int(torch.min(keys.column("o3_lambda")))
    lambda_max = int(torch.max(keys.column("o3_lambda")))
    if lambda_min < 0:
        raise ValueError("expected all o3_lambda to be >= 0 in the keys")

    if "l_1" in key_names:
        # the rank is given by the number of `l_{i}` dimensions in the keys
        tensor_rank = 1
        while f"l_{tensor_rank + 1}" in key_names:
            tensor_rank += 1

        if lambda_max > tensor_rank:
            raise ValueError(
                f"found o3_lambda={lambda_max} in the keys, which is not possible "
                f"for a tensor of rank {tensor_rank} (with {tensor_rank} `l_{{i}}` "
                "dimensions in the keys)"
            )
    else:
        # add back l_1, l_2 keys if they are missing from a rank 1 or 2 tensor
        tensor_rank = lambda_max
        if tensor_rank < 1:
            raise ValueError("expected o3_lambda to go at least to 1 in the keys")

        ones = torch.ones(len(keys), dtype=keys.values.dtype, device=keys.values.device)
        if tensor_rank <= 2:
            keys = keys.insert(0, "l_1", ones)
        if tensor_rank == 2:
            keys = keys.insert(0, "l_2", ones)

    # check the l_{i} and k_{i} keys
    key_names = keys.names
    for n in range(1, tensor_rank + 1):
        if f"l_{n}" not in key_names:
            raise ValueError(f"expected a l_{n} dimension in the keys")

        if not bool(torch.all(keys.column(f"l_{n}") == 1)):
            raise ValueError(f"expected only 1 as value for the l_{n} dimension")

        if n > 2:
            if f"k_{n - 2}" not in key_names:
                raise ValueError(f"expected a k_{n - 2} dimension in the keys")

            if int(torch.max(keys.column(f"k_{n - 2}"))) >= tensor_rank:
                raise ValueError(
                    f"expected all values for k_{n - 2} to be below {tensor_rank}"
                )

    if new_component_names is None:
        component_names = [f"xyz_{i}" for i in range(1, tensor_rank + 1)]
    else:
        if len(new_component_names) != tensor_rank:
            raise ValueError(
                f"`new_component_names` should have {tensor_rank} entries (the rank "
                f"of the tensor), got {len(new_component_names)}"
            )
        component_names = new_component_names

    # position of the (single) `o3_mu` component that will be expanded into the
    # `tensor_rank` Cartesian components
    mu_position = tensor.component_names.index("o3_mu")

    input_blocks = tensor.blocks()
    data = CouplingData.create(keys, input_blocks)

    if tensor_rank == 1:
        # transform the L=1 spherical harmonics back into Cartesian components. They
        # are stored as (mu=-1, 0, 1), i.e. (y, z, x), so a `roll` by +1 brings them
        # back to (x, y, z).
        arrays = [array.roll(1, mu_position + 1) for array in data.arrays]
        data = CouplingData(
            key_names=data.key_names,
            keys_values=data.keys_values,
            arrays=arrays,
            parents=data.parents,
        )
    else:
        # De-couple the basis back into a product of L=1 spherical harmonics.
        #
        # This reverses the iterative coupling done by `cartesian_to_spherical`: at each
        # step we split off the left-most L=1 spherical harmonic (`l_{m}`) from the
        # coupled `o3_lambda`, leaving the remaining (still coupled) intermediate to the
        # right. The intermediate is decoupled further on the next iteration, until only
        # L=1 spherical harmonics remain.
        #
        # The CG coefficients are re-ordered to directly produce the Cartesian (x, y, z)
        # order for the L=1 spherical harmonics, instead of (y, z, x) = (mu=-1, 0, 1).
        coefficients = resolve_cg_coefficients(
            cg_coefficients, tensor_rank, tensor.block(0).values
        )

        lambda_name = "o3_lambda"
        sigma_name: Optional[str] = "o3_sigma"
        axis = mu_position + 1
        for m in range(tensor_rank, 1, -1):
            if m > 2:
                lambda_right_name = f"k_{m - 2}"
            else:
                lambda_right_name = "l_1"

            data = do_decoupling(
                data=data,
                axis=axis,
                lambda_name=lambda_name,
                sigma_name=sigma_name,
                l1_name=f"l_{m}",
                l2_name=lambda_right_name,
                cartesian_1=True,
                cartesian_2=m == 2,
                cg_coefficients=coefficients,
            )

            # the next step splits the intermediate spherical harmonic
            lambda_name = lambda_right_name
            sigma_name = None
            axis += 1

    # create the output metadata, the reconstructed components occupy the positions
    # `mu_position, ..., mu_position + tensor_rank - 1`
    xyz_values = torch.arange(3, dtype=torch.int32, device=tensor.device)
    xyz_labels: List[Labels] = []
    for name in component_names:
        xyz_labels.append(Labels([name], xyz_values.reshape(-1, 1)))

    new_blocks: List[TensorBlock] = []
    for i in range(len(data.arrays)):
        parent = input_blocks[data.parents[i]]
        components = parent.components
        new_blocks.append(
            TensorBlock(
                values=data.arrays[i],
                samples=parent.samples,
                components=components[:mu_position]
                + xyz_labels
                + components[mu_position + 1 :],
                properties=parent.properties,
            )
        )

    # remove the now-unused spherical keys (`o3_lambda` and `o3_sigma` are only left
    # over for rank 1 tensors, where no decoupling happened)
    spherical_names = ["o3_lambda", "o3_sigma"]
    for n in range(1, tensor_rank + 1):
        spherical_names.append(f"l_{n}")
    kept_columns: List[int] = []
    for column, name in enumerate(data.key_names):
        if name not in spherical_names:
            kept_columns.append(column)

    if len(kept_columns) == 0:
        # there is nothing left in the keys, use a placeholder dimension
        new_keys = Labels(
            "_",
            torch.zeros((len(new_blocks), 1), dtype=torch.int32, device=keys.device),
        )
    else:
        new_keys_values: List[List[int]] = []
        for key in data.keys_values:
            new_keys_values.append([key[column] for column in kept_columns])

        # removing the spherical dimensions could create duplicated keys for invalid
        # inputs, so we don't use `assume_unique` here
        new_keys = Labels(
            [data.key_names[column] for column in kept_columns],
            torch.tensor(new_keys_values, dtype=torch.int32, device=keys.device),
        )

    return copy_tensormap_info(source=tensor, result=TensorMap(new_keys, new_blocks))
