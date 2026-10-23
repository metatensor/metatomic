"""
Conversion of :py:class:`TensorMap` between a Cartesian product basis (``xyz_1``,
``xyz_2``, ... components) and a basis of real spherical harmonics (``o3_mu`` component,
with ``o3_lambda`` and ``o3_sigma`` keys).
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


class _CouplingData:
    """
    Data of a :py:class:`TensorMap` while it is converted between Cartesian and
    spherical forms. During the conversion we only keep track of the keys and the values
    of the blocks, and create the full metadata at the end.

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
    def create(keys: Labels, blocks: List[TensorBlock]) -> "_CouplingData":
        keys_values: List[List[int]] = keys.values.to(torch.int64).tolist()
        arrays: List[torch.Tensor] = []
        parents: List[int] = []
        for block_index, block in enumerate(blocks):
            arrays.append(block.values)
            parents.append(block_index)

        return _CouplingData(keys.names, keys_values, arrays, parents)


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

    Gradients of the input blocks are not supported yet.

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

    data = _CouplingData.create(tensor.keys, input_blocks)

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

        data = _CouplingData(
            key_names=new_names + data.key_names,
            keys_values=new_keys_values,
            arrays=new_arrays,
            parents=data.parents,
        )

        spherical = _spherical_tensor(
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
        data = _do_coupling(
            data=data,
            axis_1=axes_to_convert[-2],
            axis_2=axes_to_convert[-1],
            keep_l_in_keys=keep_l,
            iteration_index=iteration_index,
            cg_coefficients=coefficients,
        )

        axes_to_convert.pop()
        iteration_index += 1

    spherical = _spherical_tensor(
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

    Gradients of the input blocks are not transformed, and are not included in the
    output.

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
    data = _CouplingData.create(keys, input_blocks)

    if tensor_rank == 1:
        # transform the L=1 spherical harmonics back into Cartesian components. They
        # are stored as (mu=-1, 0, 1), i.e. (y, z, x), so a `roll` by +1 brings them
        # back to (x, y, z).
        arrays = [array.roll(1, mu_position + 1) for array in data.arrays]
        data = _CouplingData(
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

        axis = mu_position + 1
        for m in range(tensor_rank, 1, -1):
            if m > 2:
                lambda_right_name = f"k_{m - 2}"
            else:
                lambda_right_name = "l_1"

            data = _do_decoupling(
                data=data,
                axis=axis,
                l_left_name=f"l_{m}",
                lambda_right_name=lambda_right_name,
                cartesian_right=m == 2,
                cg_coefficients=coefficients,
            )
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

    # remove the now-unused spherical keys (`l_1` is only left over for rank 1 tensors,
    # where no decoupling happened)
    spherical_names = ["o3_lambda", "o3_sigma", "l_1"]
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


def _spherical_tensor(
    tensor: TensorMap,
    data: _CouplingData,
    first_component: int,
    last_component: int,
    remove_blocks_threshold: Optional[float],
) -> TensorMap:
    """
    Create the output of :py:func:`cartesian_to_spherical`, where the components from
    ``first_component`` to ``last_component`` of the blocks in ``tensor`` have been
    replaced by a single ``o3_mu`` component.

    :param tensor: input of :py:func:`cartesian_to_spherical`
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

    all_mu_labels: List[Labels] = []
    for o3_lambda in range(last_component - first_component + 2):
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


def _do_coupling(
    data: _CouplingData,
    axis_1: int,
    axis_2: int,
    keep_l_in_keys: bool,
    iteration_index: int,
    cg_coefficients: TensorMap,
) -> _CouplingData:
    """
    Go from an uncoupled product basis that behave like a product of spherical harmonics
    to a coupled basis that behaves like a single spherical harmonic.

    This function takes in arrays where two of the axes (``axis_1`` and ``axis_2``)
    behave like spherical harmonics ``Y^m1_l1`` and ``Y^m2_l2``, and project them onto a
    single spherical harmonic ``Y^M_L``, using Clebsch-Gordan coefficients (see
    :py:func:`cg_couple_array`). ``axis_1`` is always a Cartesian axis in the ``(x, y,
    z)`` order, and ``axis_2`` is a Cartesian axis for the first iteration, and the
    result of a previous coupling for later iterations.

    The output will contain many arrays for each array in the input, matching all the
    different ``L`` (called ``o3_lambda`` in the code) required to do a full projection.

    This process can be iterated: a multi-dimensional array that is the product of many
    ``Y^m_l`` can be turned into a set of multiple terms transforming as a single
    ``Y^M_L``.

    :param data: keys and values of the blocks
    :param axis_1: first axis of the arrays behaving like spherical harmonics
    :param axis_2: second axis of the arrays behaving like spherical harmonics
    :param keep_l_in_keys: whether ``l1`` and ``l2`` (the original spherical harmonic
        degrees) should be kept in the keys. This can be useful to undo this
        transformation (or even required if there is more than one path to get to a
        given value for ``o3_lambda``)
    :param iteration_index: when iterating the coupling, this should be the number of
        iterations already done (i.e. the number of time this function has been called)
    :param cg_coefficients: pre-computed set of Clebsch-Gordan coefficients

    :return: keys and values of the blocks using the coupled basis. The keys contain
        the same dimensions as the input keys, plus ``o3_lambda`` and ``o3_sigma``. The
        axes ``axis_1`` and ``axis_2`` are replaced by a single axis.
    """
    assert axis_2 == axis_1 + 1

    key_names = data.key_names

    if iteration_index == 0:
        assert "o3_lambda" not in key_names
        new_names = ["o3_lambda", "o3_sigma"]
        if keep_l_in_keys:
            new_names.append("l_2")
            new_names.append("l_1")
        new_names = new_names + key_names
    else:
        # the previous `o3_lambda` becomes `k_{iteration_index}`, and the previous
        # `o3_sigma` is replaced by the new one
        assert keep_l_in_keys
        assert key_names[0] == "o3_lambda" and key_names[1] == "o3_sigma"
        new_names = [
            "o3_lambda",
            "o3_sigma",
            f"l_{iteration_index + 2}",
            f"k_{iteration_index}",
        ] + key_names[2:]

    # CG coefficients for each value of l2 (l1 is always 1)
    all_coefficients: Dict[int, torch.Tensor] = {}

    new_keys_values: List[List[int]] = []
    new_arrays: List[torch.Tensor] = []
    new_parents: List[int] = []
    for i in range(len(data.arrays)):
        key = data.keys_values[i]
        if iteration_index == 0:
            old_sigma = 1
            other_keys: List[int] = []
            if keep_l_in_keys:
                other_keys = [1, 1]
            other_keys = other_keys + key
        else:
            old_sigma = key[1]
            other_keys = [1, key[0]] + key[2:]

        # get l1, l2 from the array's shape
        values = data.arrays[i]
        shape = values.shape
        l1 = (shape[axis_1] - 1) // 2
        l2 = (shape[axis_2] - 1) // 2

        # generate the set of o3_lambda to compute
        o3_lambdas = [o3_lambda for o3_lambda in range(abs(l1 - l2), l1 + l2 + 1)]

        if l2 not in all_coefficients:
            all_coefficients[l2] = cg_coupling_matrix(
                cg_coefficients,
                l1,
                l2,
                o3_lambdas,
                cartesian_1=True,
                cartesian_2=iteration_index == 0,
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
        outputs = cg_couple_array(array, o3_lambdas, all_coefficients[l2])

        # create one array for each output of `cg_couple_array`
        for output_index in range(len(o3_lambdas)):
            o3_lambda = o3_lambdas[output_index]
            o3_sigma = old_sigma * int((-1) ** (l1 + l2 + o3_lambda))
            new_keys_values.append([o3_lambda, o3_sigma] + other_keys)

            new_shape: List[int] = []
            for axis in range(len(shape)):
                if axis == axis_1:
                    new_shape.append(2 * o3_lambda + 1)
                elif axis != axis_2:
                    new_shape.append(shape[axis])

            new_arrays.append(outputs[output_index].reshape(new_shape))
            new_parents.append(data.parents[i])

    return _CouplingData(new_names, new_keys_values, new_arrays, new_parents)


def _do_decoupling(
    data: _CouplingData,
    axis: int,
    l_left_name: str,
    lambda_right_name: str,
    cartesian_right: bool,
    cg_coefficients: TensorMap,
) -> _CouplingData:
    """
    Undo a single coupling step performed by :py:func:`_do_coupling`.

    The ``axis`` of the arrays (which transforms like a single spherical harmonic
    ``Y^M_L`` with ``L = o3_lambda``) is split back into a product of two spherical
    harmonics: an ``L=1`` term (given by ``l_left_name``, always equal to 1) at
    ``axis``, using the Cartesian ``(x, y, z)`` order; and an intermediate term (given
    by ``lambda_right_name``) at ``axis + 1``. This is the inverse of coupling those two
    terms into ``o3_lambda`` via Clebsch-Gordan coefficients.

    All arrays that share the same keys except for ``o3_lambda`` (and the dependent
    ``o3_sigma``) are grouped together and summed through
    :py:func:`cg_uncouple_array`, so the multiple ``o3_lambda`` arrays created by the
    forward coupling collapse back into a single array.

    :param data: keys and values of the blocks, in coupled form
    :param axis: axis of the arrays to split
    :param l_left_name: name of the key holding the degree of the left-hand spherical
        harmonic to recover (``l_{m}``); its value is always 1
    :param lambda_right_name: name of the key holding the degree of the intermediate
        (right-hand) spherical harmonic to recover (``k_{m - 2}``, or ``l_1`` for the
        last decoupling step)
    :param cartesian_right: should the right-hand spherical harmonic use the Cartesian
        ``(x, y, z)`` order? This is only possible for the last decoupling step.
    :param cg_coefficients: pre-computed Clebsch-Gordan coefficients

    :return: keys and values of the blocks, with ``o3_lambda`` set to the recovered
        intermediate degree, ``l_left_name`` and ``lambda_right_name`` removed from the
        keys, and ``axis`` replaced by two axes.
    """
    key_names = data.key_names
    keys_values = data.keys_values

    lambda_column = key_names.index("o3_lambda")
    sigma_column = key_names.index("o3_sigma")
    left_column = key_names.index(l_left_name)
    right_column = key_names.index(lambda_right_name)

    # group arrays by all key dimensions except `o3_lambda` and `o3_sigma`; and keep
    # everything except these and the two dimensions we are recovering in the output
    group_columns: List[int] = []
    kept_columns: List[int] = []
    for column in range(len(key_names)):
        if column != lambda_column and column != sigma_column:
            group_columns.append(column)
            if column != left_column and column != right_column:
                kept_columns.append(column)

    group_indexes: Dict[str, int] = {}
    group_members: List[List[int]] = []
    for i in range(len(keys_values)):
        signature = ",".join([str(keys_values[i][column]) for column in group_columns])
        if signature in group_indexes:
            group_members[group_indexes[signature]].append(i)
        else:
            group_indexes[signature] = len(group_members)
            group_members.append([i])

    # CG coefficients for each value of l2 and set of o3_lambda
    all_coefficients: Dict[str, torch.Tensor] = {}

    new_keys_values: List[List[int]] = []
    new_arrays: List[torch.Tensor] = []
    new_parents: List[int] = []
    for members in group_members:
        first = members[0]
        l1 = keys_values[first][left_column]  # always 1
        l2 = keys_values[first][right_column]

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

        coefficients_key = f"{l2} {o3_lambdas}"
        if coefficients_key not in all_coefficients:
            all_coefficients[coefficients_key] = cg_coupling_matrix(
                cg_coefficients,
                l1,
                l2,
                o3_lambdas,
                cartesian_1=True,
                cartesian_2=cartesian_right,
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

        # recover the `o3_sigma` of the intermediate. The forward coupling set
        # `o3_sigma = old_sigma * (-1)^(l1 + l2 + o3_lambda)`, and `(-1)^x` is its own
        # inverse, so we recover `old_sigma` the same way.
        old_sigma = keys_values[first][sigma_column] * int(
            (-1) ** (l1 + l2 + o3_lambdas[0])
        )

        new_key = [l2, old_sigma]
        for column in kept_columns:
            new_key.append(keys_values[first][column])
        new_keys_values.append(new_key)

    new_names = ["o3_lambda", "o3_sigma"]
    for column in kept_columns:
        new_names.append(key_names[column])

    return _CouplingData(new_names, new_keys_values, new_arrays, new_parents)
