import abc
import ctypes
import functools
import math
from typing import Optional

import metatensor as mts
import numpy as np
import torch
from ctypes_dlpack import (
    DLDevice,
    DLManagedTensorVersioned,
    DLPackArray,
    DLPackVersion,
)
from metatensor import Labels, TensorBlock, TensorMap
from metatensor._c_api import mts_array_t
from metatensor._status import check_status

from .._capabilities import ModelCapabilities
from .._quantity import Quantity
from .._system import PairListOptions, System
from ._base import BaseModel


# Standard quantities for which gradients can be computed with autograd. These are the
# only standard quantities that can contain gradients.
_AUTOGRAD_STANDARD_QUANTITIES = ("energy", "energy_ensemble", "energy_uncertainty")
_AUTOGRAD_GRADIENTS = ("positions", "strain")


def _wrap_method(implementation, name: str):
    """
    Wrap the ``implementation`` of method ``name`` to add the behavior defined in the
    corresponding ``TorchModel._<name>_wrapper`` function.
    """

    @functools.wraps(implementation)
    def wrapper(self, *args):
        if getattr(type(self), name) is not wrapper:
            # this wrapper was called through `super().<name>()` in a subclass, only
            # run the implementation and let the subclass wrapper add the behavior
            return implementation(self, *args)

        add_behavior = getattr(TorchModel, f"_{name}_wrapper")
        return add_behavior(self, implementation, *args)

    return wrapper


class TorchModel(BaseModel, torch.nn.Module):
    """
    Base class for PyTorch-based atomistic models.

    This class provides a convenient way to implement models that use PyTorch for their
    computations. It is also a :py:class:`torch.nn.Module`, so parameters, buffers and
    sub-modules can be registered as usual.

    Subclasses should implement the typical :py:class:`torch.nn.Module`'s
    :py:meth:`forward` instead of :py:meth:`BaseModel.execute_inner` from
    :py:class:`BaseModel`. The :py:meth:`execute_inner` method is implemented by
    :py:class:`TorchModel` to handle gradients with autograd.

    Compared to :py:class:`BaseModel`, this class:

    - sets the :py:attr:`System.arrays_backend` to ``"torch"`` before calling
      :py:meth:`forward`, ensuring that the system's data will be accessible as
      :py:class:`torch.Tensor`;
    - uses automatic differentiation to compute the gradients with respect to
      ``"positions"`` and ``"strain"`` of outputs with ``sample_kind="system"``, unless
      the model already declares these gradients in its :py:meth:`capabilities`. This is
      done for :ref:`standard quantities <standard-quantities>` which support gradients,
      and for any custom quantity with ``sample_kind="system"``. The gradients computed
      with autograd are automatically added to the capabilities of the model. For these
      gradients to be correct, the outputs must be computed from
      :py:attr:`System.positions`, :py:attr:`System.cell` and the pair vectors in
      :py:meth:`System.pairs`.
    - collects the pair lists and additional inputs requested by all the sub-modules of
      this model in :py:meth:`requested_pair_lists` and :py:meth:`requested_inputs`. Any
      sub-module defining a ``requested_pair_lists()`` or ``requested_inputs()`` method
      will be included, and duplicated requests will be merged together.

    .. warning::

        This class does not support :py:func:`torch.jit.script` or
        :py:func:`torch.compile` at the top level, only eager PyTorch. It is still
        possible to script or compile sub-modules of the model:

        .. code-block:: python

            class MyScriptedModel(TorchModel):
                def __init__(self):
                    super().__init__()
                    self.submodule = torch.jit.script(MySubModule())

                def forward(self, systems, outputs, selected_atoms):
                    ...
                    self.submodule(...)
                    ...


            @torch.compile(dynamic=True)
            def compiled_function(positions, cell, pair_vectors, pair_samples):
                pass


            class MyCompiledModel(TorchModel):
                def __init__(self):
                    super().__init__()

                def forward(self, systems, outputs, selected_atoms):
                    ...
                    compiled_function(...)
                    ...
    """

    def __init__(self):
        super().__init__()

    def __init_subclass__(cls, **kwargs):
        """
        Wrap the methods defined by the subclass to add the behavior of
        :py:class:`TorchModel` on top of them.
        """
        super().__init_subclass__(**kwargs)

        if "execute_inner" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} can not override execute_inner(), since it is "
                "defined by TorchModel. Implement forward() instead"
            )

        for name in ("capabilities", "requested_pair_lists", "requested_inputs"):
            implementation = cls.__dict__.get(name)

            if implementation is None:
                # this specific subclass does not override this method, it will use the
                # one from the parent. @abc.abstractmethod will make sure everything
                # is available at instantiation time.
                continue

            setattr(cls, f"_raw_{name}", implementation)
            setattr(cls, name, _wrap_method(implementation, name))

    def _raw_requested_pair_lists(
        self,
        requested_outputs: list[Quantity],
    ) -> list[PairListOptions]:
        """
        List the pair lists (neighbor lists) this model needs as input when computing
        the requested outputs.

        This method should be overridden by subclasses to specify the pair lists that
        the model requires. By default, it returns an empty list, indicating that no
        pair lists are needed.

        The pair lists requested by any sub-module of this model (any
        :py:class:`torch.nn.Module` with a ``requested_pair_lists()`` method) are
        automatically added to the ones returned by the implementation in the subclass.

        :param requested_outputs: outputs the model will compute. As for
            :py:meth:`forward`, the gradients computed with autograd are removed from
            these requests.
        """
        return []

    requested_pair_lists = _wrap_method(
        _raw_requested_pair_lists, "requested_pair_lists"
    )

    def _raw_requested_inputs(
        self,
        requested_outputs: list[Quantity],
    ) -> list[Quantity]:
        """
        List the additional per-system inputs this model needs when computing the
        requested outputs.

        This method should be overridden by subclasses to specify any additional
        per-system inputs that the model requires. By default, it returns an empty list,
        indicating that no additional inputs are needed.

        The inputs requested by any sub-module of this model (any
        :py:class:`torch.nn.Module` with a ``requested_inputs()`` method) are
        automatically added to the ones returned by the implementation in the subclass.

        :param requested_outputs: outputs the model will compute. As for
            :py:meth:`forward`, the gradients computed with autograd are removed from
            these requests.
        """
        return []

    requested_inputs = _wrap_method(_raw_requested_inputs, "requested_inputs")

    @abc.abstractmethod
    def forward(
        self,
        systems: list[System],
        selected_atoms: Optional[Labels],
        outputs: list[Quantity],
    ) -> list[TensorMap]:
        """
        Run the model and compute the requested outputs.

        :param systems: systems to run the model on, with their
            :py:attr:`System.arrays_backend` set to ``"torch"``
        :param outputs: outputs the model should compute. The gradients computed with
            autograd are removed from these requests.
        :param selected_atoms: optional selection of atoms to compute outputs for, or
            ``None`` to use all atoms
        :return: the computed outputs, one :py:class:`metatensor.TensorMap` per
            requested output, in the same order
        """

    def execute_inner(
        self,
        systems: list[System],
        selected_atoms: Optional[Labels],
        requested_outputs: list[Quantity],
    ) -> list[TensorMap]:
        """
        Run the model by calling :py:meth:`forward`, and compute the gradients that
        the model does not compute explicitly with autograd.

        This method should not be called directly, use :py:func:`execute_model`
        instead. It should also not be overridden by subclasses.
        """
        inner_outputs, autograd_gradients = self._split_gradients(requested_outputs)
        all_gradients = set(g for gradients in autograd_gradients for g in gradients)

        with torch.enable_grad():
            torch_systems = []
            autograd_inputs = []
            for system in systems:
                torch_system, inputs = _torch_system(system, all_gradients)
                torch_systems.append(torch_system)
                autograd_inputs.append(inputs)

            outputs = self(torch_systems, selected_atoms, inner_outputs)

            if not isinstance(outputs, (list, tuple)) or len(outputs) != len(
                requested_outputs
            ):
                # let the caller report the error
                return outputs

            backward_calls = _BackwardCalls(outputs, autograd_gradients)
            return [
                _finalize_output(
                    output, gradients, torch_systems, autograd_inputs, backward_calls
                )
                for output, gradients in zip(outputs, autograd_gradients, strict=True)
            ]

    ### ============================================================================ ###

    def _split_gradients(
        self, requested_outputs: list[Quantity]
    ) -> tuple[list[Quantity], list[list[str]]]:
        """
        Split the gradients in ``requested_outputs`` between the ones computed
        explicitly by the model and the ones computed with autograd.

        :return: the outputs to request from the model, containing only the explicit
            gradients; and for each output the list of gradients to compute with
            autograd.
        """
        if all(len(request.gradients) == 0 for request in requested_outputs):
            return list(requested_outputs), [[] for _ in requested_outputs]

        capabilities = self._raw_capabilities()

        inner_outputs = []
        autograd_gradients = []
        for request in requested_outputs:
            declared = capabilities.find_output(request)
            explicit = [] if declared is None else declared.gradients

            gradients = [g for g in request.gradients if g not in explicit]
            if len(gradients) != 0 and not _supports_autograd(request):
                raise ValueError(
                    f"gradients of '{request.name}' with sample kind "
                    f"'{request.sample_kind}' can not be computed with autograd"
                )
            autograd_gradients.append(gradients)

            inner_outputs.append(
                Quantity(
                    name=request.name,
                    unit=request.unit,
                    sample_kind=request.sample_kind,
                    description=request.description,
                    gradients=[g for g in request.gradients if g in explicit],
                )
            )

        return inner_outputs, autograd_gradients

    def _capabilities_wrapper(self, implementation) -> ModelCapabilities:
        """
        Wrap a subclass ``implementation`` of capabilities() to add autograd gradients
        to the outputs.
        """
        capabilities = implementation(self)
        if not isinstance(capabilities, ModelCapabilities):
            # let the caller report the error
            return capabilities

        # work on a copy, the implementation might return a cached object
        capabilities = ModelCapabilities.from_dict(capabilities.to_dict())

        outputs = []
        for output in capabilities.outputs:
            if _supports_autograd(output):
                gradients = output.gradients
                for gradient in _AUTOGRAD_GRADIENTS:
                    if gradient not in gradients:
                        gradients.append(gradient)

                output = Quantity(
                    name=output.name,
                    unit=output.unit,
                    sample_kind=output.sample_kind,
                    description=output.description,
                    gradients=gradients,
                )
            outputs.append(output)

        capabilities.outputs = outputs
        return capabilities

    def _requested_pair_lists_wrapper(
        self,
        implementation,
        requested_outputs: list[Quantity],
    ) -> list[PairListOptions]:
        """
        Wrap a subclass ``implementation`` of ``requested_pair_lists()`` to collect the
        requests from all the sub-modules of this model.
        """
        inner_outputs, _ = self._split_gradients(requested_outputs)

        requested: list[PairListOptions] = []
        for module_name, module in self._modules_with("requested_pair_lists"):
            if module is self:
                new_requests = implementation(self, inner_outputs)
            elif isinstance(module, TorchModel):
                # only get the requests of this module, its sub-modules are already
                # part of `self._modules_with()`
                new_requests = module._raw_requested_pair_lists(inner_outputs)
            else:
                new_requests = module.requested_pair_lists(inner_outputs)

            for new_options in new_requests:
                if not isinstance(new_options, PairListOptions):
                    raise TypeError(
                        f"requested_pair_lists() of {module_name} must return a "
                        f"list of PairListOptions, got an element of type "
                        f"{type(new_options)}"
                    )

                new_options = PairListOptions(
                    cutoff=new_options.cutoff,
                    full_list=new_options.full_list,
                    strict=new_options.strict,
                    requestors=new_options.requestors,
                )
                new_options.add_requestor(module_name)

                for existing in requested:
                    if existing == new_options:
                        for requestor in new_options.requestors:
                            existing.add_requestor(requestor)
                        break
                else:
                    requested.append(new_options)

        return requested

    def _requested_inputs_wrapper(
        self,
        implementation,
        requested_outputs: list[Quantity],
    ) -> list[Quantity]:
        """
        Wrap a subclass ``implementation`` of ``requested_inputs()`` to collect the
        requests from all the sub-modules of this model.
        """
        inner_outputs, _ = self._split_gradients(requested_outputs)

        requested: dict[str, Quantity] = {}
        requestors: dict[str, str] = {}
        for module_name, module in self._modules_with("requested_inputs"):
            if module is self:
                new_inputs = implementation(self, inner_outputs)
            elif isinstance(module, TorchModel):
                # only get the requests of this module, its sub-modules are already
                # part of `self._modules_with()`
                new_inputs = module._raw_requested_inputs(inner_outputs)
            else:
                new_inputs = module.requested_inputs(inner_outputs)

            for new_input in new_inputs:
                if not isinstance(new_input, Quantity):
                    raise TypeError(
                        f"requested_inputs() of {module_name} must return a list of "
                        f"Quantity, got an element of type {type(new_input)}"
                    )

                name = new_input.name
                if name not in requested:
                    requested[name] = new_input
                    requestors[name] = module_name
                    continue

                existing = requested[name]
                for attribute in ["unit", "sample_kind"]:
                    new_value = getattr(new_input, attribute)
                    existing_value = getattr(existing, attribute)
                    if new_value != existing_value:
                        raise ValueError(
                            f"the input '{name}' is requested with different "
                            f"{attribute} by '{module_name}' "
                            f"({attribute}='{new_value}') and "
                            f"'{requestors[name]}' ({attribute}='{existing_value}')"
                        )

        return list(requested.values())

    def _modules_with(self, method: str):
        """
        Iterate over all the modules in this model (including the model itself) with a
        method named ``method``, yielding ``(name, module)`` pairs.
        """
        root_name = self.__class__.__name__
        for name, module in self.named_modules():
            if hasattr(module, method):
                yield (root_name if name == "" else f"{root_name}.{name}", module)


def _supports_autograd(quantity: Quantity) -> bool:
    """Wether a given quantity can have gradients computed with autograd."""
    if quantity.sample_kind != "system":
        return False

    if "::" in quantity.name:
        # custom quantities can have any gradients
        return True

    base_name = quantity.name.split("/")[0]
    return base_name in _AUTOGRAD_STANDARD_QUANTITIES


def _torch_system(
    system: System, gradients: set[str]
) -> tuple[System, dict[str, torch.Tensor]]:
    """
    Create a new :py:class:`System` containing the same data as ``system``, using
    :py:class:`torch.Tensor` for all arrays. If ``gradients`` contains ``"positions"``
    and/or ``"strain"``, the new system is set up to compute these gradients with
    autograd.

    The data is shared with ``system`` whenever possible, instead of being copied.

    :return: the new system, and the tensors to use as inputs to compute the
        ``gradients`` with autograd.
    """
    system.arrays_backend = "torch"
    positions = system.positions
    cell = system.cell

    inputs = {}
    if "positions" in gradients:
        positions = positions.detach().requires_grad_(True)
        inputs["positions"] = positions

    if "strain" in gradients:
        strain = torch.eye(
            3, dtype=positions.dtype, device=positions.device, requires_grad=True
        )
        positions = positions @ strain
        cell = cell @ strain
        inputs["strain"] = strain

    new_system = System(
        system.length_unit,
        system.types,
        positions,
        cell,
        system.pbc,
        arrays_backend="torch",
    )

    for options in system.known_pairs():
        pairs = system.pairs(options)
        samples = pairs.samples
        vectors = _mts_array_to_torch(pairs._raw_values, positions)
        # register the pair vectors given by the engine with autograd, to make sure
        # gradients can flow from them to the positions and cell
        vectors = _PairsAutograd.apply(positions, cell, vectors, samples)

        new_pairs = TensorBlock(
            values=vectors,
            samples=samples,
            components=pairs.components,
            properties=pairs.properties,
        )
        new_system.add_pairs(options, new_pairs)

    for name in system.known_custom_data():
        data = system.custom_data(name)
        new_system.add_custom_data(name, data.to(arrays="torch"))

    return new_system, inputs


def _mts_array_to_torch(array: mts_array_t, like: torch.Tensor) -> torch.Tensor:
    """
    Get the data in ``array`` as a :py:class:`torch.Tensor` on the same device as
    ``like``, without copying it. This goes through DLPack, and thus works for arrays
    created by any language, not only Python.

    This is a temporary workaround until the :py:class:`metatensor.TensorMap` and
    :py:class:`metatensor.TensorBlock` classes can return :py:class:`torch.Tensor`
    directly even when the data origin is not ``python.torch``.
    """
    device_type, device_id = like.__dlpack_device__()
    tensor = ctypes.POINTER(DLManagedTensorVersioned)()
    status = array.as_dlpack(
        array.ptr,
        ctypes.byref(tensor),
        DLDevice(device_type=device_type, device_id=device_id),
        None,
        DLPackVersion(major=1, minor=0),
    )
    check_status(status)
    return torch.from_dlpack(DLPackArray(tensor))


class _PairsAutograd(torch.autograd.Function):
    """
    Register the pair vectors computed by the engine in the computational graph, as
    ``vectors = positions[second] - positions[first] + cell_shifts @ cell``, without
    actually re-computing them.
    """

    @staticmethod
    def forward(ctx, positions, cell, vectors, samples):
        ctx.samples = samples
        ctx.positions_shape = positions.shape
        # WARNING: we are not using `torch.autograd.Function` in the usual way here.
        # Instead we take the already computed vectors and return them directly. This
        # still causes torch to register the right function for `backward`.
        return vectors

    @staticmethod
    def backward(ctx, vectors_grad):
        vectors_grad = vectors_grad.reshape(-1, 3)

        def column(name: str) -> torch.Tensor:
            # Once metatensor can return the samples as a torch.Tensor, we can use that
            # directly instead of going through numpy here.
            values = np.array(ctx.samples.column(name), dtype=np.int64)
            return torch.from_numpy(values).to(vectors_grad.device)

        positions_grad = None
        if ctx.needs_input_grad[0]:
            positions_grad = torch.zeros(
                ctx.positions_shape,
                dtype=vectors_grad.dtype,
                device=vectors_grad.device,
            )
            positions_grad.index_add_(0, column("second_atom"), vectors_grad)
            positions_grad.index_add_(0, column("first_atom"), vectors_grad, alpha=-1)

        cell_grad = None
        if ctx.needs_input_grad[1]:
            cell_shifts = torch.stack(
                [
                    column("cell_shift_a"),
                    column("cell_shift_b"),
                    column("cell_shift_c"),
                ],
                dim=1,
            ).to(vectors_grad.dtype)
            cell_grad = cell_shifts.T @ vectors_grad

        return positions_grad, cell_grad, None, None


class _BackwardCalls:
    """
    Count the backward calls needed to compute all the gradients with autograd, to
    only keep the computational graph alive until the last of them.

    Keeping the graph with ``retain_graph=True`` in the last call wastes memory, and
    is not supported by default by backward functions created by
    :py:func:`torch.compile`.
    """

    def __init__(self, outputs: list[TensorMap], gradients: list[list[str]]):
        # this must match the calls to `torch.autograd.grad` in `_autograd_gradients`
        self.remaining = 0
        for output, output_gradients in zip(outputs, gradients, strict=True):
            if len(output_gradients) == 0 or not isinstance(output, TensorMap):
                continue

            for block in output.blocks():
                values = block.values
                if isinstance(values, torch.Tensor) and values.requires_grad:
                    # one call for each entry along the components and properties
                    self.remaining += math.prod(values.shape[1:])

    def retain_graph(self) -> bool:
        """
        Register a new backward call, and check if the graph should be kept alive
        after it.
        """
        self.remaining -= 1
        return self.remaining > 0


def _finalize_output(
    output: TensorMap,
    gradients: list[str],
    systems: list[System],
    autograd_inputs: list[dict[str, torch.Tensor]],
    backward_calls: _BackwardCalls,
) -> TensorMap:
    """
    Add the ``gradients`` computed with autograd to ``output``, and detach all the
    arrays from the computational graph to allow exporting them through DLPack.
    """
    if not isinstance(output, TensorMap):
        # let the caller report the error
        return output

    blocks = []
    for block in output.blocks():
        new_block = mts.detach_block(block)

        if len(gradients) != 0:
            for parameter, gradient in _autograd_gradients(
                block, gradients, systems, autograd_inputs, backward_calls
            ):
                new_block.add_gradient(parameter, gradient)

        blocks.append(new_block)

    return TensorMap(output.keys, blocks)


def _autograd_gradients(
    block: TensorBlock,
    gradients: list[str],
    systems: list[System],
    autograd_inputs: list[dict[str, torch.Tensor]],
    backward_calls: _BackwardCalls,
) -> list[tuple[str, TensorBlock]]:
    """
    Compute the ``gradients`` of the per-system ``block`` with respect to the positions
    and/or strain of ``systems``, using autograd.

    The gradients are computed separately for each entry along the components and
    properties of ``block``, but simultaneously for all samples, since each system has
    a single sample and the different systems are independent.
    """
    values = block.values
    if not isinstance(values, torch.Tensor):
        raise TypeError(
            "outputs of TorchModel must contain torch.Tensor to compute gradients "
            f"with autograd, got {type(values)}"
        )

    sample_systems = np.asarray(block.samples.column("system")).tolist()
    n_samples = values.shape[0]
    # shape of the components and properties, shared with the gradients
    values_shape = values.shape[1:]
    flat_values = values.reshape(n_samples, -1)

    # inputs for autograd: positions and/or strain for all the systems
    parameters = [p for p in _AUTOGRAD_GRADIENTS if p in gradients]
    inputs = []
    for system_inputs in autograd_inputs:
        for parameter in parameters:
            inputs.append(system_inputs[parameter])

    # all_grads[i][column] is the gradient of `flat_values[:, column]` w.r.t. inputs[i]
    all_grads = [[] for _ in inputs]
    for column in range(flat_values.shape[1]):
        if values.requires_grad:
            grads = torch.autograd.grad(
                flat_values[:, column].sum(),
                inputs,
                retain_graph=backward_calls.retain_graph(),
                allow_unused=True,
                materialize_grads=True,
            )
        else:
            grads = [torch.zeros_like(input) for input in inputs]

        for i, grad in enumerate(grads):
            all_grads[i].append(grad.detach())

    # stack the different components & properties as the last dimensions
    all_grads = [
        torch.stack(grads, dim=-1).reshape(*grads[0].shape, *values_shape)
        if len(grads) != 0
        else torch.zeros(
            (*input.shape, *values_shape), dtype=values.dtype, device=values.device
        )
        for grads, input in zip(all_grads, inputs, strict=True)
    ]

    def grads_for(system_id: int, parameter: str) -> torch.Tensor:
        index = system_id * len(parameters) + parameters.index(parameter)
        return all_grads[index]

    results = []
    if "positions" in gradients:
        grad_samples = []
        grad_values = []
        for sample_id, system_id in enumerate(sample_systems):
            n_atoms = len(systems[system_id])
            atoms = np.arange(n_atoms, dtype=np.int32)
            grad_samples.append(
                np.stack(
                    [
                        np.full(n_atoms, sample_id, dtype=np.int32),
                        np.full(n_atoms, system_id, dtype=np.int32),
                        atoms,
                    ],
                    axis=1,
                )
            )
            grad_values.append(grads_for(system_id, "positions"))

        if len(grad_values) == 0:
            grad_samples = np.zeros((0, 3), dtype=np.int32)
            grad_values = torch.zeros(
                (0, 3, *values_shape), dtype=values.dtype, device=values.device
            )
        else:
            grad_samples = np.concatenate(grad_samples)
            grad_values = torch.concatenate(grad_values)

        results.append(
            (
                "positions",
                TensorBlock(
                    values=grad_values,
                    samples=Labels(
                        ["sample", "system", "atom"],
                        grad_samples,
                        assume_unique=True,
                    ),
                    components=[Labels.range("xyz", 3)] + block.components,
                    properties=block.properties,
                ),
            )
        )

    if "strain" in gradients:
        grad_values = [grads_for(system_id, "strain") for system_id in sample_systems]
        if len(grad_values) == 0:
            grad_values = torch.zeros(
                (0, 3, 3, *values_shape), dtype=values.dtype, device=values.device
            )
        else:
            grad_values = torch.stack(grad_values)

        results.append(
            (
                "strain",
                TensorBlock(
                    values=grad_values,
                    samples=Labels.range("sample", n_samples),
                    components=[Labels.range("xyz_1", 3), Labels.range("xyz_2", 3)]
                    + block.components,
                    properties=block.properties,
                ),
            )
        )

    return results
