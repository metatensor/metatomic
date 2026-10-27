import abc
import ctypes
import dataclasses
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from ctypes_dlpack import DLDevice, DLManagedTensorVersioned, DLPackArray, DLPackVersion
from metatensor import Labels, TensorBlock, TensorMap
from metatensor._status import check_status

from .._quantity import Quantity
from .._system import PairListOptions, System
from ._autograd import AutogradModel


@dataclasses.dataclass
class JaxBlock:
    """
    JAX counterpart of :py:class:`metatensor.TensorBlock`: ``values`` is a
    :py:class:`jax.Array` (possibly a tracer inside :py:class:`JaxModel`), the metadata
    are regular :py:class:`metatensor.Labels`.

    ``metatensor.TensorBlock`` only stores NumPy arrays and torch tensors, and JAX can
    not trace through it, which is why :py:class:`JaxModel` uses this class instead.
    """

    values: jax.Array
    samples: Labels
    components: list[Labels]
    properties: Labels
    gradients: dict[str, "JaxBlock"] = dataclasses.field(default_factory=dict)

    def add_gradient(self, parameter: str, gradient: "JaxBlock"):
        self.gradients[parameter] = gradient

    def gradient(self, parameter: str) -> "JaxBlock":
        return self.gradients[parameter]


@dataclasses.dataclass
class JaxTensorMap:
    """JAX counterpart of :py:class:`metatensor.TensorMap`, see :py:class:`JaxBlock`."""

    keys: Labels
    blocks: list[JaxBlock]

    def block(self, index: int = 0) -> JaxBlock:
        return self.blocks[index]


class JaxSystem:
    """
    Read-only view of a :py:class:`System` with :py:class:`jax.Array` data, given to
    :py:meth:`JaxModel.forward`. The arrays may be JAX tracers, so that the gradients
    with respect to positions and strain flow through all of them, including the pair
    vectors computed by the engine.
    """

    def __init__(self, length_unit, types, positions, cell, pbc, pairs, custom_data):
        self.length_unit = length_unit
        self.types = types
        self.positions = positions
        self.cell = cell
        self.pbc = pbc
        self._pairs = pairs
        self._custom_data = custom_data

    def __len__(self) -> int:
        return self.types.shape[0]

    def pairs(self, options: PairListOptions) -> JaxBlock:
        for known, block in self._pairs:
            if known == options:
                return block
        raise ValueError(f"no pair list with {options} in this system")

    def known_pairs(self) -> list[PairListOptions]:
        return [options for options, _ in self._pairs]

    def custom_data(self, name: str) -> JaxTensorMap:
        if name not in self._custom_data:
            raise ValueError(
                f"no custom data named '{name}' in this system (known: "
                f"{list(self._custom_data)}). Request it in requested_inputs(); "
                "execute_model(check_consistency=True) then checks it is provided"
            )
        return self._custom_data[name]

    def known_custom_data(self) -> list[str]:
        return list(self._custom_data)


class JaxModel(AutogradModel):
    """
    Base class for JAX-based atomistic models.

    Subclasses implement :py:meth:`forward` instead of
    :py:meth:`BaseModel.execute_inner`. Compared to :py:class:`BaseModel`, this class:

    - gives the systems to :py:meth:`forward` as :py:class:`JaxSystem`, with all arrays
      (positions, cell, pair vectors, custom data) as :py:class:`jax.Array`, shared with
      the engine through DLPack whenever possible;
    - expects :py:meth:`forward` to return :py:class:`JaxTensorMap`, and converts them
      to :py:class:`metatensor.TensorMap`;
    - computes the gradients with respect to ``"positions"`` and ``"strain"`` of
      outputs with ``sample_kind="system"`` with :py:func:`jax.vjp`, unless the model
      already declares these gradients in its :py:meth:`capabilities`. This is done
      for the standard quantities which support gradients and for any custom quantity,
      and these gradients are added to the capabilities of the model. The pair vectors
      given by the engine are used as-is, and the gradients flow from them to the
      positions and cell;
    - collects the pair lists and inputs requested by all the sub-modules of this model
      in :py:meth:`requested_pair_lists` and :py:meth:`requested_inputs`. Sub-modules
      are found by walking the attributes of the model (including lists, tuples and
      dicts, and Flax/Equinox modules), and any object with a
      ``requested_pair_lists()`` or ``requested_inputs()`` method is included.

    :py:meth:`forward` runs eagerly, but can call :py:func:`jax.jit`-compiled functions
    on the arrays of the systems. These functions should take the pair indices as
    arrays rather than :py:class:`metatensor.Labels`, and padding the inputs to a few
    fixed sizes avoids a re-compilation every time the number of atoms or pairs changes.

    When the model ``dtype`` is ``"float64"``, ``jax_enable_x64`` must be set, otherwise
    JAX silently converts the data to ``float32``.
    """

    def _named_submodules(self):
        seen = set()
        stack = [("", self)]
        while stack:
            name, obj = stack.pop()
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            yield name, obj

            children = []
            if isinstance(obj, (list, tuple)):
                children = [(str(i), child) for i, child in enumerate(obj)]
            elif isinstance(obj, dict):
                children = [(str(key), child) for key, child in obj.items()]
            elif hasattr(obj, "__dict__") and not isinstance(obj, type):
                children = list(vars(obj).items())

            prefix = "" if name == "" else f"{name}."
            stack.extend(
                (f"{prefix}{child_name}", child)
                for child_name, child in reversed(children)
                if isinstance(child, (list, tuple, dict))
                or hasattr(child, "requested_pair_lists")
                or hasattr(child, "requested_inputs")
                or _is_jax_module(child)
            )

    @abc.abstractmethod
    def forward(
        self,
        systems: list[JaxSystem],
        selected_atoms: Optional[Labels],
        outputs: list[Quantity],
    ) -> list[JaxTensorMap]:
        """
        Run the model and compute the requested outputs.

        :param systems: systems to run the model on
        :param selected_atoms: optional selection of atoms to compute outputs for, or
            ``None`` to use all atoms
        :param outputs: outputs the model should compute. The gradients computed with
            autograd are removed from these requests.
        :return: the computed outputs, one :py:class:`JaxTensorMap` per requested
            output, in the same order
        """

    def __call__(self, systems, selected_atoms, outputs):
        return self.forward(systems, selected_atoms, outputs)

    def execute_inner(
        self,
        systems: list[System],
        selected_atoms: Optional[Labels],
        requested_outputs: list[Quantity],
    ) -> list[TensorMap]:
        """
        Run the model by calling :py:meth:`forward`, and compute the gradients that
        the model does not compute explicitly with :py:func:`jax.vjp`.

        This method should not be called directly, use :py:func:`execute_model`
        instead. It should also not be overridden by subclasses.
        """
        inner_outputs, autograd_gradients = self._split_gradients(requested_outputs)

        if self._raw_capabilities().dtype == "float64" and not jax.config.read(
            "jax_enable_x64"
        ):
            raise ValueError(
                "this model uses float64, but JAX would silently convert the data to "
                "float32: set `jax.config.update('jax_enable_x64', True)` first"
            )

        for system in systems:
            system.arrays_backend = "jax"

        data = [_system_data(system) for system in systems]
        positions = [system.positions for system in systems]
        strains = [jnp.eye(3, dtype=p.dtype) for p in positions]

        # `forward` is called inside `run`, which is traced by `jax.vjp`: the outputs
        # contain tracers there. The metadata is captured on the side, and the values
        # come back as plain arrays from `jax.vjp`. Only the values of the outputs with
        # autograd gradients are differentiated, the others (explicit gradients,
        # integer outputs, ...) go through `has_aux`.
        structure = []

        def run(positions, strains):
            jax_systems = [
                _jax_system(system, d, p, s)
                for system, d, p, s in zip(
                    systems, data, positions, strains, strict=True
                )
            ]
            outputs = self(jax_systems, selected_atoms, inner_outputs)
            if not isinstance(outputs, (list, tuple)) or not all(
                isinstance(output, JaxTensorMap) for output in outputs
            ):
                raise TypeError("JaxModel.forward() must return a list of JaxTensorMap")
            if len(outputs) != len(requested_outputs):
                raise ValueError(
                    f"model returned {len(outputs)} outputs, "
                    f"but {len(requested_outputs)} were requested"
                )

            differentiated, other = [], []
            for output, gradients in zip(outputs, autograd_gradients, strict=True):
                for block in output.blocks:
                    target = differentiated if gradients else other
                    target.append(block.values)
                    other.extend(g.values for g in block.gradients.values())
            structure.append(outputs)
            return differentiated, other

        if not any(autograd_gradients):
            _, other = run(positions, strains)
            values = iter(other)
            return [_to_tensor_map(output, values) for output in structure[0]]

        differentiated, vjp, other = jax.vjp(run, positions, strains, has_aux=True)
        (outputs,) = structure

        results = []
        other = iter(other)
        index = 0
        for output, gradients in zip(outputs, autograd_gradients, strict=True):
            blocks = []
            for block in output.blocks:
                if not gradients:
                    blocks.append(_to_block(block, other))
                    continue

                new_block = _to_block(block, other, differentiated[index])
                for parameter, gradient in _autograd_gradients(
                    block, differentiated, index, gradients, vjp
                ):
                    new_block.add_gradient(parameter, gradient)
                blocks.append(new_block)
                index += 1
            results.append(TensorMap(output.keys, blocks))

        return results


def _is_jax_module(obj) -> bool:
    """Flax NNX and Equinox modules, which can contain sub-modules."""
    return any(
        cls.__module__.split(".")[0] in ("flax", "equinox") and cls.__name__ == "Module"
        for cls in type(obj).__mro__
    )


def _mts_array_to_jax(array, like: jax.Array) -> jax.Array:
    """
    Get the data in a ``mts_array_t`` as a :py:class:`jax.Array` on the device of
    ``like``, without copying when possible. This works for arrays created by any
    language, as for ``_mts_array_to_torch`` in ``_torch.py``.
    """
    (device,) = like.devices()
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
    return jax.device_put(jnp.from_dlpack(DLPackArray(tensor)), device)


def _system_data(system: System) -> dict:
    """
    Everything ``forward`` may need from ``system``, with JAX arrays, read once outside
    of the traced function. The pairs are stored with their indices to register the
    pair vectors with autograd in ``_jax_system``.
    """
    positions = system.positions
    pairs = []
    for options in system.known_pairs():
        block = system.pairs(options)
        samples = np.asarray(block.samples.values)
        pairs.append(
            (
                options,
                block,
                _mts_array_to_jax(block._raw_values, positions),
                jnp.asarray(samples[:, 0]),
                jnp.asarray(samples[:, 1]),
                jnp.asarray(samples[:, 2:5], dtype=positions.dtype),
            )
        )

    custom_data = {}
    for name in system.known_custom_data():
        tensor = system.custom_data(name)
        custom_data[name] = JaxTensorMap(
            keys=tensor.keys,
            blocks=[
                JaxBlock(
                    values=_mts_array_to_jax(block._raw_values, positions),
                    samples=block.samples,
                    components=block.components,
                    properties=block.properties,
                )
                for block in tensor.blocks()
            ],
        )

    return {"pairs": pairs, "custom_data": custom_data}


def _jax_system(system: System, data: dict, positions, strain) -> JaxSystem:
    """
    Build the :py:class:`JaxSystem` given to ``forward``, applying the ``strain`` to
    the positions and cell.

    The engine's pair vectors are registered with autograd as
    ``vectors + (r - stop_gradient(r))`` with
    ``r = positions[second] - positions[first] + cell_shifts @ cell``: the values are
    exactly the engine's vectors, and the derivatives are those of ``r``. Unlike a
    ``custom_vjp``, this also supports forward mode and higher order derivatives.
    """
    positions = positions @ strain
    cell = system.cell @ strain

    pairs = []
    for options, block, vectors, first, second, shifts in data["pairs"]:
        r = positions[second] - positions[first] + shifts @ cell
        r = r - jax.lax.stop_gradient(r)
        pairs.append(
            (
                options,
                JaxBlock(
                    values=vectors + r.reshape(vectors.shape),
                    samples=block.samples,
                    components=block.components,
                    properties=block.properties,
                ),
            )
        )

    return JaxSystem(
        system.length_unit,
        system.types,
        positions,
        cell,
        system.pbc,
        pairs,
        data["custom_data"],
    )


try:
    from metatensor._data._array import _is_jax_array  # noqa: F401

    METATENSOR_STORES_JAX = True
except ImportError:
    METATENSOR_STORES_JAX = False


def _mts_values(array):
    """
    Values to store in a :py:class:`TensorBlock`: the JAX array itself if metatensor
    supports JAX arrays, otherwise a NumPy array (a zero-copy view on CPU, and a copy
    to the host on GPU).
    """
    if METATENSOR_STORES_JAX:
        return array
    if array.devices().pop().platform == "cpu":
        return np.from_dlpack(array)
    return np.asarray(array)


def _to_block(block: JaxBlock, other, values=None) -> TensorBlock:
    """
    Convert ``block`` to a :py:class:`TensorBlock`, taking the concrete arrays from
    ``values`` (for differentiated outputs) or the ``other`` iterator, in the order
    they were stored in ``run``.
    """
    new_block = TensorBlock(
        values=_mts_values(next(other) if values is None else values),
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )
    for parameter, gradient in block.gradients.items():
        new_block.add_gradient(
            parameter,
            TensorBlock(
                values=_mts_values(next(other)),
                samples=gradient.samples,
                components=gradient.components,
                properties=gradient.properties,
            ),
        )
    return new_block


def _to_tensor_map(output: JaxTensorMap, values) -> TensorMap:
    return TensorMap(output.keys, [_to_block(block, values) for block in output.blocks])


def _autograd_gradients(block, differentiated, index, gradients, vjp):
    """
    Compute the ``gradients`` of the per-system ``block`` (with values
    ``differentiated[index]``) with respect to the positions and/or strain of all
    systems, from the ``vjp`` function of ``jax.vjp``.

    The cotangents are one-hot along the components and properties of this block (and
    zero for all other differentiated blocks), and the same for all samples, since each
    system has a single sample and the different systems are independent. All the
    entries along components and properties are computed in one ``jax.vmap``-ed call.
    """
    values = differentiated[index]
    n_samples = values.shape[0]
    values_shape = values.shape[1:]
    size = int(np.prod(values_shape))

    basis = jnp.eye(size, dtype=values.dtype).reshape(size, 1, *values_shape)
    basis = jnp.broadcast_to(basis, (size, n_samples, *values_shape))

    def cotangent(one_hot):
        cotangents = [jnp.zeros_like(d) for d in differentiated]
        cotangents[index] = one_hot
        return vjp(cotangents)

    # each entry has a leading dimension of size `size`
    positions_grads, strain_grads = jax.vmap(cotangent)(basis)

    # move the components and properties to the end
    def finish(grad):
        return jnp.moveaxis(grad, 0, -1).reshape(*grad.shape[1:], *values_shape)

    sample_systems = np.asarray(block.samples.column("system")).tolist()
    results = []
    if "positions" in gradients:
        # positions_grads[system] has shape (size, n_atoms, 3)
        grad_samples = [
            np.stack(
                np.broadcast_arrays(sample, system, np.arange(n_atoms, dtype=np.int32)),
                axis=1,
            ).astype(np.int32)
            for sample, system in enumerate(sample_systems)
            for n_atoms in [positions_grads[system].shape[1]]
        ]
        grad_values = [finish(positions_grads[system]) for system in sample_systems]
        results.append(
            (
                "positions",
                TensorBlock(
                    values=_mts_values(
                        jnp.concatenate(grad_values)
                        if grad_values
                        else jnp.zeros((0, 3, *values_shape), dtype=values.dtype)
                    ),
                    samples=Labels(
                        ["sample", "system", "atom"],
                        np.concatenate(grad_samples)
                        if grad_samples
                        else np.zeros((0, 3), dtype=np.int32),
                        assume_unique=True,
                    ),
                    components=[Labels.range("xyz", 3)] + block.components,
                    properties=block.properties,
                ),
            )
        )

    if "strain" in gradients:
        grad_values = [finish(strain_grads[system]) for system in sample_systems]
        results.append(
            (
                "strain",
                TensorBlock(
                    values=_mts_values(
                        jnp.stack(grad_values)
                        if grad_values
                        else jnp.zeros((0, 3, 3, *values_shape), dtype=values.dtype)
                    ),
                    samples=Labels.range("sample", n_samples),
                    components=[Labels.range("xyz_1", 3), Labels.range("xyz_2", 3)]
                    + block.components,
                    properties=block.properties,
                ),
            )
        )

    return results
