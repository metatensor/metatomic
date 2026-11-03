import abc
import copy
import ctypes
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from ctypes_dlpack import DLDevice, DLManagedTensorVersioned, DLPackArray, DLPackVersion
from metatensor import Labels, TensorBlock, TensorMap
from metatensor._status import check_status

from .._quantity import Quantity
from .._system import PairListOptions, System
from .._utils import unit_conversion_factor
from ._autograd import AutogradModel


try:
    # make sure TensorMap and TensorBlock are JAX pytrees, whatever the import order
    from metatensor._jax import register_pytrees

    register_pytrees()
except ImportError:
    pass


def _check_metatensor_supports_jax():
    try:
        import metatensor._jax  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "JaxModel requires a version of metatensor-core which can store JAX "
            "arrays in TensorMap, and use TensorMap with JAX transformations"
        ) from e


class _StaticLabels:
    """
    Wrapper making :py:class:`Labels` usable as static data in a JAX pytree, which
    must be hashable: the hash uses the names and size, the equality all the values.
    """

    __slots__ = ["labels"]

    def __init__(self, labels):
        self.labels = labels

    def __hash__(self):
        return hash((tuple(self.labels.names), len(self.labels)))

    def __eq__(self, other):
        return isinstance(other, _StaticLabels) and self.labels == other.labels


@jax.tree_util.register_pytree_node_class
class JaxSystem:
    """
    A system with :py:class:`jax.Array` data, given to :py:meth:`JaxModel.forward`.

    This is a JAX pytree: the positions, cell, pair vectors and custom data are
    leaves, while the atomic types, periodic boundary conditions, length unit and the
    samples of the pair lists are static data. Systems can be given to and returned
    from :py:func:`jax.jit`, :py:func:`jax.grad`, :py:func:`jax.vmap`, ... and in
    particular to :py:meth:`JaxModel.evaluate`.

    The pair vectors are computed from the positions and cell, using the samples of
    the pair lists. When the vectors are given (for example by a simulation engine),
    they are used as-is, and the derivatives are still those of
    ``positions[second] - positions[first] + cell_shifts @ cell``.

    :param types: atomic types of all atoms
    :param positions: positions of all atoms, with shape ``(n_atoms, 3)``
    :param cell: unit cell of the system, with shape ``(3, 3)``
    :param pbc: periodic boundary conditions in each direction
    :param length_unit: unit of the positions and cell
    :param pairs: pair lists in this system, as a dictionary from the options to the
        samples of the pairs (``first_atom``, ``second_atom``, ``cell_shift_a``,
        ``cell_shift_b`` and ``cell_shift_c``), or to a tuple of the samples and the
        pair vectors, with shape ``(n_pairs, 3, 1)``
    :param custom_data: custom data in this system, indexed by name. The unit of the
        data can be given with ``TensorMap.set_info("unit", ...)``
    """

    def __init__(
        self,
        types,
        positions,
        cell,
        pbc,
        *,
        length_unit: str = "angstrom",
        pairs: Optional[dict] = None,
        custom_data: Optional[dict] = None,
    ):
        self._types = tuple(int(t) for t in np.asarray(types).reshape(-1))
        self._pbc = tuple(bool(p) for p in np.asarray(pbc).reshape(-1))
        self.positions = positions
        self.cell = cell
        self.length_unit = length_unit

        self._pairs = {}
        for options, value in (pairs or {}).items():
            samples, vectors = value if isinstance(value, tuple) else (value, None)
            if isinstance(samples, _StaticLabels):
                samples = samples.labels
            self._pairs[options] = (samples, vectors)

        self._custom_data = dict(custom_data or {})

    @classmethod
    def from_system(cls, system: System) -> "JaxSystem":
        """
        Create a :py:class:`JaxSystem` with the data of a :py:class:`System`, sharing
        memory with it whenever possible.
        """
        system.arrays_backend = "jax"
        positions = system.positions

        pairs = {}
        for options in system.known_pairs():
            block = system.pairs(options)
            pairs[options] = (
                block.samples,
                _mts_array_to_jax(block._raw_values, positions),
            )

        custom_data = {}
        for name in system.known_custom_data():
            tensor = system.custom_data(name)
            blocks = [
                TensorBlock(
                    _mts_array_to_jax(block._raw_values, positions),
                    block.samples,
                    block.components,
                    block.properties,
                )
                for block in tensor.blocks()
            ]
            custom_data[name] = TensorMap(tensor.keys, blocks)

        return cls(
            system.types,
            positions,
            system.cell,
            system.pbc,
            length_unit=system.length_unit,
            pairs=pairs,
            custom_data=custom_data,
        )

    def tree_flatten(self):
        options = tuple(self._pairs.keys())
        children = (
            self.positions,
            self.cell,
            tuple(vectors for _, vectors in self._pairs.values()),
            tuple(self._custom_data.values()),
        )
        static = (
            self._types,
            self._pbc,
            self.length_unit,
            options,
            tuple(_StaticLabels(samples) for samples, _ in self._pairs.values()),
            tuple(self._custom_data.keys()),
        )
        return children, static

    @classmethod
    def tree_unflatten(cls, static, children):
        types, pbc, length_unit, options, samples, names = static
        positions, cell, vectors, custom_data = children
        system = cls.__new__(cls)
        system._types, system._pbc, system.length_unit = types, pbc, length_unit
        system.positions, system.cell = positions, cell
        system._pairs = {
            o: (s.labels, v) for o, s, v in zip(options, samples, vectors, strict=True)
        }
        system._custom_data = dict(zip(names, custom_data, strict=True))
        return system

    def replace(self, *, positions=None, cell=None) -> "JaxSystem":
        """
        Get a copy of this system with new ``positions`` and/or ``cell``, keeping the
        same pair lists (the pair vectors are computed from the new positions).
        """
        system = copy.copy(self)
        if positions is not None:
            system.positions = positions
        if cell is not None:
            system.cell = cell
        return system

    @property
    def types(self):
        return np.array(self._types, dtype=np.int32)

    @property
    def pbc(self):
        return np.array(self._pbc, dtype=bool)

    def __len__(self) -> int:
        return len(self._types)

    def pairs(self, options: PairListOptions) -> TensorBlock:
        """Get the pair list with the given ``options``, as a :py:class:`TensorBlock`"""
        found = [pair for known, pair in self._pairs.items() if known == options]
        if not found:
            raise ValueError(f"no pair list with {options} in this system")
        ((samples, vectors),) = found

        values = np.asarray(samples.values)
        shifts = jnp.asarray(values[:, 2:5], dtype=self.positions.dtype)
        r = (
            self.positions[values[:, 1]]
            - self.positions[values[:, 0]]
            + shifts @ self.cell
        )
        r = r.reshape(-1, 3, 1)
        if vectors is not None:
            # keep the values of the given vectors, with the derivatives of `r`
            r = vectors + (r - jax.lax.stop_gradient(r))

        return TensorBlock(
            r, samples, [Labels.range("xyz", 3)], Labels.range("distance", 1)
        )

    def known_pairs(self) -> list[PairListOptions]:
        return list(self._pairs.keys())

    def custom_data(self, name: str) -> TensorMap:
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
    - expects :py:meth:`forward` to return :py:class:`metatensor.TensorMap` with JAX
      arrays. These are JAX pytrees, so they can be created inside JAX transformations;
    - computes the gradients with respect to ``"positions"`` and ``"strain"`` of
      outputs with ``sample_kind="system"`` with :py:func:`jax.vjp`, unless the model
      already declares these gradients in its :py:meth:`capabilities`. This is done
      for the standard quantities which support gradients and for any custom quantity,
      and these gradients are added to the capabilities of the model;
    - collects the pair lists and inputs requested by all the sub-modules of this model
      in :py:meth:`requested_pair_lists` and :py:meth:`requested_inputs`. Sub-modules
      are found by walking the attributes of the model (including lists, tuples and
      dicts, and Flax/Equinox modules), and any object with a
      ``requested_pair_lists()`` or ``requested_inputs()`` method is included;
    - can be used directly from JAX with :py:meth:`evaluate`, which works inside
      :py:func:`jax.jit`, :py:func:`jax.grad`, :py:func:`jax.vmap`, ...

    :py:meth:`forward` runs eagerly when called through :py:func:`execute_model`, but
    can call :py:func:`jax.jit`-compiled functions on the arrays of the systems. These
    functions should take the pair indices as arrays rather than
    :py:class:`metatensor.Labels`, and padding the inputs to a few fixed sizes avoids a
    re-compilation every time the number of atoms or pairs changes.

    When the model ``dtype`` is ``"float64"``, ``jax_enable_x64`` must be set, otherwise
    JAX silently converts the data to ``float32``.
    """

    def __init_subclass__(cls, **kwargs):
        _check_metatensor_supports_jax()
        super().__init_subclass__(**kwargs)

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
    ) -> list[TensorMap]:
        """
        Run the model and compute the requested outputs.

        :param systems: systems to run the model on
        :param selected_atoms: optional selection of atoms to compute outputs for, or
            ``None`` to use all atoms
        :param outputs: outputs the model should compute. The gradients computed with
            autograd are removed from these requests.
        :return: the computed outputs, one :py:class:`metatensor.TensorMap` per
            requested output, in the same order
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
        systems = [JaxSystem.from_system(system) for system in systems]
        return self._execute(systems, selected_atoms, requested_outputs)

    def evaluate(
        self,
        systems: list[JaxSystem],
        outputs: list[Quantity],
        *,
        selected_atoms: Optional[Labels] = None,
    ) -> list[TensorMap]:
        """
        Run the model on JAX ``systems``, like :py:func:`execute_model` does for
        :py:class:`System`, but entirely with JAX: this function can be used inside
        :py:func:`jax.jit`, differentiated with :py:func:`jax.grad`, batched with
        :py:func:`jax.vmap`, ...

        The systems are converted to the length unit of the model, the custom data to
        the units requested by the model (using the ``"unit"`` info of each
        :py:class:`metatensor.TensorMap`), and the outputs to the requested units. The
        gradients with respect to positions are given in the length unit of the
        systems.

        The systems must contain the pair lists returned by
        :py:meth:`requested_pair_lists` for these ``outputs``. The ``outputs`` are not
        JAX types, use them from the enclosing scope when using :py:func:`jax.jit`:

        .. code-block:: python

            energy = Quantity(
                name="energy", unit="eV", sample_kind="system", gradients=["positions"]
            )
            compute = jax.jit(lambda systems: model.evaluate(systems, [energy]))

        :param systems: systems to run the model on
        :param outputs: outputs the model should compute, with their unit and
            gradients
        :param selected_atoms: optional selection of atoms to compute outputs for
        :return: one :py:class:`metatensor.TensorMap` per requested output
        """
        capabilities = self.capabilities()
        declared = []
        for request in outputs:
            output = capabilities.find_output(request)
            if output is None:
                raise ValueError(
                    f"this model can not compute '{request.name}' with sample kind "
                    f"'{request.sample_kind}'"
                )
            missing = [g for g in request.gradients if g not in output.gradients]
            if missing:
                raise ValueError(
                    f"this model can not compute gradients of '{request.name}' with "
                    f"respect to {missing}"
                )
            declared.append(output)

        inputs = {q.name: q for q in self.requested_inputs(outputs)}
        converted = []
        for system in systems:
            factor = unit_conversion_factor(
                system.length_unit, capabilities.length_unit
            )
            pairs = {
                options: (samples, None if vectors is None else vectors * factor)
                for options, (samples, vectors) in system._pairs.items()
            }
            custom_data = {}
            for name, data in system._custom_data.items():
                if name in inputs:
                    data_unit = data.get_info("unit") or ""
                    data = _scale(
                        data, unit_conversion_factor(data_unit, inputs[name].unit)
                    )
                custom_data[name] = data
            converted.append((system, factor, pairs, custom_data))

        def model_units(index, positions, cell):
            original, factor, pairs, custom_data = converted[index]
            system = original.replace(positions=positions * factor, cell=cell * factor)
            system._pairs, system._custom_data = pairs, custom_data
            return system

        results = self._execute(systems, selected_atoms, outputs, model_units)

        # explicit gradients with respect to positions are per unit of the model's
        # length, while the autograd ones are already in the unit of the systems
        _, autograd = self._split_gradients(outputs)
        length_factors = {factor for _, factor, _, _ in converted}
        if any("positions" in o.gradients for o in outputs) and len(length_factors) > 1:
            raise ValueError(
                "all systems must use the same length unit when requesting gradients "
                "with respect to positions"
            )
        length_factor = length_factors.pop() if length_factors else 1.0

        scaled = []
        for result, request, output, autograd_gradients in zip(
            results, outputs, declared, autograd, strict=True
        ):
            factor = unit_conversion_factor(output.unit, request.unit)
            explicit_positions = (
                "positions" in request.gradients
                and "positions" not in autograd_gradients
            )
            scaled.append(
                _scale(
                    result,
                    factor,
                    positions_factor=length_factor if explicit_positions else 1.0,
                )
            )
        return scaled

    def _execute(self, systems, selected_atoms, requested_outputs, transform=None):
        """
        Run the model on JAX systems, and compute the autograd gradients. If given,
        ``transform(index, positions, cell)`` creates the system given to the model
        from the (deformed) positions and cell of system ``index``.
        """
        inner_outputs, autograd_gradients = self._split_gradients(requested_outputs)

        if self._raw_capabilities().dtype == "float64" and not jax.config.read(
            "jax_enable_x64"
        ):
            raise ValueError(
                "this model uses float64, but JAX would silently convert the data to "
                "float32: set `jax.config.update('jax_enable_x64', True)` first"
            )

        if transform is None:

            def transform(index, positions, cell):
                return systems[index].replace(positions=positions, cell=cell)

        positions = [system.positions for system in systems]
        strains = [jnp.eye(3, dtype=p.dtype) for p in positions]

        def run(positions, strains):
            deformed = [
                transform(i, p @ strain, system.cell @ strain)
                for i, (system, p, strain) in enumerate(
                    zip(systems, positions, strains, strict=True)
                )
            ]
            outputs = self(deformed, selected_atoms, inner_outputs)
            if not isinstance(outputs, (list, tuple)) or not all(
                isinstance(output, TensorMap) for output in outputs
            ):
                raise TypeError(
                    "JaxModel.forward() must return a list of metatensor.TensorMap"
                )
            if len(outputs) != len(requested_outputs):
                raise ValueError(
                    f"model returned {len(outputs)} outputs, "
                    f"but {len(requested_outputs)} were requested"
                )

            # only the values of outputs with autograd gradients are differentiated,
            # the outputs are given back as auxiliary data
            differentiated = [
                block.values
                for output, gradients in zip(outputs, autograd_gradients, strict=True)
                if gradients
                for block in output.blocks()
            ]
            return differentiated, list(outputs)

        if not any(autograd_gradients):
            return run(positions, strains)[1]

        differentiated, vjp, outputs = jax.vjp(run, positions, strains, has_aux=True)

        results = []
        index = 0
        for output, gradients in zip(outputs, autograd_gradients, strict=True):
            if not gradients:
                results.append(output)
                continue

            blocks = []
            for block in output.blocks():
                new_block = _copy_block(block)
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


def _copy_block(block: TensorBlock, factor=1.0, positions_factor=1.0) -> TensorBlock:
    """a new block with the data of ``block`` (which can be a view), scaled"""
    new = TensorBlock(
        block.values * factor if factor != 1.0 else block.values,
        block.samples,
        block.components,
        block.properties,
    )
    for parameter, gradient in block.gradients():
        scale = factor * (positions_factor if parameter == "positions" else 1.0)
        new.add_gradient(parameter, _copy_block(gradient, scale))
    return new


def _scale(tensor: TensorMap, factor: float, positions_factor=1.0) -> TensorMap:
    if factor == 1.0 and positions_factor == 1.0:
        return tensor
    blocks = [_copy_block(block, factor, positions_factor) for block in tensor.blocks()]
    return TensorMap(tensor.keys, blocks)


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
                    values=jnp.concatenate(grad_values)
                    if grad_values
                    else jnp.zeros((0, 3, *values_shape), dtype=values.dtype),
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
                    values=jnp.stack(grad_values)
                    if grad_values
                    else jnp.zeros((0, 3, 3, *values_shape), dtype=values.dtype),
                    samples=Labels.range("sample", n_samples),
                    components=[Labels.range("xyz_1", 3), Labels.range("xyz_2", 3)]
                    + block.components,
                    properties=block.properties,
                ),
            )
        )

    return results
