.. _migrate-to-torch-model:

Migrating to ``metatomic.TorchModel``
=====================================

Models written for ``metatomic.torch`` (a :py:class:`metatomic.torch.ModelInterface`
module, wrapped in an :py:class:`metatomic.torch.AtomisticModel` and saved as a
TorchScript file) can be used with the new :py:class:`metatomic.TorchModel` API in two
ways:

- **wrap the existing model** with ``metatomic_torch.legacy.LegacyModel``, without
  changing any of its code;
- **port the model** to inherit from :py:class:`metatomic.TorchModel`, following the
  correspondence table below.

Wrapping an existing model
--------------------------

``LegacyModel`` turns an :py:class:`AtomisticModel <metatomic.torch.AtomisticModel>`,
either a Python object or a TorchScript file, into a :py:class:`metatomic.TorchModel`
that can be executed with :py:func:`metatomic.execute_model`:

.. code-block:: python

    import metatomic
    from metatomic_torch.legacy import LegacyModel, system_with_pairs

    model = LegacyModel.load("model.pt", extensions_directory="extensions/")
    # or, from an AtomisticModel object
    model = LegacyModel(atomistic_model)

    energy = metatomic.Quantity(
        name="energy", unit="eV", sample_kind="system", gradients=["positions"]
    )
    # the systems must contain the pair lists requested by the model
    system = system_with_pairs(
        types, positions, cell, pbc, model.requested_pair_lists([energy]),
        length_unit=model.capabilities().length_unit,
        model_length_unit=model.capabilities().length_unit,
    )
    (energy,) = metatomic.execute_model(model, [system], [energy])
    forces = -energy.block().gradient("positions").values

The gradients of the energy are computed with autograd by
:py:class:`metatomic.TorchModel`, so the engine no longer needs to call
``backward()`` itself. Outputs and inputs with a deprecated name (e.g.
``non_conservative_forces``) are available under their current name (e.g.
``non_conservative_force``).

Units and current limitations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The positions and cell in the example are in the model's declared length unit.
``system_with_pairs`` also accepts a ``length_unit`` for the input arrays and a
``model_length_unit`` for the requested cutoffs.

.. warning::

    On the current core branch, ``execute_model`` does not include the inverse
    length conversion factor when converting positions gradients back to the
    engine's units. Until this is fixed in core, use the same length unit for the
    input systems and the model when requesting forces. Converting energy units
    alone is supported.

Some older models declare an empty length unit, which meant no length conversion.
For these models, pass the actual unit explicitly, for example
``LegacyModel.load("model.pt", length_unit="angstrom")``. Otherwise the wrapper
assumes angstrom and emits a warning. This supplies missing metadata; it does not
rescale the model parameters.

The wrapper is for Python engines. Loading a legacy TorchScript model directly
from a C++ engine requires a separate plugin integration.

Porting a model
---------------

The table below lists what changes between the two APIs. Names on the left are in
``metatomic.torch`` and ``metatensor.torch``, names on the right in ``metatomic`` and
``metatensor``.

.. list-table::
    :header-rows: 1
    :widths: 50 50

    * - ``metatomic.torch``
      - ``metatomic``
    * - ``class Model(torch.nn.Module)``, then ``AtomisticModel(model, metadata,
        capabilities)``
      - ``class Model(metatomic.TorchModel)``, which also defines ``capabilities()``
        and ``metadata()``
    * - ``forward(systems, outputs, selected_atoms)``
      - ``forward(systems, selected_atoms, outputs)``, note the different order
    * - ``outputs: Dict[str, ModelOutput]``
      - ``outputs: list[Quantity]``, with the name in ``Quantity.name``
    * - returns ``Dict[str, TensorMap]``
      - returns ``list[TensorMap]``, in the same order as ``outputs``
    * - ``ModelOutput(unit=..., sample_kind=..., explicit_gradients=[...])``
      - ``Quantity(name=..., unit=..., sample_kind=..., gradients=[...])``
    * - ``ModelOutput.per_atom``
      - ``Quantity.sample_kind``, one of ``"system"``, ``"atom"`` or ``"atom_pair"``
    * - ``ModelCapabilities(outputs={name: ModelOutput})``
      - ``ModelCapabilities(outputs=[Quantity])``, all arguments are keyword-only
    * - ``requested_neighbor_lists() -> List[NeighborListOptions]``
      - ``requested_pair_lists(requested_outputs) -> list[PairListOptions]``
    * - ``NeighborListOptions(cutoff, full_list, strict, requestor)``
      - ``PairListOptions(cutoff=..., full_list=..., strict=..., requestors=[...])``
    * - ``system.get_neighbor_list(options)``
      - ``system.pairs(options)``, with the same samples, components and properties
    * - ``requested_inputs() -> Dict[str, ModelOutput]``
      - ``requested_inputs(requested_outputs) -> list[Quantity]``
    * - ``system.get_data(name)`` / ``system.known_data()``
      - ``system.custom_data(name)`` / ``system.known_custom_data()``
    * - ``metatensor.torch.Labels``, ``TensorBlock``, ``TensorMap``
      - ``metatensor.Labels``, ``TensorBlock``, ``TensorMap``, with ``torch.Tensor``
        values and NumPy arrays for the labels
    * - ``metatensor.torch.<operation>``
      - ``metatensor.<operation>`` (from ``metatensor-operations``)
    * - ``AtomisticModel.save("model.pt")``
      - a Python script defining a ``model`` variable, loaded with
        :py:func:`metatomic.load_model` after loading the Python plugin with
        ``metatomic.load_plugin(metatomic.utils.python_plugin_path)``

``requested_pair_lists()`` and ``requested_inputs()`` now receive the outputs that
will be computed, so a model can request different inputs for different outputs. As
before, these methods can be defined on any sub-module, and the requests of all
sub-modules are merged.

Gradients
^^^^^^^^^

With ``metatomic.torch``, the engine computes gradients by calling ``backward()`` on
the outputs, unless the model declares them in ``ModelOutput.explicit_gradients``.

:py:class:`metatomic.TorchModel` instead computes the gradients with respect to
``"positions"`` and ``"strain"`` itself, with autograd, for the ``energy``,
``energy_ensemble`` and ``energy_uncertainty`` outputs and for custom outputs with
``sample_kind="system"``. These gradients are added to the model capabilities
automatically, and are removed from the ``outputs`` given to ``forward()``.

If a model computes some gradients explicitly (for example forces from an analytical
expression), it should list them in ``Quantity.gradients`` in its capabilities.
:py:class:`metatomic.TorchModel` then only uses autograd for the remaining gradients.
For autograd to give the right answer, outputs must be computed from
:py:attr:`System.positions <metatomic.System.positions>`,
:py:attr:`System.cell <metatomic.System.cell>` and the pair vectors in
:py:meth:`System.pairs() <metatomic.System.pairs>`. There is no need to call
``register_autograd_neighbors()``.

Non-conservative outputs (``non_conservative_force`` and ``non_conservative_stress``)
are separate outputs, not gradients, and are not affected by this.

TorchScript
^^^^^^^^^^^

:py:class:`metatomic.TorchModel` runs in eager PyTorch and can not be scripted or
compiled as a whole. Sub-modules and functions can still use
:py:func:`torch.jit.script` or :py:func:`torch.compile`, which is also how
``LegacyModel`` runs models saved as TorchScript files.

Existing explicit-gradient implementations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Metatrain's LLPR model assembles ``energy_ensemble`` positions and strain gradients
with ``_systems_with_grad`` and ``_add_energy_ensemble_gradients``, calling
``torch.autograd.grad`` for each ensemble member. A port to ``TorchModel`` can
replace this assembly with its automatic gradients. Wrapping the existing model
preserves its declared explicit gradients and does not remove this code.

Metatrain's experimental SPACE model uses ``["sample", "atom"]`` samples for its
explicit positions gradients. Core requires ``["sample", "system", "atom"]``;
these labels must be updated when porting that model. This incompatibility was
confirmed with a minimal output carrying SPACE's label schema through
``execute_model(check_consistency=True)``, rather than by running the full SPACE
model.

The runnable examples in ``python/metatomic_core/tests/model/split_gradients.py``
compare explicit positions gradients plus autograd strain gradients against an
all-autograd model, including ``energy_ensemble``. They also demonstrate that a
separate non-conservative force output does not change the energy gradients.
Run them from the repository root with::

    python -m pytest python/metatomic_core/tests/model/split_gradients.py

The real-model checks in ``regtests/tests/legacy_wrapper.py`` cover the stored
PET-MAD, FlashMD and ShiftML references, as well as eager and saved versions of the
pure-PyTorch and custom-extension LJ models. They compare LJ energy gradients to
the legacy engine's backward pass, and compare the pure model's non-conservative
outputs using the same random seed in both engines::

    python -m pytest regtests/tests/legacy_wrapper.py --import-mode=append
