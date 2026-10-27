.. _engine-ase:

ASE
===

.. list-table::
   :header-rows: 1

   * - Official website
     - How is metatomic supported?
   * - https://ase-lib.org/
     - Via the ``metatomic-ase`` package

Supported model outputs
^^^^^^^^^^^^^^^^^^^^^^^

.. py:currentmodule:: metatomic_ase

- the :ref:`energy <energy-quantity>`, non-conservative :ref:`forces
  <non-conservative-force-quantity>` and :ref:`stress
  <non-conservative-stress-quantity>` including their :ref:`variants
  <quantity-variants>` are supported and fully integrated with ASE calculator
  interface (i.e. :py:meth:`ase.Atoms.get_potential_energy`,
  :py:meth:`ase.Atoms.get_forces`, …);
- arbitrary outputs can be computed for any :py:class:`ase.Atoms` using
  :py:meth:`MetatomicCalculator.run_model`;


How to install the code
^^^^^^^^^^^^^^^^^^^^^^^

The code is available in the ``metatomic-ase`` package, which can be installed
using ``pip install metatomic-ase``.

How to use the code
^^^^^^^^^^^^^^^^^^^

We offer two ASE calculators: :py:class:`metatomic_ase.MetatomicCalculator` is
the default one, and support all the features described above, while
:py:class:`metatomic_ase.SymmetrizedCalculator` is a wrapper around the former
that allows to compute rotationally-averaged energies, forces, and stresses for
non-equivariant architectures. Both calculators are designed to be used as
drop-in replacements for any ASE calculator, and can be used in any ASE
workflow. You can also check the :ref:`corresponding tutorial
<atomistic-tutorial-md>`.

Core models
^^^^^^^^^^^

``MetatomicCalculator`` accepts :py:class:`metatomic.BaseModel` instances,
including :py:class:`metatomic.NumpyModel`, :py:class:`metatomic.TorchModel`, and
models loaded through Core plugins. Core models are evaluated with
:py:func:`metatomic.execute_model`; the calculator reads the returned position
and strain gradients to obtain forces and stress. TorchModel computes missing
supported gradients with autograd. ASE coordinates remain in angstrom and
energies in eV; Core handles conversion to and from model units.

.. code-block:: python

    from metatomic import Quantity
    from metatomic_ase import MetatomicCalculator

    # A Core BaseModel instance can also be passed directly.
    calculator = MetatomicCalculator(
        "model.py", plugin_name="python", model_options={}
    )
    atoms.calc = calculator
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    atomic, total = calculator.run_model(atoms, [
        Quantity(name="energy", unit="eV", sample_kind="atom"),
        Quantity(name="energy", unit="eV", sample_kind="system",
                 gradients=["positions", "strain"]),
    ])

For Core models, ``run_model`` accepts a sequence of quantities and returns a
list of :py:class:`metatensor.TensorMap` objects in the same order. A mapping of
aliases to quantities returns a mapping of aliases to TensorMaps. Use Core
:py:class:`metatensor.Labels` for ``selected_atoms``. Values retain the backend
returned by the model; ordinary ASE results are NumPy arrays.

``additional_outputs`` is a mapping of arbitrary aliases to quantities. Aliases
allow requesting the same name with different sample kinds. These outputs are
stored in ``calculator.additional_outputs``. Custom outputs, including
pair-sampled quantities, are available through these interfaces.

Input and pair-list requests are evaluated for each output request. Standard
ASE inputs (mass, momentum, velocity, charge, spin multiplicity, initial magnetic
moments, initial charges, and tags) and ``ase::arrays::`` / ``ase::info::`` inputs
are supported. For other inputs, pass ``input_provider(atoms, quantity)`` returning
a Core TensorMap with unit metadata, appropriate labels, dtype and device. Return
``None`` to use a built-in input. Changes to inputs used by cached calculations
invalidate the ASE cache, including callback-provided inputs.

Core neighbor searches currently use CPU vesin with Verlet caching, transferring
pair vectors to the selected device when necessary. ``neighbor_skin`` remains in
angstrom; model cutoffs are converted before searching. The legacy GPU neighbor
backends are unchanged. Core device names ``metal`` and ``rocm`` are translated
to Torch device names when preparing GPU arrays. External model parameter/device
configuration belongs in the plugin's ``model_options``; only Torch modules are
moved with ``.to(device)``.

Existing ``AtomisticModel`` objects, scripted models, and model paths without
``plugin_name`` or ``model_options`` continue using the legacy Torch interface.
Their ``run_model`` and ``additional_outputs`` still accept dictionaries of
``ModelOutput`` objects and return Torch TensorMaps. Legacy device selection,
heat-flux wrappers, and neighbor implementations are preserved. Set
``plugin_name`` or ``model_options`` explicitly when loading a Core model source.
The default package dependencies retain Torch for installation compatibility;
the Core execution path itself does not import it.

``compute_energy`` preserves single-system and batch return formats, and
``SymmetrizedCalculator`` supports both model interfaces. Core serialization with
``todict`` requires a model source and does not support an input-provider callback.

.. _ase-integration-api:

API documentation
-----------------

.. _calculator: https://ase-lib.org/ase/calculators/calculators.html

.. autoclass:: metatomic_ase.MetatomicCalculator
    :show-inheritance:
    :members:

.. autoclass:: metatomic_ase.SymmetrizedCalculator
    :show-inheritance:
    :members:
