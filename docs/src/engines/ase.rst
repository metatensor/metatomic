.. _engine-ase:

ASE
===

.. list-table::
   :header-rows: 1

   * - Official website
     - How is metatomic supported?
   * - https://ase-lib.org/
     - As part of the ``metatomic-torch`` package

Supported model outputs
^^^^^^^^^^^^^^^^^^^^^^^

.. py:currentmodule:: metatomic.torch.ase_calculator

- the :ref:`energy <energy-output>`, non-conservative :ref:`forces
  <non-conservative-forces-output>` and :ref:`stress <non-conservative-stress-output>`
  including their :ref:`variants <output-variants>` are supported and fully integrated
  with ASE calculator interface (i.e. :py:meth:`ase.Atoms.get_potential_energy`,
  :py:meth:`ase.Atoms.get_forces`, …);
- arbitrary outputs can be computed for any :py:class:`ase.Atoms` using
  :py:meth:`MetatomicCalculator.run_model`;
- for non-equivariant architectures like
  `PET <https://docs.metatensor.org/metatrain/latest/architectures/pet.html>`_,
  rotationally-averaged energies, forces, and stresses can be computed using
  :py:class:`metatomic.torch.ase_calculator.SymmetrizedCalculator`.

How to install the code
^^^^^^^^^^^^^^^^^^^^^^^

The code is available in the ``metatomic-torch`` package, in the
:py:class:`metatomic.torch.ase_calculator.MetatomicCalculator` class.

Supported model inputs
^^^^^^^^^^^^^^^^^^^^^^

The ASE calculator can provide per-atom inputs (e.g. ``"charges"``,
``"momenta"``, ``"velocities"``) as well as the following **system-level**
integer inputs used for model conditioning:

.. list-table::
   :header-rows: 1
   :widths: 2 3 5

   * - Input name
     - Default
     - How to set
   * - ``"charge"``
     - ``0``
     - ``atoms.info["charge"] = <int>``
   * - ``"spin"``
     - ``1``
     - ``atoms.info["spin"] = <int>``

``"charge"`` is the total charge of the simulation cell in elementary
charges.  ``"spin"`` is the spin multiplicity (2S+1) — a singlet is
``spin=1``, a doublet is ``spin=2``, a triplet is ``spin=3``, and so on.
Both values are read as integers from ``atoms.info`` and stored in the
system as the model's floating-point dtype (float32 or float64); the model
converts them back to integers internally for the embedding lookup.

How to use the code
^^^^^^^^^^^^^^^^^^^

See the :ref:`corresponding tutorial <atomistic-tutorial-md>`, and API
documentation of the :py:class:`MetatomicCalculator` class.
