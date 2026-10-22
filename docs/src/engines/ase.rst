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

:py:class:`metatomic_ase.MetatomicCalculator` supports all the features
described above, and is designed to be used as a drop-in replacement for any ASE
calculator in any ASE workflow. You can also check the :ref:`corresponding
tutorial <atomistic-tutorial-md>`. To compute rotationally-averaged energies,
forces, and stresses with non-equivariant architectures, wrap the model with
:py:meth:`metatomic.torch.o3.SymmetrizedModel.wrap` before passing it to the
calculator.

.. _ase-integration-api:

API documentation
-----------------

.. _calculator: https://ase-lib.org/ase/calculators/calculators.html

.. autoclass:: metatomic_ase.MetatomicCalculator
    :show-inheritance:
    :members:
