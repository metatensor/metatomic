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
