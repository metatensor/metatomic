.. _engine-ase:

ASE
===

.. list-table::
   :header-rows: 1

   * - Official website
     - How is metatomic supported?
   * - https://wiki.fysik.dtu.dk/ase/
     - As part of the ``metatomic-torch`` package

Supported model outputs
^^^^^^^^^^^^^^^^^^^^^^^

.. py:currentmodule:: metatomic.torch.ase_calculator

- the :ref:`energy <energy-output>` output is supported and fully integrated
  with ASE calculator interface (i.e. :py:meth:`ase.Atoms.get_potential_energy`,
  :py:meth:`ase.Atoms.get_forces`, …);
- arbitrary outputs can be computed for any :py:class:`ase.Atoms` using
  :py:meth:`MetatomicCalculator.run_model`;

How to install the code
^^^^^^^^^^^^^^^^^^^^^^^

The code is available in the ``metatomic-torch`` package, in the
:py:class:`metatomic.torch.ase_calculator.MetatomicCalculator` class.

How to use the code
^^^^^^^^^^^^^^^^^^^

See the :ref:`corresponding tutorial <atomistic-tutorial-md>`, and API
documentation of the :py:class:`MetatomicCalculator` class.
