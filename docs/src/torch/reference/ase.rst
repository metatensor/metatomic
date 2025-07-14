Atomic Simulation Environment (ASE) integration
===============================================

.. py:currentmodule:: metatomic.torch

The code in ``metatomic.torch.ase_calculator`` defines a class that
allows using a :py:class:`AtomisticModel` which predicts the energy and forces of a
system as an ASE `calculator`_; enabling the use of machine learning interatomic
potentials to drive calculations compatible with ASE calculators.

Additionally, it allows using arbitrary models with prediction targets which are
not just the energy, through the
:py:meth:`ase_calculator.MetatomicCalculator.run_model` function.

.. _calculator: https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html

.. autoclass:: metatomic.torch.ase_calculator.MetatomicCalculator
    :show-inheritance:
    :members:
